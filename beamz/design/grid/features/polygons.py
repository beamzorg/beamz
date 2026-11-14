"""Extract meshing features from polygon geometry."""

from __future__ import annotations

from typing import cast

import numpy as np

from beamz.design.geometry.bounds import planar_geometry
from beamz.design.geometry.shapes import Shape
from beamz.design.grid.features.constraints import (
    _clamp_spacing,
    _MeshFeatureRegion,
    _OpposingFeature,
)
from beamz.design.grid.spec import AxisValues, GridSpec, OptionalAxisValues

_AXIS_NORMAL_THRESHOLD = 0.2


_EQUIVALENT_ENVELOPE_RATIO = 1.05


def _polygon_opposing_features(structure: Shape) -> list[_OpposingFeature]:
    """Return reliable local distances between opposing material faces.

    Boundary vertex spacing is deliberately excluded: refining a curved polygon's
    tessellation must not change its physical mesh target.
    """
    from shapely.geometry import LineString

    geometry = planar_geometry(structure)
    if geometry is None or not geometry.is_valid:
        return []
    if not geometry.interiors and geometry.equals(geometry.convex_hull):
        coordinates = np.asarray(geometry.exterior.coords[:-1], dtype=np.float64)
        edges = np.roll(coordinates, -1, axis=0) - coordinates
        edge_lengths = np.linalg.norm(edges, axis=1)
        normals = np.column_stack((-edges[:, 1], edges[:, 0])) / edge_lengths[:, None]
        widths = np.asarray([float(np.ptp(coordinates @ normal)) for normal in normals])
        index = int(np.argmin(widths))
        width = float(widths[index])
        if not np.isfinite(width) or width <= 0.0:
            return []
        bounds = tuple(map(float, geometry.bounds))
        return [
            _OpposingFeature(
                (bounds[0], bounds[1]),
                (bounds[2], bounds[3]),
                width,
                (float(normals[index, 0]), float(normals[index, 1])),
                "minimum convex Feret width",
            )
        ]

    def canonical_ring(ring) -> np.ndarray:
        coordinates = np.asarray(ring.coords[:-1], dtype=np.float64)
        if coordinates.shape[0] < 2:
            return coordinates
        start = min(
            range(coordinates.shape[0]),
            key=lambda index: (
                *coordinates[index],
                *coordinates[(index + 1) % coordinates.shape[0]],
                *coordinates[index - 1],
            ),
        )
        return np.roll(coordinates, -start, axis=0)

    exterior = canonical_ring(geometry.exterior)
    interiors = sorted(
        (canonical_ring(ring) for ring in geometry.interiors),
        key=lambda coordinates: tuple(coordinates.reshape(-1)),
    )
    rings = (exterior, *interiors)
    segments = []
    ring_sizes = [coordinates.shape[0] for coordinates in rings]
    for ring_id, coordinates in enumerate(rings):
        for local_id, start in enumerate(coordinates):
            end = coordinates[(local_id + 1) % coordinates.shape[0]]
            if np.linalg.norm(end - start) <= np.finfo(float).tiny:
                continue
            segments.append((start, end, ring_id, local_id))
    segments.sort(
        key=lambda segment: (
            *segment[0],
            *segment[1],
            segment[2],
            segment[3],
        )
    )
    starts = [segment[0] for segment in segments]
    ends = [segment[1] for segment in segments]
    ring_ids = [segment[2] for segment in segments]
    local_ids = [segment[3] for segment in segments]
    if len(starts) < 2:
        return []

    start_array = np.asarray(starts)
    end_array = np.asarray(ends)
    vectors = end_array - start_array
    lengths = np.linalg.norm(vectors, axis=1)
    tangents = vectors / lengths[:, None]
    # Oriented exterior and interior rings both keep material on their left.
    inward_normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    midpoints = 0.5 * (start_array + end_array)
    ring_ids_array = np.asarray(ring_ids)
    local_ids_array = np.asarray(local_ids)
    tolerance = 128.0 * np.finfo(float).eps * max(1.0, *geometry.bounds)
    geometry_with_tolerance = geometry.buffer(tolerance)
    features = []

    for index in range(len(starts)):
        normal = inward_normals[index]
        offsets = start_array - midpoints[index]
        denominators = normal[0] * vectors[:, 1] - normal[1] * vectors[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            ray_distances = (
                offsets[:, 0] * vectors[:, 1] - offsets[:, 1] * vectors[:, 0]
            ) / denominators
            segment_positions = (
                offsets[:, 0] * normal[1] - offsets[:, 1] * normal[0]
            ) / denominators
        valid = np.arange(len(starts)) != index
        same_ring = ring_ids_array == ring_ids_array[index]
        local_delta = np.abs(local_ids_array - local_ids_array[index])
        ring_size = ring_sizes[ring_ids[index]]
        adjacent = same_ring & ((local_delta <= 1) | (local_delta >= ring_size - 1))
        valid &= ~adjacent
        valid &= np.einsum("ij,j->i", inward_normals, inward_normals[index]) < -0.95
        valid &= np.abs(denominators) > tolerance
        valid &= np.abs(ray_distances) > tolerance
        valid &= segment_positions >= -1e-12
        valid &= segment_positions <= 1.0 + 1e-12
        candidates = np.flatnonzero(valid)
        if candidates.size == 0:
            continue
        candidates = candidates[
            np.argsort(np.abs(ray_distances[candidates]), kind="stable")
        ]
        first_point = midpoints[index]
        for other in candidates:
            signed_distance = float(ray_distances[other])
            distance = abs(signed_distance)
            second_point = first_point + signed_distance * normal
            connector = LineString((first_point, second_point))
            material_between = geometry_with_tolerance.covers(connector)
            gap_between = float(connector.intersection(geometry).length) <= tolerance
            if not material_between and not gap_between:
                continue
            direction = second_point - first_point
            feature_normal = direction / distance
            padding = 0.5 * distance
            feature_points = np.vstack(
                (start_array[index], end_array[index], first_point, second_point)
            )
            lower = np.min(feature_points, axis=0) - padding
            upper = np.max(feature_points, axis=0) + padding
            features.append(
                _OpposingFeature(
                    (float(lower[0]), float(lower[1])),
                    (float(upper[0]), float(upper[1])),
                    distance,
                    (float(feature_normal[0]), float(feature_normal[1])),
                    (
                        "opposing polygon faces"
                        if material_between
                        else "opposing polygon gap faces"
                    ),
                )
            )
            # The nearest full connector defines the local thickness. Farther
            # crossings are bulk polygon dimensions and should not add constraints.
            break

    return features


def _merge_mesh_feature_regions(
    regions: list[_MeshFeatureRegion],
) -> list[_MeshFeatureRegion]:
    """Canonicalize overlapping local corridors into a 1D spacing envelope."""
    if not regions:
        return []
    global_lower = tuple(
        min(region.lower[axis] for region in regions) for axis in range(3)
    )
    global_upper = tuple(
        max(region.upper[axis] for region in regions) for axis in range(3)
    )
    axis_envelopes: list[list[tuple[float, float, float, str]]] = []
    for axis in range(2):
        selected = [region for region in regions if region.spacing[axis] is not None]
        if not selected:
            axis_envelopes.append([])
            continue

        def active_spacing(
            region: _MeshFeatureRegion, selected_axis: int = axis
        ) -> float:
            value = region.spacing[selected_axis]
            assert value is not None
            return float(value)

        raw_coordinates = sorted(
            {
                value
                for region in selected
                for value in (region.lower[axis], region.upper[axis])
            }
        )
        smallest_spacing = min(active_spacing(region) for region in selected)
        coordinate_clusters: list[list[float]] = []
        for value in raw_coordinates:
            if (
                not coordinate_clusters
                or value - coordinate_clusters[-1][0] >= smallest_spacing
            ):
                coordinate_clusters.append([value])
            else:
                coordinate_clusters[-1].append(value)
        coordinates = [float(np.mean(cluster)) for cluster in coordinate_clusters]
        intervals: list[tuple[float, float, float, str]] = []
        for lower, upper in zip(coordinates[:-1], coordinates[1:], strict=True):
            if upper <= lower:
                continue
            midpoint = 0.5 * (lower + upper)
            covering = [
                region
                for region in selected
                if region.lower[axis] <= midpoint <= region.upper[axis]
            ]
            if not covering:
                continue
            spacing = min(active_spacing(region) for region in covering)
            finest = min(covering, key=active_spacing)
            reason = finest.reason
            if intervals:
                old_lower, old_upper, old_spacing, old_reason = intervals[-1]
                ratio = max(old_spacing, spacing) / min(old_spacing, spacing)
                if old_upper == lower and ratio <= 1.25:
                    intervals[-1] = (
                        old_lower,
                        upper,
                        min(old_spacing, spacing),
                        old_reason if old_spacing <= spacing else reason,
                    )
                    continue
            intervals.append((lower, upper, spacing, reason))
        axis_envelopes.append(intervals)

    x_intervals, y_intervals = axis_envelopes
    if len(x_intervals) == len(y_intervals) and all(
        max(x_item[2], y_item[2]) / min(x_item[2], y_item[2])
        <= _EQUIVALENT_ENVELOPE_RATIO
        and abs(x_item[0] - y_item[0]) <= max(x_item[2], y_item[2])
        and abs(x_item[1] - y_item[1]) <= max(x_item[2], y_item[2])
        for x_item, y_item in zip(x_intervals, y_intervals, strict=True)
    ):
        canonical = [
            (
                0.5 * (x_item[0] + y_item[0]),
                0.5 * (x_item[1] + y_item[1]),
                min(x_item[2], y_item[2]),
                x_item[3] if x_item[2] <= y_item[2] else y_item[3],
            )
            for x_item, y_item in zip(x_intervals, y_intervals, strict=True)
        ]
        axis_envelopes = [canonical, canonical]

    result = []
    for axis, intervals in enumerate(axis_envelopes):
        for lower, upper, spacing, reason in intervals:
            region_lower = list(global_lower)
            region_upper = list(global_upper)
            region_lower[axis], region_upper[axis] = lower, upper
            axis_spacing: list[float | None] = [None, None, None]
            axis_spacing[axis] = spacing
            result.append(
                _MeshFeatureRegion(
                    cast(AxisValues, tuple(region_lower)),
                    cast(AxisValues, tuple(region_upper)),
                    cast(OptionalAxisValues, tuple(axis_spacing)),
                    reason,
                )
            )
    return result


def _polygon_mesh_feature_regions(
    structure: Shape,
    *,
    material_spacing: float,
    spec: GridSpec,
) -> list[_MeshFeatureRegion]:
    """Convert opposing polygon faces into local, axis-specific constraints."""
    depth = float(structure.depth)
    z_lower = float(structure.z)
    z_upper = z_lower + depth if depth > 0.0 else 1.0
    regions = []
    for feature in _polygon_opposing_features(structure):
        target = _clamp_spacing(feature.width / spec.min_feature_cells, spec)
        if target >= material_spacing * (1.0 - 64.0 * np.finfo(float).eps):
            continue
        direction = np.abs(np.asarray(feature.normal, dtype=np.float64))
        # Ignore nearly tangential projections. The threshold also provides enough
        # angular overlap for coarse curve tessellations to form one stable envelope.
        active = direction >= _AXIS_NORMAL_THRESHOLD
        if not np.any(active):
            active[int(np.argmax(direction))] = True
        for axis in np.flatnonzero(active):
            spacing = cast(
                OptionalAxisValues,
                tuple(target if index == int(axis) else None for index in range(3)),
            )
            regions.append(
                _MeshFeatureRegion(
                    (feature.lower[0], feature.lower[1], z_lower),
                    (feature.upper[0], feature.upper[1], z_upper),
                    spacing,
                    f"{feature.reason} ({feature.width:.6g})",
                )
            )
    return _merge_mesh_feature_regions(regions)
