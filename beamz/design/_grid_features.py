"""Local polygon thicknesses and gaps that tighten a graded grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from beamz.design._grid_constraints import (
    _AXIS_NORMAL_THRESHOLD,
    _EQUIVALENT_ENVELOPE_RATIO,
    AxisValues,
    _axis_only_spacing,
    _clamp_spacing,
    _MeshFeatureRegion,
    _OpposingFeature,
)
from beamz.design.structures import Polygon

if TYPE_CHECKING:
    from beamz.design.grid_spec import GridSpec

# Neighboring envelope bins within this ratio describe one feature.
_SIMILAR_SPACING_RATIO = 1.25
_Envelope = tuple[float, float, float, str]


@dataclass(frozen=True, slots=True)
class _BoundarySegments:
    """Sorted polygon edges and the ring each edge belongs to."""

    starts: np.ndarray
    ends: np.ndarray
    vectors: np.ndarray
    inward_normals: np.ndarray
    midpoints: np.ndarray
    ring_ids: np.ndarray
    local_ids: np.ndarray
    ring_sizes: np.ndarray

    @property
    def count(self) -> int:
        return int(self.starts.shape[0])


def _polygon_opposing_features(structure: Polygon) -> list[_OpposingFeature]:
    """Return reliable local distances between opposing material faces.

    Boundary vertex spacing is deliberately excluded: refining a curved polygon's
    tessellation must not change its physical mesh target.
    """
    geometry = _oriented_polygon(structure)
    if geometry is None:
        return []
    if not geometry.interiors and geometry.equals(geometry.convex_hull):
        return _convex_feret_features(geometry)
    return _nonconvex_opposing_features(geometry)


def _oriented_polygon(structure: Polygon):
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry.polygon import orient

    shell = [(float(x), float(y)) for x, y, _z in structure.vertices]
    holes = [
        [(float(x), float(y)) for x, y, _z in path] for path in structure.interiors
    ]
    geometry = orient(ShapelyPolygon(shell, holes=holes), sign=1.0)
    if geometry.is_empty or not geometry.is_valid:
        return None
    return geometry


def _convex_feret_features(geometry) -> list[_OpposingFeature]:
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


def _canonical_ring(ring) -> np.ndarray:
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


def _boundary_segments(geometry) -> _BoundarySegments | None:
    """Return sorted edge arrays, or None when fewer than two edges exist."""
    exterior = _canonical_ring(geometry.exterior)
    interiors = sorted(
        (_canonical_ring(ring) for ring in geometry.interiors),
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
        key=lambda segment: (*segment[0], *segment[1], segment[2], segment[3])
    )
    if len(segments) < 2:
        return None
    starts = np.asarray([segment[0] for segment in segments])
    ends = np.asarray([segment[1] for segment in segments])
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    tangents = vectors / lengths[:, None]
    # Oriented exterior and interior rings both keep material on their left.
    inward_normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    return _BoundarySegments(
        starts,
        ends,
        vectors,
        inward_normals,
        0.5 * (starts + ends),
        np.asarray([segment[2] for segment in segments]),
        np.asarray([segment[3] for segment in segments]),
        np.asarray(ring_sizes),
    )


def _nonconvex_opposing_features(geometry) -> list[_OpposingFeature]:
    from shapely.geometry import LineString

    segments = _boundary_segments(geometry)
    if segments is None:
        return []
    tolerance = 128.0 * np.finfo(float).eps * max(1.0, *geometry.bounds)
    covered = geometry.buffer(tolerance)
    features = []
    for index in range(segments.count):
        feature = _nearest_opposing_feature(
            index, segments, geometry, covered, tolerance, LineString
        )
        if feature is not None:
            features.append(feature)
    return features


def _nearest_opposing_feature(
    index: int,
    segments: _BoundarySegments,
    geometry,
    covered,
    tolerance: float,
    line_string,
) -> _OpposingFeature | None:
    normal = segments.inward_normals[index]
    offsets = segments.starts - segments.midpoints[index]
    denominators = (
        normal[0] * segments.vectors[:, 1] - normal[1] * segments.vectors[:, 0]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ray_distances = (
            offsets[:, 0] * segments.vectors[:, 1]
            - offsets[:, 1] * segments.vectors[:, 0]
        ) / denominators
        segment_positions = (
            offsets[:, 0] * normal[1] - offsets[:, 1] * normal[0]
        ) / denominators
    candidates = _opposing_candidates(
        index, segments, denominators, ray_distances, segment_positions, tolerance
    )
    if candidates.size == 0:
        return None
    ordered = candidates[np.argsort(np.abs(ray_distances[candidates]), kind="stable")]
    first_point = segments.midpoints[index]
    for other in ordered:
        feature = _feature_if_connected(
            index,
            int(other),
            first_point,
            normal,
            float(ray_distances[other]),
            segments.starts,
            segments.ends,
            geometry,
            covered,
            tolerance,
            line_string,
        )
        if feature is not None:
            # The nearest full connector defines the local thickness. Farther
            # crossings are bulk polygon dimensions and should not add constraints.
            return feature
    return None


def _opposing_candidates(
    index: int,
    segments: _BoundarySegments,
    denominators: np.ndarray,
    ray_distances: np.ndarray,
    segment_positions: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    valid = np.arange(segments.count) != index
    same_ring = segments.ring_ids == segments.ring_ids[index]
    local_delta = np.abs(segments.local_ids - segments.local_ids[index])
    ring_size = int(segments.ring_sizes[int(segments.ring_ids[index])])
    adjacent = same_ring & ((local_delta <= 1) | (local_delta >= ring_size - 1))
    valid &= ~adjacent
    valid &= (
        np.einsum("ij,j->i", segments.inward_normals, segments.inward_normals[index])
        < -0.95
    )
    valid &= np.abs(denominators) > tolerance
    valid &= np.abs(ray_distances) > tolerance
    valid &= segment_positions >= -1e-12
    valid &= segment_positions <= 1.0 + 1e-12
    return np.flatnonzero(valid)


def _feature_if_connected(
    index: int,
    other: int,
    first_point: np.ndarray,
    normal: np.ndarray,
    signed_distance: float,
    starts: np.ndarray,
    ends: np.ndarray,
    geometry,
    covered,
    tolerance: float,
    line_string,
) -> _OpposingFeature | None:
    distance = abs(signed_distance)
    second_point = first_point + signed_distance * normal
    connector = line_string((first_point, second_point))
    material_between = covered.covers(connector)
    gap_between = float(connector.intersection(geometry).length) <= tolerance
    if not material_between and not gap_between:
        return None
    direction = second_point - first_point
    padding = 0.5 * distance
    feature_points = np.vstack((starts[index], ends[index], first_point, second_point))
    lower = np.min(feature_points, axis=0) - padding
    upper = np.max(feature_points, axis=0) + padding
    reason = (
        "opposing polygon faces" if material_between else "opposing polygon gap faces"
    )
    return _OpposingFeature(
        (float(lower[0]), float(lower[1])),
        (float(upper[0]), float(upper[1])),
        distance,
        (float(direction[0] / distance), float(direction[1] / distance)),
        reason,
    )


def _polygon_mesh_feature_regions(
    structure: Polygon,
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
        for axis in _active_normal_axes(feature.normal):
            regions.append(
                _MeshFeatureRegion(
                    (feature.lower[0], feature.lower[1], z_lower),
                    (feature.upper[0], feature.upper[1], z_upper),
                    _axis_only_spacing(int(axis), target),
                    f"{feature.reason} ({feature.width:.6g})",
                )
            )
    return _merge_mesh_feature_regions(regions)


def _active_normal_axes(normal: tuple[float, float]) -> np.ndarray:
    direction = np.abs(np.asarray(normal, dtype=np.float64))
    active = direction >= _AXIS_NORMAL_THRESHOLD
    if not np.any(active):
        active[int(np.argmax(direction))] = True
    return np.flatnonzero(active)


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
    envelopes = [_axis_envelope(regions, axis) for axis in range(2)]
    if _envelopes_match(envelopes[0], envelopes[1]):
        shared = _shared_envelope(envelopes[0], envelopes[1])
        envelopes = [shared, shared]
    return _regions_from_envelopes(
        envelopes,
        cast(AxisValues, global_lower),
        cast(AxisValues, global_upper),
    )


def _spacing_on(region: _MeshFeatureRegion, axis: int) -> float:
    value = region.spacing[axis]
    assert value is not None
    return float(value)


def _axis_envelope(regions: list[_MeshFeatureRegion], axis: int) -> list[_Envelope]:
    selected = [region for region in regions if region.spacing[axis] is not None]
    if not selected:
        return []
    raw_coordinates = sorted(
        {
            value
            for region in selected
            for value in (region.lower[axis], region.upper[axis])
        }
    )
    smallest_spacing = min(_spacing_on(region, axis) for region in selected)
    coordinates = _cluster_coordinates(raw_coordinates, smallest_spacing)
    return _envelope_intervals(coordinates, selected, axis)


def _cluster_coordinates(
    raw_coordinates: list[float], smallest_spacing: float
) -> list[float]:
    clusters: list[list[float]] = []
    for value in raw_coordinates:
        if not clusters or value - clusters[-1][0] >= smallest_spacing:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [float(np.mean(cluster)) for cluster in clusters]


def _envelope_intervals(
    coordinates: list[float],
    selected: list[_MeshFeatureRegion],
    axis: int,
) -> list[_Envelope]:
    intervals: list[_Envelope] = []
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
        spacing = min(_spacing_on(region, axis) for region in covering)
        reason = min(covering, key=lambda region: _spacing_on(region, axis)).reason
        if intervals and _can_merge_bins(intervals[-1], lower, spacing):
            intervals[-1] = _merged_bin(intervals[-1], upper, spacing, reason)
            continue
        intervals.append((lower, upper, spacing, reason))
    return intervals


def _can_merge_bins(previous: _Envelope, lower: float, spacing: float) -> bool:
    _old_lower, old_upper, old_spacing, _old_reason = previous
    ratio = max(old_spacing, spacing) / min(old_spacing, spacing)
    return old_upper == lower and ratio <= _SIMILAR_SPACING_RATIO


def _merged_bin(
    previous: _Envelope, upper: float, spacing: float, reason: str
) -> _Envelope:
    old_lower, _old_upper, old_spacing, old_reason = previous
    kept_reason = old_reason if old_spacing <= spacing else reason
    return (old_lower, upper, min(old_spacing, spacing), kept_reason)


def _envelopes_match(
    x_intervals: list[_Envelope], y_intervals: list[_Envelope]
) -> bool:
    if len(x_intervals) != len(y_intervals):
        return False
    return all(
        max(x_item[2], y_item[2]) / min(x_item[2], y_item[2])
        <= _EQUIVALENT_ENVELOPE_RATIO
        and abs(x_item[0] - y_item[0]) <= max(x_item[2], y_item[2])
        and abs(x_item[1] - y_item[1]) <= max(x_item[2], y_item[2])
        for x_item, y_item in zip(x_intervals, y_intervals, strict=True)
    )


def _shared_envelope(
    x_intervals: list[_Envelope], y_intervals: list[_Envelope]
) -> list[_Envelope]:
    return [
        (
            0.5 * (x_item[0] + y_item[0]),
            0.5 * (x_item[1] + y_item[1]),
            min(x_item[2], y_item[2]),
            x_item[3] if x_item[2] <= y_item[2] else y_item[3],
        )
        for x_item, y_item in zip(x_intervals, y_intervals, strict=True)
    ]


def _regions_from_envelopes(
    envelopes: list[list[_Envelope]],
    global_lower: AxisValues,
    global_upper: AxisValues,
) -> list[_MeshFeatureRegion]:
    result = []
    for axis, intervals in enumerate(envelopes):
        for lower, upper, spacing, reason in intervals:
            region_lower = list(global_lower)
            region_upper = list(global_upper)
            region_lower[axis], region_upper[axis] = lower, upper
            result.append(
                _MeshFeatureRegion(
                    cast(AxisValues, tuple(region_lower)),
                    cast(AxisValues, tuple(region_upper)),
                    _axis_only_spacing(axis, spacing),
                    reason,
                )
            )
    return result
