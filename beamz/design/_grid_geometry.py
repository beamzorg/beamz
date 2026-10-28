"""Turn a design into per-axis spacing caps for the graded mesher."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from beamz.design._grid_constraints import (
    AxisValues,
    OptionalAxisValues,
    _axis_only_spacing,
    _clamp_spacing,
    _clip_to_domain,
    _collapsed,
    _Region,
)
from beamz.design._grid_features import _polygon_mesh_feature_regions
from beamz.design.structures import Polygon

if TYPE_CHECKING:
    from beamz.design.grid_spec import GridSpec


def _material_index(material: Any) -> float:
    maximum = getattr(material, "max_permittivity", None)
    if maximum is None:
        raise ValueError("Automatic meshing requires finite material permittivity.")
    value = float(np.real(maximum))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Automatic meshing requires positive material permittivity.")
    return float(np.sqrt(max(value, 1.0)))


def _design_extents(design: Any) -> tuple[float, float, float]:
    depth = float(design.depth)
    return float(design.width), float(design.height), depth if depth > 0.0 else 1.0


def _design_coordinate_offset(design: Any) -> AxisValues:
    """Translate centered public geometry into the positive raster domain."""
    if not bool(getattr(design, "_centered_coordinates", False)):
        return (0.0, 0.0, 0.0)
    extents = _design_extents(design)
    return (
        0.5 * extents[0],
        0.5 * extents[1],
        0.5 * extents[2] if float(design.depth) > 0.0 else 0.0,
    )


def _structure_bounds(
    structure: Any, *, two_dimensional: bool
) -> tuple[AxisValues, AxisValues]:
    if hasattr(structure, "lower") and hasattr(structure, "upper"):
        return cast(AxisValues, tuple(map(float, structure.lower))), cast(
            AxisValues, tuple(map(float, structure.upper))
        )
    if hasattr(structure, "radius") and not getattr(structure, "vertices", ()):
        center = tuple(map(float, structure.center))
        radius = float(structure.radius)
        return cast(AxisValues, tuple(value - radius for value in center)), cast(
            AxisValues, tuple(value + radius for value in center)
        )
    vertices = np.asarray(getattr(structure, "vertices", ()), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] < 3:
        raise TypeError(f"Cannot derive meshing bounds for {type(structure).__name__}.")
    lower = [float(np.min(vertices[:, 0])), float(np.min(vertices[:, 1]))]
    upper = [float(np.max(vertices[:, 0])), float(np.max(vertices[:, 1]))]
    if two_dimensional:
        return cast(AxisValues, (*lower, 0.0)), cast(AxisValues, (*upper, 1.0))
    z = float(getattr(structure, "z", np.min(vertices[:, 2])))
    depth = float(getattr(structure, "depth", 0.0))
    return cast(AxisValues, (*lower, z)), cast(AxisValues, (*upper, z + depth))


def _feature_sizes(
    structure: Any,
) -> tuple[float | None, float | None, float | None]:
    semantic = structure._mesh_feature_sizes()
    if semantic is not None:
        return cast(
            tuple[float | None, float | None, float | None],
            tuple(
                value
                if value is None
                else (float(value) if np.isfinite(value) and value > 0.0 else None)
                for value in semantic
            ),
        )
    width = getattr(structure, "width", None)
    height = getattr(structure, "height", None)
    depth = getattr(structure, "depth", None)
    return tuple(
        value if value is not None and np.isfinite(value) and value > 0.0 else None
        for value in (width, height, depth)
    )  # type: ignore[return-value]


def _geometry_regions(
    design: Any, spec: GridSpec, coordinate_offset: AxisValues
) -> list[_Region]:
    """Collect material, feature, gap, and override caps in raster coordinates."""
    extents = _design_extents(design)
    wavelength = _required_wavelength(spec)
    two_dimensional = float(design.depth) == 0.0
    active_count = 2 if two_dimensional else 3
    regions: list[_Region] = []
    material_regions: list[_Region] = []
    for structure in getattr(design, "structures", ()):
        placed = _material_region(
            structure,
            spec,
            wavelength,
            coordinate_offset,
            extents,
            two_dimensional=two_dimensional,
            active_count=active_count,
        )
        if placed is None:
            continue
        material_region, material_spacing = placed
        regions.append(material_region)
        material_regions.append(material_region)
        if isinstance(structure, Polygon) and spec.min_feature_cells > 1.0:
            regions.extend(
                _polygon_regions(
                    structure,
                    material_region,
                    material_spacing=material_spacing,
                    spec=spec,
                    coordinate_offset=coordinate_offset,
                    extents=extents,
                    active_count=active_count,
                )
            )
    regions.extend(_gap_regions(material_regions, extents, spec, active_count))
    regions.extend(_override_regions(spec, coordinate_offset, extents, active_count))
    return regions


def _required_wavelength(spec: GridSpec) -> float:
    if spec.wavelength is None:
        raise ValueError("GridSpec.auto requires wavelength.")
    return float(spec.wavelength)


def _material_region(
    structure: Any,
    spec: GridSpec,
    wavelength: float,
    coordinate_offset: AxisValues,
    extents: tuple[float, float, float],
    *,
    two_dimensional: bool,
    active_count: int,
) -> tuple[_Region, float] | None:
    raw_lower, raw_upper = _structure_bounds(structure, two_dimensional=two_dimensional)
    lower, upper = _clip_to_domain(raw_lower, raw_upper, coordinate_offset, extents)
    if _collapsed(lower, upper, active_count):
        return None
    material_spacing = _clamp_spacing(
        wavelength / (_material_index(structure.material) * spec.min_steps_per_wvl),
        spec,
    )
    spacing = tuple(
        _clamp_spacing(
            min(material_spacing, feature / spec.min_feature_cells)
            if feature is not None
            else material_spacing,
            spec,
        )
        for feature in _feature_sizes(structure)
    )
    region = _Region(lower, upper, cast(OptionalAxisValues, spacing))
    return region, material_spacing


def _polygon_regions(
    structure: Polygon,
    material_region: _Region,
    *,
    material_spacing: float,
    spec: GridSpec,
    coordinate_offset: AxisValues,
    extents: tuple[float, float, float],
    active_count: int,
) -> list[_Region]:
    regions = []
    for feature_region in _polygon_mesh_feature_regions(
        structure,
        material_spacing=material_spacing,
        spec=spec,
    ):
        lower, upper = _clip_to_domain(
            feature_region.lower,
            feature_region.upper,
            coordinate_offset,
            extents,
        )
        lower, upper = _snap_to_material(
            lower, upper, material_region, feature_region.spacing
        )
        if _collapsed(lower, upper, active_count):
            continue
        regions.append(_Region(lower, upper, feature_region.spacing))
    return regions


def _snap_to_material(
    lower: AxisValues,
    upper: AxisValues,
    material_region: _Region,
    spacing: OptionalAxisValues,
) -> tuple[AxisValues, AxisValues]:
    """Absorb a feature edge that merely restates the material boundary."""
    snapped_lower = list(lower)
    snapped_upper = list(upper)
    for axis, feature_spacing in enumerate(spacing):
        if feature_spacing is None:
            continue
        if abs(snapped_lower[axis] - material_region.lower[axis]) < feature_spacing:
            snapped_lower[axis] = material_region.lower[axis]
        if abs(snapped_upper[axis] - material_region.upper[axis]) < feature_spacing:
            snapped_upper[axis] = material_region.upper[axis]
    return cast(AxisValues, tuple(snapped_lower)), cast(
        AxisValues, tuple(snapped_upper)
    )


def _gap_regions(
    material_regions: list[_Region],
    extents: tuple[float, float, float],
    spec: GridSpec,
    active_count: int,
) -> list[_Region]:
    regions = []
    for first_index, first in enumerate(material_regions):
        for second in material_regions[first_index + 1 :]:
            regions.extend(_gaps_between(first, second, extents, spec, active_count))
    return regions


def _gaps_between(
    first: _Region,
    second: _Region,
    extents: tuple[float, float, float],
    spec: GridSpec,
    active_count: int,
) -> list[_Region]:
    regions = []
    for axis in range(active_count):
        transverse = tuple(index for index in range(active_count) if index != axis)
        gap = _separated_gap(first, second, axis, transverse)
        if gap is None:
            continue
        gap_lower, gap_upper = gap
        lower = [0.0, 0.0, 0.0]
        upper = list(extents)
        lower[axis], upper[axis] = gap_lower, gap_upper
        regions.append(
            _Region(
                cast(AxisValues, tuple(lower)),
                cast(AxisValues, tuple(upper)),
                _axis_only_spacing(
                    axis,
                    _clamp_spacing(
                        (gap_upper - gap_lower) / spec.min_feature_cells, spec
                    ),
                ),
            )
        )
    return regions


def _separated_gap(
    first: _Region,
    second: _Region,
    axis: int,
    transverse: tuple[int, ...],
) -> tuple[float, float] | None:
    if not _overlap(first, second, transverse):
        return None
    if first.upper[axis] <= second.lower[axis]:
        return first.upper[axis], second.lower[axis]
    if second.upper[axis] <= first.lower[axis]:
        return second.upper[axis], first.lower[axis]
    return None


def _overlap(first: _Region, second: _Region, transverse_axes: tuple[int, ...]) -> bool:
    return all(
        first.lower[axis] < second.upper[axis]
        and second.lower[axis] < first.upper[axis]
        for axis in transverse_axes
    )


def _override_regions(
    spec: GridSpec,
    coordinate_offset: AxisValues,
    extents: tuple[float, float, float],
    active_count: int,
) -> list[_Region]:
    regions = []
    for override in spec.overrides:
        raw_lower, raw_upper = override.bounds
        lower, upper = _clip_to_domain(raw_lower, raw_upper, coordinate_offset, extents)
        if _collapsed(lower, upper, active_count):
            continue
        spacing = tuple(
            None if value is None else _clamp_spacing(value, spec)
            for value in cast(OptionalAxisValues, override.dl)
        )
        regions.append(
            _Region(lower, upper, cast(OptionalAxisValues, spacing), override.enforced)
        )
    return regions


def _unique_coordinates(values: list[float], extent: float) -> np.ndarray:
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(extent))
    result = []
    for value in sorted(float(np.clip(item, 0.0, extent)) for item in values):
        if not result or value - result[-1] > tolerance:
            result.append(value)
    result[0], result[-1] = 0.0, extent
    return np.asarray(result, dtype=np.float64)


def _axis_constraints(
    axis: int,
    extent: float,
    background_spacing: float,
    regions: list[_Region],
    spec: GridSpec,
    coordinate_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = [
        0.0,
        extent,
        *_snapping_coordinates(spec, axis, extent, coordinate_offset),
    ]
    effective_regions = []
    for region in regions:
        if region.spacing[axis] is None:
            continue
        coordinates.extend((region.lower[axis], region.upper[axis]))
        effective_regions.append(region)
    coords = _unique_coordinates(coordinates, extent)
    limits = np.empty(coords.size - 1, dtype=np.float64)
    for index, midpoint in enumerate(0.5 * (coords[:-1] + coords[1:])):
        limits[index] = _spacing_limit(
            midpoint, axis, background_spacing, effective_regions, spec
        )
    return coords, limits


def _snapping_coordinates(
    spec: GridSpec, axis: int, extent: float, offset: float
) -> list[float]:
    coordinates = []
    for point in spec.snapping_points:
        value = point[axis]
        if value is not None:
            value = float(value) + offset
        if value is not None and 0.0 < value < extent:
            coordinates.append(value)
    return coordinates


def _spacing_limit(
    midpoint,
    axis: int,
    background_spacing: float,
    regions: list[_Region],
    spec: GridSpec,
) -> float:
    """Apply soft caps first, then let an enforced override replace them."""
    limit = background_spacing
    for region in regions:
        if not region.enforced and region.lower[axis] <= midpoint <= region.upper[axis]:
            limit = min(limit, float(region.spacing[axis]))  # type: ignore[arg-type]
    for region in regions:
        if region.enforced and region.lower[axis] <= midpoint <= region.upper[axis]:
            limit = float(region.spacing[axis])  # type: ignore[arg-type]
    return _clamp_spacing(limit, spec)
