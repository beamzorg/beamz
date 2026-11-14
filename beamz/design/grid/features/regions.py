"""Design implementation."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from beamz.design._geometry import structure_bounds as _structure_bounds
from beamz.design.grid.features.constraints import _clamp_spacing, _Region
from beamz.design.grid.features.polygons import _polygon_mesh_feature_regions
from beamz.design.grid.spec import AxisValues, GridSpec, OptionalAxisValues


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


def _overlap(first: _Region, second: _Region, transverse_axes: tuple[int, ...]) -> bool:
    return all(
        first.lower[axis] < second.upper[axis]
        and second.lower[axis] < first.upper[axis]
        for axis in transverse_axes
    )


def _clip_region_bounds(lower, upper, offset, extents) -> tuple[AxisValues, AxisValues]:
    """Apply the design coordinate frame consistently to all meshing regions."""
    return (
        cast(
            AxisValues,
            tuple(
                max(0.0, value + delta)
                for value, delta in zip(lower, offset, strict=True)
            ),
        ),
        cast(
            AxisValues,
            tuple(
                min(extent, value + delta)
                for value, delta, extent in zip(upper, offset, extents, strict=True)
            ),
        ),
    )


def _geometry_regions(
    design: Any, spec: GridSpec, coordinate_offset: AxisValues
) -> list[_Region]:
    extents = _design_extents(design)
    if spec.wavelength is None:
        raise ValueError("GridSpec.auto requires wavelength.")
    wavelength = float(spec.wavelength)
    two_dimensional = float(design.depth) == 0.0
    regions = []
    material_regions = []
    for structure in getattr(design, "structures", ()):
        raw_lower, raw_upper = _structure_bounds(
            structure, two_dimensional=two_dimensional
        )
        lower, upper = _clip_region_bounds(
            raw_lower, raw_upper, coordinate_offset, extents
        )
        if any(
            upper[axis] <= lower[axis] for axis in range(2 if two_dimensional else 3)
        ):
            continue
        material_spacing = _clamp_spacing(
            wavelength / (_material_index(structure.material) * spec.min_steps_per_wvl),
            spec,
        )
        features = _feature_sizes(structure)
        spacing = tuple(
            _clamp_spacing(
                min(material_spacing, feature / spec.min_feature_cells)
                if feature is not None
                else material_spacing,
                spec,
            )
            for feature in features
        )
        material_region = _Region(lower, upper, cast(OptionalAxisValues, spacing))
        regions.append(material_region)
        material_regions.append(material_region)
        if structure.shape_kind == "polygon" and spec.min_feature_cells > 1.0:
            for feature_region in _polygon_mesh_feature_regions(
                structure,
                material_spacing=material_spacing,
                spec=spec,
            ):
                feature_lower, feature_upper = _clip_region_bounds(
                    feature_region.lower,
                    feature_region.upper,
                    coordinate_offset,
                    extents,
                )
                snapped_lower = list(feature_lower)
                snapped_upper = list(feature_upper)
                for axis, feature_spacing in enumerate(feature_region.spacing):
                    if feature_spacing is None:
                        continue
                    if abs(snapped_lower[axis] - lower[axis]) < feature_spacing:
                        snapped_lower[axis] = lower[axis]
                    if abs(snapped_upper[axis] - upper[axis]) < feature_spacing:
                        snapped_upper[axis] = upper[axis]
                feature_lower = cast(AxisValues, tuple(snapped_lower))
                feature_upper = cast(AxisValues, tuple(snapped_upper))
                if any(
                    feature_upper[axis] <= feature_lower[axis]
                    for axis in range(2 if two_dimensional else 3)
                ):
                    continue
                regions.append(
                    _Region(
                        feature_lower,
                        feature_upper,
                        feature_region.spacing,
                    )
                )

    active_count = 2 if two_dimensional else 3
    for first_index, first in enumerate(material_regions):
        for second in material_regions[first_index + 1 :]:
            for axis in range(active_count):
                transverse = tuple(
                    index for index in range(active_count) if index != axis
                )
                if not _overlap(first, second, transverse):
                    continue
                if first.upper[axis] <= second.lower[axis]:
                    gap_lower, gap_upper = first.upper[axis], second.lower[axis]
                elif second.upper[axis] <= first.lower[axis]:
                    gap_lower, gap_upper = second.upper[axis], first.lower[axis]
                else:
                    continue
                gap = gap_upper - gap_lower
                gap_spacing = _clamp_spacing(gap / spec.min_feature_cells, spec)
                lower = [0.0, 0.0, 0.0]
                upper = list(extents)
                lower[axis], upper[axis] = gap_lower, gap_upper
                regions.append(
                    _Region(
                        cast(AxisValues, tuple(lower)),
                        cast(AxisValues, tuple(upper)),
                        cast(
                            OptionalAxisValues,
                            tuple(
                                gap_spacing if index == axis else None
                                for index in range(3)
                            ),
                        ),
                    )
                )

    for override in spec.overrides:
        raw_lower, raw_upper = override.bounds
        lower, upper = _clip_region_bounds(
            raw_lower, raw_upper, coordinate_offset, extents
        )
        if any(upper[axis] <= lower[axis] for axis in range(active_count)):
            continue
        override_spacing = cast(OptionalAxisValues, override.dl)
        spacing = tuple(
            None if value is None else _clamp_spacing(value, spec)
            for value in override_spacing
        )
        regions.append(
            _Region(lower, upper, cast(OptionalAxisValues, spacing), override.enforced)
        )
    return regions
