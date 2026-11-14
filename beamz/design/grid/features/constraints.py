"""Region spacing constraints for graded grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from beamz.design.grid.spec import AxisValues, GridSpec, OptionalAxisValues


@dataclass(frozen=True, slots=True)
class _Region:
    lower: AxisValues
    upper: AxisValues
    spacing: OptionalAxisValues
    enforced: bool = False


@dataclass(frozen=True, slots=True)
class _MeshFeatureRegion:
    """Localized, axis-aware spacing constraint produced by geometry analysis."""

    lower: AxisValues
    upper: AxisValues
    spacing: OptionalAxisValues
    reason: str


@dataclass(frozen=True, slots=True)
class _OpposingFeature:
    """A reliable local distance between opposing polygon boundary faces."""

    lower: tuple[float, float]
    upper: tuple[float, float]
    width: float
    normal: tuple[float, float]
    reason: str


def _clamp_spacing(value: float, spec: GridSpec) -> float:
    if spec.dl_min is not None:
        value = max(value, float(spec.dl_min))
    if spec.dl_max is not None:
        value = min(value, float(spec.dl_max))
    return value


def _unique_coordinates(values: list[float], extent: float) -> np.ndarray:
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(extent))
    result: list[float] = []
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
    coordinates = [0.0, extent]
    effective_regions = []
    for region in regions:
        if region.spacing[axis] is None:
            continue
        coordinates.extend((region.lower[axis], region.upper[axis]))
        effective_regions.append(region)
    for point in spec.snapping_points:
        value = point[axis]
        if value is not None:
            value = float(value) + coordinate_offset
        if value is not None and 0.0 < value < extent:
            coordinates.append(value)
    coords = _unique_coordinates(coordinates, extent)
    limits = np.empty(coords.size - 1, dtype=np.float64)
    for index, midpoint in enumerate(0.5 * (coords[:-1] + coords[1:])):
        limit = background_spacing
        for region in effective_regions:
            if (
                not region.enforced
                and region.lower[axis] <= midpoint <= region.upper[axis]
            ):
                limit = min(limit, float(cast(float, region.spacing[axis])))
        for region in effective_regions:
            if region.enforced and region.lower[axis] <= midpoint <= region.upper[axis]:
                limit = float(cast(float, region.spacing[axis]))
        limits[index] = _clamp_spacing(limit, spec)
    return coords, limits
