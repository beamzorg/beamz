"""Spacing-constraint types shared by graded-grid construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from beamz.design.grid_spec import GridSpec

AxisValues = tuple[float, float, float]
OptionalAxisValues = tuple[float | None, float | None, float | None]

# Projections gentler than this are not given their own axis cap. The overlap
# also lets a coarsely tessellated curve collapse to one envelope.
_AXIS_NORMAL_THRESHOLD = 0.2
# X and Y envelopes within this spacing ratio, and nearly the same interval,
# are stored once and applied to both axes.
_EQUIVALENT_ENVELOPE_RATIO = 1.05


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


def _axis_only_spacing(axis: int, spacing: float) -> OptionalAxisValues:
    values: list[float | None] = [None, None, None]
    values[axis] = spacing
    return cast(OptionalAxisValues, tuple(values))


def _clip_to_domain(
    raw_lower: tuple[float, ...],
    raw_upper: tuple[float, ...],
    offset: AxisValues,
    extents: tuple[float, float, float],
) -> tuple[AxisValues, AxisValues]:
    """Translate centered geometry into the raster domain and clip to it."""
    lower = tuple(max(0.0, raw_lower[axis] + offset[axis]) for axis in range(3))
    upper = tuple(
        min(extents[axis], raw_upper[axis] + offset[axis]) for axis in range(3)
    )
    return cast(AxisValues, lower), cast(AxisValues, upper)


def _collapsed(lower: AxisValues, upper: AxisValues, active_count: int) -> bool:
    return any(upper[axis] <= lower[axis] for axis in range(active_count))
