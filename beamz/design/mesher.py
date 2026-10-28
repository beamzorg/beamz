"""Deterministic one-dimensional grading for rectilinear FDTD grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Below this slope a spacing segment is treated as constant, avoiding a log of 1.
_FLAT_GRADIENT = 32.0 * np.finfo(float).eps
_WIDTH_TOLERANCE = 128.0 * np.finfo(float).eps


def _positive_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values.")
    return array


def _cell_count(value: float) -> int:
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(value))
    return max(1, int(np.ceil(value - tolerance)))


def _refined_cell_count(current: int, required_factor: float) -> int:
    """Increase an interval count without overshooting a coupled boundary."""
    # An interval participates in constraints at both ends. Jumping directly by
    # the apparent boundary-width ratio can reverse the violation and make the
    # neighboring interval over-refine in response. A bounded proportional step
    # converges much faster than +1 on long intervals while remaining well damped.
    fractional_growth = min(max(float(required_factor) - 1.0, 0.0), 0.05)
    increment = max(1, int(np.ceil(int(current) * fractional_growth)))
    return int(current) + increment


def _density_integral(length: float, start_spacing: float, end_spacing: float):
    """Integrate 1/h along a linearly changing spacing segment."""
    gradient = (end_spacing - start_spacing) / length
    if abs(gradient) <= _FLAT_GRADIENT:
        return length / start_spacing
    return np.log(end_spacing / start_spacing) / gradient


def _offset_at_density(local_density: float, start_spacing: float, gradient: float):
    """Invert a linear spacing segment to a coordinate offset."""
    if abs(gradient) <= _FLAT_GRADIENT:
        return local_density * start_spacing
    return start_spacing * np.expm1(gradient * local_density) / gradient


def _profile_coordinates(
    lower: float,
    upper: float,
    cap: float,
    left: float,
    right: float,
    slope: float,
) -> list[float]:
    """Breakpoints where a capped linear spacing envelope changes slope."""
    candidates = [lower, upper]
    if slope > 0.0:
        candidates.extend(
            (
                lower + (cap - left) / slope,
                upper - (cap - right) / slope,
                0.5 * (lower + upper + (right - left) / slope),
            )
        )
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))
    coordinates = []
    for value in sorted(candidates):
        clipped = float(np.clip(value, lower, upper))
        if not coordinates or clipped - coordinates[-1] > tolerance:
            coordinates.append(clipped)
    if coordinates[-1] != upper:
        coordinates[-1] = upper
    return coordinates


def _cumulative_density(coordinates: np.ndarray, spacings: np.ndarray) -> np.ndarray:
    density = np.zeros(coordinates.size, dtype=np.float64)
    for index, (x0, x1, h0, h1) in enumerate(
        zip(
            coordinates[:-1], coordinates[1:], spacings[:-1], spacings[1:], strict=True
        ),
        start=1,
    ):
        density[index] = density[index - 1] + _density_integral(x1 - x0, h0, h1)
    return density


@dataclass(frozen=True, slots=True)
class _SpacingProfile:
    """Piecewise-linear spacing profile and its integrated cell density."""

    coordinates: np.ndarray
    spacings: np.ndarray
    cumulative_density: np.ndarray

    @classmethod
    def from_interval(
        cls,
        lower: float,
        upper: float,
        cap: float,
        left: float,
        right: float,
        slope: float,
    ) -> _SpacingProfile:
        coordinates = _profile_coordinates(lower, upper, cap, left, right, slope)
        x = np.asarray(coordinates, dtype=np.float64)
        h = np.minimum(
            cap,
            np.minimum(left + slope * (x - lower), right + slope * (upper - x)),
        )
        h = np.maximum(h, np.finfo(float).tiny)
        return cls(x, h, _cumulative_density(x, h))

    @property
    def total_density(self) -> float:
        return float(self.cumulative_density[-1])

    def coordinate_at_density(self, density: np.ndarray) -> np.ndarray:
        values = np.asarray(density, dtype=np.float64)
        result = np.empty_like(values)
        for output_index, target in np.ndenumerate(values):
            result[output_index] = self._coordinate_at(float(target))
        return result

    def _coordinate_at(self, target: float) -> float:
        if target <= 0.0:
            return float(self.coordinates[0])
        if target >= self.cumulative_density[-1]:
            return float(self.coordinates[-1])
        segment = int(
            np.searchsorted(self.cumulative_density, target, side="right") - 1
        )
        x0, x1 = self.coordinates[segment : segment + 2]
        h0, h1 = self.spacings[segment : segment + 2]
        gradient = (h1 - h0) / (x1 - x0)
        offset = _offset_at_density(
            target - self.cumulative_density[segment], h0, gradient
        )
        return float(x0 + offset)


def _axis_inputs(interval_coords, max_spacings) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(interval_coords, dtype=np.float64)
    if (
        coords.ndim != 1
        or coords.size < 2
        or not np.all(np.isfinite(coords))
        or np.any(np.diff(coords) <= 0.0)
    ):
        raise ValueError(
            "interval_coords must be finite, one-dimensional, and strictly increasing."
        )
    caps = _positive_array(max_spacings, "max_spacings")
    if caps.size != coords.size - 1:
        raise ValueError("max_spacings must contain one value per coordinate interval.")
    return coords, caps


def _neighbor_ratios(widths: np.ndarray) -> np.ndarray:
    return np.maximum(widths[1:] / widths[:-1], widths[:-1] / widths[1:])


def _increase_counts_over_cap(
    counts: np.ndarray,
    owners: np.ndarray,
    widths: np.ndarray,
    target_limits: np.ndarray,
) -> None:
    for interval in np.unique(owners[widths > target_limits]):
        selected = owners == interval
        required_factor = float(np.max(widths[selected] / target_limits[selected]))
        counts[interval] = _refined_cell_count(int(counts[interval]), required_factor)


def _refine_ratio_violations(
    counts: np.ndarray,
    owners: np.ndarray,
    widths: np.ndarray,
    max_scale: float,
    violating_pairs: np.ndarray,
) -> bool:
    """Grow intervals that own the larger cell of a violating pair.

    Returns False when a violation sits inside one interval. Adding cells cannot
    repair that case, so the caller must try a gentler spacing slope.
    """
    refinement_factors: dict[int, float] = {}
    for pair in violating_pairs:
        left_owner, right_owner = owners[pair : pair + 2]
        if left_owner == right_owner:
            return False
        larger_cell = pair if widths[pair] > widths[pair + 1] else pair + 1
        smaller_cell = pair + 1 if larger_cell == pair else pair
        interval = int(owners[larger_cell])
        required_factor = float(
            widths[larger_cell] / (max_scale * widths[smaller_cell])
        )
        refinement_factors[interval] = max(
            refinement_factors.get(interval, 1.0), required_factor
        )
    for interval, required_factor in refinement_factors.items():
        counts[interval] = _refined_cell_count(counts[interval], required_factor)
    return True


@dataclass(frozen=True, slots=True)
class _GradedMesher:
    """Generate smoothly graded cell edges across constrained 1D intervals.

    Parameters
    ----------
    max_scale : float, default=1.3
        Maximum permitted ratio between the widths of adjacent cells.
    transition_safety : float, default=0.85
        Fraction of the theoretical spacing-profile slope used before the exact
        cell-ratio verification pass. Values below one leave room for cell-count
        quantization at snapped interval boundaries.
    max_refinement_passes : int, default=64
        Maximum number of local cell-count corrections for one slope attempt.
    max_slope_passes : int, default=12
        Maximum number of progressively gentler spacing-envelope attempts.

    Notes
    -----
    ``interval_coords`` passed to :meth:`make_axis_edges` are mandatory grid
    boundaries. ``max_spacings[i]`` is an upper bound on cell width within the
    corresponding interval. The result preserves every boundary, never exceeds
    an interval limit, and verifies the adjacent-cell ratio before returning.
    """

    max_scale: float = 1.3
    transition_safety: float = 0.85
    max_refinement_passes: int = 64
    max_slope_passes: int = 12

    @staticmethod
    def _check_cell_budget(
        counts,
        *,
        max_cells: int | None,
        axis_name: str | None,
        coords: np.ndarray,
        caps: np.ndarray,
        reason: str,
    ) -> None:
        if max_cells is None:
            return
        estimated = sum(int(value) for value in counts)
        if estimated <= int(max_cells):
            return
        limiting = int(np.argmin(caps))
        estimated_mib = estimated * 16 / 1024**2
        label = "axis" if axis_name is None else f"{axis_name} axis"
        raise ValueError(
            f"Grid budget exceeded on {label} before allocation: estimated "
            f"{estimated:,} cells exceeds the {int(max_cells):,}-cell axis limit "
            f"({estimated_mib:.2f} MiB for temporary edge/owner arrays). The "
            f"smallest requested spacing is {caps[limiting]:.6g} in interval "
            f"[{coords[limiting]:.6g}, {coords[limiting + 1]:.6g}] ({reason}). "
            "Increase max_cells_per_axis explicitly, raise dl_min, or relax "
            "local refinement."
        )

    def __post_init__(self) -> None:
        max_scale = float(self.max_scale)
        transition_safety = float(self.transition_safety)
        if not np.isfinite(max_scale) or max_scale <= 1.0 or max_scale >= 2.0:
            raise ValueError("max_scale must be finite and strictly between 1 and 2.")
        if (
            not np.isfinite(transition_safety)
            or transition_safety <= 0.0
            or transition_safety > 1.0
        ):
            raise ValueError("transition_safety must be in the interval (0, 1].")
        if int(self.max_refinement_passes) <= 0 or int(self.max_slope_passes) <= 0:
            raise ValueError("Mesher pass limits must be positive integers.")
        object.__setattr__(self, "max_scale", max_scale)
        object.__setattr__(self, "transition_safety", transition_safety)
        object.__setattr__(
            self, "max_refinement_passes", int(self.max_refinement_passes)
        )
        object.__setattr__(self, "max_slope_passes", int(self.max_slope_passes))

    @staticmethod
    def _boundary_spacings(
        lengths: np.ndarray, caps: np.ndarray, slope: float
    ) -> np.ndarray:
        boundaries = _seed_boundary_caps(caps)
        _limit_boundaries_forward(boundaries, lengths, caps, slope)
        _limit_boundaries_backward(boundaries, lengths, caps, slope)
        return boundaries

    @staticmethod
    def _profiles(
        coords: np.ndarray,
        caps: np.ndarray,
        slope: float,
    ) -> list[_SpacingProfile]:
        lengths = np.diff(coords)
        # An exact snapped interval always contains at least one cell. Treat its
        # length as an additional local spacing constraint so a tiny interval
        # grades smoothly into its neighbors instead of creating a sliver cell.
        effective_caps = np.minimum(caps, lengths)
        boundary_spacings = _GradedMesher._boundary_spacings(
            lengths, effective_caps, slope
        )
        return [
            _SpacingProfile.from_interval(
                coords[index],
                coords[index + 1],
                effective_caps[index],
                boundary_spacings[index],
                boundary_spacings[index + 1],
                slope,
            )
            for index in range(caps.size)
        ]

    @staticmethod
    def _realize(
        profiles: list[_SpacingProfile], counts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pieces = []
        owners = []
        for interval, (profile, count) in enumerate(zip(profiles, counts, strict=True)):
            density = np.linspace(
                0.0, profile.total_density, int(count) + 1, dtype=np.float64
            )
            edges = profile.coordinate_at_density(density)
            edges[[0, -1]] = profile.coordinates[[0, -1]]
            pieces.append(edges if interval == 0 else edges[1:])
            owners.extend([interval] * int(count))
        return np.concatenate(pieces), np.asarray(owners, dtype=np.int64)

    def make_axis_edges(
        self,
        interval_coords,
        max_spacings,
        *,
        max_cells: int | None = None,
        axis_name: str | None = None,
    ) -> np.ndarray:
        """Return graded edges satisfying interval and neighbor constraints."""
        coords, caps = _axis_inputs(interval_coords, max_spacings)
        slope = self._opening_slope()
        for _slope_pass in range(self.max_slope_passes):
            edges = self._grade_at_slope(coords, caps, slope, max_cells, axis_name)
            if edges is not None:
                return edges
            slope *= 0.65
        raise RuntimeError(
            "Could not construct a graded axis within the configured pass limits. "
            "Relax max_scale, remove near-coincident snapped boundaries, or increase "
            "the mesher pass limits."
        )

    def _opening_slope(self) -> float:
        """Largest spacing slope that can still meet ``max_scale`` after sampling."""
        theoretical_slope = 2.0 * (self.max_scale - 1.0) / (self.max_scale + 1.0)
        return theoretical_slope * self.transition_safety

    def _grade_at_slope(
        self,
        coords: np.ndarray,
        caps: np.ndarray,
        slope: float,
        max_cells: int | None,
        axis_name: str | None,
    ) -> np.ndarray | None:
        """Realize one slope, or return None so the caller can ease the slope."""
        profiles = self._profiles(coords, caps, slope)
        counts = self._initial_counts(profiles, coords, caps, max_cells, axis_name)
        for _refinement_pass in range(self.max_refinement_passes):
            self._check_cell_budget(
                counts,
                max_cells=max_cells,
                axis_name=axis_name,
                coords=coords,
                caps=caps,
                reason="neighbor-ratio refinement",
            )
            edges, owners = self._realize(profiles, counts)
            if self._repair_cap_violations(
                counts, owners, edges, caps, coords, max_cells, axis_name
            ):
                continue
            widths = np.diff(edges)
            if widths.size < 2:
                return edges
            violating_pairs = np.flatnonzero(
                _neighbor_ratios(widths) > self.max_scale * (1.0 + _WIDTH_TOLERANCE)
            )
            if violating_pairs.size == 0:
                return edges
            if not _refine_ratio_violations(
                counts, owners, widths, self.max_scale, violating_pairs
            ):
                return None
            self._check_cell_budget(
                counts,
                max_cells=max_cells,
                axis_name=axis_name,
                coords=coords,
                caps=caps,
                reason="neighbor-ratio refinement",
            )
        return None

    def _initial_counts(
        self,
        profiles: list[_SpacingProfile],
        coords: np.ndarray,
        caps: np.ndarray,
        max_cells: int | None,
        axis_name: str | None,
    ) -> np.ndarray:
        counts = np.asarray(
            [_cell_count(profile.total_density) for profile in profiles], dtype=np.int64
        )
        self._check_cell_budget(
            counts,
            max_cells=max_cells,
            axis_name=axis_name,
            coords=coords,
            caps=caps,
            reason="initial graded-spacing envelope",
        )
        # Keep a mandatory interval as one cell when it already satisfies its
        # hard cap. Its profile density can be just above one because the
        # envelope also grades neighboring cells; blindly taking ceil would
        # split it into the new sliver that this propagation is meant to avoid.
        counts[np.diff(coords) <= caps * (1.0 + _WIDTH_TOLERANCE)] = 1
        return counts

    def _repair_cap_violations(
        self,
        counts: np.ndarray,
        owners: np.ndarray,
        edges: np.ndarray,
        caps: np.ndarray,
        coords: np.ndarray,
        max_cells: int | None,
        axis_name: str | None,
    ) -> bool:
        """Subdivide intervals whose realized cells exceed their hard cap."""
        widths = np.diff(edges)
        target_limits = caps[owners]
        if not np.any(widths > target_limits * (1.0 + _WIDTH_TOLERANCE)):
            return False
        _increase_counts_over_cap(counts, owners, widths, target_limits)
        self._check_cell_budget(
            counts,
            max_cells=max_cells,
            axis_name=axis_name,
            coords=coords,
            caps=caps,
            reason="hard-spacing refinement",
        )
        return True


def _seed_boundary_caps(caps: np.ndarray) -> np.ndarray:
    boundaries = np.empty(caps.size + 1, dtype=np.float64)
    boundaries[0] = caps[0]
    boundaries[-1] = caps[-1]
    if caps.size > 1:
        boundaries[1:-1] = np.minimum(caps[:-1], caps[1:])
    return boundaries


def _limit_boundaries_forward(
    boundaries: np.ndarray, lengths: np.ndarray, caps: np.ndarray, slope: float
) -> None:
    for index in range(caps.size):
        boundaries[index + 1] = min(
            boundaries[index + 1],
            caps[index],
            boundaries[index] + slope * lengths[index],
        )


def _limit_boundaries_backward(
    boundaries: np.ndarray, lengths: np.ndarray, caps: np.ndarray, slope: float
) -> None:
    for index in range(caps.size - 1, -1, -1):
        boundaries[index] = min(
            boundaries[index],
            caps[index],
            boundaries[index + 1] + slope * lengths[index],
        )


__all__: list[str] = []
