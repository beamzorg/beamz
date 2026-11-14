"""Physics- and geometry-aware grid specification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from beamz.const import LIGHT_SPEED
from beamz.design.grid import Axis, RectilinearGrid

AxisValues = tuple[float, float, float]


OptionalAxisValues = tuple[float | None, float | None, float | None]


def _xyz(values, name: str, *, fill: float) -> tuple[float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) == 2:
        result = (*result, fill)
    if len(result) != 3:
        raise ValueError(f"{name} must contain two or three values.")
    return result


@dataclass(frozen=True, slots=True)
class MeshOverride:
    """Specify a meshing-only rectangular region with explicit axis spacings.

    ``None`` disables the override along an axis. An enforced override replaces
    material-derived spacing constraints in its region; a normal override can
    only refine them.
    """

    center: tuple[float, ...]
    size: tuple[float, ...]
    dl: float | tuple[float | None, ...]
    enforced: bool = False

    def __post_init__(self) -> None:
        center = _xyz(self.center, "MeshOverride center", fill=0.0)
        size = _xyz(self.size, "MeshOverride size", fill=np.inf)
        if not np.all(np.isfinite(center)):
            raise ValueError("MeshOverride center must be finite.")
        if any(np.isnan(value) or value <= 0.0 for value in size):
            raise ValueError("MeshOverride size must contain positive values.")
        if np.isscalar(self.dl):
            scalar = float(cast(float, self.dl))
            spacing: OptionalAxisValues = (scalar, scalar, scalar)
        else:
            raw = tuple(cast(tuple[float | None, ...], self.dl))
            if len(raw) == 2:
                raw = (*raw, None)
            if len(raw) != 3:
                raise ValueError("MeshOverride dl must contain two or three values.")
            spacing = cast(
                OptionalAxisValues,
                tuple(None if value is None else float(value) for value in raw),
            )
        if all(value is None for value in spacing) or any(
            value is not None and (not np.isfinite(value) or value <= 0.0)
            for value in spacing
        ):
            raise ValueError("MeshOverride dl must contain a positive spacing.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "dl", spacing)
        object.__setattr__(self, "enforced", bool(self.enforced))

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return (
            tuple(c - 0.5 * s for c, s in zip(self.center, self.size, strict=True)),
            tuple(c + 0.5 * s for c, s in zip(self.center, self.size, strict=True)),
        )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Configure automatic or explicit FDTD spatial discretization.

    Parameters
    ----------
    min_steps_per_wvl : float, default=10.0
        Minimum cells per wavelength inside the highest-index material.
    wavelength : float, optional
        Vacuum wavelength in metres used to derive automatic resolution.
    resolution : float, optional
        Explicit uniform cell size in metres. When present, it takes precedence
        over ``wavelength`` and ``min_steps_per_wvl``.
    courant : float, default=0.99
        Fraction of the dimensional Courant stability limit used for time steps.
    max_scale : float, default=1.3
        Hard maximum ratio between neighboring cell widths on a graded axis.
    min_steps_per_sim_size : float, default=10.0
        Minimum number of cells across each active simulation dimension.
    min_feature_cells : float, default=1.0
        Minimum target cells across detected structure thicknesses and gaps.
        Values above one opt into additional geometry-driven refinement beyond
        the material-wavelength target.
    dl_min, dl_max : float, optional
        Global lower and upper bounds for local automatic cell-width targets.
    overrides : tuple[MeshOverride, ...]
        Rectangular regions that refine or replace automatic spacing targets.
    snapping_points : tuple[tuple[float | None, ...], ...]
        User coordinates that must be grid edges along the selected axes.
    max_cells_per_axis : int, optional
        Safety limit for realized cells along any active axis. ``None`` disables it.
    max_total_cells : int, optional
        Safety limit for the active Cartesian cell product. ``None`` disables it.
    """

    min_steps_per_wvl: float = 10.0
    wavelength: float | None = None
    resolution: float | None = None
    courant: float = 0.99
    max_scale: float = 1.3
    min_steps_per_sim_size: float = 10.0
    min_feature_cells: float = 1.0
    dl_min: float | None = None
    dl_max: float | None = None
    overrides: tuple[MeshOverride, ...] = ()
    snapping_points: tuple[tuple[float | None, ...], ...] = ()
    max_cells_per_axis: int | None = 200_000
    max_total_cells: int | None = 20_000_000

    @property
    def is_automatic(self) -> bool:
        """Return whether this policy realizes a geometry-aware grid.

        Returns
        -------
        bool
            ``True`` when no explicit uniform resolution was supplied.
        """
        return self.resolution is None

    def __post_init__(self) -> None:
        positive = {
            "min_steps_per_wvl": self.min_steps_per_wvl,
            "courant": self.courant,
            "max_scale": self.max_scale,
            "min_steps_per_sim_size": self.min_steps_per_sim_size,
            "min_feature_cells": self.min_feature_cells,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"GridSpec {name} must be finite and positive.")
        if float(self.courant) > 1.0:
            raise ValueError("GridSpec courant cannot exceed one.")
        if not 1.0 < float(self.max_scale) < 2.0:
            raise ValueError("GridSpec max_scale must be strictly between 1 and 2.")
        for name in ("wavelength", "resolution", "dl_min", "dl_max"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or float(value) <= 0.0):
                raise ValueError(f"GridSpec {name} must be positive and finite.")
        for name in ("max_cells_per_axis", "max_total_cells"):
            value = getattr(self, name)
            if value is None:
                continue
            integer = int(value)
            if isinstance(value, (bool, np.bool_)) or integer != value or integer <= 0:
                raise ValueError(f"GridSpec {name} must be a positive integer or None.")
            object.__setattr__(self, name, integer)
        if (
            self.dl_min is not None
            and self.dl_max is not None
            and self.dl_min > self.dl_max
        ):
            raise ValueError("GridSpec dl_min cannot exceed dl_max.")
        overrides = tuple(self.overrides)
        if any(not isinstance(value, MeshOverride) for value in overrides):
            raise TypeError("GridSpec overrides must contain MeshOverride values.")
        points = []
        for point in self.snapping_points:
            values = tuple(point)
            if len(values) == 2:
                values = (*values, None)
            if len(values) != 3:
                raise ValueError(
                    "GridSpec snapping points must have two or three values."
                )
            normalized = tuple(
                None if value is None else float(value) for value in values
            )
            if any(
                value is not None and not np.isfinite(value) for value in normalized
            ):
                raise ValueError("GridSpec snapping points must be finite.")
            points.append(normalized)
        object.__setattr__(self, "overrides", overrides)
        object.__setattr__(self, "snapping_points", tuple(points))

    @classmethod
    def auto(
        cls,
        *,
        min_steps_per_wvl: float = 10.0,
        wavelength: float | None = None,
        courant: float = 0.99,
        max_scale: float = 1.3,
        min_steps_per_sim_size: float = 10.0,
        min_feature_cells: float = 1.0,
        dl_min: float | None = None,
        dl_max: float | None = None,
        overrides: tuple[MeshOverride, ...] = (),
        snapping_points: tuple[tuple[float | None, ...], ...] = (),
        max_cells_per_axis: int | None = 200_000,
        max_total_cells: int | None = 20_000_000,
    ) -> GridSpec:
        """Create a wavelength- and geometry-aware nonuniform grid policy.

        Returns
        -------
        GridSpec
            Immutable wavelength-driven grid policy.
        """
        return cls(
            min_steps_per_wvl=float(min_steps_per_wvl),
            wavelength=wavelength,
            courant=float(courant),
            max_scale=float(max_scale),
            min_steps_per_sim_size=float(min_steps_per_sim_size),
            min_feature_cells=float(min_feature_cells),
            dl_min=dl_min,
            dl_max=dl_max,
            overrides=overrides,
            snapping_points=snapping_points,
            max_cells_per_axis=max_cells_per_axis,
            max_total_cells=max_total_cells,
        )

    @classmethod
    def uniform(
        cls,
        resolution: float,
        *,
        courant: float = 0.99,
        max_cells_per_axis: int | None = 200_000,
        max_total_cells: int | None = 20_000_000,
    ) -> GridSpec:
        """Create a grid specification with an explicit uniform cell size.

        Returns
        -------
        GridSpec
            Immutable uniform-grid policy.
        """
        return cls(
            resolution=float(resolution),
            courant=float(courant),
            max_cells_per_axis=max_cells_per_axis,
            max_total_cells=max_total_cells,
        )

    def _target_spacing(self, *, max_index: float = 1.0) -> float:
        """Return the explicit or wavelength-derived target spacing."""
        if self.resolution is not None:
            return float(self.resolution)
        if self.wavelength is None:
            raise ValueError(
                "GridSpec.auto requires wavelength when resolution is absent."
            )
        return float(self.wavelength) / (
            max(float(max_index), 1.0) * float(self.min_steps_per_wvl)
        )

    def realize(self, design: Any) -> RectilinearGrid:
        """Realize this policy for a concrete design domain and material stack.

        Returns
        -------
        RectilinearGrid
            Physical grid edges satisfying this policy for ``design``.
        """
        from beamz.design.grid.realize import (
            _realize_graded_grid,
            _realize_uniform_grid,
        )

        if self.is_automatic:
            if self.wavelength is None:
                raise ValueError("GridSpec.auto requires wavelength.")
            grid = _realize_graded_grid(design, self)
        else:
            grid = _realize_uniform_grid(design, self._target_spacing(), spec=self)
        dimensions = 3 if float(design.depth) > 0.0 else 2
        return _validate_grid_budget(grid, self, dimensions=dimensions)

    def resolve_time_step(
        self, resolution: float | RectilinearGrid, *, dims: int
    ) -> float:
        """Return a Courant-limited time step in seconds.

        Returns
        -------
        float
            Courant-limited time step in seconds.
        """
        if isinstance(resolution, RectilinearGrid):
            active_axes: tuple[Axis, ...] = (
                ("x", "y", "z") if int(dims) == 3 else ("x", "y")
            )
            return resolution.cfl_time_step(
                self.courant,
                active_axes=active_axes,
            )
        return (
            float(self.courant)
            * float(resolution)
            / (LIGHT_SPEED * np.sqrt(float(max(1, int(dims)))))
        )


def _validate_grid_budget(
    grid: RectilinearGrid, spec: GridSpec, *, dimensions: int
) -> RectilinearGrid:
    """Reject a realized grid before material and field arrays can exhaust memory."""
    _validate_grid_shape_budget(
        grid.shape,
        spec,
        dimensions=dimensions,
        context="realized grid",
        minimum_spacing=grid.minimum_spacing,
    )
    return grid


def _validate_grid_shape_budget(
    shape: tuple[int, int, int],
    spec: GridSpec,
    *,
    dimensions: int,
    context: str,
    minimum_spacing: float | None = None,
) -> None:
    """Reject predicted cell counts before allocating edge or material arrays."""
    dims = 3 if int(dimensions) == 3 else 2
    active_shape = shape[:dims]
    total_cells = math.prod(active_shape)
    violations = []
    if spec.max_cells_per_axis is not None:
        oversized = [
            f"{'xyz'[axis]}={count:,}"
            for axis, count in enumerate(active_shape)
            if count > spec.max_cells_per_axis
        ]
        if oversized:
            violations.append(
                "axis limit "
                f"{spec.max_cells_per_axis:,} exceeded by {', '.join(oversized)}"
            )
    if spec.max_total_cells is not None and total_cells > spec.max_total_cells:
        violations.append(
            f"total limit {spec.max_total_cells:,} exceeded by {total_cells:,} cells"
        )
    if not violations:
        return
    bytes_per_cell = 128 if dims == 3 else 64
    estimated_gib = total_cells * bytes_per_cell / 1024**3
    spacing_detail = (
        ""
        if minimum_spacing is None
        else f" The smallest requested spacing is {minimum_spacing:.6g}."
    )
    raise ValueError(
        f"Grid budget exceeded before allocation for predicted active shape "
        f"{active_shape} ({context}; estimated setup storage {estimated_gib:.2f} "
        f"GiB): {'; '.join(violations)}.{spacing_detail} Increase the matching "
        "GridSpec budget explicitly, raise dl_min, or relax local refinement."
    )
