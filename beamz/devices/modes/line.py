"""Shared preparation and selection for 2D preview, injection and projection.

The eigensolver sees cell materials, never a consumer's staggered samples.
Yee conversion and power normalization belong to the individual consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from beamz.devices._placement import (
    SnappedRegion,
    snap_mode_source_region,
    snap_mode_source_region_grid,
)

from .policy import candidate_rank, solve_interval


@dataclass(frozen=True)
class ModeLineContext:
    eps: np.ndarray
    crop: slice
    aperture: slice
    edges: np.ndarray
    snapped: SnappedRegion | None
    transverse_axis: str


def prepare_mode_line(
    *,
    permittivity,
    center,
    size,
    axis,
    spec,
    resolution,
    grid=None,
    origin=(0, 0),
    full_shape=None,
    snapped=None,
):
    """Extract the same finite, explicitly padded cell profile for every consumer."""
    permittivity = np.asarray(permittivity)
    if permittivity.ndim != 2 or axis not in {"x", "y"}:
        raise ValueError("2D mode ports require an x- or y-normal line in an xy grid")
    shape = permittivity.shape if full_shape is None else full_shape
    transverse = "y" if axis == "x" else "x"
    width = size[1 if axis == "x" else 0]
    if snapped is None:
        kwargs: dict[str, Any] = dict(
            center=center,
            width=width,
            height=None,
            axis=axis,
            direction_sign=1.0,
            is_3d=False,
        )
        snapped = (
            snap_mode_source_region_grid(**kwargs, grid=grid)
            if grid is not None
            and grid.metric_kind_for(("x", "y")) != "isotropic_uniform"
            else snap_mode_source_region(
                **kwargs, grid_shape=shape, resolution=resolution
            )
        )
    interval = snapped.axis_interval(transverse)
    if interval is None:
        raise ValueError("A 2D mode port requires a transverse aperture")
    normal_dim, transverse_dim = (1, 0) if axis == "x" else (0, 1)
    normal = int(snapped.plane_index) - int(origin[normal_dim])
    padding = spec.solver_padding_cells
    pad = (
        0
        if padding is None
        else int(padding[0] if isinstance(padding, tuple) else padding)
    )
    lo, hi = solve_interval(interval.start, interval.stop, shape[transverse_dim], pad)
    local_lo, local_hi = lo - origin[transverse_dim], hi - origin[transverse_dim]
    if (
        normal < 0
        or normal >= permittivity.shape[normal_dim]
        or local_lo < 0
        or local_hi > permittivity.shape[transverse_dim]
    ):
        raise ValueError(
            "Stored modal materials do not contain the requested solve bounds"
        )
    profile = permittivity[:, normal] if axis == "x" else permittivity[normal, :]
    edges = (
        np.asarray(grid.axis_edges(transverse))
        if grid is not None
        else np.arange(shape[transverse_dim] + 1) * resolution
    )
    return ModeLineContext(
        np.asarray(profile[local_lo:local_hi]),
        slice(lo, hi),
        slice(int(interval.start) - lo, int(interval.stop) - lo),
        edges[lo : hi + 1],
        snapped,
        transverse,
    )


def solve_mode_line(
    context,
    *,
    spec,
    frequency,
    resolution,
    direction,
    polarization,
    solver,
    diagnostics=None,
    use_grid_edges=True,
):
    """Use a common search target, candidate count, ordering and confinement policy."""
    requested = max(spec.num_modes, spec.mode_index + 1)
    count = requested
    minimum = spec.min_aperture_fraction
    limit = max(count, min(64, max(1, 2 * context.eps.size - 3)))
    weights = np.diff(context.edges)
    while True:
        neffs, electric, magnetic, _ = solver(
            eps=context.eps,
            omega=2 * np.pi * frequency,
            dL=resolution,
            m=count,
            direction=direction,
            filter_pol=polarization,
            target_neff=spec.target_neff,
            force_symmetry=spec.force_symmetry,
            return_fields=True,
            grid_edges=(context.edges,) if use_grid_edges else None,
        )
        electric, magnetic = np.asarray(electric), np.asarray(magnetic)
        # Both tangential E components contribute to confinement; this is an
        # intensity diagnostic, not modal power or electromagnetic energy.
        tangential = [1, 2] if direction[-1] == "x" else [0, 2]
        intensity = np.sum(abs(electric[:, tangential]) ** 2, axis=1)
        fractions = np.sum(
            intensity[:, context.aperture] * weights[context.aperture], axis=1
        ) / np.maximum(np.sum(intensity * weights, axis=1), np.finfo(float).tiny)
        first = np.sum(abs(electric[:, tangential[0]]) ** 2, axis=1)
        second = np.sum(abs(electric[:, tangential[1]]) ** 2, axis=1)
        matches = (
            np.ones(len(neffs), dtype=bool)
            if polarization is None
            else (first >= second if polarization == "te" else second > first)
        )
        qualifying = matches if minimum is None else matches & (fractions >= minimum)
        if (
            minimum is None
            or np.count_nonzero(qualifying) >= requested
            or count >= limit
        ):
            break
        count = min(2 * count, limit)
    order = sorted(
        range(len(neffs)),
        key=lambda i: candidate_rank(
            neffs[i],
            polarization_matches=matches[i],
            aperture_fraction=fractions[i],
            minimum=minimum,
        ),
        reverse=True,
    )
    if spec.mode_index >= len(order):
        raise ValueError(
            f"Requested mode_index={spec.mode_index}, but only {len(order)} modes are available"
        )
    if minimum is not None and np.count_nonzero(qualifying) < requested:
        raise ValueError(
            "No requested mode meets min_aperture_fraction in the candidate pool"
        )
    if diagnostics is not None:
        diagnostics.update(
            solve_bounds=(float(context.edges[0]), float(context.edges[-1])),
            solver_padding_cells=spec.solver_padding_cells,
            candidate_count=count,
            target_neff=spec.target_neff,
            aperture_fractions=tuple(float(fractions[i]) for i in order[:requested]),
            field_representation="solver",
            processing_stage="solver_fields_before_yee_conversion",
        )
    return (
        np.asarray(neffs)[order[:requested]],
        electric[order[:requested]],
        magnetic[order[:requested]],
    )
