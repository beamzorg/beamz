"""Realize a GridSpec as a uniform or graded rectilinear grid."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from beamz.design._grid_budget import _validate_grid_shape_budget
from beamz.design._grid_constraints import _clamp_spacing
from beamz.design._grid_geometry import (
    _axis_constraints,
    _design_coordinate_offset,
    _design_extents,
    _geometry_regions,
    _material_index,
)
from beamz.design.grid import RectilinearGrid
from beamz.design.mesher import _GradedMesher

if TYPE_CHECKING:
    from beamz.design.grid_spec import GridSpec


def _realize_uniform_grid(
    design: Any, resolution: float, *, spec: GridSpec | None = None
) -> RectilinearGrid:
    """Cover a design with the exact isotropic spacing used by scalar rasterization."""
    spacing = float(resolution)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Uniform grid resolution must be positive and finite.")

    extents = _design_extents(design)
    nx, ny = (
        _uniform_cell_count(extents[0], spacing),
        _uniform_cell_count(extents[1], spacing),
    )
    dimensions = 3 if float(design.depth) > 0.0 else 2
    nz = _uniform_cell_count(extents[2], spacing) if dimensions == 3 else 1
    if spec is not None:
        _validate_grid_shape_budget(
            (nx, ny, nz),
            spec,
            dimensions=dimensions,
            context="uniform spacing preflight",
            minimum_spacing=spacing,
        )
    if float(design.depth) > 0.0:
        return RectilinearGrid.uniform(
            (0.0, 0.0, 0.0),
            (nx * spacing, ny * spacing, nz * spacing),
            (nx, ny, nz),
        )
    return RectilinearGrid.uniform(
        (0.0, 0.0, 0.0),
        (nx * spacing, ny * spacing, 1.0),
        (nx, ny, 1),
    )


def _uniform_cell_count(extent: float, spacing: float) -> int:
    ratio = float(extent) / spacing
    tolerance = 16.0 * np.finfo(float).eps * max(1.0, abs(ratio))
    return max(1, math.ceil(ratio - tolerance))


def _lower_bound_cells(coords: np.ndarray, limits: np.ndarray) -> int:
    return sum(
        max(1, math.ceil(float(length) / float(limit) - 1e-12))
        for length, limit in zip(np.diff(coords), limits, strict=True)
    )


def _realize_graded_grid(design: Any, spec: GridSpec) -> RectilinearGrid:
    if spec.wavelength is None:
        raise ValueError("GridSpec.auto requires wavelength.")
    wavelength = float(spec.wavelength)
    extents = _design_extents(design)
    coordinate_offset = _design_coordinate_offset(design)
    regions = _geometry_regions(design, spec, coordinate_offset)
    background_spacing = _clamp_spacing(
        wavelength / (_material_index(design.background) * spec.min_steps_per_wvl),
        spec,
    )
    active_count = 2 if float(design.depth) == 0.0 else 3
    constraints = []
    predicted_counts = []
    minimum_spacing = np.inf
    for axis in range(active_count):
        domain_limit = extents[axis] / spec.min_steps_per_sim_size
        coords, limits = _axis_constraints(
            axis,
            extents[axis],
            min(background_spacing, domain_limit),
            regions,
            spec,
            coordinate_offset[axis],
        )
        constraints.append((coords, limits))
        minimum_spacing = min(minimum_spacing, float(np.min(limits)))
        predicted_counts.append(_lower_bound_cells(coords, limits))
    if active_count == 2:
        predicted_counts.append(1)
    _validate_grid_shape_budget(
        cast(tuple[int, int, int], tuple(predicted_counts)),
        spec,
        dimensions=active_count,
        context="graded-spacing lower-bound preflight",
        minimum_spacing=float(minimum_spacing),
    )
    return _mesh_constraints(
        constraints,
        predicted_counts,
        spec,
        active_count=active_count,
        minimum_spacing=float(minimum_spacing),
    )


def _mesh_constraints(
    constraints: list[tuple[np.ndarray, np.ndarray]],
    predicted_counts: list[int],
    spec: GridSpec,
    *,
    active_count: int,
    minimum_spacing: float,
) -> RectilinearGrid:
    mesher = _GradedMesher(max_scale=spec.max_scale)
    edges = []
    for axis, (coords, limits) in enumerate(constraints):
        axis_edges = mesher.make_axis_edges(
            coords,
            limits,
            max_cells=spec.max_cells_per_axis,
            axis_name="xyz"[axis],
        )
        edges.append(axis_edges)
        predicted_counts[axis] = int(axis_edges.size - 1)
        _validate_grid_shape_budget(
            cast(tuple[int, int, int], tuple(predicted_counts)),
            spec,
            dimensions=active_count,
            context=f"after realizing {'xyz'[axis]} axis",
            minimum_spacing=minimum_spacing,
        )
    if active_count == 2:
        edges.append(np.asarray([0.0, 1.0]))
    return RectilinearGrid(*edges)
