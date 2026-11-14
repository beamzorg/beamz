"""Realize uniform and geometry-aware rectilinear grids."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from beamz.design.grid import RectilinearGrid
from beamz.design.grid.features.constraints import _axis_constraints, _clamp_spacing
from beamz.design.grid.features.regions import (
    _design_extents,
    _geometry_regions,
    _material_index,
)
from beamz.design.grid.mesher import _GradedMesher
from beamz.design.grid.spec import AxisValues, GridSpec, _validate_grid_shape_budget


def _realize_uniform_grid(
    design: Any,
    resolution: float,
    *,
    spec: GridSpec | None = None,
    dimensions: int | None = None,
) -> RectilinearGrid:
    """Cover a design with the exact isotropic spacing used by scalar rasterization."""
    spacing = float(resolution)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Uniform grid resolution must be positive and finite.")

    def count(extent: float) -> int:
        ratio = float(extent) / spacing
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, abs(ratio))
        return max(1, math.ceil(ratio - tolerance))

    extents = _design_extents(design)
    nx, ny = count(extents[0]), count(extents[1])
    dimensions = (
        (3 if float(design.depth) > 0.0 else 2) if dimensions is None else dimensions
    )
    nz = count(float(design.depth)) if dimensions == 3 else 1
    if spec is not None:
        _validate_grid_shape_budget(
            (nx, ny, nz),
            spec,
            dimensions=dimensions,
            context="uniform spacing preflight",
            minimum_spacing=spacing,
        )
    return RectilinearGrid.uniform(
        (0.0, 0.0, 0.0),
        (nx * spacing, ny * spacing, nz * spacing if dimensions == 3 else 1.0),
        (nx, ny, nz),
    )


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
    dimensions = active_count
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
        predicted_counts.append(
            sum(
                max(1, math.ceil(float(length) / float(limit) - 1e-12))
                for length, limit in zip(np.diff(coords), limits, strict=True)
            )
        )
    if active_count == 2:
        predicted_counts.append(1)
    _validate_grid_shape_budget(
        cast(tuple[int, int, int], tuple(predicted_counts)),
        spec,
        dimensions=dimensions,
        context="graded-spacing lower-bound preflight",
        minimum_spacing=float(minimum_spacing),
    )

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
            dimensions=dimensions,
            context=f"after realizing {'xyz'[axis]} axis",
            minimum_spacing=float(minimum_spacing),
        )
    if active_count == 2:
        edges.append(np.asarray([0.0, 1.0]))
    return RectilinearGrid(*edges)
