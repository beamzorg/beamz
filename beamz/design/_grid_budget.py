"""Reject realized grids before edge or material arrays are allocated."""

from __future__ import annotations

import math

from beamz.design.grid import RectilinearGrid

# Bytes used only to explain a rejected shape. The check itself never allocates.
_BYTES_PER_CELL = {2: 64, 3: 128}


def _validate_grid_budget(
    grid: RectilinearGrid, spec, *, dimensions: int
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
    spec,
    *,
    dimensions: int,
    context: str,
    minimum_spacing: float | None = None,
) -> None:
    """Reject predicted cell counts before allocating edge or material arrays."""
    dims = 3 if int(dimensions) == 3 else 2
    active_shape = shape[:dims]
    total_cells = math.prod(active_shape)
    violations = _budget_violations(active_shape, total_cells, spec)
    if not violations:
        return
    estimated_gib = total_cells * _BYTES_PER_CELL[dims] / 1024**3
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


def _budget_violations(
    active_shape: tuple[int, ...], total_cells: int, spec
) -> list[str]:
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
    return violations
