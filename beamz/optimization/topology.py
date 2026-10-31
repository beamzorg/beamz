"""Mathematical helpers for custom density-optimization loops."""

from __future__ import annotations

from typing import Any

import numpy as np

from beamz.design.core import Design
from beamz.design.materials import Material
from beamz.design.structures import Structure


def compute_overlap_gradient(
    forward_fields_history,
    adjoint_fields_history,
    field_key="Ez",
    forward_start=0,
    adjoint_start=0,
):
    """
    Compute the gradient of the overlap integral with respect to epsilon.
    Gradient = Re(E_fwd * E_adj) integrated over time.
    """
    if len(forward_fields_history) == 0 or len(adjoint_fields_history) == 0:
        raise ValueError("Field histories must be non-empty.")

    forward_start = max(int(forward_start), 0)
    adjoint_start = max(int(adjoint_start), 0)
    fwd_hist = forward_fields_history[forward_start:]
    adj_hist = adjoint_fields_history[adjoint_start:]

    grad = np.zeros_like(forward_fields_history[0], dtype=float)

    n_steps = min(len(fwd_hist), len(adj_hist))
    if n_steps <= 0:
        return grad

    for i in range(n_steps):
        grad += fwd_hist[i] * adj_hist[n_steps - 1 - i]

    return grad


def fold_high_side_yee_padding_to_shape(
    grad: np.ndarray,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Fold high-side Yee padding planes back onto a material-grid shape.

    Some full-Yee field components include the high boundary sample explicitly,
    so their field-derived gradients can be one element larger than the
    cell-centered material grid. Material sampling clips those high-side samples
    to the last material cell; this helper applies the corresponding adjoint by
    accumulating the high-side plane into the last target index.
    """

    out = np.asarray(grad, dtype=float)
    target_shape = tuple(int(v) for v in target_shape)
    if out.shape == target_shape:
        return out
    if out.ndim != len(target_shape):
        raise ValueError(
            f"Gradient ndim {out.ndim} does not match target ndim {len(target_shape)}."
        )

    for axis, target in enumerate(target_shape):
        size = out.shape[axis]
        if size == target:
            continue
        if size != target + 1:
            raise ValueError(
                "Cannot align gradient shape "
                f"{out.shape} to target shape {target_shape}: axis {axis} has "
                f"size {size}, expected {target} or {target + 1}."
            )

        core: list[Any] = [slice(None)] * out.ndim
        core[axis] = slice(0, target)
        folded = out[tuple(core)].copy()

        last: list[Any] = [slice(None)] * out.ndim
        last[axis] = target - 1
        high: list[Any] = [slice(None)] * out.ndim
        high[axis] = target
        folded[tuple(last)] += out[tuple(high)]
        out = folded

    return out


def create_optimization_mask(grid, region_structure):
    """
    Helper to create a boolean mask from a structure on a grid.
    Uses rasterization to ensure exact alignment with how structures are mapped to the grid.
    """
    # Create temp design to rasterize mask exactly as grid does
    temp_design = Design(
        width=grid.width, height=grid.height, material=Material(permittivity=1.0)
    )

    # Set to a distinct permittivity to detect it without mutating the input.
    marker = Material(permittivity=2.0)
    if not isinstance(region_structure, Structure):
        raise TypeError("Optimization regions must be canonical Structure objects.")
    temp_design += region_structure.with_material(marker)

    # Use the production Rust rasterizer so optimization masks follow the same
    # geometry classification as the solver material grid.
    temp_grid = temp_design.rasterize(resolution=grid.dx)

    # Mask is where permittivity > background
    # Use a safe threshold to include any partial fill
    mask = np.asarray(temp_grid.permittivity) > 1.001

    return mask
