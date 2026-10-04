"""Circular-bend coordinate transformation for the raster mode solver."""

from __future__ import annotations

import numpy as np

from .models import Materials


def bend_materials(
    materials: Materials, radius: float, axis: str
) -> tuple[Materials, np.ndarray]:
    """Return equivalent straight tensors and the longitudinal metric h.

    In the local orthonormal (x, y, tangent) frame, ds_physical=h ds,
    h=1+u/R. Maxwell's pullback gives T'=h A T A, A=diag(1,1,1/h),
    for BOTH epsilon and mu. Input tensors must be invariant along the arc
    in this rotating frame. See Shyroki, arXiv:physics/0605002, Eqs. (7,12,13).
    """
    if axis not in {"x", "y"}:
        raise ValueError("bend_axis must be 'x' or 'y'")
    if not np.isfinite(radius) or radius == 0:
        raise ValueError(
            "bend_radius must be finite and nonzero; use None for straight"
        )
    edges = np.asarray(
        materials.grid.x_edges if axis == "x" else materials.grid.y_edges
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        edge_metric = 1.0 + edges / radius
    if not np.all(np.isfinite(edge_metric)) or np.any(edge_metric <= 0):
        raise ValueError("bend domain must satisfy 1 + coordinate / bend_radius > 0")
    # Match the supplied raster material sampling and Result's cell centers.
    metric = (edge_metric[:-1] + edge_metric[1:]) / 2
    metric = metric[:, None] if axis == "x" else metric[None, :]
    metric = np.broadcast_to(metric, materials.shape)
    factor = np.ones((3, 3, *materials.shape))
    factor[:2, :2] = metric
    factor[2, 2] = 1.0 / metric
    return (
        Materials(
            materials.grid, materials.eps_tensor * factor, materials.mu_tensor * factor
        ),
        metric,
    )
