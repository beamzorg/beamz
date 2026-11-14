"""Calculate electromagnetic overlap gradients."""

from __future__ import annotations

import numpy as np


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
