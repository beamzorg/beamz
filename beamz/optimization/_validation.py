"""Validation shared by density APIs and optimizer continuation."""

from numbers import Integral

import numpy as np


def execution_backend(backend, gradient_backend):
    """Validate native execution separately from the gradient method."""
    if backend not in ("jax", "cuda"):
        raise ValueError("Inverse-design backend must be 'jax' or 'cuda'.")
    if backend == "cuda" and gradient_backend != "autodiff":
        raise ValueError(
            "backend='cuda' requires gradient_backend='autodiff'; "
            "the spectral adjoint requires backend='jax'."
        )
    return backend


def step_count(value, name="step", *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        bound = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a {bound} integer.")
    return int(value)


def check_density(values, shape, name="Density"):
    values = np.asarray(values)
    if (
        values.shape != shape
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError(f"{name} must match shape {shape} and be finite in [0, 1].")


def fixed_density_masks(solid, void, shape):
    """Copy optional, disjoint boolean constraints on physical density cells."""
    masks = []
    for name, value in (("fixed_solid", solid), ("fixed_void", void)):
        if value is None:
            masks.append(None)
            continue
        mask = np.array(value, copy=True)
        if mask.shape != shape or mask.dtype != np.bool_:
            raise ValueError(f"{name} must be a boolean array with shape {shape}.")
        mask.setflags(write=False)
        masks.append(mask)
    if all(m is not None for m in masks) and np.any(masks[0] & masks[1]):
        raise ValueError("fixed_solid and fixed_void must not overlap.")
    return tuple(masks)
