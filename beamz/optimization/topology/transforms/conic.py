"""Conic density kernels and masked filtering."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve


def _conic_kernel(radius, ndim, xp):
    """Normalized radial weights; callers choose host or traced construction."""
    coordinates = xp.meshgrid(*([xp.arange(-radius, radius + 1)] * ndim), indexing="ij")
    distance = xp.sqrt(sum(axis**2 for axis in coordinates))
    weights = xp.maximum(0.0, 1.0 - distance / radius)
    return weights / weights.sum()


def _conic_filter(values, radius, padding, kernel):
    """Convolve a region-local halo, keeping the 3D FFT specialization."""
    return convolve(
        jnp.pad(values, radius, mode=padding),
        kernel,
        mode="valid",
        method="fft" if values.ndim == 3 else "direct",
    )


@partial(jax.jit, static_argnames=["radius"])
def generate_conic_kernel(radius: int):
    """
    Generate a 2D conic kernel (linear decay).
    w(r) = max(0, 1 - r/R)
    """
    return _conic_kernel(int(max(1, radius)), 2, jnp)


@partial(jax.jit, static_argnames=["radius"])
def masked_conic_filter(values, mask, radius: int, fixed_structure_mask=None):
    """
    Apply conic filter with hard mask boundaries (literature-standard).
    Smoothness comes ONLY from the filter kernel radius.

    This follows standard topology optimization practice (Bendsøe, Sigmund, Hammond):
    - Hard boolean mask defines optimization region
    - Filter kernel radius controls minimum feature size
    - No soft boundary blending or protection zones
    """
    radius = int(max(0, radius))
    if radius <= 0:
        return jnp.where(mask, values, 0.0), jnp.ones_like(mask)

    # Fixed waveguides supply filter context without extending the design mask.
    filter_input = (
        values
        if fixed_structure_mask is None
        else jnp.where(fixed_structure_mask, 1.0, values)
    )
    filtered = _conic_filter(
        jnp.asarray(filter_input), radius, "edge", generate_conic_kernel(radius)
    )
    return jnp.where(mask, filtered, 0.0), jnp.ones_like(mask)
