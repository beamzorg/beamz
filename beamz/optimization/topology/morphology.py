"""Differentiable grayscale morphology."""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=["axis"])
def smooth_max(x, axis=None, tau=0.1):
    """
    Smooth maximum approximation: tau * log(sum(exp(x/tau)))
    Also known as LogSumExp.
    """
    return tau * jax.scipy.special.logsumexp(x / tau, axis=axis)


@partial(jax.jit, static_argnames=["axis"])
def smooth_min(x, axis=None, tau=0.1):
    """
    Smooth minimum approximation: -smooth_max(-x)
    """
    return -smooth_max(-x, axis=axis, tau=tau)


@partial(jax.jit, static_argnames=["radius"])
def grayscale_erosion(values, radius, tau=0.05):
    """
    Grayscale erosion using smooth minimum filter with a disk structuring element.
    Uses 2D shifts to implement isotropic erosion.
    """
    radius = int(max(0, radius))
    if radius <= 0:
        return values

    # One halo handles every disk offset, including radii larger than the array.
    padded = jnp.pad(values, radius, mode="edge")
    ny, nx = values.shape
    neighbors = [
        padded[radius - dy : radius - dy + ny, radius - dx : radius - dx + nx]
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dy * dy + dx * dx <= radius * radius
    ]
    return smooth_min(jnp.stack(neighbors), axis=0, tau=tau)


@partial(jax.jit, static_argnames=["radius"])
def grayscale_dilation(values, radius, tau=0.05):
    """Disk dilation, the dual of erosion under sign reversal."""
    return -grayscale_erosion(-values, radius, tau)


@partial(jax.jit, static_argnames=["radius"])
def grayscale_opening(values, radius, tau=0.05):
    """Opening: Erosion followed by Dilation."""
    return grayscale_dilation(grayscale_erosion(values, radius, tau), radius, tau)


@partial(jax.jit, static_argnames=["radius"])
def grayscale_closing(values, radius, tau=0.05):
    """Closing: Dilation followed by Erosion."""
    return grayscale_erosion(grayscale_dilation(values, radius, tau), radius, tau)
