"""JAX-based autodifferentiation helpers for topology optimization."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d

from .projections import project_density, smoothed_heaviside  # noqa: F401

_TRANSFORM_STATIC_ARGS = (
    "radius",
    "filter_type",
    "morphology_operation",
    "projection_type",
    "ssp_smoothing_radius",
)


@partial(jax.jit, static_argnames=["radius"])
def generate_conic_kernel(radius: int):
    """Return a normalized 2D conic kernel, ``max(0, 1 - r/R)``."""
    radius = int(max(1, radius))
    y, x = jnp.ogrid[-radius : radius + 1, -radius : radius + 1]
    weights = jnp.maximum(0.0, 1.0 - jnp.sqrt(x**2 + y**2) / radius)
    return weights / jnp.sum(weights)


@partial(jax.jit, static_argnames=["radius"])
def masked_conic_filter(values, mask, radius: int, fixed_structure_mask=None):
    """Conic-filter inside a hard design mask. The radius is the feature scale."""
    radius = int(max(0, radius))
    if radius <= 0:
        return jnp.where(mask, values, 0.0), jnp.ones_like(mask)

    # Fixed waveguides supply filter context without extending the design mask.
    filter_input = (
        values
        if fixed_structure_mask is None
        else jnp.where(fixed_structure_mask, 1.0, values)
    )
    padded = jnp.pad(jnp.asarray(filter_input), radius, mode="edge")
    filtered = convolve2d(padded, generate_conic_kernel(radius), mode="valid")
    return jnp.where(mask, filtered, 0.0), jnp.ones_like(mask)


@partial(jax.jit, static_argnames=["axis"])
def smooth_max(x, axis=None, tau=0.1):
    """Smooth maximum, ``tau * logsumexp(x / tau)``."""
    return tau * jax.scipy.special.logsumexp(x / tau, axis=axis)


@partial(jax.jit, static_argnames=["axis"])
def smooth_min(x, axis=None, tau=0.1):
    """Smooth minimum, ``-smooth_max(-x)``."""
    return -smooth_max(-x, axis=axis, tau=tau)


@partial(jax.jit, static_argnames=["radius"])
def grayscale_erosion(values, radius, tau=0.05):
    """Isotropic grayscale erosion with a smooth disk minimum."""
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


@partial(jax.jit, static_argnames=["radius", "operation"])
def masked_morphological_filter(
    values, mask, radius, operation="openclose", tau=0.05, fixed_structure_mask=None
):
    """Morphological filter inside a hard mask.

    ``operation`` is erosion, dilation, opening, closing, or openclose.
    Fixed-solid cells supply filter context and are not written into the result.
    """
    # Treat fixed structures as solid context, then mask the filtered result.
    filtered = (
        values
        if fixed_structure_mask is None
        else jnp.where(fixed_structure_mask, 1.0, values)
    )
    if operation == "erosion":
        filtered = grayscale_erosion(filtered, radius, tau)
    elif operation == "dilation":
        filtered = grayscale_dilation(filtered, radius, tau)
    elif operation == "opening":
        filtered = grayscale_opening(filtered, radius, tau)
    elif operation == "closing":
        filtered = grayscale_closing(filtered, radius, tau)
    elif operation == "openclose":
        # Opening then Closing is a standard noise removal filter
        filtered = grayscale_closing(
            grayscale_opening(filtered, radius, tau), radius, tau
        )

    return jnp.where(mask, jnp.asarray(filtered), 0.0)


def apply_fixed_density(density, fixed_solid=None, fixed_void=None):
    """Apply hard material constraints with zero gradients at fixed cells."""
    density = jnp.asarray(density)
    if fixed_solid is not None:
        density = jnp.where(fixed_solid, 1.0, density)
    if fixed_void is not None:
        density = jnp.where(fixed_void, 0.0, density)
    return density


@partial(jax.jit, static_argnames=_TRANSFORM_STATIC_ARGS)
def transform_density(
    density,
    mask,
    beta,
    eta,
    radius,
    filter_type="conic",
    morphology_operation="openclose",
    morphology_tau=0.05,
    fixed_structure_mask=None,
    projection_type="heaviside",
    ssp_smoothing_radius=0.55,
    fixed_solid=None,
    fixed_void=None,
):
    """Filter then project density onto ``[0, 1]`` inside the hard mask.

    ``filter_type`` is ``"conic"`` or ``"morphological"``. ``projection_type`` is
    ``"heaviside"``, ``"ssp"``, or ``"identity"``.
    """
    density = apply_fixed_density(density, fixed_solid, fixed_void)
    # Apply filter (returns hard-masked result)
    if filter_type == "morphological":
        filtered = masked_morphological_filter(
            density,
            mask,
            radius,
            morphology_operation,
            morphology_tau,
            fixed_structure_mask,
        )
    elif filter_type == "conic":
        # Conic filter (for geometric constraints)
        filtered, _ = masked_conic_filter(density, mask, radius, fixed_structure_mask)
    else:
        raise ValueError(
            f"Unknown filter_type: {filter_type}. Use 'conic' or 'morphological'."
        )

    physical = project_density(
        filtered,
        beta,
        eta,
        projection_type=projection_type,
        ssp_smoothing_radius=ssp_smoothing_radius,
    )

    return apply_fixed_density(physical, fixed_solid, fixed_void)


@partial(jax.jit, static_argnames=_TRANSFORM_STATIC_ARGS)
def compute_parameter_gradient_vjp(
    density,
    grad_physical,
    mask,
    beta,
    eta,
    radius,
    filter_type="conic",
    morphology_operation="openclose",
    morphology_tau=0.05,
    fixed_structure_mask=None,
    projection_type="heaviside",
    ssp_smoothing_radius=0.55,
    fixed_solid=None,
    fixed_void=None,
):
    """Return the latent-density VJP of :func:`transform_density`."""

    # Define a wrapper for the transform to differentiate
    def transform_wrapper(d):
        return transform_density(
            d,
            mask,
            beta,
            eta,
            radius,
            filter_type,
            morphology_operation,
            morphology_tau,
            fixed_structure_mask,
            projection_type,
            ssp_smoothing_radius,
            fixed_solid,
            fixed_void,
        )

    _, pullback = jax.vjp(transform_wrapper, density)
    return pullback(grad_physical)[0]
