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
    """
    Generate a 2D conic kernel (linear decay).
    w(r) = max(0, 1 - r/R)
    """
    radius = int(max(1, radius))
    y, x = jnp.ogrid[-radius : radius + 1, -radius : radius + 1]
    weights = jnp.maximum(0.0, 1.0 - jnp.sqrt(x**2 + y**2) / radius)
    return weights / jnp.sum(weights)


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
    padded = jnp.pad(jnp.asarray(filter_input), radius, mode="edge")
    filtered = convolve2d(padded, generate_conic_kernel(radius), mode="valid")
    return jnp.where(mask, filtered, 0.0), jnp.ones_like(mask)


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


@partial(jax.jit, static_argnames=["radius", "operation"])
def masked_morphological_filter(
    values, mask, radius, operation="openclose", tau=0.05, fixed_structure_mask=None
):
    """
    Apply masked morphological filtering with hard boundaries (literature-standard).

    Args:
        values: Density field
        mask: Design region mask (hard boolean boundary)
        radius: Filter radius in cells
        operation: 'erosion', 'dilation', 'opening', 'closing', 'openclose' (opening then closing)
        tau: Smoothness temperature for differentiable min/max
        fixed_structure_mask: Optional boolean mask of fixed solid structures (e.g. waveguides)
                              used to provide context for filtering without forcing values.
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
):
    """
    Full density transform: Filter -> Project (literature-standard).
    Returns the physical density [0, 1] with hard boundary masking.

    Args:
        filter_type: 'morphological' or 'conic' (recommended)
        morphology_operation: 'opening', 'closing', 'openclose'
        fixed_structure_mask: Optional mask for fixed structures
        projection_type: 'heaviside' (default) or 'ssp'
        ssp_smoothing_radius: SSP smoothing radius in grid-cell units
    """
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

    return project_density(
        filtered,
        beta,
        eta,
        projection_type=projection_type,
        ssp_smoothing_radius=ssp_smoothing_radius,
    )


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
):
    """
    Compute gradient w.r.t. design density using VJP.
    Supports both morphological and conic filters with hard boundary masking (literature-standard).
    """

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
        )

    _, pullback = jax.vjp(transform_wrapper, density)
    return pullback(grad_physical)[0]
