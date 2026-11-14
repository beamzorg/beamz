"""Masked density morphology filters."""

from functools import partial

import jax
import jax.numpy as jnp

from beamz.optimization.topology.morphology import (
    grayscale_closing,
    grayscale_dilation,
    grayscale_erosion,
    grayscale_opening,
)


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
