"""Apply fixed constraints and density transforms."""

from functools import partial

import jax
import jax.numpy as jnp

from beamz.optimization.topology.filtering import masked_morphological_filter
from beamz.optimization.topology.projections import project_density
from beamz.optimization.topology.transforms.conic import masked_conic_filter

_TRANSFORM_STATIC_ARGS = (
    "radius",
    "filter_type",
    "morphology_operation",
    "projection_type",
    "ssp_smoothing_radius",
)


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
