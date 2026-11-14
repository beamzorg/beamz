"""Backpropagate gradients through density transforms."""

import jax

from beamz.optimization.topology.transforms.density import transform_density


def compute_parameter_gradient_vjp(density, grad_physical, *args, **options):
    """Compatibility VJP of the authoritative density transform.

    Positional and keyword transform options retain their original ordering;
    JIT specialization belongs to transform_density, not a second signature.
    """
    _, pullback = jax.vjp(
        lambda values: transform_density(values, *args, **options), density
    )
    return pullback(grad_physical)[0]
