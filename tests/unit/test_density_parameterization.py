"""Native parameter maps retain transform and interpolation adjoints."""

import jax
import jax.numpy as jnp
import numpy as np

from beamz.optimization.autodiff import transform_density
from beamz.optimization.region import (
    DensityParameterization,
    FilterProject,
    TopologyDesignRegion,
)


def test_region_map_interpolation_and_fixed_cell_pullback():
    solid = np.zeros((2, 3), dtype=bool)
    solid[0, 0] = True
    region = TopologyDesignRegion(
        size=(3.0, 2.0, 2.0),
        center=(0, 0, 0),
        eps_bounds=(1, 4),
        pixel_size=1.0,
        transformations=(FilterProject(1.1),),
        fixed_solid=solid,
    )
    mask = np.zeros((4, 8, 10), dtype=bool)
    slices = (slice(1, 3), slice(2, 6), slice(2, 8))
    mask[slices] = True
    mapping = DensityParameterization(
        region.params_shape,
        1.0,
        (1, 4),
        region.material_density,
        solid,
        None,
        mask,
        slices,
    )
    params = jnp.full(region.params_shape, 0.4)
    base = jnp.full(mask.shape, 2.0)

    def reference(values):
        density = jax.image.resize(
            region.material_density(values, beta=3.0),
            (4, 6),
            method="linear",
            antialias=False,
        )
        return base.at[slices].set(1 + 3 * density)

    def bound(values):
        return mapping.permittivity(values, 3.0, base)

    np.testing.assert_allclose(bound(params), reference(params))
    weights = jnp.arange(base.size).reshape(base.shape)
    expected = jax.grad(lambda values: jnp.sum(reference(values) * weights))(params)
    actual = jax.grad(lambda values: jnp.sum(bound(values) * weights))(params)
    np.testing.assert_allclose(actual, expected)
    assert actual[0, 0] == 0


def test_masked_map_ignores_external_latent_values_but_keeps_fixed_context():
    from functools import partial

    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    context = np.zeros_like(mask)
    context[3, :2] = True
    transform = partial(
        transform_density,
        mask=mask,
        eta=0.5,
        radius=2,
        fixed_structure_mask=context,
    )
    mapping = DensityParameterization(mask.shape, 1.0, (1, 4), transform, mask=mask)
    params = jnp.full(mask.shape, 0.4)
    expected = transform(jnp.where(mask, params, 0), beta=2.0)
    np.testing.assert_allclose(mapping.physical(params, 2.0), expected)
    gradient = jax.grad(lambda values: jnp.sum(mapping.physical(values, 2.0)))(params)
    assert np.all(np.asarray(gradient)[~mask] == 0)
