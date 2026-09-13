"""Region transforms/penalties are differentiable, SI-based, and reproducible."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.plugins.invdes import (
    CustomInitializationSpec,
    ErosionDilationPenalty,
    FilterProject,
    RandomInitializationSpec,
    TopologyDesignRegion,
)


def test_filter_and_penalty_gradient_and_constant_limits():
    values = np.random.default_rng(5).uniform(0.2, 0.8, (11, 12)).astype(np.float32)
    transform = FilterProject(230e-9, beta=4)
    penalty = ErosionDilationPenalty(230e-9, beta=8, weight=0.8)

    def objective(x):
        density = transform(x, 100e-9)
        return jnp.mean(density**2) - penalty(density, 100e-9)

    gradient = jax.grad(objective)(values)
    direction = gradient / jnp.linalg.norm(gradient)
    h = 0.003
    fd = (objective(values + h * direction) - objective(values - h * direction)) / (
        2 * h
    )
    np.testing.assert_allclose(fd, jnp.sum(gradient * direction), rtol=0.002)
    for value in (0, 1):
        constant = jnp.full((11, 12), value, dtype=jnp.float32)
        np.testing.assert_allclose(transform(constant, 100e-9), value, atol=3e-7)
        assert np.all(np.asarray(transform(constant, 100e-9)) >= 0)
        assert np.all(np.asarray(transform(constant, 100e-9)) <= 1)
        assert abs(float(penalty(constant, 100e-9))) < 1e-6
        assert np.isfinite(jax.grad(lambda x: penalty(x, 100e-9))(constant)).all()


def test_region_initialization_and_transform_order():
    r = TopologyDesignRegion(
        size=(1.2e-6, 1e-6, np.inf),
        center=(0, 0, 0),
        eps_bounds=(1, 4),
        pixel_size=100e-9,
        transformations=(FilterProject(150e-9, beta=2), FilterProject(100e-9, beta=3)),
        initialization_spec=RandomInitializationSpec(seed=3),
    )
    x = r.initial_parameters
    assert x.shape == (10, 12)
    np.testing.assert_array_equal(x, r.initial_parameters)
    np.testing.assert_allclose(
        r.material_density(x),
        r.transformations[1](r.transformations[0](x, 100e-9), 100e-9),
    )
    copy = CustomInitializationSpec(x)
    x[:] = 0
    assert np.any(copy.params != 0)
    with pytest.raises(ValueError):
        copy.create_parameters((3, 4))
