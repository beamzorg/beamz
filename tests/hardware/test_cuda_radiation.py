"""Small native CUDA radiation objectives versus JAX and finite differences."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.const import LIGHT_SPEED
from beamz.optimization import CollectionConePower, DirectionalPower, TargetBeamOverlap
from beamz.simulation.backend import cuda_backend_status
from tests.far_field_case import box_quadrature, dipole_fields, sphere_quadrature
from tests.integration.test_radiation_inverse_design import radiation_design

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(
    not STATUS.available, reason=STATUS.reason or "CUDA unavailable"
)


@pytest.mark.parametrize("kind", ["direction", "cone", "overlap"])
def test_native_cuda_radiation_material_gradient_matches_jax_and_finite_difference(
    kind,
):
    # 9,600 cells and about 210 steps; no device-scale optimization run.
    reference, projection = radiation_design(run_time=40e-15)
    reference = reference.updated_copy(output_monitor_names=("radiation",))
    native = reference.updated_copy(backend="cuda")
    assert native._problem.program.config.backend == "cuda_streamed"
    target = np.array([[1, 0, 0], [0.8, 0, -0.6], [0.8, 0, 0.6]], complex)
    objective = {
        "direction": DirectionalPower(projection),
        "cone": CollectionConePower(projection, 0.9),
        "overlap": TargetBeamOverlap(projection, target),
    }[kind]
    params = jnp.asarray(reference.design_region.initial_parameters)
    expected = jax.value_and_grad(reference.make_objective_fn(objective))(params)
    fun = native.make_objective_fn(objective)
    actual = jax.jit(jax.value_and_grad(fun))(params)
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-4, atol=1e-8)
    gradient = actual[1]
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0
    assert np.linalg.norm(gradient - expected[1]) / np.linalg.norm(expected[1]) < 0.003
    direction = gradient / jnp.linalg.norm(gradient)
    fd = (fun(params + 0.005 * direction) - fun(params - 0.005 * direction)) / 0.01
    np.testing.assert_allclose(fd, jnp.sum(gradient * direction), rtol=0.015, atol=1e-8)
    # Check angular complex fields, not only a scalar power reduction.
    expected_far = projection(reference.to_simulation_data(params))
    actual_far = projection(native.to_simulation_data(params))
    np.testing.assert_allclose(
        actual_far.electric, expected_far.electric, rtol=4e-4, atol=1e-7
    )


def test_native_cuda_enclosing_box_projection_and_gradients():
    from examples.optimization.radiation_cuda import make_design

    reference, projection = make_design(backend="jax")
    native = reference.updated_copy(backend="cuda")
    objective = CollectionConePower(projection, 1.0)
    params = jnp.asarray(reference.design_region.initial_parameters)
    expected = jax.value_and_grad(reference.make_objective_fn(objective))(params)
    actual = jax.value_and_grad(native.make_objective_fn(objective))(params)
    np.testing.assert_allclose(actual[0], expected[0], rtol=5e-4, atol=1e-8)
    assert np.linalg.norm(expected[1]) > 0
    assert np.linalg.norm(actual[1] - expected[1]) / np.linalg.norm(expected[1]) < 0.005


def test_angular_projection_jit_and_field_vjp_stay_on_gpu():
    from beamz.analysis import near_to_far

    points, normals, areas = box_quadrature(count=8)
    e, h = dipole_fields(points)
    directions, weights = sphere_quadrature(4, 8)

    def evaluate(amplitude):
        far = near_to_far(
            amplitude * e[None],
            amplitude * h[None],
            points,
            normals,
            areas,
            np.array([LIGHT_SPEED]),
            directions,
        )
        return far.power(weights)[0]

    with jax.default_device(jax.devices("gpu")[0]):
        amplitude = jnp.float32(0.7)
        value, gradient = jax.jit(jax.value_and_grad(evaluate))(amplitude)
    assert all(d.platform == "gpu" for d in value.devices() | gradient.devices())
    np.testing.assert_allclose(gradient, 2 * value / amplitude, rtol=2e-5)
