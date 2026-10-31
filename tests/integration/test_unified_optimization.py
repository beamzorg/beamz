"""One optimizer respects masked material constraints and callback penalties."""

import jax
import jax.numpy as jnp
import numpy as np

from beamz.optimization import OptimizationSchedule, Optimizer
from tests.topology_case import make_problem


def test_masked_sgd_penalty_gradient_and_frozen_external_parameters(tmp_path):
    base = make_problem(target_mode=0)
    mask = base.design_region.region_mask
    solid, void = np.zeros_like(mask), np.zeros_like(mask)
    solid[15, 25], void[16, 25] = True, True

    def penalty(density, pixel_size):
        del pixel_size
        return 0.2 * jnp.mean(density[mask] ** 2)

    region = base.design_region.updated_copy(
        fixed_solid=solid, fixed_void=void, penalties=(penalty,)
    )
    design = base.updated_copy(design_region=region)
    params = np.full(mask.shape, 0.4, dtype=np.float32)
    # Outside latent values are irrelevant to both the material and its penalty.
    changed = np.where(mask, params, 0.9).astype(np.float32)
    objective = design.make_objective_fn(design.objective, beta=2)
    value, gradient = jax.value_and_grad(objective)(params)
    changed_value, changed_gradient = jax.value_and_grad(objective)(changed)
    np.testing.assert_array_equal(value, changed_value)
    np.testing.assert_array_equal(gradient, changed_gradient)
    np.testing.assert_array_equal(np.asarray(gradient)[~mask | solid | void], 0)
    step = 0.02
    optimizer = Optimizer(
        design,
        step,
        method="sgd",
        schedule=OptimizationSchedule(beta=2),
        store_full_results=True,
        evaluate_after=True,
    )
    result = optimizer.run(
        steps=1, initial_params=params, checkpoint=tmp_path / "sgd.npz"
    )
    expected = np.where(mask, np.clip(params + step * gradient, 0, 1), params)
    np.testing.assert_allclose(result.final_params, expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_array_equal(result.final_params[~mask], params[~mask])
    physical = design.physical_density(result.final_params, beta=2)
    np.testing.assert_array_equal(physical[solid], 1)
    np.testing.assert_array_equal(physical[void], 0)
    assert result.objective is not None and result.completed_steps == 1
    restored = optimizer.load_result(tmp_path / "sgd.npz")
    np.testing.assert_array_equal(restored.final_params, result.final_params)
