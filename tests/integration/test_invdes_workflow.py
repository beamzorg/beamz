"""Tidy3D-style post-processing, gradient selection, and optimizer lifecycle."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz.plugins.invdes as bi
from tests.topology_case import make_problem

pytestmark = pytest.mark.optimization


def transmission(data):
    out = data["output"].amps.sel(direction="+", mode_index=0).values
    incoming = data["input"].amps.sel(direction="+", mode_index=0).values
    return jnp.mean(jnp.abs(out / incoming) ** 2)


def phase_objective(data):
    out = data["output"].amps.sel(direction="+", mode_index=0).values
    incoming = data["input"].amps.sel(direction="+", mode_index=0).values
    return jnp.mean(jnp.real(out / incoming))


@pytest.fixture(scope="module")
def design():
    p = make_problem(target_mode=0, run_time=320e-15)
    region = bi.TopologyDesignRegion(
        size=(2e-6, 1.6e-6, np.inf),
        center=(3e-6, 2e-6, 0),
        eps_bounds=(1, 4),
        pixel_size=100e-9,
        transformations=[bi.FilterProject(radius=200e-9, beta=2)],
        initialization_spec=bi.RandomInitializationSpec(
            min_value=0.45, max_value=0.55, seed=7
        ),
    )
    return bi.InverseDesign(
        p.simulation, region, output_monitor_names=["input", "output"]
    )


@pytest.mark.parametrize("backend", ["autodiff", "adjoint"])
@pytest.mark.parametrize("post_process_fn", [transmission, phase_objective])
def test_differentiable_user_objective(design, backend, post_process_fn):
    design = design.updated_copy(gradient_backend=backend)
    params = design.design_region.initial_parameters
    objective = design.make_objective_fn(post_process_fn)
    value, gradient = jax.value_and_grad(objective)(params)
    assert np.isfinite(value) and np.linalg.norm(gradient) > 1e-3
    direction = gradient / jnp.linalg.norm(gradient)
    h = 0.01
    fd = (objective(params + h * direction) - objective(params - h * direction)) / (
        2 * h
    )
    np.testing.assert_allclose(fd, jnp.sum(gradient * direction), rtol=0.004)
    neg_value, neg_grad = jax.value_and_grad(
        design.make_objective_fn(post_process_fn, maximize=False)
    )(params)
    np.testing.assert_allclose(neg_value, -value, rtol=1e-6)
    np.testing.assert_allclose(neg_grad, -gradient, rtol=1e-5, atol=1e-8)


def test_complex_amplitudes_match_ordinary_results_and_selection(design):
    params = design.design_region.initial_parameters
    data = design.to_simulation_data(params)
    ordinary = design.to_simulation(params).run(backend="jax", performance=False)
    for name in ("input", "output"):
        expected = ordinary.mode(name).amps
        actual = data[name].amps
        np.testing.assert_allclose(
            actual.values,
            expected.transpose("direction", "f", "mode_index").values,
            rtol=4e-5,
            atol=1e-7,
        )
        frequency = design.simulation.monitors[0].freqs[0]
        selected = actual.sel(direction="+", mode_index=0, f=frequency)
        assert selected.shape == () and selected.dims == ()
        np.testing.assert_allclose(selected.values, actual.values[0, 0, 0])
        assert actual.sel(direction=["-", "+"]).shape == actual.shape
        with pytest.raises(KeyError):
            actual.sel(f=1.0)


@pytest.mark.parametrize("store_full_results", [True, False])
def test_optimizer_history_resume_and_checkpoint_identity(
    design, tmp_path, store_full_results
):
    path = tmp_path / "history.npz"
    optimizer = bi.AdamOptimizer(
        design,
        learning_rate=0.02,
        num_steps=3,
        store_full_results=store_full_results,
        results_cache_fname=str(path),
    )
    full = optimizer.run(post_process_fn=transmission)
    partial = optimizer.updated_copy(num_steps=1).run(post_process_fn=transmission)
    seen = []
    resumed = optimizer.complete_run_from_history(
        post_process_fn=transmission,
        callback=lambda result, step_index, aux_data: seen.append(step_index),
    )
    assert seen == [1, 2]
    np.testing.assert_array_equal(full.get_last("params"), resumed.get_last("params"))
    np.testing.assert_allclose(
        full.objective_fn_val, resumed.objective_fn_val, rtol=0, atol=0
    )
    assert len(partial.objective_fn_val) == 1
    assert full.post_process_val[-1] > full.post_process_val[0]
    assert len(full.params) == (4 if store_full_results else 1)
    assert full.sim_last is not None
    assert transmission(full.sim_data_last()) > full.post_process_val[0]
    with pytest.raises(ValueError, match="different design"):
        optimizer.load_result(path, phase_objective)
    with pytest.raises(ValueError, match="different design"):
        optimizer.updated_copy(
            design=design.updated_copy(gradient_backend="autodiff")
        ).load_result(path, transmission)

    # Captured values affect identity, even when function names are identical.
    def make_objective(weight):
        return lambda data: weight * transmission(data)

    assert optimizer._fingerprint(make_objective(1)) != optimizer._fingerprint(
        make_objective(2)
    )


def test_workflow_rejects_bad_parameters_and_objectives(design):
    with pytest.raises(ValueError, match="integer multiples"):
        design.design_region.updated_copy(size=(2.01e-6, 1.6e-6, 0))
    with pytest.raises(ValueError, match="align"):
        design.updated_copy(
            design_region=design.design_region.updated_copy(center=(3.01e-6, 2e-6, 0))
        )
    with pytest.raises(ValueError, match="pixel_size"):
        design.updated_copy(
            design_region=design.design_region.updated_copy(pixel_size=50e-9)
        )
    with pytest.raises(ValueError, match="Parameters"):
        design.to_simulation_data(np.zeros((2, 3)))
    params = design.design_region.initial_parameters
    with pytest.raises(ValueError, match="real scalars"):
        design.make_objective_fn(lambda data: data["output"].amps.values)(params)
