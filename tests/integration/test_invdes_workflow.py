"""Tidy3D-style post-processing, gradient selection, and optimizer lifecycle."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz.optimization as bi
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
        store_full_results=store_full_results,
    )
    full = optimizer.run(transmission, steps=3)
    partial = optimizer.run(transmission, steps=1, checkpoint=path)
    seen = []

    def callback(result):
        step_index = result.completed_steps - 1
        saved = optimizer.load_result(path, transmission)
        assert saved.completed_steps == result.completed_steps == step_index + 1
        np.testing.assert_array_equal(saved.params[-1], result.params[-1])
        assert saved.history[-1].post_process_val == result.history[-1].post_process_val
        seen.append(step_index)

    resumed = optimizer.run(
        transmission, steps=3, resume=path, checkpoint=path, callback=callback
    )
    assert seen == [1, 2]
    np.testing.assert_array_equal(full.final_params, resumed.final_params)
    np.testing.assert_allclose(
        [h.objective_before for h in full.history],
        [h.objective_before for h in resumed.history],
        rtol=0,
        atol=0,
    )
    assert partial.completed_steps == 1
    assert full.history[-1].post_process_val > full.history[0].post_process_val
    assert len(full.params) == (4 if store_full_results else 1)
    assert full.to_simulation() is not None
    assert transmission(full.simulation_data()) > full.history[0].post_process_val
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


def test_checkpoint_rejects_old_schema_and_wrong_identity(design, tmp_path):
    optimizer = bi.AdamOptimizer(design, learning_rate=0.02)
    result = optimizer.run(transmission, steps=1)
    path = tmp_path / "old.npz"
    result.save(path)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    meta = json.loads(str(arrays["metadata"]))
    np.savez_compressed(
        path, **{**arrays, "metadata": json.dumps({**meta, "schema": 1})}
    )
    with pytest.raises(ValueError, match="schema"):
        optimizer.run(transmission, steps=2, resume=path)
    result.save(path)
    with pytest.raises(ValueError, match="different design"):
        optimizer.updated_copy(learning_rate=0.03).load_result(path, transmission)
    with pytest.raises(ValueError, match="different design"):
        optimizer.load_result(path, phase_objective)


@pytest.mark.parametrize("interface", ["optimizer", "topology"])
def test_checkpoint_rejects_pre_merge_solver_numerics(
    design, tmp_path, monkeypatch, interface
):
    if interface == "optimizer":
        import beamz.optimization.optimizer as module

        optimizer = bi.AdamOptimizer(design, learning_rate=0.02)
        current = optimizer._fingerprint(transmission)
        with monkeypatch.context() as previous:
            previous.setattr(module, "TOPOLOGY_NUMERICS_VERSION", 3)
            old = optimizer._fingerprint(transmission)

        def load(path):
            return optimizer.load_result(path, transmission)
    else:
        import beamz.optimization.problem as module

        problem = make_problem()
        current = problem.fingerprint
        with monkeypatch.context() as previous:
            previous.setattr(module, "TOPOLOGY_NUMERICS_VERSION", 3)
            old = make_problem().fingerprint
        load = problem.load
    assert old != current
    # Identity is checked before loading optimizer arrays or scalar history.
    path = tmp_path / "previous-solver.npz"
    np.savez_compressed(path, metadata=json.dumps({"schema": 2, "fingerprint": old}))
    with pytest.raises(ValueError, match="different design"):
        load(path)


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


def test_checkpoint_rejects_vector_valued_scalar_history(design, tmp_path):
    optimizer = bi.AdamOptimizer(design, learning_rate=0.02)
    path = tmp_path / "history.npz"
    optimizer.run(transmission, steps=1).save(path)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    meta = json.loads(str(arrays["metadata"]))
    for key in ("objective_before", "post_process_val", "penalty"):
        corrupt = {**meta, "history": [{**meta["history"][0], key: [0.1]}]}
        np.savez_compressed(path, **{**arrays, "metadata": json.dumps(corrupt)})
        with pytest.raises(ValueError, match="scalar checkpoint history"):
            optimizer.load_result(path, transmission)


@pytest.mark.parametrize("backend", ["autodiff", "adjoint"])
@pytest.mark.parametrize("store_full_results", [True, False])
def test_scheduled_resume_and_material_history(
    design, tmp_path, backend, store_full_results
):
    design = design.updated_copy(gradient_backend=backend)
    schedule = bi.OptimizationSchedule(
        beta=bi.LinearSchedule(1, 5, 4),
        penalty_weight=bi.LinearSchedule(0, 1, 4),
        objective_kwargs={"weight": bi.StepSchedule(0.5, 1, 2)},
    )

    def objective(data, weight):
        return weight * transmission(data)

    optimizer = bi.AdamOptimizer(
        design,
        learning_rate=0.02,
        schedule=schedule,
        store_full_results=store_full_results,
    )
    full = optimizer.run(objective, steps=4)
    partial = optimizer.run(objective, steps=2)
    path = tmp_path / "scheduled.npz"
    partial.save(path)
    restored = optimizer.load_result(path, objective)
    np.testing.assert_array_equal(partial.final_params, restored.final_params)
    for expected_state, restored_state in zip(
        jax.tree.leaves(partial.optimizer_state),
        jax.tree.leaves(restored.optimizer_state),
        strict=True,
    ):
        np.testing.assert_array_equal(expected_state, restored_state)
    assert restored.settings()["beta"] == schedule.values(1)["beta"]
    resumed = optimizer.run(objective, steps=4, resume=restored)
    # Independent JAX compilations can differ by a few float32 ULPs, even before
    # saving the partial run. Check serialization exactly and recomputation to
    # float32 precision.
    np.testing.assert_allclose(
        full.params[-1], resumed.params[-1], rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        [h.objective_before for h in full.history],
        [h.objective_before for h in resumed.history],
        rtol=1e-6,
        atol=1e-7,
    )
    assert resumed.schedule_history == full.schedule_history
    assert resumed.settings()["beta"] == 5
    expected = design.to_simulation_data(resumed.params[-1], beta=5)
    np.testing.assert_allclose(
        transmission(resumed.simulation_data()), transmission(expected), rtol=1e-6
    )
    assert resumed.export_design() is not None
    with pytest.raises(ValueError, match="different design"):
        optimizer.updated_copy(
            schedule=schedule.updated_copy(beta=bi.LinearSchedule(1, 6, 4))
        ).load_result(path, objective)


@pytest.mark.parametrize("backend", ["autodiff", "adjoint"])
def test_builtin_objectives_share_callback_api_and_gradient(design, backend):
    design = design.updated_copy(gradient_backend=backend)
    params = design.design_region.initial_parameters
    power = bi.ModePower("output", reference_monitor="input")
    data = design.to_simulation_data(params)
    np.testing.assert_allclose(power(data), transmission(data), rtol=2e-6)
    for objective in (
        power,
        bi.SoftMinModePower("output", reference_monitor="input"),
        0.8 * power - 0.2 * bi.ModePower("input", direction="-"),
    ):
        value, gradient = jax.value_and_grad(design.make_objective_fn(objective))(
            params
        )
        assert np.isfinite(value) and np.isfinite(gradient).all()
        assert np.linalg.norm(gradient) > 1e-3
        optimizer = bi.AdamOptimizer(design, learning_rate=0.02)
        result = optimizer.run(objective, steps=1)
        assert result.completed_steps == 1 and len(result.params) == 1
        assert result.history[0].objective is None
        np.testing.assert_allclose(result.history[0].objective_before, value, rtol=2e-6)


def test_target_steps_checkpoint_interval_and_snapshot_policy(design, tmp_path):
    optimizer = bi.AdamOptimizer(design, learning_rate=0.02)
    path = tmp_path / "compact.npz"
    zero = optimizer.run(transmission, steps=0, checkpoint=path)
    assert optimizer.load_result(path, transmission).completed_steps == 0
    result = optimizer.run(
        transmission, steps=3, resume=zero, checkpoint=path, checkpoint_every=2
    )
    restored = optimizer.load_result(path, transmission)
    np.testing.assert_array_equal(result.final_params, restored.final_params)
    assert len(restored.history) == 3 and len(restored.params) == 1
    assert all(h.objective is None for h in restored.history)
    with np.load(path, allow_pickle=False) as saved:
        assert json.loads(str(saved["metadata"]))["leaf_count"] == len(
            jax.tree.leaves(result.optimizer_state)
        )
    # Snapshot storage does not change the physical problem or optimizer identity.
    full = optimizer.updated_copy(store_full_results=True).run(
        transmission, steps=4, resume=restored
    )
    expected = optimizer.run(transmission, steps=4)
    full.save(path)
    np.testing.assert_array_equal(
        optimizer.load_result(path, transmission).final_params, full.final_params
    )
    np.testing.assert_array_equal(full.final_params, expected.final_params)
    with pytest.raises(ValueError, match="completed steps"):
        optimizer.run(transmission, steps=2, resume=result)
    for value in (True, -1, 1.5, np.nan):
        with pytest.raises(ValueError, match="steps"):
            optimizer.run(transmission, steps=value)
    with pytest.raises(ValueError, match="checkpoint_every"):
        optimizer.run(transmission, steps=1, checkpoint_every=0)
