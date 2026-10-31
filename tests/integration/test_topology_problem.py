"""User workflow, checkpoint identity, and design export contracts."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from beamz.optimization import InverseDesign, ModePower
from tests.topology_case import make_optimizer, make_problem

pytestmark = pytest.mark.optimization


@pytest.fixture(scope="module")
def problem():
    return make_problem()


def test_optimization_resume_and_independent_export(problem, tmp_path):
    mask = problem.design_region.region_mask
    density = problem.design_region.initial_parameters.copy()
    density[mask] += 0.05 * np.random.default_rng(10).normal(size=mask.sum())
    seen = []
    full = make_optimizer(problem, 6).run(steps=6, initial_params=density)
    partial = make_optimizer(problem, 6).run(
        steps=6,
        initial_params=density,
        stop_after=2,
        checkpoint=tmp_path / "state.npz",
        callback=lambda r: seen.append(r.completed_steps),
    )
    assert seen == [1, 2]
    fresh_problem = InverseDesign(
        problem.simulation,
        problem.design_region,
        problem.objective,
        gradient_backend="autodiff",
    )
    loaded = make_optimizer(fresh_problem, 6).load_result(tmp_path / "state.npz")
    resumed = make_optimizer(fresh_problem, 6).run(steps=6, resume=loaded)
    assert partial.completed_steps == loaded.completed_steps == 2
    np.testing.assert_array_equal(full.final_params, resumed.final_params)
    np.testing.assert_array_equal(
        [h.objective for h in full.history], [h.objective for h in resumed.history]
    )
    for a, b in zip(
        jax.tree.leaves(full.optimizer_state),
        jax.tree.leaves(resumed.optimizer_state),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(full.final_params[~mask], density[~mask])
    assert full.objective > full.initial_objective + 0.1
    np.testing.assert_allclose(
        full.objective, problem.value(full.final_params, beta=full.beta), rtol=1e-6
    )
    exported = full.export_design()
    assert exported != problem.simulation.design
    assert len(problem.simulation.design.structures) == 2
    simulation = problem.simulation.updated_copy(design=exported)
    results = simulation.run(backend="jax", performance=False)
    power = (
        abs(results.mode("output").amps.sel(direction="+", mode_index=1).item()) ** 2
    )
    power /= (
        abs(results.mode("input").amps.sel(direction="+", mode_index=0).item()) ** 2
    )
    assert np.isfinite(power) and power > full.initial_objective
    original = np.asarray(problem.program.grid.permittivity)
    raster = np.asarray(simulation.compile(backend="jax").grid.permittivity)
    # Fixed source/monitor cross-sections survive geometry export.
    np.testing.assert_array_equal(raster[:, :15], original[:, :15])
    np.testing.assert_array_equal(raster[:, 45:], original[:, 45:])


def test_checkpoint_rejects_different_problem_and_schedule(problem, tmp_path):
    result = make_optimizer(problem, 3).run(
        steps=3, stop_after=0, checkpoint=tmp_path / "initial.npz"
    )
    assert (
        make_optimizer(problem, 3).load_result(tmp_path / "initial.npz").completed_steps
        == 0
    )
    with pytest.raises(ValueError, match="different design|schedule"):
        make_optimizer(problem, 4).run(steps=4, resume=result)
    other = make_optimizer(problem, 3, learning_rate=0.01)
    with pytest.raises(ValueError, match="different optimization problem"):
        other.load_result(tmp_path / "initial.npz")
    with pytest.raises(ValueError, match="different design"):
        other.run(steps=3, resume=result)


def test_sgd_continuation_preserves_update_history(problem, tmp_path):
    optimizer = make_optimizer(problem, 3, method="sgd")
    full = optimizer.run(steps=3)
    path = tmp_path / "sgd.npz"
    optimizer.run(steps=3, stop_after=1, checkpoint=path)
    resumed = optimizer.run(steps=3, resume=optimizer.load_result(path))
    np.testing.assert_array_equal(full.final_params, resumed.final_params)
    np.testing.assert_array_equal(
        [h.objective for h in full.history], [h.objective for h in resumed.history]
    )


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"eps_bounds": (-1, 4)}, "eps_bounds"),
        ({"projection_eta": 0}, "projection_eta"),
    ],
)
def test_rejects_invalid_topology_settings(problem, changes, match):
    with pytest.raises(ValueError, match=match):
        InverseDesign(
            problem.simulation,
            replace(problem.design_region, **changes),
            problem.objective,
            gradient_backend="autodiff",
        )


def test_rejects_moving_ports_and_absorber_overlap(problem):
    mask = problem.design_region.region_mask.copy()
    mask[20, 10] = True
    with pytest.raises(ValueError, match="source and mode-monitor"):
        InverseDesign(
            problem.simulation,
            replace(problem.design_region, region_mask=mask),
            problem.objective,
            gradient_backend="autodiff",
        )
    mask = problem.design_region.region_mask.copy()
    mask[2, 30] = True
    with pytest.raises(ValueError, match="absorber"):
        InverseDesign(
            problem.simulation,
            replace(problem.design_region, region_mask=mask),
            problem.objective,
            gradient_backend="autodiff",
        )


def test_rejects_invalid_objective_and_density(problem):
    with pytest.raises(TypeError, match="modal objectives"):
        InverseDesign(
            problem.simulation,
            problem.design_region,
            lambda data: 1,
            gradient_backend="autodiff",
        )
    # Data-returning inverse designs accept a user objective at optimizer time.
    unbound = InverseDesign(
        problem.simulation, problem.design_region, gradient_backend="autodiff"
    )
    with pytest.raises(TypeError, match="callable"):
        unbound.make_objective_fn(None)

    for count in (True, -1, 1.5, np.nan, np.inf):
        with pytest.raises(ValueError, match="steps"):
            make_optimizer(problem, count).run(steps=count)
    with pytest.raises(ValueError, match="mode_index"):
        InverseDesign(
            problem.simulation,
            problem.design_region,
            ModePower("output", mode_index=10),
            gradient_backend="autodiff",
        )
    with pytest.raises(ValueError, match="Reference monitor"):
        InverseDesign(
            problem.simulation,
            problem.design_region,
            ModePower("output", reference_monitor="missing"),
            gradient_backend="autodiff",
        )
    with pytest.raises(ValueError, match="beta"):
        problem.value(problem.design_region.initial_parameters, beta=0)
    with pytest.raises(ValueError, match="Parameters"):
        problem.value(np.full(problem.design_region.region_mask.shape, np.nan))
    for interval in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="checkpoint_interval"):
            InverseDesign(
                problem.simulation,
                problem.design_region,
                problem.objective,
                checkpoint_interval=interval,
                gradient_backend="autodiff",
            )


def test_rejects_out_of_plane_2d_spectral_monitor(problem):
    from beamz import FieldMonitor

    monitor = FieldMonitor(
        center=(3e-6, 2e-6, 0),
        size=(2e-6, 2e-6, 0),
        freqs=problem.simulation.monitors[0].freqs,
        name="plane",
    )
    simulation = problem.simulation.updated_copy(
        monitors=[*problem.simulation.monitors, monitor]
    )
    with pytest.raises(ValueError, match="lines normal"):
        InverseDesign(
            simulation,
            problem.design_region,
            problem.objective,
            gradient_backend="autodiff",
        )
