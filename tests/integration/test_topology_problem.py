"""User workflow, checkpoint identity, and design export contracts."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from beamz.optimization import ModePower, TopologyProblem
from tests.topology_case import make_problem

pytestmark = pytest.mark.optimization


@pytest.fixture(scope="module")
def problem():
    return make_problem()


def test_optimization_resume_and_independent_export(problem, tmp_path):
    mask = problem.topology.region_mask
    density = problem.topology.initial_state().density.copy()
    density[mask] += 0.05 * np.random.default_rng(10).normal(size=mask.sum())
    seen = []
    full = problem.run(6, initial_density=density)
    partial = problem.run(
        6,
        initial_density=density,
        stop_after=2,
        checkpoint=tmp_path / "state.npz",
        callback=lambda r: seen.append(r.completed_steps),
    )
    assert seen == [1, 2]
    fresh_problem = TopologyProblem(
        problem.simulation, problem.topology, problem.objective
    )
    loaded = fresh_problem.load(tmp_path / "state.npz")
    resumed = fresh_problem.run(6, resume=loaded)
    assert partial.completed_steps == loaded.completed_steps == 2
    np.testing.assert_array_equal(full.state.density, resumed.state.density)
    np.testing.assert_array_equal(
        full.state.objective_history, resumed.state.objective_history
    )
    for a, b in zip(
        jax.tree.leaves(full.state.optimizer_state),
        jax.tree.leaves(resumed.state.optimizer_state),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(full.state.density[~mask], density[~mask])
    assert full.objective > full.initial_objective + 0.1
    np.testing.assert_allclose(
        full.objective, problem.value(full.state.density, beta=full.beta), rtol=1e-6
    )
    exported = problem.export_design(full)
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
    result = problem.run(3, stop_after=0, checkpoint=tmp_path / "initial.npz")
    assert problem.load(tmp_path / "initial.npz").completed_steps == 0
    with pytest.raises(ValueError, match="schedule"):
        problem.run(4, resume=result)
    other = TopologyProblem(
        problem.simulation,
        replace(problem.topology, learning_rate=0.01),
        problem.objective,
    )
    with pytest.raises(ValueError, match="different optimization problem"):
        other.load(tmp_path / "initial.npz")
    with pytest.raises(ValueError, match="different optimization problem"):
        other.export_design(result)


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"learning_rate": 0}, "learning rate"),
        ({"beta_schedule": (1, float("inf"))}, "beta schedule"),
        ({"eps_min": -1}, "eps_min"),
        ({"projection_eta": 0}, "projection_eta"),
    ],
)
def test_rejects_invalid_topology_settings(problem, changes, match):
    with pytest.raises(ValueError, match=match):
        TopologyProblem(
            problem.simulation, replace(problem.topology, **changes), problem.objective
        )


def test_rejects_moving_ports_and_absorber_overlap(problem):
    mask = problem.topology.region_mask.copy()
    mask[20, 10] = True
    with pytest.raises(ValueError, match="source and mode-monitor"):
        TopologyProblem(
            problem.simulation,
            replace(problem.topology, region_mask=mask),
            problem.objective,
        )
    mask = problem.topology.region_mask.copy()
    mask[2, 30] = True
    with pytest.raises(ValueError, match="absorber"):
        TopologyProblem(
            problem.simulation,
            replace(problem.topology, region_mask=mask),
            problem.objective,
        )


def test_rejects_invalid_objective_and_density(problem):
    with pytest.raises(ValueError, match="mode_index"):
        TopologyProblem(
            problem.simulation, problem.topology, ModePower("output", mode_index=10)
        )
    with pytest.raises(ValueError, match="Reference monitor"):
        TopologyProblem(
            problem.simulation,
            problem.topology,
            ModePower("output", reference_monitor="missing"),
        )
    with pytest.raises(ValueError, match="beta"):
        problem.value(problem.topology.initial_state().density, beta=0)
    with pytest.raises(ValueError, match="Density"):
        problem.value(np.full(problem.topology.region_mask.shape, np.nan))
    for interval in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="checkpoint_interval"):
            TopologyProblem(
                problem.simulation,
                problem.topology,
                problem.objective,
                checkpoint_interval=interval,
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
        TopologyProblem(simulation, problem.topology, problem.objective)
