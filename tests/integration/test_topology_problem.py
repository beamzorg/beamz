"""User workflow, checkpoint identity, and design export contracts."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from beamz import SampledSignal
from beamz.optimization import ModePower, TopologyProblem
from beamz.simulation.observe import normalization_from_result
from tests.topology_3d_case import make_3d_design
from tests.topology_case import legacy_checkpoint_fingerprint, make_problem

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
    legacy = legacy_checkpoint_fingerprint(problem._fingerprint)
    assert legacy != fresh_problem.fingerprint
    path = tmp_path / "legacy.npz"
    replace(partial, fingerprint=legacy).save(path)
    loaded = fresh_problem.load(path)
    assert loaded.fingerprint == fresh_problem.fingerprint
    resumed = fresh_problem.run(6, resume=loaded)
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


@pytest.mark.parametrize("quadrature_only", [False, True])
def test_2d_mode_taper_matches_emitted_waveform_and_normalization(
    problem, quadrature_only
):
    simulation = problem.simulation
    source = simulation.sources[0]
    source = source.updated_copy(
        mode_spec=replace(source.mode_spec, num_freqs=3), band_taper=(47e-15, 52e-15)
    )
    if quadrature_only:
        _, q = source.source_time.sample(simulation.time)
        source = source.updated_copy(
            source_time=SampledSignal(
                np.zeros_like(q),
                dt=simulation.dt,
                quadrature=q,
                freq0=source.frequency,
            )
        )
    simulation = simulation.updated_copy(sources=(source,))
    times = simulation.time + 0.5 * simulation.dt
    start, end = source.band_taper
    fraction = np.clip((times - start) / (end - start), 0.0, 1.0)
    weights = 0.5 * (1.0 + np.cos(np.pi * fraction))
    signal, quadrature = source.source_time.sample(times)
    expected = [(wave * weights).astype(np.float32) for wave in (signal, quadrature)]
    half_step_samples = [
        np.asarray(spec.waveform) for spec in simulation.compile(backend="jax").sources
    ]
    for waveform in half_step_samples:
        np.testing.assert_array_equal(waveform[times >= end], 0)
    for wave in expected:
        if np.any(wave):
            assert any(np.array_equal(wave, emitted) for emitted in half_step_samples)
    assert np.any(expected[1][times < end])

    result = simulation.run(backend="jax", performance=False)
    monitor = result["output"]
    normalization = normalization_from_result(result, monitor)
    phase = np.exp(2j * np.pi * times * monitor.dft_frequencies[0])
    drive = expected[0].astype(float) - 1j * expected[1].astype(float)
    np.testing.assert_allclose(
        normalization.waveform_spectrum, [np.mean(drive * phase)], rtol=1e-12
    )
    assert result.normalization_source == 0
    raw = result.renormalize(None)["output"]
    assert any(np.any(values) for values in raw.dft_fields.values())
    for component, values in raw.dft_fields.items():
        np.testing.assert_allclose(
            monitor.dft_fields[component],
            values / normalization.field_amplitude_norm[0],
        )
    if quadrature_only:
        TopologyProblem(simulation, problem.topology, problem.objective)

    silent = simulation.updated_copy(
        sources=(source.updated_copy(band_taper=(-2e-15, -1e-15)),)
    ).run(backend="jax", performance=False)
    assert normalization_from_result(silent, silent["output"]) is None
    assert silent.normalization_source is None
    assert silent.renormalize(0).normalization_source is None


def test_3d_tapered_mode_normalizes_a_single_step_prefix():
    simulation = make_3d_design(run_time=100e-15).simulation
    source = simulation.sources[0]
    source = source.updated_copy(
        mode_spec=replace(source.mode_spec, num_freqs=3),
        source_time=replace(source.source_time, offset=0.0),
        band_taper=(40e-15, 50e-15),
    )
    simulation = simulation.updated_copy(sources=(source,))
    result = simulation.advance(num_steps=1, backend="jax", performance=False).results
    monitor = result["output"]
    np.testing.assert_array_equal(result.metadata.time, simulation.time[:1])
    assert result.metadata.dt == simulation.dt
    assert result.normalization_source == 0
    normalization = normalization_from_result(result, monitor)
    signal, quadrature = source.source_time.sample(simulation.time[:1])
    np.testing.assert_allclose(
        normalization.waveform_spectrum, signal - 1j * quadrature
    )


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


def test_sgd_continuation_preserves_update_history(problem, tmp_path):
    problem = TopologyProblem(
        problem.simulation,
        replace(problem.topology, optimizer="sgd"),
        problem.objective,
    )
    full = problem.run(3)
    path = tmp_path / "sgd.npz"
    problem.run(3, stop_after=1, checkpoint=path)
    resumed = problem.run(3, resume=problem.load(path))
    np.testing.assert_array_equal(full.final_params, resumed.final_params)
    np.testing.assert_array_equal(
        [h.objective for h in full.history], [h.objective for h in resumed.history]
    )


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
    with pytest.raises(TypeError, match="modal objectives"):
        TopologyProblem(problem.simulation, problem.topology, lambda data: 1)
    with pytest.raises(ValueError, match="Supply an objective"):
        TopologyProblem(problem.simulation, problem.topology)

    for count in (True, -1, 1.5, np.nan, np.inf):
        with pytest.raises(ValueError, match="num_steps"):
            problem.run(count)
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
