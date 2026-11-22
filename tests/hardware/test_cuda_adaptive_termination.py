"""Continuation, stopping and normalization on resident native CUDA state."""

from dataclasses import replace
from time import perf_counter

import jax
import numpy as np
import pytest

import beamz as bz
from beamz.simulation import execute
from beamz.simulation.backend import _load_extension, cuda_backend_status
from beamz.simulation.cuda.engine._contracts import _FIELDS
from beamz.simulation.cuda.engine._runtime import _SESSIONS, _Session

STATUS = cuda_backend_status()
pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        not STATUS.available, reason=STATUS.reason or "CUDA unavailable"
    ),
]


@pytest.fixture
def scene():
    """A bounded heterogeneous CPML12 scene with an exactly sampled source."""
    if not getattr(_load_extension(), "coherent_backend_version", 0):
        pytest.skip("Coherent native component required")
    steps, dt = 113, 3e-17
    frequency = bz.LIGHT_SPEED / 700e-9
    # Simulation(domain=...) uses centered public coordinates; reconstruction
    # translates replacement sources into domain coordinates exactly once.
    design = bz.Design(background=bz.Material(1.0))
    design += bz.Box(
        center=(0.0,) * 3,
        size=(0.20e-6,) * 3,
        material=bz.Material(2.25),
    )
    simulation = bz.Simulation(
        design=design,
        domain=(1.28e-6,) * 3,
        resolution=20e-9,
        sources=[
            bz.PlaneWaveSource(
                center=(-0.28e-6, 0.0, 0.0),
                size=(0, 1.28e-6, 1.28e-6),
                direction="+x",
                source_time=bz.SampledSignal(
                    np.ones(steps), dt, quadrature=np.zeros(steps), freq0=frequency
                ),
            )
        ],
        monitors=[
            bz.FieldMonitor(
                center=(0.28e-6, 0.0, 0.0),
                size=(0, 0.4e-6, 0.4e-6),
                freqs=[frequency],
                fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                name="field",
            )
        ],
        boundaries=[bz.PML(formulation="cpml")],
        time=np.arange(steps) * dt,
    )
    yield simulation
    simulation.clear_compiled_cache()


def legacy_run(simulation, policy, *, performance=True):
    """Exercise the pre-native termination loop, bypassing public dispatch."""
    return execute.run_until_terminated(
        simulation,
        policy,
        progress=False,
        store_full_materials=False,
        sharding=None,
        backend="cuda_streamed",
        schedule=None,
        performance=performance,
    )


def assert_same_acquisition(actual, expected, *, bitwise=False):
    np.testing.assert_array_equal(actual.metadata.time, expected.metadata.time)
    for normalization in (None, 0):
        a, b = actual.renormalize(normalization), expected.renormalize(normalization)
        for component in a["field"].dft_fields:
            x = np.asarray(a["field"].get_dft_component(component))
            y = np.asarray(b["field"].get_dft_component(component))
            if bitwise:
                np.testing.assert_array_equal(
                    np.ascontiguousarray(x).view(np.uint8),
                    np.ascontiguousarray(y).view(np.uint8),
                )
            else:
                # Scale weak components to the full acquired vector. A tiny
                # symmetry-plane component does not define its own error scale.
                peak = max(
                    np.max(np.abs(b["field"].get_dft_component(c)), initial=0)
                    for c in b["field"].dft_fields
                )
                np.testing.assert_allclose(x, y, rtol=5e-5, atol=peak * 1e-6)
        np.testing.assert_array_equal(
            a["field"].dft_weight_sum, b["field"].dft_weight_sum
        )


def test_device_monitor_reduction_matches_host_checks(scene, monkeypatch):
    if not hasattr(_load_extension().NativeSession, "_monitor_diagnostics"):
        pytest.skip("Native CUDA monitor diagnostics required")
    policy = bz.AutoTermination(chunk_steps=17, source_decay=0.0)
    native = scene.run(backend="cuda", termination=policy, performance=False)
    assert native.cuda_plan["monitor_convergence"] == "device"
    assert native.cuda_plan["monitor_convergence_gpu_bytes"] > 0
    assert _SESSIONS[scene].native.metadata()["monitor_diagnostic_gpu_bytes"] == 0

    monkeypatch.setattr(_Session, "_configure_monitor_checks", lambda *_: -1)
    host = scene.run(backend="cuda", termination=policy, performance=False)
    assert host.cuda_plan["monitor_convergence"] == "host"
    assert host.termination.steps == native.termination.steps
    assert host.termination.reason == native.termination.reason
    assert host.termination.monitor_change == pytest.approx(
        native.termination.monitor_change, rel=1e-12
    )
    assert_same_acquisition(native, host, bitwise=True)


@pytest.mark.parametrize("backend", ["cuda_streamed", "cuda_mstep"])
@pytest.mark.parametrize("frequency_count", [3, 19])
def test_compact_dft_preserves_selected_fields_and_weights(
    scene, monkeypatch, backend, frequency_count
):
    if not getattr(_load_extension(), "compiled_compact_dft_components", False):
        pytest.skip("Compact native CUDA DFT arenas required")
    frequency = bz.LIGHT_SPEED / 700e-9
    sim = scene.updated_copy(
        monitors=(
            bz.FluxMonitor(
                name="flux",
                center=(0.22e-6, 0, 0),
                size=(0, 0.4e-6, 0.4e-6),
                freqs=np.linspace(0.9 * frequency, 1.1 * frequency, frequency_count),
            ),
            bz.FieldMonitor(
                name="field",
                center=(0.26e-6, 0, 0),
                size=(0, 0.5e-6, 0.5e-6),
                freqs=[frequency],
                fields=("Ey", "Hz"),
            ),
        )
    )
    policy = bz.AutoTermination(chunk_steps=17, source_decay=0.0)
    packed = sim.run(
        backend=backend, termination=policy, performance=False
    ).renormalize(None)
    assert _SESSIONS[sim]._compact_monitors
    sim.clear_compiled_cache()
    monkeypatch.setattr(
        _Session, "_pack_monitor_components", staticmethod(lambda program: program)
    )
    dense = sim.run(backend=backend, termination=policy, performance=False).renormalize(
        None
    )
    assert packed.termination.steps == dense.termination.steps
    for name in ("flux", "field"):
        assert packed[name].dft_fields.keys() == dense[name].dft_fields.keys()
        for component in packed[name].dft_fields:
            np.testing.assert_array_equal(
                packed[name].dft_fields[component], dense[name].dft_fields[component]
            )
        np.testing.assert_array_equal(
            packed[name].dft_weight_sum, dense[name].dft_weight_sum
        )


@pytest.mark.parametrize("performance", [False, True])
def test_native_temporal_source_guard_horizon_tails_and_reset(scene, performance):
    policy = bz.AutoTermination(
        chunk_steps=17,
        field_decay=1.0,
        monitor_change=1e6,
        source_decay=0.0,
        consecutive_checks=2,
    )
    ordinary = scene.run(
        backend="cuda_streamed", termination=policy, performance=performance
    )
    temporal = scene.run(
        backend="cuda_mstep", termination=policy, performance=performance
    )
    repeated = scene.run(
        backend="cuda_mstep", termination=policy, performance=performance
    )
    assert temporal.cuda_plan["binding"] == "native"
    assert temporal.cuda_plan["schedule"] == "temporal"
    assert temporal.cuda_plan["final_step"] == scene.num_steps
    for result in (ordinary, temporal, repeated):
        report = result.termination
        assert report.reason == "time_limit" and not report.converged
        assert report.steps == scene.num_steps == 113
        assert report.energy > 0 and np.isfinite(report.energy)
        assert report.max_field > 0
        assert report.consecutive_checks < policy.consecutive_checks
        assert (result.performance is not None) is performance
        if performance:
            assert result.performance.steps == 113
    assert_same_acquisition(temporal, ordinary, bitwise=True)
    assert_same_acquisition(repeated, temporal, bitwise=True)
    reference = legacy_run(scene, policy, performance=performance)
    assert reference.termination.steps == temporal.termination.steps
    assert reference.termination.reason == temporal.termination.reason
    assert_same_acquisition(temporal, reference)


@pytest.mark.parametrize("field_decay", [0.0, 1.0])
def test_native_convergence_keeps_full_dft_window_and_source_normalization(
    scene, field_decay
):
    # Deliberately permissive stopping tolerances isolate controller semantics,
    # not pulse-decay physics. The separate test above enforces the future-source
    # guard using the identical continuously driven source and source_decay=0.
    policy = bz.AutoTermination(
        chunk_steps=17,
        field_decay=field_decay,
        monitor_change=1e6,
        source_decay=1.0,
        consecutive_checks=2,
        monitor_names=("field",),
    )
    ordinary = scene.run(backend="cuda_streamed", termination=policy)
    temporal = scene.run(backend="cuda_mstep", termination=policy)
    for result in (ordinary, temporal):
        report = result.termination
        assert report.reason == "converged" and report.converged
        assert report.steps == 51
        assert result.cuda_plan["final_step"] == 51
        assert len(result.metadata.time) == scene.num_steps
        np.testing.assert_array_equal(
            result["field"].dft_weight_sum, np.asarray([scene.num_steps])
        )
        assert report.consecutive_checks == 2
    assert_same_acquisition(temporal, ordinary, bitwise=True)
    reference = legacy_run(scene, policy)
    assert reference.termination.steps == 51
    assert_same_acquisition(temporal, reference)


def test_native_wall_budget_includes_setup_and_returns_exact_prefix(scene, monkeypatch):
    # A budget far shorter than even setup must still produce one chunk, then
    # stop. This catches the native path starting its clock after compilation.
    policy = bz.AutoTermination(
        chunk_steps=17, max_wall_time=1e-12, consecutive_checks=10
    )
    from beamz.simulation.cuda.engine._runtime import _Session

    preparation_started = []
    constructor = _Session.__init__
    shared_controller = execute.run_until_terminated

    def record_preparation(self, *args, **kwargs):
        preparation_started.append(perf_counter())
        constructor(self, *args, **kwargs)

    def record_wall_origin(*args, **kwargs):
        assert kwargs["_wall_start"] <= preparation_started[0]
        return shared_controller(*args, **kwargs)

    monkeypatch.setattr(_Session, "__init__", record_preparation)
    monkeypatch.setattr(execute, "run_until_terminated", record_wall_origin)
    result = scene.run(backend="cuda_mstep", termination=policy)
    assert result.cuda_plan["binding"] == "native"
    assert result.cuda_plan["schedule"] == "temporal"
    assert result.cuda_plan["final_step"] == 17
    assert result.termination.reason == "wall_time_limit"
    assert result.termination.steps == result.performance.steps == 17
    assert len(result.metadata.time) == 17
    np.testing.assert_array_equal(result["field"].dft_weight_sum, [17])


def test_native_gpu_energy_matches_authoritative_material_and_yee_measures(scene):
    policy = bz.AutoTermination(chunk_steps=17, max_wall_time=1e-12)
    result = scene.run(backend="cuda_mstep", termination=policy)
    session = _SESSIONS[scene]
    fields = session.native.download()[:6]
    state = session.initial._replace(**dict(zip(_FIELDS, fields, strict=True)))
    with jax.default_device(jax.devices("cpu")[0]):
        expected = execute._field_diagnostics(
            state, execute._energy_terms(session.program)
        )
    assert result.termination.energy == pytest.approx(expected[0], rel=2e-6)
    assert result.termination.max_field == expected[1]
    assert expected[2]


def test_native_rectilinear_gpu_energy_uses_component_control_volumes():
    from beamz.design.discretization import MaterialGrid
    from beamz.design.grid import RectilinearGrid
    from beamz.design.raster import Scene, rasterize

    edges = []
    for count, spacing in ((35, 20e-9), (37, 23e-9), (39, 26e-9)):
        widths = spacing * (1 + 0.15 * np.sin(np.arange(count) * 0.21))
        edges.append(np.r_[0.0, np.cumsum(widths)])
    grid = RectilinearGrid(*edges)
    materials = MaterialGrid.from_raster_result(
        rasterize(Scene((bz.Material(2.25),)), grid), dimensions=3
    )
    center = tuple(e[-1] / 2 for e in edges)
    simulation = bz.Simulation(
        material_grid=materials,
        sources=[bz.GaussianSource(position=center, width=40e-9, signal=np.ones(33))],
        monitors=[
            bz.FieldMonitor(
                center=center,
                size=(0, 100e-9, 100e-9),
                freqs=[4e14],
                fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                name="field",
            )
        ],
        time=np.arange(33) * grid.cfl_time_step(0.5),
        boundaries=[bz.PML(formulation="cpml")],
    )
    try:
        result = simulation.run(
            backend="cuda_streamed",
            termination=bz.AutoTermination(chunk_steps=17, max_wall_time=1e-12),
        )
        assert result.cuda_plan["binding"] == "native"
        assert result.cuda_plan["schedule"] == "ordinary"
        assert result.termination.steps == 17
        session = _SESSIONS[simulation]
        state = session.initial._replace(
            **dict(zip(_FIELDS, session.native.download()[:6], strict=True))
        )
        with jax.default_device(jax.devices("cpu")[0]):
            expected = execute._field_diagnostics(
                state, execute._energy_terms(session.program)
            )
        assert expected[0] > 0
        assert result.termination.energy == pytest.approx(expected[0], rel=2e-6)
        assert result.termination.max_field == expected[1]
        assert expected[2]
    finally:
        simulation.clear_compiled_cache()


def test_native_nonfinite_and_growth_checks_are_not_bypassed(scene, monkeypatch):
    from beamz.simulation.cuda.engine._runtime import _Session

    monkeypatch.setattr(
        _Session, "_field_diagnostics", lambda self: (np.nan, 1.0, False)
    )
    policy = bz.AutoTermination(chunk_steps=17)
    result = scene.run(backend="cuda_mstep", termination=policy)
    assert result.termination.reason == "nonfinite"
    assert not result.termination.converged
    assert result.termination.steps == 17
    energy = iter((1.0, 2.0, 4.0))
    monkeypatch.setattr(
        _Session, "_field_diagnostics", lambda self: (next(energy), 1.0, True)
    )
    diverged = scene.run(
        backend="cuda_mstep",
        termination=replace(
            policy,
            source_decay=1.0,
            growth_factor=1.1,
            growth_checks=2,
            consecutive_checks=10,
        ),
    )
    assert diverged.termination.reason == "diverged"
    assert diverged.termination.steps == 51


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_resident_diagnostics_detect_actual_nonfinite_field_values(scene, bad_value):
    from beamz.simulation.cuda.engine._prepare import _diagnostic_inputs

    scene.run(
        backend="cuda_streamed",
        termination=bz.AutoTermination(chunk_steps=17, max_wall_time=1e-12),
    )
    session = _SESSIONS[scene]
    inputs = list(session.packet.arguments)
    field = np.zeros(inputs[0].shape, dtype=np.float32)
    field.flat[field.size // 2] = bad_value
    inputs[0] = field
    module = _load_extension()
    oracle = module.NativeSession(
        tuple(inputs),
        dict(session.packet.attributes),
        preserve_inputs=False,
        kind="generated",
        initialize_scratch=False,
    )
    try:
        materials, axes, volume, _ = _diagnostic_inputs(session.program)
        oracle.configure_diagnostics(materials, axes, volume)
        diagnostics = oracle.field_diagnostics()
        assert not diagnostics["finite"][0]
        assert all(diagnostics["finite"][1:])
    finally:
        oracle.close()


def test_native_rejects_invalid_monitor_policy(scene):
    with pytest.raises(ValueError, match="missing"):
        scene.run(
            backend="cuda_mstep",
            termination=bz.AutoTermination(monitor_names=("missing",)),
        )


def test_radiated_pulse_converges_without_changing_full_window_spectrum(scene):
    frequency = bz.LIGHT_SPEED / 700e-9
    simulation = scene.updated_copy(
        sources=[
            bz.PlaneWaveSource(
                center=(-0.28e-6, 0.0, 0.0),
                size=(0, 1.28e-6, 1.28e-6),
                direction="+x",
                source_time=bz.GaussianPulse(frequency, frequency * 0.4, offset=0.6),
            )
        ],
        time=np.arange(1025) * 3e-17,
    )
    signal, quadrature = simulation.sources[0].source_time.sample(simulation.time)
    assert np.max(np.hypot(signal, quadrature)) > 0.9
    assert np.max(np.hypot(signal[512:], quadrature[512:])) < 1e-6
    policy = bz.AutoTermination(
        field_decay=1e-5,
        monitor_change=1e-3,
        source_decay=1e-6,
        chunk_steps=61,
        consecutive_checks=3,
    )
    try:
        bounded = simulation.run(backend="cuda_mstep", termination=policy)
        complete = simulation.run(backend="cuda_mstep")
        report = bounded.termination
        assert report.reason == "converged" and report.converged
        assert policy.chunk_steps < report.steps < simulation.num_steps
        assert report.field_decay <= policy.field_decay
        assert report.monitor_change <= policy.monitor_change
        assert report.source_decay <= policy.source_decay
        np.testing.assert_array_equal(bounded.metadata.time, complete.metadata.time)
        for normalization in (None, 0):
            a = bounded.renormalize(normalization)["field"]
            b = complete.renormalize(normalization)["field"]
            np.testing.assert_array_equal(a.dft_weight_sum, b.dft_weight_sum)
            actual = np.concatenate(
                [a.get_dft_component(c).ravel() for c in a.dft_fields]
            )
            expected = np.concatenate(
                [b.get_dft_component(c).ravel() for c in b.dft_fields]
            )
            assert np.linalg.norm(expected) > 0
            # Same spectral-tail gate as the existing pulse-convergence invariant.
            assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 5e-3
    finally:
        simulation.clear_compiled_cache()
