"""Run the native CUDA host controller against an in-memory ABI endpoint.

The endpoint returns deterministic downloads. Hardware tests independently check
the field updates, reductions, and peer copies.
"""

import gc
import weakref
from math import ceil
from types import SimpleNamespace

import numpy as np
import pytest

import beamz as bz
from beamz.lattice import component_shape_3d
from beamz.simulation import _cuda_abi as abi
from beamz.simulation import backend
from beamz.simulation.cuda.engine import _runtime
from beamz.simulation.cuda.engine._contracts import _CPML, _DFT, _YEE


@pytest.fixture
def driver(monkeypatch):
    device = 0
    sessions = []
    default = dict(
        name="CPU test endpoint",
        major=8,
        minor=6,
        total_memory_bytes=2 << 30,
        free_memory_bytes=2 << 30,
    )
    driver = SimpleNamespace(
        sessions=sessions,
        fail_shard=None,
        drift_shard=None,
        extra_effect_bytes=0,
        device_info=lambda ordinal=None: {
            **default,
            "uuid": f"cpu-device-{device if ordinal is None else ordinal}",
        },
        _device_count=lambda: 2,
        coherent_backend_version=1,
        __abi_version__=abi.CUDA_ABI_VERSION,
        jax_independent=True,
        compiled_program_sha256="host-contract",
        compiled_general_ordinary=True,
        compiled_rectilinear=True,
        compiled_compact_dft_components=True,
        compiled_ragged_dft_gather=True,
        compiled_clamped_source_tails=True,
        compiled_coefficient_staging_bytes=1 << 20,
        compiled_electric_coefficient_math="cuda_fast_divide_f32",
        electric_coefficient_table=lambda epsilon, dt, eps0: dt / (epsilon * eps0),
        ZeroBuffer=lambda shape, dtype: np.zeros(shape, dtype=dtype),
    )

    def select_device(ordinal):
        nonlocal device
        previous, device = device, ordinal
        return previous

    class Session:
        def __init__(self, arguments, attributes, **options):
            assert not options["preserve_inputs"]
            self.device = device
            self.arguments, self.attributes = arguments, attributes
            self.layout = _CPML if attributes["cpml_enabled"] else _YEE
            self.sharded = len(arguments) != self.layout.input_count
            monitor = self.layout.monitor - (2 * abi.FIELD_COUNT if self.sharded else 0)
            self.origin = float(arguments[monitor + abi.MONITOR_TIME_INPUT])
            self.nonempty = np.any(arguments[monitor + abi.MONITOR_WEIGHTS_INPUT])
            self.initial_origin = self.origin
            self.dt = attributes["dt"]
            self.closed = False
            self.restores = []
            self.downloads = []
            self.recorders = ()
            self.checks = None
            self.reset()
            sessions.append(self)

        def reset(self):
            self.origin = self.initial_origin
            self.step = self.elapsed = 0

        def run_schedule(self, chunks):
            assert device == self.device and not self.closed
            for count in chunks:
                start = self.step
                self.step += count
                self.elapsed += count
                for _, interval, _, _, _, history in self.recorders:
                    indices = np.arange(
                        start // interval + 1, self.step // interval + 1
                    )
                    history[indices - 1] = (interval * indices)[:, None]

        @staticmethod
        def _run_shards(owners, steps):
            for owner in owners:
                owner.step += steps
                owner.elapsed += steps - (owner.device == driver.drift_shard)
            return 0.125

        def clocks(self):
            return dict(
                current_step=self.step,
                elapsed_steps=self.elapsed,
                observation_origin=self.origin,
                time=np.float32(self.origin + self.elapsed * self.dt),
            )

        def restore_state(self, leaves, origin, step, elapsed):
            self.restores.append((leaves, origin, step, elapsed))
            self.origin, self.step, self.elapsed = origin, step, elapsed

        def download_into(self, output, target):
            assert device == self.device and not self.closed
            self.downloads.append(output)
            start = self.layout.outputs[_DFT[0]] - (
                2 * abi.FIELD_COUNT if self.sharded else 0
            )
            value = (
                2 * self.step * self.nonempty
                if output == start
                else 0
                if output == start + 1
                else self.elapsed
                if output == start + 2
                else self.step
            )
            target.fill(value)

        def configure_effects(self, tensor, scatters, recorders, chunk):
            self.recorders = recorders
            self.scatters = scatters
            return (
                sum(a.nbytes for a in tensor)
                + sum(a.nbytes for _, _, *buffers in scatters for a in buffers)
                + sum(
                    i.nbytes
                    + w.nbytes
                    + min(len(h), ceil(chunk / interval)) * h.shape[1] * h.itemsize
                    for _, interval, _, i, w, h in recorders
                )
                + driver.extra_effect_bytes
            )

        def rebind_recordings(self, histories):
            self.recorders = tuple(
                (*spec[:-1], history)
                for spec, history in zip(self.recorders, histories, strict=True)
            )

        def configure_diagnostics(
            self, materials, axes, domain_measure, selection=None
        ):
            assert len(materials) == len(axes) == 6
            assert domain_measure > 0
            self.diagnostics = (materials, axes)
            self.diagnostic_selection = selection
            return {
                "gpu_bytes": sum(a.nbytes for item in materials for a in item.values())
            }

        def field_diagnostics(self):
            return {"sums": np.ones(6), "max_fields": [2.0], "finite": [True]}

        def _configure_monitor_diagnostics(self, re, im, ranges, budget):
            self.checks = re, im, ranges, budget
            self.check_request = self.checks
            return sum(8 * freqs * points for _, freqs, points in ranges)

        def _monitor_diagnostics(self):
            assert self.checks is not None
            return 0.0, True

        def _clear_monitor_diagnostics(self):
            self.checks = None

        def _configure_shard(self, geometry):
            self.geometry = geometry
            if self.device == driver.fail_shard:
                raise RuntimeError("test shard setup failure")

        def metadata(self):
            return {"allocated_payload_bytes": sum(a.nbytes for a in self.arguments)}

        def close(self):
            assert device == self.device
            self.closed = True

    driver._set_device = select_device
    driver.NativeSession = Session
    _runtime._clear()
    monkeypatch.setattr(backend, "_load_extension", lambda: driver)
    monkeypatch.setattr(backend, "_has_cuda_legacy_options", lambda **_: False)
    yield driver
    _runtime._clear()


def scene(*, recorder=False, graded=False, cpml=True, shape=(30, 32, 34), steps=11):
    dx = 20e-9
    source = bz.CustomSource(
        "Ez",
        "e",
        (np.array([14, 14, 15]), slice(14, 18, 2), 17)
        if recorder
        else (14, slice(14, 18), 17),
        np.ones((3, 2) if recorder else (4,), dtype=np.float32),
        [1] + [0] * 10,
        component_shape_3d("Ez", shape),
    )
    monitor = bz.FieldMonitor(
        center=(16 * dx, 15 * dx, 20 * dx),
        size=(10 * dx, 10 * dx, 0),
        freqs=(2e14, 3e14),
        fields=("Ez",),
        name="dft",
    )
    monitors = (
        (monitor, bz.FieldRecorder(("Ez",), interval=2, name="frames"))
        if recorder
        else (monitor,)
    )
    geometry = {}
    if graded:
        from beamz.design.discretization import MaterialGrid
        from beamz.design.grid import RectilinearGrid
        from beamz.design.raster import Scene, rasterize

        grid = RectilinearGrid(
            *[
                np.r_[0.0, np.cumsum(dx * (1 + 0.08 * np.sin(np.arange(n))))]
                for n in reversed(shape)
            ]
        )
        material = bz.Material(permittivity=2.25)
        geometry["material_grid"] = MaterialGrid.from_raster_result(
            rasterize(Scene((material,)), grid),
            dimensions=3,
        )
    else:
        geometry.update(domain=tuple(n * dx for n in reversed(shape)), resolution=dx)
    return bz.Simulation(
        **geometry,
        time=1e-16 + np.arange(steps) * 2.5e-17,
        sources=(source,),
        monitors=monitors,
        normalize_source=None,
        boundaries=(bz.PML(),) if cpml else (bz.PEC(),),
        setup_device="cpu",
    )


def test_compact_result_and_device_checks_share_the_native_clock(driver):
    sim = scene()
    fixed = sim.run(backend="cuda", performance=False)
    assert fixed.cuda_plan["final_step"] == 11
    np.testing.assert_array_equal(fixed["dft"].dft_fields["Ez"], 22)
    policy = bz.AutoTermination(
        chunk_steps=3,
        min_steps=6,
        field_decay=0,
        consecutive_checks=1,
        monitor_names=("dft",),
    )
    result = sim.run(backend="cuda", termination=policy, performance=False)
    owner = driver.sessions[-1]
    assert driver.sessions[0].closed
    assert result.termination.converged and result.termination.steps == 6
    assert result.cuda_plan["dft_storage"] == "compact"
    assert result.cuda_plan["dft_gather_layout"] == "ragged"
    assert result.cuda_plan["monitor_convergence_gpu_bytes"] > 0
    assert result.cuda_plan["source_waveform_bytes"] == 2 * 4
    assert len(owner.downloads) == 3
    assert owner.checks is None
    re, im, ranges, budget = owner.check_request
    assert (re, im) == tuple(owner.layout.outputs[name] for name in _DFT[:2])
    assert len(ranges) == 1 and ranges[0][:2] == (0, 2)
    assert budget == 256 << 20
    np.testing.assert_array_equal(
        result["dft"].dft_weight_sum, fixed["dft"].dft_weight_sum
    )
    np.testing.assert_array_equal(result["dft"].dft_fields["Ez"], 12)


def test_recorded_checkpoints_are_detached_and_restored_on_branch(driver):
    sim = scene(recorder=True)
    lead = sim.advance(num_steps=4, backend="cuda", performance=False)
    continuation = sim.advance(
        state=lead.state,
        num_steps=2,
        backend="cuda",
        performance=False,
    )
    branch = sim.advance(
        state=lead.state,
        num_steps=4,
        backend="cuda",
        performance=False,
    )
    native = driver.sessions[-1]
    assert continuation.results.cuda_plan["reused_resident_checkpoint"]
    assert not branch.results.cuda_plan["reused_resident_checkpoint"]
    assert native.restores[-1][2:] == (4, 4)
    assert native.scatters[0][2].size == 4
    assert len(driver.sessions) == 1
    for run, n in ((lead, 4), (continuation, 6), (branch, 8)):
        monitor = run.results["frames"]
        expected = np.arange(2, n + 1, 2)
        np.testing.assert_array_equal(monitor.field_steps, expected)
        np.testing.assert_array_equal(
            monitor.fields["Ez"].reshape(len(expected), -1)[:, 0], expected
        )


def test_complete_recorder_transfers_ownership_before_session_reuse(driver):
    sim = scene(recorder=True)
    first = sim.run(backend="cuda", performance=False)
    session = driver.sessions[-1]
    original = first["frames"].fields["Ez"]
    assert np.shares_memory(original, session.recorders[0][-1])
    assert not original.flags.writeable
    expected = original.copy()
    second = sim.run(backend="cuda", performance=False)
    assert driver.sessions[-1] is session
    assert not np.shares_memory(original, session.recorders[0][-1])
    np.testing.assert_array_equal(second["frames"].fields["Ez"], expected)
    np.testing.assert_array_equal(original, expected)


def test_explicit_temporal_recording_can_stage_a_full_block(monkeypatch):
    from beamz.simulation.cuda.engine import _prepare

    monkeypatch.setattr(_prepare, "_RECORDING_STAGING_BYTES", 16)
    spec = SimpleNamespace(
        recorder_index=0,
        canonical_components=("Ez",),
        component_signs=(1.0,),
        field_buffer_indices=(0,),
        field_interp_flat_idx=(np.empty(0, np.int32),),
        field_interp_weights=(np.empty(0, np.float32),),
        record_interval=1,
    )
    program = SimpleNamespace(monitors=(spec,))
    state = SimpleNamespace(recorded_fields=(np.zeros((80, 2, 2), np.float32),))
    assert _prepare._recorder_inputs(program, state)[1:] == (1, 16)
    assert _prepare._recorder_inputs(program, state, minimum_chunk=16)[1:] == (16, 256)


@pytest.mark.parametrize("graded,cpml", [(True, True), (False, False)])
def test_sharded_packets_keep_dft_window_and_device_ownership(driver, graded, cpml):
    setup_device = "cpu" if graded else "default"
    sim = scene(graded=graded, cpml=cpml).updated_copy(setup_device=setup_device)
    convert = driver.electric_coefficient_table
    calls = []

    def track_table(values, dt, eps0):
        calls.append(len(values))
        return convert(values, dt, eps0)

    driver.electric_coefficient_table = track_table
    sharding = dict(axis="z", num_devices=2)
    result = sim.run(backend="cuda", sharding=sharding, progress=True)
    plan = result.cuda_plan
    assert plan["shard_devices"] == (0, 1)
    assert bool(calls) == (setup_device == "default")
    assert plan["shard_edges"] == (0, 15, 31)
    assert plan["final_step"] == 11
    first = tuple(driver.sessions)
    assert len(first) == 2 and all(not owner.closed for owner in first)
    for owner, lo in zip(driver.sessions, plan["shard_edges"][:-1], strict=True):
        assert owner.geometry[0][0, 1] == lo
    assert any(
        owner.arguments[owner.layout.monitor - 2 * abi.FIELD_COUNT].ndim == 2
        for owner in driver.sessions
    )
    np.testing.assert_array_equal(result["dft"].dft_weight_sum, 11)
    np.testing.assert_array_equal(result["dft"].dft_fields["Ez"], 22)

    bounded = sim.run(
        backend="cuda",
        sharding=sharding,
        performance=False,
        termination=bz.AutoTermination(chunk_steps=3, max_wall_time=1e-12),
    )
    assert bounded.termination.reason == "wall_time_limit"
    assert all(owner.closed for owner in first)
    sim.clear_compiled_cache()
    assert all(owner.closed for owner in driver.sessions)
    assert bounded.termination.steps == 3
    assert bounded.termination.energy > 0
    assert all(owner.closed for owner in driver.sessions)


def test_collecting_sharded_simulation_releases_both_devices(driver):
    sim = scene()
    sim.run(backend="cuda", sharding=dict(axis="z", num_devices=2))
    assert len(driver.sessions) == 2 and all(
        not owner.closed for owner in driver.sessions
    )
    ref = weakref.ref(sim)
    del sim
    gc.collect()
    assert ref() is None
    assert all(owner.closed for owner in driver.sessions)


def test_shard_setup_failure_closes_allocated_owners(driver):
    driver.fail_shard = 1
    with pytest.raises(RuntimeError, match="test shard setup"):
        scene().run(backend="cuda", sharding=dict(axis="z", num_devices=2))
    assert [owner.device for owner in driver.sessions] == [0, 1]
    assert all(owner.closed for owner in driver.sessions)


def test_shard_clock_mismatch_discards_partial_results(driver):
    driver.drift_shard = 1
    with pytest.raises(RuntimeError, match="diverged in their physical clocks"):
        scene().run(backend="cuda", sharding=dict(axis="z", num_devices=2))
    assert len(driver.sessions) == 2 and all(owner.closed for owner in driver.sessions)


def test_off_grid_continuation_preserves_the_recorder_prefix(driver):
    sim = scene(recorder=True)
    lead = sim.advance(num_steps=4, backend="cuda", performance=False)
    origin = np.float32(lead.state.t) + 10 * sim.dt
    result = sim.advance(
        state=lead.state._replace(t=np.float32(origin)),
        num_steps=2,
        backend="cuda",
        performance=False,
    )
    assert driver.sessions[-1].restores[-1][2:] == (4, 0)
    assert not result.results.cuda_plan["reused_resident_checkpoint"]
    observed = result.results["frames"].field_times
    np.testing.assert_array_equal(observed[:2], lead.results["frames"].field_times)
    assert observed[-1] == pytest.approx(origin + 2 * sim.dt, rel=2e-7, abs=0)


def test_effect_budget_mismatch_closes_session_before_publication(driver):
    driver.extra_effect_bytes = 1
    sim = scene(recorder=True)
    with pytest.raises(RuntimeError, match="effect allocations differ"):
        sim.run(backend="cuda")
    assert driver.sessions[0].closed
    assert sim not in _runtime._SESSIONS


@pytest.mark.parametrize(
    ("shape", "count", "expected"),
    [
        ((216, 690, 1742), 2, 1),
        ((128, 128, 128), 2, 0),
        ((8, 16, 1024), 2, 2),
        ((2, 4, 8), 8, 2),
    ],
)
def test_native_automatic_shard_axis_accounts_for_strided_faces(shape, count, expected):
    from beamz.simulation.cuda.runtime.native_sharding import _native_shard_axis

    assert _native_shard_axis("auto", shape, count) == expected
    for axis, name in enumerate("zyx"):
        assert _native_shard_axis(name, shape, count) == axis


@pytest.mark.parametrize("sharding", [False, {"enabled": False}, {"num_devices": 1}])
def test_single_device_continuation_keeps_native_dispatch(driver, sharding):
    sim = scene()
    first = sim.advance(
        backend="cuda", sharding=sharding, num_steps=3, performance=False
    )
    tail = sim.advance(
        backend="cuda", sharding=sharding, state=first.state, performance=False
    )
    assert (
        first.results.cuda_plan["binding"]
        == tail.results.cuda_plan["binding"]
        == "native"
    )
    assert tail.results.cuda_plan["reused_resident_checkpoint"]


@pytest.mark.parametrize("sharding", [None, {"num_devices": 1}])
def test_abi_mismatch_falls_back_only_for_auto(driver, monkeypatch, sharding):
    driver.__abi_version__ -= 1
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setattr(backend, "_gpu_devices", lambda: ())
    sim = scene()
    assert (
        sim.run(backend="auto", sharding=sharding, performance=False).cuda_plan[
            "binding"
        ]
        == "jax"
    )
    with pytest.raises(backend.CudaBackendUnavailable, match="ABI mismatch"):
        sim.run(backend="cuda", sharding=sharding, performance=False)
    assert not driver.sessions


def test_sharded_abi_mismatch_rejects_before_allocating_ranks(driver, monkeypatch):
    driver.__abi_version__ -= 1
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    with pytest.raises(backend.CudaBackendUnavailable, match="ABI mismatch"):
        scene().run(backend="cuda", sharding={"num_devices": 2}, performance=False)
    assert not driver.sessions


def test_auto_temporal_shard_capability_rejection_uses_jax(driver, monkeypatch):
    from beamz.simulation.cuda.runtime import native_sharding

    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setattr(driver, "_has_direct_exchange", lambda *_: True, raising=False)
    monkeypatch.setattr(
        driver.NativeSession, "_run_temporal_partitions", True, raising=False
    )
    monkeypatch.setattr(
        native_sharding, "_temporal_shard_choice", lambda *_, **__: (32, 16, 6)
    )
    monkeypatch.setattr(native_sharding, "_minimum_temporal_slabs", lambda *_, **__: 2)
    sim = scene().updated_copy(boundaries=(bz.Periodic(axes=("x",)),))
    sharding = {"num_devices": 1}
    with pytest.raises(backend.CudaBackendUnavailable, match="periodic"):
        sim.run(backend="cuda_mstep", sharding=sharding, performance=False)
    with pytest.raises(backend.CudaBackendUnavailable, match="periodic"):
        sim.run(backend="auto", schedule="mstep", sharding=sharding, performance=False)
    assert (
        sim.run(backend="auto", sharding=sharding, performance=False).cuda_plan[
            "binding"
        ]
        == "jax"
    )
    assert not driver.sessions


@pytest.mark.parametrize(
    ("option", "setting", "attribute", "expected"),
    [
        ("BEAMZ_CUDA_DISABLE_GRAPH_CACHE", "1", "cuda_flags", 0),
        ("BEAMZ_CUDA_GRAPH_CACHE_CAPACITY", "7", "graph_cache_capacity", 7),
    ],
)
def test_sharded_graph_options_replace_resident_owner(
    driver, monkeypatch, option, setting, attribute, expected
):
    monkeypatch.delenv("BEAMZ_CUDA_DISABLE_GRAPH_CACHE", raising=False)
    monkeypatch.delenv("BEAMZ_CUDA_GRAPH_CACHE_CAPACITY", raising=False)
    sim = scene()
    options = dict(backend="cuda", sharding={"num_devices": 2}, performance=False)
    sim.run(**options)
    original = tuple(driver.sessions)
    monkeypatch.setenv(option, setting)
    sim.run(**options)
    assert len(driver.sessions) == 4
    assert all(owner.closed for owner in original)
    for owner in driver.sessions[2:]:
        value = owner.attributes[attribute]
        assert (
            value & abi.CUDA_GRAPH_CACHE if attribute == "cuda_flags" else value
        ) == expected


def test_temporal_shards_restore_fp32_cpml_and_observations(driver, monkeypatch):
    import jax.numpy as jnp

    launched = []

    def run_temporal(owners, spans, axis, steps, window, depth, slots, overlaps):
        assert axis == 2 and window == 2 * depth and slots > 0 and overlaps
        assert spans[0][0] == spans[0][1] == 0
        assert spans[1][0] < spans[1][1] == spans[0][2]
        for owner in owners:
            previous = driver._set_device(owner.device)
            try:
                owner.run_schedule((steps,))
            finally:
                driver._set_device(previous)
        launched.append((steps, window, spans))
        return 0.125

    monkeypatch.setattr(
        driver.NativeSession,
        "_run_temporal_partitions",
        staticmethod(run_temporal),
        raising=False,
    )
    monkeypatch.setattr(driver, "_has_direct_exchange", lambda *_: False, raising=False)
    monkeypatch.setattr(driver, "compiled_transparent_halos", True, raising=False)
    monkeypatch.setattr(driver, "compiled_shard_resource_cuts", True, raising=False)
    monkeypatch.setattr(
        driver,
        "moving_resource_plan",
        lambda *_, **__: {"extra_gpu_bytes": 0},
        raising=False,
    )
    sim = scene(recorder=True, graded=True, shape=(30, 32, 76), steps=21).updated_copy(
        boundaries=(bz.PML(formulation="cpml", thickness=85e-9),)
    )
    options = dict(
        backend="cuda_mstep",
        sharding={"axis": "x", "num_devices": 2},
        performance=False,
    )
    fixed = sim.run(**options)
    assert fixed.cuda_plan["schedule"] == "mstep"
    assert fixed.cuda_plan["partition_devices"] == (0, 1)
    np.testing.assert_array_equal(fixed["frames"].field_steps, np.arange(2, 21, 2))
    np.testing.assert_array_equal(fixed["dft"].dft_fields["Ez"], 42)

    adaptive = sim.run(**options, termination=bz.AutoTermination(chunk_steps=17))
    check_window = adaptive.cuda_plan["exchange_window"]
    assert 0 < check_window <= 17
    assert adaptive.cuda_plan["temporal_windows"] >= 1
    assert adaptive.termination.steps == 21
    rank_zero, rank_one = launched[-1][2]
    assert [owner.diagnostic_selection for owner in driver.sessions[-2:]] == [
        (2, 0, rank_zero[2]),
        (2, rank_one[1] - rank_one[0], -1),
    ]

    head = sim.advance(**options, num_steps=10)
    legacy = head.state._replace(
        cpml_psi_h_terms=tuple(
            a.astype(jnp.bfloat16) for a in head.state.cpml_psi_h_terms
        ),
        cpml_psi_e_terms=tuple(
            a.astype(jnp.bfloat16) for a in head.state.cpml_psi_e_terms
        ),
    )
    resumed = sim.advance(**options, state=legacy)
    assert resumed.results.cuda_plan["schedule"] == "mstep"
    assert [(steps, window) for steps, window, _ in launched] == [
        (21, fixed.cuda_plan["exchange_window"]),
        (17, check_window),
        (4, check_window),
        (10, head.results.cuda_plan["exchange_window"]),
        (11, resumed.results.cuda_plan["exchange_window"]),
    ]
    for owner in driver.sessions[-2:]:
        packet, _, step, elapsed = owner.restores[0]
        assert (step, elapsed) == (10, 10)
        cpml = packet[abi.FIELD_COUNT : abi.FIELD_COUNT + 2 * abi.CPML_TERM_COUNT]
        assert any(leaf.size for leaf in cpml)
        for leaf in cpml:
            assert leaf.dtype == np.float32
            np.testing.assert_array_equal(leaf, 10)
    for result in (adaptive, resumed.results):
        np.testing.assert_array_equal(
            result["frames"].field_times, fixed["frames"].field_times
        )
        np.testing.assert_array_equal(
            result["frames"].fields["Ez"], fixed["frames"].fields["Ez"]
        )
