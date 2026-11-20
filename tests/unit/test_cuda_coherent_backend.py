"""The production CUDA controller has one deterministic, observable plan."""

import runpy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from beamz.simulation.cuda.engine._contracts import _CPML, _YEE
from beamz.simulation.cuda.engine._plan import _choose

_physics = runpy.run_path("cuda/compiler/physics.py")
_cell, _terms = _physics["_cell"], _physics["_terms"]


def packet(n=384):
    fields = [SimpleNamespace(nbytes=4 * n**3, shape=(n, n, n)) for _ in range(6)]
    inputs = fields + [np.empty((0,), np.float32) for _ in range(128)]
    inputs[6:12] = [np.asarray(1, np.float32) for _ in range(6)]
    inputs[40:43] = [np.empty((1, 0), np.int32) for _ in range(3)]
    return tuple(inputs)


def module(extra=1 << 30):
    return SimpleNamespace(
        compiled_program_sha256="physics",
        compiled_general_ordinary=True,
        moving_resource_plan=lambda inputs, **kw: {
            "extra_gpu_bytes": extra,
            "policy_tile_counts": [
                int(np.prod(inputs[0].shape)) // 3255,
                0,
                0,
                0,
                0,
                0,
            ],
            "max_frontier": 512,
            "macro_levels": 256,
            "tile_count": int(np.prod(inputs[0].shape)) // 3255,
        },
    )


@pytest.mark.parametrize(
    "interval,volume,expected",
    [(4, True, 0), (17, True, 64), (64, True, 64), (65, True, 0), (17, False, 0)],
)
def test_full_volume_rolling_requires_measured_cadence_and_fitting_storage(
    interval, volume, expected
):
    gpu = module()
    gpu.compiled_rolling_volume_windows = gpu.compiled_rolling_tiles = True
    resource = gpu.moving_resource_plan
    gpu.moving_resource_plan = lambda inputs, **kw: dict(
        resource(inputs, **kw),
        macro_slab_count=8,
        rolling_band_bytes={64: 128, 32: 64},
    )
    indices = np.empty((0 if volume else 1, 1), np.int32)
    recorder = (
        5,
        interval,
        1.0,
        indices,
        np.empty_like(indices, np.float32),
        np.empty((2, 1)),
    )
    inputs = packet(512)
    options = dict(
        steps=128,
        requested="temporal",
        attributes={"boundary_code": 12 << 8},
        recorders=(recorder,),
    )
    plan = _choose(inputs, gpu, device(), **options)
    assert plan.rolling_band == expected
    if expected:
        smaller = _choose(
            inputs,
            gpu,
            device(),
            budget=plan.payload_bytes + plan.reserve_bytes - 1,
            **options,
        )
        assert smaller.rolling_band == 32
        assert plan.estimated_first_solve_s < smaller.estimated_first_solve_s
        gpu.compiled_rolling_volume_windows = False
        assert _choose(inputs, gpu, device(), **options).rolling_band == 0


def device(name="NVIDIA GeForce RTX 3090"):
    return dict(name=name, major=8, minor=6, total_memory_bytes=24 << 30)


def test_direct_recording_charges_only_the_field_bank_and_keeps_staged_fallback():
    gpu = module()
    gpu.compiled_rolling_volume_windows = gpu.compiled_rolling_tiles = True
    base_resource, seen = gpu.moving_resource_plan, []

    def resource(inputs, **options):
        seen.append(options)
        return dict(
            base_resource(inputs, **options),
            macro_slab_count=8,
            rolling_band_bytes={64: 256 << 20, 32: 128 << 20},
        )

    gpu.moving_resource_plan = resource
    fields = packet(512)
    staging = fields[5].nbytes
    empty = np.empty((0, 0), np.int32)
    rec = (5, 17, -1.0, empty, empty, np.empty((2, 1), np.float32))
    options = dict(
        steps=128,
        attributes={"boundary_code": 12 << 8},
        extra_payload_bytes=staging,
        recorder_staging_bytes=staging,
    )
    direct = _choose(fields, gpu, device(), recorders=(rec,), **options)
    assert direct.direct_volume_recording and "recorders" not in seen[-1]
    staged = _choose(
        fields, gpu, device(), recorders=(rec[:1] + (23,) + rec[2:],), **options
    )
    assert not staged.direct_volume_recording and "recorders" in seen[-1]
    assert staged.payload_bytes - direct.payload_bytes == staging
    pair = dict(
        options, recorder_staging_bytes=2 * staging, extra_payload_bytes=2 * staging
    )
    slow = (2, 64, *rec[2:])
    common = _choose(fields, gpu, device(), recorders=((5, 32, *rec[2:]), slow), **pair)
    assert (
        common.direct_volume_recording and common.payload_bytes == direct.payload_bytes
    )
    separate = _choose(fields, gpu, device(), recorders=(rec, slow), **pair)
    assert not separate.direct_volume_recording
    budget = direct.payload_bytes + direct.reserve_bytes
    assert _choose(
        fields, gpu, device(), budget=budget, recorders=(rec,), **options
    ) == replace(direct, memory_budget_bytes=budget)
    with pytest.raises(MemoryError):
        _choose(
            fields,
            gpu,
            device(),
            budget=budget,
            requested="ordinary",
            recorders=(rec,),
            **options,
        )


def test_boundary_gathers_price_one_interpolated_frame_and_keep_staged_fallback():
    from beamz.simulation.cuda.engine._plan import _boundary_recorder_bytes

    gpu = SimpleNamespace(compiled_boundary_acquisition=True)
    idx = np.zeros((12, 4), np.int32)
    weights = np.ones_like(idx, dtype=np.float32)
    history = np.empty((5, len(idx)), np.float32)
    record = (5, 32, -1, idx, weights, history)
    shape = (512, 384, 512)
    assert _boundary_recorder_bytes((record, record), shape, gpu) == 2 * (
        idx.nbytes + weights.nbytes + history[0].nbytes
    )
    for cadence in (16, 33, 65):
        assert _boundary_recorder_bytes(((5, cadence, *record[2:]),), shape, gpu) is None
    assert _boundary_recorder_bytes((record,), shape, SimpleNamespace()) is None
    assert _boundary_recorder_bytes((record, (5, 64, 1, idx[:0], weights[:0], history)), shape, gpu) is None
    assert _boundary_recorder_bytes((record[:-1] + (history[:, ::-1],),), shape, gpu) is None


def test_boundary_gather_keeps_ordinary_when_no_band_fits_its_actual_payload():
    gpu = module()
    gpu.compiled_rolling_tiles = gpu.compiled_rolling_volume_windows = True
    gpu.compiled_boundary_acquisition = True
    resource = gpu.moving_resource_plan
    gpu.moving_resource_plan = lambda inputs, **options: dict(
        resource(inputs, **options), macro_slab_count=8, policy_tile_counts=[10**7] * 6,
        rolling_band_bytes={32: 1 << 20, 64: 2 << 20},
    )
    indices = np.zeros((16, 4), np.int32)
    record = (5, 32, 1, indices, indices.astype(np.float32), np.empty((4, 16), np.float32))
    plan = _choose(
        packet(512), gpu, device(), steps=128, recorders=(record,), extra_payload_bytes=4096,
        recorder_staging_bytes=4096, attributes={"boundary_code": 12 << 8},
    )
    assert plan.schedule == "ordinary" and plan.boundary_gather_bytes == 0


def test_single_frozen_plan_depends_on_workload_not_prior_runs():
    p = _choose(packet(), module(), device(), steps=4096)
    assert (p.schedule, p.depth, p.raw_slots) == ("temporal", 16, 4)
    with pytest.raises(FrozenInstanceError):
        p.depth = 4
    assert _choose(packet(), module(), device(), steps=4096) == p
    assert _choose(packet(136), module(), device(), steps=4096).schedule == "ordinary"
    assert _choose(packet(), module(), device(), steps=16).schedule == "ordinary"
    assert (
        _choose(packet(), module(), device("NVIDIA H100"), steps=4096).schedule
        == "ordinary"
    )


def test_budget_rejects_temporal_before_allocating_and_never_replaces_explicit_request():
    inputs = packet()
    ordinary = _choose(inputs, module(), device(), steps=4096, requested="ordinary")
    budget = ordinary.payload_bytes + ordinary.reserve_bytes
    assert (
        _choose(inputs, module(), device(), steps=4096, budget=budget).schedule
        == "ordinary"
    )
    with pytest.raises(MemoryError):
        _choose(
            inputs, module(), device(), steps=4096, budget=budget, requested="temporal"
        )
    with pytest.raises(MemoryError):
        _choose(inputs, module(), device(), steps=4096, budget=budget - 1)
    inputs = list(inputs)
    inputs[86] = np.ones(1, np.float32)
    assert _choose(tuple(inputs), module(), device(), steps=4096).schedule == "ordinary"
    with pytest.raises(ValueError, match="pre-electric"):
        _choose(tuple(inputs), module(), device(), steps=4096, requested="temporal")


def test_temporal_planner_accounts_for_recorders_and_scatter_in_a_yee_packet():
    inputs = list(packet()[: _YEE.input_count])
    for axis in "xyz":
        inputs[_YEE.inputs[f"e_source_{axis}"]] = np.empty((1, 0), np.int32)
    inputs[_YEE.inputs["source_pre_e_x_coefficients"]] = np.ones(1, np.float32)
    flags = module()
    flags.compiled_temporal_boundary_widths = flags.compiled_temporal_sources = True
    original = flags.moving_resource_plan
    recorder = (("recorder",),)
    scatter = (("scatter",),)
    received = []

    def estimate(args, **kwargs):
        received.append(kwargs)
        return original(args, **kwargs)

    flags.moving_resource_plan = estimate
    plan = _choose(
        tuple(inputs),
        flags,
        device(),
        steps=4096,
        requested="temporal",
        recorders=recorder,
        scatters=scatter,
        extra_payload_bytes=256,
    )
    assert plan.schedule == "temporal"
    assert received == [
        dict(depth=16, raw_slots=4, recorders=recorder, scatters=scatter)
    ]
    plain = _choose(tuple(inputs), flags, device(), steps=4096, requested="temporal")
    assert plan.payload_bytes == plain.payload_bytes + 256


def test_maintained_physics_cannot_drift_between_build_and_source_mirror():
    assert Path("cuda/src/physics/cell.cuh").read_text() == _cell()
    assert Path("cuda/src/physics/curl_terms.inc").read_text() == _terms()


def test_tensor_temporal_planner_admits_the_wider_dependence_stencil():
    plugin = module()
    plugin.compiled_temporal_diagonal_materials = True
    requested = []
    estimate = plugin.moving_resource_plan

    def record(inputs, **kwargs):
        requested.append(kwargs)
        return estimate(inputs, **kwargs)

    plugin.moving_resource_plan = record
    plan = _choose(
        packet(), plugin, device(), steps=65, requested="temporal", tensor=True
    )
    assert plan.depth == 4
    assert requested == [{"depth": 4, "raw_slots": 4, "tensor": True}]
    automatic = _choose(packet(), plugin, device(), steps=16384, tensor=True)
    assert automatic.schedule == "ordinary"
    assert requested == [{"depth": 4, "raw_slots": 4, "tensor": True}]


def test_cuda_source_layout_and_runtime_are_self_contained():
    for root in (Path("beamz/simulation/cuda"), Path("cuda/src")):
        for directory in [
            root,
            *[
                p
                for p in root.rglob("*")
                if p.is_dir() and "__pycache__" not in p.parts
            ],
        ]:
            items = [
                p
                for p in directory.iterdir()
                if p.name != "__init__.py"
                and not p.name.startswith(".")
                and p.name != "__pycache__"
                and (
                    p.is_dir()
                    or p.suffix in (".py", ".cu", ".cc", ".h", ".cuh", ".inc")
                )
            ]
            assert len(items) <= 6, (directory, items)
    for source in Path("beamz/simulation/cuda/engine").glob("*.py"):
        assert (
            "from scripts" not in source.read_text()
            and "import scripts" not in source.read_text()
        )


def test_explicit_cuda_unavailability_does_not_silently_become_jax(monkeypatch):
    from beamz.simulation import backend
    from beamz.simulation.cuda.engine._runtime import _dispatch

    def unavailable():
        raise RuntimeError("no CUDA device")

    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setattr(
        backend,
        "_load_extension",
        lambda: SimpleNamespace(coherent_backend_version=1, device_info=unavailable),
    )
    simulation = SimpleNamespace(is_3d=True)
    with pytest.raises(backend.CudaBackendUnavailable, match="no CUDA device"):
        _dispatch(simulation, backend="cuda")
    result, plan = _dispatch(simulation, backend="auto")
    assert result is None and plan.binding == "jax"


def test_explicit_legacy_default_tile_keeps_the_interop_binding(monkeypatch):
    from beamz.simulation import backend
    from beamz.simulation.cuda.engine._runtime import _dispatch

    class Simulation:
        is_3d = True

    monkeypatch.setenv("BEAMZ_CUDA_CPML_SHELL_TILE", "64x4")
    monkeypatch.setattr(
        backend, "_load_extension", lambda: SimpleNamespace(coherent_backend_version=1)
    )
    result, plan = _dispatch(Simulation(), backend="cuda_streamed")
    assert result is None and plan.binding == "ffi"
    assert plan.reason == "explicit legacy CUDA options"


def test_moved_module_imports_preserve_existing_object_identity():
    import importlib

    for old, new in {
        "capabilities": "runtime.capabilities",
        "sharding": "runtime.sharding",
        "tensor": "storage.tensor",
        "material_blocks": "storage.material_blocks",
        "schedule": "mstep.schedule",
        "mstep_config": "mstep.mstep_config",
        "tuning": "mstep.tuning",
    }.items():
        assert importlib.import_module(
            "beamz.simulation.cuda." + old
        ) is importlib.import_module("beamz.simulation.cuda." + new)


@pytest.mark.parametrize("steps,expected", [(32, "ordinary"), (20981, "temporal")])
def test_horizon_amortizes_temporal_work_on_a_rectangular_grid(steps, expected):
    inputs = list(packet(96))
    for i in range(6):
        inputs[i] = SimpleNamespace(nbytes=4 * 96 * 385 * 344, shape=(96, 385, 344))
    native = SimpleNamespace(
        compiled_program_sha256="physics",
        moving_resource_plan=lambda *a, **k: dict(
            extra_gpu_bytes=856808580,
            policy_tile_counts=[719, 3426, 801, 32, 32, 498],
            max_frontier=612,
            macro_levels=189,
            tile_count=5508,
        ),
    )
    plan = _choose(tuple(inputs), native, device(), steps=steps)
    assert plan.schedule == expected
    if expected == "temporal":
        assert plan.estimated_first_solve_s < plan.estimated_ordinary_s


@pytest.mark.parametrize("steps, schedule", [(64, "ordinary"), (512, "temporal")])
def test_mid_sized_device_workload_amortizes_measured_startup(steps, schedule):
    shape = (108, 344, 867)
    inputs = list(packet())
    for field in range(6):
        inputs[field] = SimpleNamespace(shape=shape, nbytes=4 * np.prod(shape))
    native = module()
    native.moving_resource_plan = lambda *args, **kwargs: dict(
        extra_gpu_bytes=600 << 20,
        policy_tile_counts=[4608, 8023, 0, 27, 45, 347],
        tile_count=13050,
        rank_count=70,
        macro_levels=340 if steps >= 256 else 124,
        max_frontier=725,
        observation_keys=205806,
    )
    assert _choose(tuple(inputs), native, device(), steps=steps).schedule == schedule


@pytest.mark.parametrize("slots", [3, 2])
def test_memory_budget_admits_lower_overlap_without_changing_explicit_physics(slots):
    native = module()

    def resources(inputs, **kw):
        return {
            **module().moving_resource_plan(inputs, **kw),
            "extra_gpu_bytes": kw["raw_slots"] * (1 << 28),
        }

    native.moving_resource_plan = resources
    base = _choose(packet(), native, device(), steps=4096, requested="ordinary")
    plan = _choose(
        packet(),
        native,
        device(),
        steps=4096,
        budget=base.payload_bytes + base.reserve_bytes + slots * (1 << 28),
    )
    assert (plan.schedule, plan.raw_slots) == ("temporal", slots)


@pytest.mark.parametrize(
    "depths,expected",
    [((4, 8, 12, 16, 32), 12), ((8, 16), 8), ((4, 16), 4), ((16,), 0)],
)
def test_automatic_schedule_uses_the_longest_supported_block_that_fits(
    depths, expected
):
    native = module()
    native.compiled_depths = depths
    estimate = native.moving_resource_plan
    tried = []

    def resources(inputs, *, depth, raw_slots):
        tried.append((depth, raw_slots))
        return {
            **estimate(inputs),
            "extra_gpu_bytes": 2**depth + raw_slots,
            "observation_slots": 1,
        }

    native.moving_resource_plan = resources
    base = _choose(packet(), native, device(), steps=4096, requested="ordinary")
    plan = _choose(
        packet(),
        native,
        device(),
        steps=4096,
        budget=base.payload_bytes + base.reserve_bytes + 2**12 + 3,
    )
    assert tried[:3] == [(16, 4), (16, 3), (16, 2)]
    if expected:
        assert (plan.schedule, plan.depth, plan.observation_slots) == (
            "temporal",
            expected,
            1,
        )
        assert plan.raw_slots == (3 if expected == 12 else 4)
    else:
        assert plan.schedule == "ordinary"
        assert len(tried) == 3


def test_native_overlap_search_receives_the_remaining_device_budget_once():
    inputs, native = packet(), module()
    ordinary = _choose(inputs, native, device(), steps=4096, requested="ordinary")
    native.compiled_budgeted_overlap = True
    estimate = native.moving_resource_plan
    calls = []

    def bounded(values, **options):
        calls.append(options)
        return {
            **estimate(values),
            "extra_gpu_bytes": 1024,
            "raw_slots": 2,
            "observation_slots": 1,
        }

    native.moving_resource_plan = bounded
    plan = _choose(
        inputs,
        native,
        device(),
        steps=4096,
        budget=ordinary.payload_bytes + ordinary.reserve_bytes + 1024,
    )
    assert calls == [dict(depth=16, raw_slots=4, budget=1024, minimum_overlap=2)]
    assert (plan.depth, plan.raw_slots, plan.observation_slots) == (16, 2, 1)


@pytest.mark.parametrize(
    "requested,policy,enabled",
    [("auto", "auto", True), ("auto", "speed", False), ("temporal", "auto", False)],
)
def test_only_the_automatic_balanced_plan_can_trade_wave_spacing_for_observation_memory(
    requested, policy, enabled
):
    native = module()
    native.compiled_observation_spacing = True
    estimate = native.moving_resource_plan
    calls = []

    def resources(inputs, **options):
        calls.append(options)
        return {**estimate(inputs), "macro_level_stride": 13, "macro_slab_count": 16}

    native.moving_resource_plan = resources
    plan = _choose(
        packet(),
        native,
        {**device(), "multiprocessors": 82},
        steps=4096,
        requested=requested,
        memory_policy=policy,
    )
    assert plan.schedule == "temporal"
    assert plan.scheduling_sms == (82 if enabled else 0)
    assert calls[0].get("scheduling_sms", 0) == plan.scheduling_sms
    assert plan.macro_level_stride == 13


def test_driver_graph_reserve_can_reduce_overlap_before_field_allocation():
    inputs, native = packet(), module()
    native.compiled_budgeted_overlap = True
    ordinary = _choose(inputs, native, device(), steps=4096, requested="ordinary")
    estimate = native.moving_resource_plan
    calls = []

    def resources(values, *, raw_slots, **kwargs):
        calls.append(raw_slots)
        return {
            **estimate(values),
            "raw_slots": raw_slots,
            "extra_gpu_bytes": raw_slots * (128 << 20),
            "macro_levels": 1536,
        }

    native.moving_resource_plan = resources
    budget = ordinary.payload_bytes + (1110 << 20)
    result = _choose(inputs, native, device(), steps=4096, budget=budget)
    assert calls == [4, 3]
    assert result.raw_slots == 3
    assert result.reserve_bytes == 704 << 20
    assert result.payload_bytes + result.reserve_bytes <= budget


def test_explicit_temporal_can_shorten_depth_to_fit_memory_budget():
    native = module()
    native.compiled_depths = (4, 16)
    estimate = native.moving_resource_plan

    def resources(inputs, **options):
        return {
            **estimate(inputs, **options),
            "extra_gpu_bytes": 512 if options["depth"] == 16 else 256,
        }

    native.moving_resource_plan = resources
    base = _choose(packet(), native, device(), steps=4096, requested="ordinary")
    plan = _choose(
        packet(),
        native,
        device(),
        steps=4096,
        requested="temporal",
        budget=base.payload_bytes + base.reserve_bytes + 256,
    )
    assert (plan.schedule, plan.depth, plan.raw_slots) == ("temporal", 4, 4)


@pytest.mark.parametrize("steps,depth", [(513, 4), (9260, 12), (12347, 16)])
@pytest.mark.parametrize("progress", [False, True])
def test_progress_preserves_full_horizon_with_one_final_temporal_tail(
    monkeypatch, steps, depth, progress
):
    import threading

    from beamz import _helpers
    from beamz.simulation import execute
    from beamz.simulation.cuda.engine._contracts import _StateContract
    from beamz.simulation.cuda.engine._plan import _Plan
    from beamz.simulation.cuda.engine._runtime import _Session
    from beamz.simulation.results import SimulationResults

    calls = []
    native = SimpleNamespace(elapsed=0)

    def run(chunks, *args):
        calls.append(chunks)
        native.elapsed += sum(chunks)

    native.reset = lambda: setattr(native, "elapsed", 0)
    native.run_moving = run
    native.clocks = lambda: dict(
        current_step=native.elapsed, elapsed_steps=native.elapsed, time=1.0
    )
    downloads = []
    native.download_into = lambda output, destination: downloads.append(output)
    monkeypatch.setattr(_helpers, "_print_inline_progress", lambda *a, **k: None)
    monkeypatch.setattr(_helpers, "_finish_inline_progress", lambda: None)
    monkeypatch.setattr(execute, "_decode_monitor_results", lambda *a: {})
    monkeypatch.setattr(execute, "_compiled_source_launch_powers", lambda *a: ())
    monkeypatch.setattr(
        SimulationResults, "from_run", staticmethod(lambda *a, **k: SimpleNamespace())
    )
    owner = SimpleNamespace(
        lock=threading.RLock(),
        native=native,
        _completed=0,
        state_contract=_StateContract(total_steps=steps, steps=steps),
        plan=_Plan(schedule="temporal", depth=depth, raw_slots=6, macro_slab_count=16),
        program=SimpleNamespace(grid=None),
        packet=SimpleNamespace(
            _layout=_CPML,
            _buffer=lambda n: (
                np.asarray(0, np.int32)
                if n == "elapsed_steps"
                else np.empty((0,), np.float32)
            ),
        ),
        initial=SimpleNamespace(_replace=lambda **kw: None),
    )
    owner._advance = MethodType(_Session._advance, owner)
    owner._snapshot = MethodType(_Session._snapshot, owner)
    owner._execute = MethodType(_Session._execute, owner)
    owner._reset_recording_destination = MethodType(
        _Session._reset_recording_destination, owner
    )
    owner._recording_result_options = MethodType(
        _Session._recording_result_options, owner
    )
    owner._observation_report = lambda: {}
    _Session._run(
        owner, SimpleNamespace(sources=()), progress=progress, performance=False
    )
    chunks = [n for call in calls for n in call]
    assert sum(chunks) == native.elapsed == steps
    assert all(n % (16 * depth) == 0 for n in chunks[:-1])
    assert all(len(call) == 1 for call in calls)
    assert (len(calls) == 1) if not progress else len(calls) <= 20
    assert len(downloads) == 3  # One terminal DFT snapshot, independent of progress.


def test_current_free_memory_limits_planning_before_allocation():
    native = module()
    ordinary = _choose(packet(), native, device(), steps=4096, requested="ordinary")
    busy = {
        **device(),
        "free_memory_bytes": ordinary.payload_bytes
        + ordinary.reserve_bytes
        + (64 << 20),
    }
    plan = _choose(packet(), native, busy, steps=4096)
    assert plan.schedule == "ordinary"
    with pytest.raises(MemoryError):
        _choose(packet(), native, busy, steps=4096, requested="temporal")


def test_capacity_policy_never_queries_temporal_resources():
    from beamz.simulation.backend import CudaBackendUnavailable

    native = module()
    native.moving_resource_plan = lambda *a, **k: pytest.fail(
        "capacity requested temporal resources"
    )
    plan = _choose(packet(), native, device(), steps=4096, memory_policy="capacity")
    assert plan.schedule == "ordinary" and plan.memory_policy == "capacity"
    assert "capacity" in plan.reason
    inputs = list(packet())
    inputs[6] = np.ones((3, 3, 3), np.float32)
    assert _choose(
        tuple(inputs), native, device(), steps=4096, memory_policy="capacity"
    ).ordinary_general
    with pytest.raises(CudaBackendUnavailable, match="capacity"):
        _choose(
            packet(),
            native,
            device(),
            steps=4096,
            requested="temporal",
            memory_policy="capacity",
        )


def test_capacity_temporal_conflict_rejects_before_native_preparation(monkeypatch):
    import beamz.simulation.cuda.engine._runtime as runtime
    from beamz.simulation.backend import CudaBackendUnavailable

    monkeypatch.setenv("BEAMZ_CUDA_MEMORY_POLICY", "capacity")
    monkeypatch.setattr(
        runtime,
        "_Session",
        lambda *a, **k: pytest.fail("conflicting native session prepared"),
    )
    with pytest.raises(CudaBackendUnavailable, match="capacity"):
        runtime._dispatch(backend="cuda", schedule="mstep")


def test_alias_budget_requires_explicit_module_capability_and_preserves_mutable_storage():
    inputs = list(packet())
    palette = np.arange(8192, dtype=np.float32)
    inputs[37:40] = [palette] * 3
    # Repeated mutable descriptors still represent separate storage obligations.
    inputs[1] = inputs[0]
    native = module()
    before = _choose(tuple(inputs), native, device(), steps=4096, requested="ordinary")
    native.compiled_immutable_aliases = True
    after = _choose(tuple(inputs), native, device(), steps=4096, requested="ordinary")
    assert before.payload_bytes - after.payload_bytes == 2 * palette.nbytes


@pytest.mark.parametrize("memory_budget", [0, 128 << 20])
def test_small_budget_rejects_before_coefficient_preparation(
    monkeypatch, memory_budget
):
    import beamz.simulation.compile as compiler
    import beamz.simulation.cuda.engine._prepare as preparation
    from beamz.simulation.backend import CUDA_ABI_VERSION

    monkeypatch.setattr(compiler, "compile_program", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(preparation, "_validate", lambda *a, **k: None)
    monkeypatch.setattr(
        preparation,
        "_lower_coefficients",
        lambda *a, **k: pytest.fail(
            "GPU coefficients prepared before reserve admission"
        ),
    )
    native = SimpleNamespace(
        jax_independent=True,
        __abi_version__=CUDA_ABI_VERSION,
        compiled_electric_coefficient_math="cuda_fast_divide_f32",
    )
    with pytest.raises(MemoryError, match="preparation"):
        preparation._compile(
            SimpleNamespace(
                is_3d=True, grid=SimpleNamespace(metric_kind="isotropic_uniform")
            ),
            native,
            memory_budget=memory_budget,
        )


@pytest.mark.parametrize("budget,free", [(128 << 20, 24 << 30), (0, 128 << 20)])
def test_unsupported_scene_declines_before_native_reserve_guard(
    monkeypatch, budget, free
):
    import beamz.simulation.cuda.engine._runtime as runtime
    from beamz.simulation.cuda.engine._capabilities import _Unsupported

    def decline(*args, **kwargs):
        assert kwargs["memory_budget"] == min(budget or int((24 << 30) * 0.9), free)
        raise _Unsupported("capability before native reserve")

    monkeypatch.setattr(runtime, "_compile", decline)
    native = SimpleNamespace(
        compiled_coefficient_staging_bytes=32 << 20,
        device_info=lambda: {
            "uuid": "test-native-device",
            "total_memory_bytes": 24 << 30,
            "free_memory_bytes": free,
        },
    )
    with pytest.raises(_Unsupported, match="capability before native reserve"):
        runtime._Session(SimpleNamespace(num_steps=33), native, budget=budget)


@pytest.mark.parametrize("failure", ["capability", "runtime", "physics"])
def test_native_preflight_declines_capability_without_swallowing_faults(
    monkeypatch, failure
):
    import beamz.simulation.compile as compiler
    from beamz.simulation.backend import CUDA_ABI_VERSION, CudaBackendUnavailable
    from beamz.simulation.cuda.engine._capabilities import _Unsupported
    from beamz.simulation.cuda.engine._prepare import _compile

    error = {
        "capability": CudaBackendUnavailable,
        "runtime": RuntimeError,
        "physics": ValueError,
    }[failure]

    def reject(*args, **kwargs):
        raise error("preflight witness")

    monkeypatch.setattr(compiler, "compile_program", reject)
    simulation = SimpleNamespace(
        is_3d=True, grid=SimpleNamespace(metric_kind="isotropic_uniform")
    )
    native = SimpleNamespace(jax_independent=True, __abi_version__=CUDA_ABI_VERSION)
    with pytest.raises(
        _Unsupported if failure == "capability" else error, match="preflight witness"
    ):
        _compile(simulation, native, steps=3)


@pytest.mark.parametrize(
    ("name", "capability", "depth", "slots"),
    [
        ("NVIDIA GeForce RTX 5090", (12, 0), 32, 4),
        ("NVIDIA H100 NVL", (9, 0), 16, 6),
        ("NVIDIA H100 80GB HBM3", (9, 0), 16, 6),
        ("NVIDIA A100-SXM4-80GB", (8, 0), 16, 6),
    ],
)
def test_temporal_profile_respects_device_identity_and_memory(
    name, capability, depth, slots
):
    info = {**device(name), "major": capability[0], "minor": capability[1]}
    runtime = module()
    runtime.compiled_depths = (4, 8, 12, 16, 32)
    inputs = packet()
    plan = _choose(inputs, runtime, info, steps=4096)
    assert (plan.schedule, plan.depth, plan.raw_slots) == ("temporal", depth, slots)
    assert _choose(inputs, runtime, info, steps=8).schedule == "ordinary"
    speed = _choose(inputs, runtime, info, steps=4096, memory_policy="speed")
    assert speed.raw_slots == {
        "NVIDIA H100 80GB HBM3": 12,
        "NVIDIA GeForce RTX 5090": 8,
    }.get(name, slots)
    assert (
        _choose(inputs, runtime, info, steps=4096, memory_policy="capacity").schedule
        == "ordinary"
    )
    assert (
        _choose(inputs, runtime, dict(info, minor=1), steps=4096).schedule == "ordinary"
    )
