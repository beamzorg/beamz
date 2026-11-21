"""Exercise real admission/ownership with fake devices, without CUDA contexts."""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from beamz.simulation import backend
from beamz.simulation import compile as compiler
from beamz.simulation.cuda.engine import _runtime as runtime
from beamz.simulation.cuda.engine._contracts import _CPML
from beamz.simulation.cuda.engine._plan import _choose


@dataclass(frozen=True)
class CacheKey:
    token: str = ""
    native_preparation: bool = True


class Scene:
    is_3d = True

    def __init__(self, name: str = "", device: str = "device-A") -> None:
        self.name, self.device = name, device


@pytest.fixture
def ownership(monkeypatch):
    runtime._clear()
    cache, created = {}, []
    local = threading.local()
    gates = {}
    module = SimpleNamespace(
        coherent_backend_version=1,
        __abi_version__=backend.CUDA_ABI_VERSION,
        device_info=lambda: {"uuid": getattr(local, "uuid", "device-A")},
    )

    class Session:
        def __init__(
            self,
            simulation,
            module,
            *,
            requested,
            budget,
            memory_policy,
            diagnostics=False,
            compact_monitors=False,
        ):
            self.key = (requested, budget, memory_policy)
            self.cuda_options = (
                backend.cuda_flags_from_env() & backend.CUDA_GRAPH_CACHE,
                backend.cuda_graph_cache_capacity_from_env(),
            )
            self.device_uuid = module.device_info()["uuid"]
            self.device_uuids = (self.device_uuid,)
            self.has_diagnostics = diagnostics
            self._compact_monitors = compact_monitors
            self.plan = runtime._Plan()
            self.lock = threading.RLock()
            self.active = self.closed = False
            self.calls = 0
            created.append(self)
            cache[CacheKey(simulation.name)] = object()
            cache[CacheKey(simulation.name, False)] = object()
            if simulation.name == "unsupported":
                raise runtime._Unsupported("deliberate native capability rejection")

        def _run(self, simulation, **kwargs):
            assert not self.closed
            self.active = True
            self.calls += 1
            try:
                if simulation.name in gates:
                    entered, release = gates[simulation.name]
                    entered.set()
                    assert release.wait(3), "Test failed to release active solver"
                return simulation.name
            finally:
                self.active = False

        def _close(self):
            with self.lock:
                assert not self.active, "An active solver was evicted"
                self.closed = True

    monkeypatch.setattr(runtime, "_DEVICE_LOCKS", {})
    monkeypatch.setattr(compiler, "_PROGRAM_CACHE", cache)
    monkeypatch.setattr(runtime, "_Session", Session)
    monkeypatch.setattr(backend, "_load_extension", lambda: module)
    monkeypatch.setattr(
        backend, "cuda_flags_from_env", lambda: backend.CUDA_DEFAULT_FLAGS
    )
    monkeypatch.setattr(backend, "cuda_storage_axes_from_env", lambda: (0, 1, 2))
    monkeypatch.setattr(backend, "_cuda_legacy_schedule_options", tuple)
    monkeypatch.delenv("BEAMZ_CUDA_MEMORY_POLICY", raising=False)
    monkeypatch.delenv("BEAMZ_CUDA_MEMORY_BUDGET_BYTES", raising=False)
    monkeypatch.delenv("BEAMZ_CUDA_GRAPH_CACHE_CAPACITY", raising=False)

    def run(simulation):
        local.uuid = simulation.device
        return runtime._dispatch(simulation, backend="cuda_streamed")[0]

    yield SimpleNamespace(run=run, cache=cache, created=created, gates=gates)
    for _, release in gates.values():
        release.set()
    runtime._clear()


def test_same_scene_reuses_owner_and_cross_scene_retires_before_replacement(ownership):
    a, b = Scene("a"), Scene("b")
    assert ownership.run(a) == "a"
    first = cast(Any, runtime._SESSIONS[a])
    assert ownership.run(a) == "a"
    assert runtime._SESSIONS[a] is first and first.calls == 2
    assert len(ownership.created) == 1
    assert ownership.run(b) == "b"
    assert first.closed and a not in runtime._SESSIONS
    assert list(runtime._SESSIONS) == [b]
    assert len(ownership.created) == 2


def test_graph_cache_options_remain_native_and_replace_the_resident_owner(
    ownership, monkeypatch
):
    scene = Scene("a")
    assert ownership.run(scene) == "a"
    first = ownership.created[-1]
    monkeypatch.setattr(
        backend,
        "cuda_flags_from_env",
        lambda: backend.CUDA_DEFAULT_FLAGS & ~backend.CUDA_GRAPH_CACHE,
    )
    assert ownership.run(scene) == "a"
    second = ownership.created[-1]
    assert first.closed
    assert not second.cuda_options[0] & backend.CUDA_GRAPH_CACHE
    monkeypatch.setenv("BEAMZ_CUDA_GRAPH_CACHE_CAPACITY", "0")
    assert ownership.run(scene) == "a"
    assert second.closed
    assert ownership.created[-1].cuda_options[1] == 0


@pytest.mark.parametrize("name", ["a", "unsupported"])
def test_only_new_native_preparation_cache_entries_are_pruned(ownership, name):
    existing = CacheKey("preexisting")
    original = object()
    ownership.cache[existing] = original
    simulation = Scene(name)
    result = ownership.run(simulation)
    assert result == (None if name == "unsupported" else name)
    assert ownership.cache[existing] is original
    assert CacheKey(name) not in ownership.cache
    assert CacheKey(name, False) in ownership.cache
    assert (simulation in runtime._SESSIONS) == (name != "unsupported")


@pytest.mark.parametrize("action", ["cross_scene", "clear"])
def test_active_run_cannot_be_evicted_and_explicit_clear_waits(ownership, action):
    entered, release = threading.Event(), threading.Event()
    attempted, finished = threading.Event(), threading.Event()
    ownership.gates["blocking"] = entered, release
    active, replacement = Scene("blocking"), Scene("replacement")

    def waiting_action():
        attempted.set()
        result = (
            ownership.run(replacement)
            if action == "cross_scene"
            else runtime._clear(active)
        )
        finished.set()
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(ownership.run, active)
        try:
            assert entered.wait(2)
            session = ownership.created[-1]
            waiting = pool.submit(waiting_action)
            assert attempted.wait(2)
            assert not finished.wait(0.05)
            assert session.active and not session.closed
        finally:
            release.set()
        assert running.result(timeout=3) == "blocking"
        assert waiting.result(timeout=3) == (
            "replacement" if action == "cross_scene" else None
        )
    assert session.closed and active not in runtime._SESSIONS
    assert len(runtime._SESSIONS) == (1 if action == "cross_scene" else 0)


def test_independent_device_uuids_can_execute_concurrently(ownership):
    entered, release = threading.Event(), threading.Event()
    ownership.gates["blocking"] = entered, release
    a, b = Scene("blocking", "device-A"), Scene("independent", "device-B")
    with ThreadPoolExecutor(max_workers=2) as pool:
        blocked = pool.submit(ownership.run, a)
        try:
            assert entered.wait(2)
            # This completes while A is still blocked, proving no global execution lock.
            assert pool.submit(ownership.run, b).result(timeout=2) == "independent"
            assert not blocked.done()
            assert len(runtime._SESSIONS) == 2
            assert {cast(Any, s).device_uuid for s in runtime._SESSIONS.values()} == {
                "device-A",
                "device-B",
            }
        finally:
            release.set()
        assert blocked.result(timeout=3) == "blocking"


def test_auto_requires_measured_benefit_margin_but_preserves_explicit_overrides():
    n = 384
    inputs: list[Any] = [np.empty(0, np.float32) for _ in range(_CPML.input_count)]
    inputs[:6] = [SimpleNamespace(nbytes=4 * n**3, shape=(n, n, n)) for _ in range(6)]
    inputs[6:12] = [np.asarray(1, np.float32) for _ in range(6)]
    for axis in "xyz":
        inputs[_CPML.inputs[f"e_source_{axis}"]] = np.empty((1, 0), np.int32)
    device = dict(
        name="NVIDIA GeForce RTX 3090", major=8, minor=6, total_memory_bytes=24 << 30
    )
    module: Any = SimpleNamespace(
        compiled_program_sha256="physics",
        moving_resource_plan=lambda *a, **kw: dict(
            extra_gpu_bytes=1 << 30,
            policy_tile_counts=[70700, 0, 0, 0, 0, 0],
            max_frontier=512,
            macro_levels=1,
            tile_count=70700,
        ),
    )
    arguments = tuple(inputs)
    speed = _choose(arguments, module, device, steps=4096, memory_policy="speed")
    assert speed.schedule == "temporal"
    assert 0.9 < speed.estimated_first_solve_s / speed.estimated_ordinary_s < 1
    automatic = _choose(arguments, module, device, steps=4096)
    assert automatic.schedule == "ordinary"
    explicit = _choose(arguments, module, device, steps=4096, requested="temporal")
    assert explicit.schedule == "temporal"
    assert explicit.estimated_first_solve_s == speed.estimated_first_solve_s
    module.moving_resource_plan = lambda *a, **kw: dict(
        extra_gpu_bytes=1 << 30,
        policy_tile_counts=[50000, 0, 0, 0, 0, 0],
        max_frontier=512,
        macro_levels=1,
        tile_count=50000,
    )
    strong = _choose(arguments, module, device, steps=4096)
    assert strong.schedule == "temporal"
    assert strong.estimated_first_solve_s < 0.9 * strong.estimated_ordinary_s


def test_temporal_shard_choice_uses_local_domain_and_hardware():
    from beamz.simulation.cuda.runtime.native_sharding import _temporal_shard_choice

    h100 = {"name": "NVIDIA H100 80GB HBM3"}
    mmi = (171, 545, 1375)
    assert _temporal_shard_choice(mmi, 2, 4, 512, h100, automatic=True) == ()
    assert _temporal_shard_choice(mmi, 2, 2, 512, h100, automatic=True) == (64, 16, 6)
    assert _temporal_shard_choice(
        (8192, 512, 1024), 0, 4, 512, h100, automatic=True
    ) == (128, 16, 8)
    for extent in (4096, 8192):
        assert _temporal_shard_choice(
            (extent, 512, 1024),
            0,
            4,
            256,
            dict(h100, multiprocessors=132),
            automatic=True,
            tile=(7, 15, 31),
        ) == (64, 16, 8)
    local_ampere = {"name": "NVIDIA GeForce RTX 3090"}
    thin = (8192, 96, 256)
    assert _temporal_shard_choice(thin, 0, 1, 512, local_ampere, automatic=True) == (
        32,
        16,
        4,
    )
    assert not _temporal_shard_choice(thin, 0, 2, 512, local_ampere, automatic=True)
    assert not _temporal_shard_choice(mmi, 2, 2, 32, h100, automatic=True)
    assert not _temporal_shard_choice(mmi, 2, 2, 512, {"name": "other"}, automatic=True)
    assert _temporal_shard_choice(mmi, 2, 2, 512, {"name": "other"}, automatic=False)


def test_shard_source_padding_stays_inside_both_local_faces():
    from types import SimpleNamespace

    import numpy as np

    from beamz.simulation.cuda.runtime.native_sharding import _local_sources

    coeffs = np.arange(12, dtype=np.float32).reshape(2, 1, 1, 6) + 1
    group = SimpleNamespace(
        coeffs=coeffs,
        waveforms=np.ones((2, 3), np.float32),
        starts=np.array([[0, 0, 3], [0, 0, 13]], np.int32),
    )
    (values, _, starts), _, _ = _local_sources([group], 2, 5, {"ex": 15}, None)
    np.testing.assert_array_equal(starts[:, 2], [0, 6])
    np.testing.assert_array_equal(values[:, 0, 0], [[3, 4, 5, 6], [0, 0, 7, 8]])


def test_sharded_capacity_policy_uses_ordinary_and_rejects_forced_workspace(
    monkeypatch,
):
    from types import SimpleNamespace

    from beamz.simulation import backend
    from beamz.simulation.cuda.runtime import native_sharding as sharding

    simulation = SimpleNamespace(
        is_3d=True, num_steps=512, grid=SimpleNamespace(shape_zyx=(4096, 256, 512))
    )
    module = SimpleNamespace(
        __abi_version__=backend.CUDA_ABI_VERSION,
        _device_count=lambda: 2,
        NativeSession=SimpleNamespace(_run_temporal_partitions=True),
        _has_direct_exchange=lambda a, b: True,
        device_info=lambda _: {"name": "NVIDIA H100 80GB HBM3"},
    )
    monkeypatch.setattr(backend, "_load_extension", lambda: module)
    monkeypatch.setattr(backend, "_has_cuda_legacy_options", lambda **_: False)
    monkeypatch.setattr(
        sharding, "_run_shards", lambda *a, **kw: kw.get("temporal", ())
    )
    args = (simulation, {"axis": "auto", "num_devices": 2})
    options = dict(
        backend="cuda",
        termination=None,
        wall_start=None,
        progress=False,
        store_full_materials=False,
        performance=True,
    )
    monkeypatch.setenv("BEAMZ_CUDA_MEMORY_POLICY", "auto")
    assert sharding._try_sharded_run(*args, schedule=None, **options) == (
        (128, 16, 8),
        False,
    )
    monkeypatch.setenv("BEAMZ_CUDA_MEMORY_POLICY", "capacity")
    assert sharding._try_sharded_run(*args, schedule=None, **options) == ((), False)
    with pytest.raises(backend.CudaBackendUnavailable, match="capacity"):
        sharding._try_sharded_run(*args, schedule="mstep", **options)


@pytest.mark.parametrize(
    "shape,count,expected,window",
    [
        ((8192, 128, 512), 1, (0, 0), 64),
        ((8192, 128, 512), 2, (0, 0, 1, 1), 64),
        ((8192, 512, 1024), 2, (0, 0, 1, 1), 128),
        ((171, 545, 1375), 2, (0, 1), 64),
    ],
)
def test_temporal_sharding_sizes_frontier_per_physical_device(
    monkeypatch, shape, count, expected, window
):
    from beamz.simulation.cuda.runtime import native_sharding as sharding

    simulation = SimpleNamespace(
        is_3d=True, num_steps=512, grid=SimpleNamespace(shape_zyx=shape)
    )
    module = SimpleNamespace(
        __abi_version__=backend.CUDA_ABI_VERSION,
        _device_count=lambda: 2,
        NativeSession=SimpleNamespace(_run_temporal_partitions=True),
        _has_direct_exchange=lambda a, b: True,
        compiled_owned_zyx=(7, 15, 31),
        device_info=lambda _: {
            "name": "NVIDIA GeForce RTX 5090",
            "multiprocessors": 170,
        },
    )
    monkeypatch.setattr(backend, "_load_extension", lambda: module)
    monkeypatch.setattr(backend, "_has_cuda_legacy_options", lambda **_: False)
    monkeypatch.setenv("BEAMZ_CUDA_MEMORY_POLICY", "auto")

    def dispatch(sim, mod, devices, axis, **options):
        return devices, options["temporal"]

    monkeypatch.setattr(sharding, "_run_shards", dispatch)
    options = dict(
        backend="cuda",
        schedule=None,
        termination=None,
        wall_start=None,
        progress=False,
        store_full_materials=False,
        performance=False,
    )
    config = {"axis": "auto", "num_devices": count}
    assert sharding._try_sharded_run(simulation, config, **options) == (
        (expected, (window, 32, 4)),
        False,
    )

    def limited_memory(sim, mod, devices, axis, **options):
        if len(devices) > count:
            raise MemoryError("two partitions exceed the device budget")
        return dispatch(sim, mod, devices, axis, **options)

    if len(expected) > count:
        monkeypatch.setattr(sharding, "_run_shards", limited_memory)
        if np.prod(shape) // count >= 2**31:
            with pytest.raises(MemoryError, match="device budget"):
                sharding._try_sharded_run(simulation, config, **options)
            return
        result, interop = sharding._try_sharded_run(simulation, config, **options)
        assert not interop
        if count == 1:
            assert result is None
        else:
            assert result[0] == tuple(range(count))


def test_large_domain_partitions_keep_staggered_local_offsets_addressable():
    from beamz.simulation.cuda.runtime.native_sharding import _minimum_temporal_slabs

    shape = (8192, 512, 1024)
    assert _minimum_temporal_slabs(shape, 0, 2, 64) == 2
    assert _minimum_temporal_slabs(shape, 0, 4, 64) == 1
    assert _minimum_temporal_slabs((4096, 512, 896), 0, 1, 64) == 1


def test_rolling_band_respects_the_previous_workspace_and_device_headroom():
    from beamz.simulation.cuda.engine._plan import _admit_rolling

    resource = {
        "extra_gpu_bytes": 100,
        "rolling_extra_bytes": 160,
        "rolling_band_bytes": {32: 40, 64: 80, 128: 160},
        "observation_keys": 8,
    }
    assert _admit_rolling(resource, 1024, 128, 500) == (64, 80)
    assert _admit_rolling(resource, 1024, 128, 75, envelope=200) == (32, 40)
    assert _admit_rolling(resource, 1024, 128, 170, envelope=200) == (128, 160)
    assert _admit_rolling(resource, 1023, 128, 500, envelope=200) == (0, None)
    assert _admit_rolling(resource, 1024, 128, 39) == (0, None)
    # A narrow source can use the field storage already charged to wavefront.
    wider = dict(resource, extra_gpu_bytes=60, field_frontier_bytes=50)
    assert _admit_rolling(wider, 1024, 128, 500) == (64, 80)
    assert _admit_rolling(dict(wider, scatter_sweep_planes=2), 1024, 128, 500) == (
        32,
        40,
    )
    assert _admit_rolling(dict(wider, scatter_sweep_planes=3), 1024, 128, 500) == (
        0,
        None,
    )


def test_a100_temporal_sharding_uses_measured_long_axis():
    from beamz.simulation.cuda.runtime.native_sharding import _temporal_shard_choice

    board = {"name": "NVIDIA A100-SXM4-80GB", "multiprocessors": 108}
    options = dict(automatic=True, tile=(7, 15, 31))
    assert _temporal_shard_choice((4096, 256, 512), 0, 2, 256, board, **options)
    assert not _temporal_shard_choice((171, 545, 1375), 2, 2, 512, board, **options)


def test_small_a100_exchange_band_falls_back_on_auto_and_remains_forceable(monkeypatch):
    from types import SimpleNamespace

    from beamz.simulation import backend
    from beamz.simulation.cuda.runtime import native_sharding as sharding

    board = {"name": "NVIDIA A100-SXM4-80GB", "multiprocessors": 108}
    module = SimpleNamespace(
        __abi_version__=backend.CUDA_ABI_VERSION,
        _device_count=lambda: 2,
        device_info=lambda _: board,
        _has_direct_exchange=lambda a, b: True,
        compiled_owned_zyx=(7, 15, 31),
        NativeSession=SimpleNamespace(_run_temporal_partitions=True),
    )
    simulation = SimpleNamespace(
        is_3d=True, num_steps=256, grid=SimpleNamespace(shape_zyx=(4096, 256, 512))
    )

    def dispatch(*args, minimum_band=0, temporal=(), **kwargs):
        if minimum_band > 32:
            raise sharding._Unsupported("The admitted rolling band underfills this GPU")
        return "temporal" if temporal else "ordinary"

    monkeypatch.setattr(backend, "_load_extension", lambda: module)
    monkeypatch.setattr(backend, "_has_cuda_legacy_options", lambda **_: False)
    monkeypatch.setattr(sharding, "_run_shards", dispatch)
    monkeypatch.setenv("BEAMZ_CUDA_MEMORY_POLICY", "auto")
    config = {"num_devices": 2, "axis": "auto"}
    options = dict(
        backend="cuda",
        termination=None,
        wall_start=None,
        progress=False,
        store_full_materials=False,
        performance=False,
    )
    assert sharding._try_sharded_run(simulation, config, schedule=None, **options) == (
        "ordinary",
        False,
    )
    assert sharding._try_sharded_run(
        simulation, config, schedule="mstep", **options
    ) == ("temporal", False)


def test_rolling_admits_compact_apertures_inside_the_wavefront_memory_budget():
    from beamz.simulation.cuda.engine._plan import _admit_rolling, _rolling_band

    shape = (171, 545, 1375)
    preferred = _rolling_band(shape, {"name": "NVIDIA GeForce RTX 5090"}, 512)
    resource = dict(
        observation_keys=511178,
        extra_gpu_bytes=2800582812,
        rolling_band_bytes={32: 1061000000, 64: 1640087560, 128: 2839812872},
    )
    assert _admit_rolling(resource, 128143125, preferred, 4 << 30) == (64, 1640087560)
    assert _admit_rolling(resource, 128143125, preferred, 1 << 29) == (0, None)
    crowded = dict(resource, observation_keys=2343684)
    assert _admit_rolling(crowded, 384**3, preferred, 4 << 30) == (0, None)


@pytest.mark.parametrize(
    "devices,room,direct,links,charges",
    [
        ((0, 1, 2), (8, 7, 8), (), (False, False), (0, 0, 0)),
        ((0, 1, 2), (7, 8, 8), (), (False, True), (0, 8, 8)),
        ((0, 1, 2), (8, 16, 8), (), (True, True), (8, 16, 8)),
        ((0, 1, 0, 1), (16, 16), (), (True, True, False), (8, 16, 8, 0)),
        ((0, 0, 1, 2), (8, 16, 8), ((1, 2),), (False, True, False), (0, 8, 8, 0)),
    ],
)
def test_peer_packet_scratch_belongs_to_both_endpoints(
    devices, room, direct, links, charges
):
    from beamz.simulation.cuda.runtime.native_sharding import _admit_peer_packets

    module = SimpleNamespace(
        temporal_peer_staging_bytes=8,
        _has_direct_exchange=lambda a, b: (a, b) in direct,
    )
    remaining = dict(enumerate(room))
    payloads = [100] * len(devices)
    assert _admit_peer_packets(devices, remaining, payloads, module) == links
    assert payloads == [100 + charge for charge in charges]
    for device, available in enumerate(room):
        assert remaining[device] == available - sum(
            charge
            for dev, charge in zip(devices, charges, strict=True)
            if dev == device
        )
