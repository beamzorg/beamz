"""A measured schedule choice must preserve capability and memory contracts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from beamz.lattice import component_shapes
from beamz.simulation import _cuda_abi as abi
from beamz.simulation.backend import CUDA_DEFAULT_FLAGS
from beamz.simulation.compile import CompiledProgramKey, _preparation_key
from beamz.simulation.cuda import schedule


@pytest.mark.parametrize(
    "backend,choice,expected",
    [
        ("cuda", None, "streamed"),
        ("auto", None, "streamed"),
        ("cuda", "auto", "auto"),
        ("auto", "auto", "auto"),
        ("cuda_streamed", None, "streamed"),
        ("cuda_mstep", None, "mstep"),
        ("cuda", "mstep", "mstep"),
        ("cuda", "streamed", "streamed"),
    ],
)
def test_schedule_request_and_compatibility_aliases(backend, choice, expected):
    assert schedule.normalize_schedule_request(backend, choice, 3) == (expected, 3)


@pytest.mark.parametrize(
    "backend,choice,runs",
    [
        ("cuda", "unknown", 1),
        ("cuda_streamed", "mstep", 1),
        ("cuda_mstep", "auto", 1),
        ("jax", "mstep", 1),
        ("cuda", "auto", 0),
        ("cuda", "auto", True),
        ("cuda", "auto", 1.5),
    ],
)
def test_invalid_schedule_requests_fail(backend, choice, runs):
    with pytest.raises(ValueError):
        schedule.normalize_schedule_request(backend, choice, runs)


def _program(shape=(256, 512, 512), steps=256, runs=1):
    cfg = SimpleNamespace(
        backend="cuda_streamed",
        cuda_schedule="auto",
        cuda_expected_runs=runs,
        num_steps=steps,
        cuda_flags=CUDA_DEFAULT_FLAGS,
        is_3d=True,
        sharding=SimpleNamespace(enabled=False),
        cuda_storage_axes=(0, 1, 2),
        metric_kind="isotropic_uniform",
        cuda_memory_policy="auto",
        cuda_graph_cache_capacity=8,
    )
    coeffs = SimpleNamespace(e_inverse_offdiagonal=np.empty(0))
    for axis in "xyz":
        setattr(coeffs, f"e_decay_{axis}", np.ones(2, np.float32))
        setattr(coeffs, f"e_source_{axis}", np.zeros(1, np.int32))
        for kind in ("decay", "source"):
            setattr(coeffs, f"h_{kind}_{axis}", np.array(1, np.float32))
    terms = tuple(
        SimpleNamespace(slab=SimpleNamespace(low=12, high=12, shape=(1, 1, 1)))
        for _ in range(6)
    )
    cpml = SimpleNamespace(
        enabled=True,
        h_terms=terms,
        e_terms=terms,
        metallic_edges=frozenset(),
        magnetic_edges=frozenset(),
    )
    supports = component_shapes(shape)
    return SimpleNamespace(
        config=cfg,
        coefficients=coeffs,
        boundary=SimpleNamespace(cpml=cpml, logical_component_shapes=supports),
        monitors=(SimpleNamespace(freq_count=3, dft_point_count=100),) * 2,
        sharding=SimpleNamespace(
            mesh=None,
            layout=SimpleNamespace(
                enabled=False, num_devices=1, padded_shapes=supports, axis=None
            ),
        ),
    )


def _geometry(shape=(257, 513, 513), active=480, depth=229):
    return SimpleNamespace(
        shape=shape,
        core_cells=52_000_000,
        summary=lambda: {
            "core_fraction": 0.78,
            "active_tiles": active,
            "column_depth": depth,
            "owned_quadrants": active * 4,
        },
    )


def test_selector_requires_startup_to_be_amortized():
    program = _program()
    candidate, reason, details = schedule.automatic_candidate(
        program, _geometry(), device_kind="NVIDIA GeForce RTX 3090"
    )
    assert not candidate
    assert "startup" in reason
    assert details["predicted_saved_execution_s"] <= 1.25 * details["extra_startup_s"]
    program.config.cuda_expected_runs = 16
    assert schedule.automatic_candidate(
        program, _geometry(), device_kind="NVIDIA GeForce RTX 3090"
    )[0]


@pytest.mark.parametrize(
    "change",
    [
        "gpu",
        "columns",
        "dense",
        "monitors",
        "precision",
        "graph_cache",
        "graph_capacity",
    ],
)
def test_selector_stays_streamed_outside_measured_envelope(change):
    program = _program(runs=32)
    geometry = _geometry()
    gpu = "NVIDIA GeForce RTX 3090"
    if change == "gpu":
        gpu = "NVIDIA H100"
    if change == "columns":
        geometry = _geometry(active=28)
    if change == "dense":
        program.coefficients.h_source_x = np.ones((3, 3, 3), np.float32)
    if change == "monitors":
        program.monitors *= 2
    if change == "precision":
        program.config.cuda_flags |= abi.CUDA_BF16_PSI
    if change == "graph_cache":
        program.config.cuda_flags &= ~abi.CUDA_GRAPH_CACHE
    if change == "graph_capacity":
        program.config.cuda_graph_cache_capacity = 0
    assert not schedule.automatic_candidate(program, geometry, device_kind=gpu)[0]


def test_known_short_chunk_reuse_can_repay_startup():
    program = _program(steps=16, runs=256)
    candidate, _, details = schedule.automatic_candidate(
        program, _geometry(), device_kind="NVIDIA GeForce RTX 3090"
    )
    assert candidate
    assert details["expected_runs"] == 256
    program.config.cuda_expected_runs = 1
    assert not schedule.automatic_candidate(
        program, _geometry(), device_kind="NVIDIA GeForce RTX 3090"
    )[0]


@pytest.mark.parametrize("family", ["all", "pec", "right"])
def test_measured_boundary_families_select_their_validated_variant(family):
    program = _program(steps=1024, runs=32)
    cpml = program.boundary.cpml
    if family == "pec":
        cpml.enabled = False
        cpml.metallic_edges = frozenset(
            {"front", "back", "bottom", "top", "left", "right"}
        )
    if family == "right":
        terms = tuple(
            SimpleNamespace(
                axis=a, slab=SimpleNamespace(low=0, high=12 if a == 2 else 0)
            )
            for a in (0, 1, 2, 0, 1, 2)
        )
        cpml.h_terms = cpml.e_terms = terms
        cpml.metallic_edges = frozenset({"front", "back", "bottom", "top", "left"})
    candidate, _, details = schedule.automatic_candidate(
        program, _geometry(), device_kind="NVIDIA GeForce RTX 3090"
    )
    assert candidate
    assert details["boundary_family"] == family
    expected = abi.CUDA_GRAPH_CACHE
    if family != "all":
        expected |= abi.CUDA_MSTEP_TILE8X16 | abi.CUDA_MSTEP_SEGMENT32
    assert details["mstep_cuda_flags"] == expected


def test_short_single_call_does_not_query_gpu_or_build_dependency_plan(monkeypatch):
    from beamz.simulation.cuda import mstep

    def forbidden(*args, **kwargs):
        pytest.fail("Short single call queried hardware or built a plan")

    monkeypatch.setattr(schedule, "cuda_device_kind", forbidden)
    monkeypatch.setattr(mstep, "build_mstep_geometry", forbidden)
    _, decision = schedule.choose_execution_schedule(
        _program(steps=16), (None,) * 9, None, native_supported=True, donate_state=False
    )
    assert decision["selected"] == "streamed"


@pytest.mark.parametrize(
    "available,chosen", [(1, "streamed"), (None, "streamed"), (20 << 30, "mstep")]
)
def test_auto_selection_checks_actual_workspace(monkeypatch, available, chosen):
    from beamz.simulation import backend, memory
    from beamz.simulation.cuda import mstep

    program = _program(runs=32)
    monkeypatch.setattr(schedule, "cuda_device_kind", lambda: "NVIDIA GeForce RTX 3090")
    monkeypatch.setattr(
        backend,
        "cuda_backend_status",
        lambda **kw: SimpleNamespace(targets=(abi.CUDA_MSTEP_TARGET,)),
    )

    def geometry(*args):
        assert available is not None and available > 1, (
            "planned before budget rejection"
        )
        return _geometry()

    monkeypatch.setattr(mstep, "build_mstep_geometry", geometry)
    plan = SimpleNamespace(summary=lambda: {"scratch_bytes": 8192, "plan_bytes": 1024})
    monkeypatch.setattr(mstep, "build_mstep_plan", lambda *args: plan)
    monkeypatch.setattr(memory, "cuda_available_bytes", lambda program: available)
    result, decision = schedule.choose_execution_schedule(
        program, (None,) * 9, None, native_supported=True, donate_state=False
    )
    assert decision["selected"] == chosen
    assert (result is plan) == (chosen == "mstep")


def test_auto_selection_still_checks_full_plan_and_refreshes_budget(monkeypatch):
    from beamz.simulation import backend, memory
    from beamz.simulation.cuda import mstep

    program = _program(runs=32)
    monkeypatch.setattr(schedule, "cuda_device_kind", lambda: "NVIDIA GeForce RTX 3090")
    monkeypatch.setattr(
        backend,
        "cuda_backend_status",
        lambda **kw: SimpleNamespace(targets=(abi.CUDA_MSTEP_TARGET,)),
    )
    monkeypatch.setattr(mstep, "build_mstep_geometry", lambda *args: _geometry())
    plan = SimpleNamespace(summary=lambda: {"scratch_bytes": 8192, "plan_bytes": 1024})
    monkeypatch.setattr(mstep, "build_mstep_plan", lambda *args: plan)
    budgets = iter((20 << 30, 1))
    monkeypatch.setattr(memory, "cuda_available_bytes", lambda _: next(budgets))
    result, decision = schedule.choose_execution_schedule(
        program, (None,) * 9, None, native_supported=True, donate_state=False
    )
    assert result is None
    assert decision["selected"] == "streamed"
    assert decision["available_bytes"] == 1
    assert "workspace_bytes" in decision


@pytest.mark.parametrize("progress", [False, True])
@pytest.mark.parametrize("runs", [True, False, 0, -1, 1.5, "2"])
def test_progress_preserves_reuse_hint_validation(progress, runs):
    import beamz as bz

    sim = bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um),
        resolution=0.1 * bz.um,
        time=np.arange(5) * 1e-17,
    )
    with pytest.raises(ValueError, match="expected_runs must be a positive integer"):
        sim.advance(
            num_steps=2,
            backend="jax",
            progress=progress,
            expected_runs=runs,
            performance=False,
        )


def test_schedule_and_reuse_are_execution_cache_inputs():
    import beamz as bz

    sim = bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um),
        resolution=0.1 * bz.um,
        time=np.arange(3) * 1e-17,
    )
    request = sim.to_request(
        num_steps=2, backend="cuda", schedule="auto", expected_runs=4
    )
    keys = [
        CompiledProgramKey.from_request(
            replace(
                request,
                run=replace(request.run, cuda_schedule=choice, cuda_expected_runs=runs),
            )
        )
        for choice, runs in (("auto", 4), ("streamed", 4), ("auto", 1))
    ]
    assert len(set(keys)) == 3
    assert len({_preparation_key(key) for key in keys}) == 1
    cuda_keys = [
        replace(
            keys[0],
            backend=backend,
            cuda_flags=flags,
            cuda_storage_axes=axes,
            cuda_memory_policy=policy,
        )
        for backend, flags, axes, policy in (
            ("cuda_streamed", abi.CUDA_GRAPH_CACHE, (0, 1, 2), "auto"),
            (
                "cuda_mstep",
                abi.CUDA_GRAPH_CACHE | abi.CUDA_MSTEP_TILE8X32,
                (0, 1, 2),
                "speed",
            ),
            (
                "cuda_streamed",
                abi.CUDA_GRAPH_CACHE | abi.CUDA_TEMPORAL_PAIR,
                (1, 2, 0),
                "capacity",
            ),
        )
    ]
    assert len(set(cuda_keys)) == 3
    assert len({_preparation_key(key) for key in cuda_keys}) == 1
    assert _preparation_key(
        replace(cuda_keys[0], cuda_flags=abi.CUDA_BF16_PSI)
    ) != _preparation_key(cuda_keys[0])


def test_auto_backend_cannot_silently_accept_mstep_on_2d():
    import beamz as bz
    from beamz.simulation.backend import CudaBackendUnavailable

    sim = bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um),
        resolution=0.1 * bz.um,
        time=np.arange(3) * 1e-17,
    )
    with pytest.raises(CudaBackendUnavailable, match="3D"):
        sim.compile(backend="auto", schedule="mstep")


@pytest.mark.parametrize(
    "tile,segment",
    [("8x32", "32"), ("8x16", "128"), ("16x16", "256"), ("16x32", "whole")],
)
def test_mstep_variant_flags_are_validated_and_cacheable(monkeypatch, tile, segment):
    from beamz.simulation.cuda.mstep_config import (
        variant_flags_from_env,
        variant_summary,
    )

    monkeypatch.setenv("BEAMZ_CUDA_MSTEP_TILE", tile)
    monkeypatch.setenv("BEAMZ_CUDA_MSTEP_SEGMENT", segment)
    flags = variant_flags_from_env()
    assert variant_summary(flags) == {
        "tile": tuple(map(int, tile.split("x"))),
        "column_segment": 0 if segment == "whole" else int(segment),
    }
    monkeypatch.setenv("BEAMZ_CUDA_MSTEP_TILE", "bad")
    with pytest.raises(ValueError, match="TILE"):
        variant_flags_from_env()


def test_progress_uses_known_chunk_reuse_and_preserves_state(monkeypatch):
    import jax

    import beamz as bz
    from beamz.simulation.execute import initial_program_state

    sim = bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um),
        resolution=0.1 * bz.um,
        time=np.arange(44) * 1e-17,
    )
    program = sim.compile(num_steps=43, backend="jax")
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=43)
    rng = np.random.default_rng(73)
    state = state._replace(
        **{
            name: rng.normal(size=getattr(state, name).shape).astype(np.float32) * 1e-3
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        }
    )
    expected = sim.advance(state=state, num_steps=43, backend="jax", progress=False)
    calls = []
    compile_program = type(sim).compile

    def record(self, **kwargs):
        calls.append((kwargs["num_steps"], kwargs["expected_runs"]))
        return compile_program(self, **kwargs)

    monkeypatch.setattr(type(sim), "compile", record)
    actual = sim.advance(
        state=state, num_steps=43, backend="jax", progress=True, expected_runs=2
    )
    assert sorted(calls) == [(1, 2), (3, 28)]
    for reference, result in zip(
        jax.tree.leaves(expected.state), jax.tree.leaves(actual.state), strict=True
    ):
        np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-8)


def test_pmc_does_not_reuse_measured_pec_or_cpml_performance_models():
    from beamz.simulation.cuda.schedule import measured_boundary_family

    cpml = SimpleNamespace(magnetic_edges=frozenset({"left"}))
    assert measured_boundary_family(cpml) is None
