"""Freeze the explicit working-set policy across CUDA compilation and execution."""

import numpy as np
import pytest

import beamz as bz
from beamz.simulation.compile import compile_program
from beamz.simulation.memory import _cuda_working_set


def test_cuda_budget_is_frozen_in_program_key_and_shares_material_preparation(
    monkeypatch,
):
    sim = bz.Simulation(
        domain=(0.64e-6,) * 3,
        resolution=20e-9,
        time=np.arange(3) * 1e-17,
        boundaries=[bz.PML(formulation="cpml")],
    )
    try:
        monkeypatch.setenv("BEAMZ_CUDA_MEMORY_BUDGET_BYTES", str(1 << 30))
        first = compile_program(
            sim, backend="cuda_streamed", backend_resolver=lambda _: "cuda_streamed"
        )
        monkeypatch.setenv("BEAMZ_CUDA_MEMORY_BUDGET_BYTES", str(2 << 30))
        second = compile_program(
            sim, backend="cuda_streamed", backend_resolver=lambda _: "cuda_streamed"
        )
        assert first is not second
        assert first.grid is second.grid
        assert first.config._cuda_memory_budget_bytes == 1 << 30
        assert second.config._cuda_memory_budget_bytes == 2 << 30
        required = _cuda_working_set(
            first, donate_state=False, low_memory=True, enforce=True
        )
        assert 512 << 20 < required < 1 << 30
    finally:
        sim.clear_compiled_cache()


def test_interop_budget_rejects_small_working_set_before_runtime_placement(monkeypatch):
    sim = bz.Simulation(
        domain=(0.64e-6,) * 3,
        resolution=20e-9,
        time=np.arange(3) * 1e-17,
        boundaries=[bz.PML(formulation="cpml")],
    )
    try:
        monkeypatch.setenv("BEAMZ_CUDA_MEMORY_BUDGET_BYTES", str(128 << 20))
        with pytest.raises(MemoryError, match="per-run storage"):
            compile_program(
                sim, backend="cuda_streamed", backend_resolver=lambda _: "cuda_streamed"
            )
    finally:
        sim.clear_compiled_cache()


def test_explicit_speed_cannot_bypass_capacity_sized_budget(monkeypatch):
    from dataclasses import replace

    from beamz.simulation.memory import cuda_capacity_schedule

    sim = bz.Simulation(
        domain=(0.64e-6,) * 3,
        resolution=20e-9,
        time=np.arange(3) * 1e-17,
        boundaries=[bz.PML(formulation="cpml")],
    )
    try:
        monkeypatch.setenv("BEAMZ_CUDA_MEMORY_BUDGET_BYTES", str(1 << 30))
        program = compile_program(
            sim, backend="cuda_streamed", backend_resolver=lambda _: "cuda_streamed"
        )
        capacity = _cuda_working_set(program, donate_state=True, low_memory=True)
        speed = _cuda_working_set(program, donate_state=True, low_memory=False)
        assert capacity < speed
        budget = (capacity + speed) // 2
        bounded = replace(
            program,
            config=replace(
                program.config,
                _cuda_memory_budget_bytes=budget,
                cuda_memory_policy="speed",
            ),
        )
        with pytest.raises(MemoryError, match="per-run storage"):
            _cuda_working_set(bounded, donate_state=True, enforce=True)
        with pytest.raises(MemoryError, match="per-run storage"):
            cuda_capacity_schedule(bounded, donate_state=True)
        monkeypatch.setenv("BEAMZ_CUDA_MEMORY_BUDGET_BYTES", str(budget))
        monkeypatch.setenv("BEAMZ_CUDA_MEMORY_POLICY", "speed")
        with pytest.raises(MemoryError, match="per-run storage"):
            compile_program(
                sim, backend="cuda_streamed", backend_resolver=lambda _: "cuda_streamed"
            )
        monkeypatch.setenv("BEAMZ_CUDA_MEMORY_POLICY", "capacity")
        allowed = compile_program(
            sim, backend="cuda_streamed", backend_resolver=lambda _: "cuda_streamed"
        )
        assert cuda_capacity_schedule(allowed, donate_state=True)
    finally:
        sim.clear_compiled_cache()


def test_native_capture_does_not_apply_interop_bank_admission(monkeypatch):
    from beamz.simulation import execute
    from tests.unit.test_cuda_runtime_contract import _program_and_state

    program, _, _ = _program_and_state(cpml=True)
    monkeypatch.setattr(
        execute,
        "cuda_capacity_schedule",
        lambda *a, **k: pytest.fail(
            "native capture applied interoperability memory admission"
        ),
    )
    ctx = execute.build_step_context(program, low_memory=True)
    assert ctx.low_memory is True
