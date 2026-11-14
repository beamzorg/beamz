"""Memory pressure changes scheduling, never the caller's ownership contract."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from beamz.simulation.compile import storage as memory
from beamz.simulation.compile import CompiledProgramKey
from tests.unit.test_cuda_runtime_contract import _program_and_state


@pytest.mark.parametrize("donate", [False, True])
@pytest.mark.parametrize("available,capacity", [(1 << 30, False), (1, True)])
def test_auto_schedule_uses_budget_not_donation(
    monkeypatch, donate, available, capacity
):
    program, _, context = _program_and_state(cpml=True)
    program = replace(
        program, config=replace(context.config, cuda_memory_policy="auto")
    )
    device = SimpleNamespace(
        platform="gpu",
        memory_stats=lambda: dict(
            bytes_limit=2 << 30, bytes_in_use=(2 << 30) - available
        ),
    )
    monkeypatch.setattr(memory.jax, "devices", lambda: [device])
    assert memory.cuda_capacity_schedule(program, donate_state=donate) is capacity


@pytest.mark.parametrize("policy,capacity", [("capacity", True), ("speed", False)])
@pytest.mark.parametrize("donate", [False, True])
def test_explicit_schedule_does_not_query_devices(
    monkeypatch, policy, capacity, donate
):
    program, _, context = _program_and_state(cpml=True)
    program = replace(
        program, config=replace(context.config, cuda_memory_policy=policy)
    )

    def unexpected():
        raise AssertionError("explicit policy must not query devices")

    monkeypatch.setattr(memory.jax, "devices", unexpected)
    assert memory.cuda_capacity_schedule(program, donate_state=donate) is capacity


def test_policy_validated_and_in_program_identity(monkeypatch):
    import numpy as np

    import beamz as bz

    sim = bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um),
        resolution=0.1 * bz.um,
        time=np.arange(3) * 1e-17,
    )
    request = sim.to_request(num_steps=2, backend="cuda_streamed")
    keys = []
    for policy in ("auto", "speed", "capacity"):
        monkeypatch.setenv("BEAMZ_CUDA_MEMORY_POLICY", policy)
        assert memory.cuda_memory_policy_from_env() == policy
        keys.append(
            CompiledProgramKey.from_request(
                replace(request, run=replace(request.run, cuda_memory_policy=policy))
            )
        )
    assert len(set(keys)) == 3
    monkeypatch.setenv("BEAMZ_CUDA_MEMORY_POLICY", "fastest")
    with pytest.raises(ValueError, match="auto, speed, or capacity"):
        memory.cuda_memory_policy_from_env()


def test_initial_state_copies_each_field_only_once(monkeypatch):
    from beamz.simulation import execute

    program, _, _ = _program_and_state(cpml=True)
    original = execute._copy_initial_field
    copies = []

    def copy(value):
        copies.append(value.shape)
        return original(value)

    monkeypatch.setattr(execute, "_copy_initial_field", copy)
    execute.initial_program_state(program, t=0, current_step=0)
    assert len(copies) == 6


def test_monitor_continuation_does_not_allocate_discarded_history(monkeypatch):
    import numpy as np

    import beamz as bz
    from beamz.simulation.results import observe

    sim = bz.Simulation(
        size=(0.8e-6, 0.6e-6, 0.4e-6),
        resolution=0.1e-6,
        time=np.arange(8) * 1e-17,
        monitors=[
            bz.FieldRecorder(("Ez",), interval=1, name="frames"),
            bz.FieldMonitor(
                center=(0, 0, 0),
                size=(0, 0.2e-6, 0.2e-6),
                freqs=(2e14,),
                fields=("Ez",),
                name="dft",
            ),
        ],
    )
    original = observe.empty_monitor_values
    allocations = []

    def allocate(*args, **kwargs):
        allocations.append(kwargs.get("num_steps"))
        return original(*args, **kwargs)

    monkeypatch.setattr(observe, "empty_monitor_values", allocate)
    first = sim.advance(num_steps=4, backend="jax", performance=False)
    second = sim.advance(
        state=first.state, num_steps=4, backend="jax", performance=False
    )
    np.testing.assert_array_equal(second.results["frames"].field_steps, np.arange(1, 9))
    assert int(second.state.recorded_counts[0]) == 8
    assert np.all(np.asarray(second.state.dft_weight_sum) == 8)
    assert allocations == [8], (
        "Compatible monitor history must be reused before allocation"
    )


def test_memory_policy_does_not_disable_layout_selection(monkeypatch):
    from beamz.simulation.cuda.tuning import tuning_policy_from_env

    monkeypatch.setenv("BEAMZ_CUDA_MEMORY_POLICY", "speed")
    assert not any(
        key == "BEAMZ_CUDA_MEMORY_POLICY" for key, _ in tuning_policy_from_env()[2]
    )


def test_scan_remains_composable_under_jit_and_grad():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from beamz.simulation.execute import build_scan

    program, state, _ = _program_and_state(cpml=True)
    scan = build_scan(program)
    pulse = jnp.zeros_like(state.ex).at[3, 3, 3].set(1e-3)

    def energy(amplitude):
        result = scan(state._replace(ex=pulse * amplitude), program.coefficients)
        return jnp.sum(result.ex**2)

    expected = float(energy(1.0))
    assert expected > 0
    np.testing.assert_allclose(jax.jit(energy)(1.0), expected, rtol=1e-5)
    np.testing.assert_allclose(jax.grad(energy)(1.0), 2 * expected, rtol=1e-5)
    np.testing.assert_allclose(jax.jit(jax.grad(energy))(1.0), 2 * expected, rtol=1e-5)


def test_mstep_estimate_counts_compact_fields_and_plan():
    import math

    program, _, context = _program_and_state(cpml=True)
    program = replace(
        program, config=replace(context.config, cuda_memory_policy="auto")
    )
    plan = SimpleNamespace(summary=lambda: {"scratch_bytes": 8192, "plan_bytes": 1024})
    fields = sum(
        math.prod(shape) * 4 for shape in program.sharding.layout.padded_shapes.values()
    )
    psi = sum(
        math.prod(t.slab.shape) * 4
        for t in (*program.boundary.cpml.h_terms, *program.boundary.cpml.e_terms)
    )
    donated = memory.cuda_workspace_estimate(
        program, donate_state=True, mstep_plan=plan
    )
    retained = memory.cuda_workspace_estimate(
        program, donate_state=False, mstep_plan=plan
    )
    assert donated == 2 * fields + psi + 8192 + 1024
    assert retained - donated == fields + psi
    for donate, full in ((True, donated), (False, retained)):
        minimum = memory.cuda_workspace_estimate(
            program, donate_state=donate, mstep=True
        )
        assert full - minimum == 8192 + 1024


@pytest.mark.parametrize("policy,available", [("auto", 1), ("capacity", 1 << 30)])
def test_forced_mstep_reports_memory_limits(monkeypatch, policy, available):
    from beamz.simulation.compile.backend import CudaBackendUnavailable

    program, _, context = _program_and_state(cpml=True)
    program = replace(
        program,
        config=replace(context.config, backend="cuda_mstep", cuda_memory_policy=policy),
    )
    plan = SimpleNamespace(summary=lambda: {"scratch_bytes": 8192, "plan_bytes": 1024})
    monkeypatch.setattr(memory, "cuda_available_bytes", lambda p: available)
    with pytest.raises(CudaBackendUnavailable, match="schedule='mstep'"):
        memory.check_mstep_workspace(program, plan, donate_state=False)
