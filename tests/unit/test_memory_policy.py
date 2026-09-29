"""Memory pressure changes scheduling, never the caller's ownership contract."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from beamz.simulation import memory
from beamz.simulation.compile import CompiledProgramKey
from tests.unit.test_cuda_runtime_contract import _program_and_state


@pytest.mark.parametrize("donate", [False, True])
@pytest.mark.parametrize("available,capacity", [(1 << 30, False), (1, True)])
def test_auto_schedule_uses_budget_not_donation(
    monkeypatch, donate, available, capacity
):
    program, _, context = _program_and_state(cpml=True)
    program = replace(program, config=context.config)
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
