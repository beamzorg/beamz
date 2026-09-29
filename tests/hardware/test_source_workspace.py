"""Mode-source layout conversion must not retain extra full field shards."""

from argparse import Namespace
from dataclasses import replace

import jax
import pytest

from beamz.simulation import observe, sharding
from beamz.simulation.backend import cuda_backend_status
from beamz.simulation.execute import build_scan, initial_program_state
from scripts.benchmark_cuda_realistic import build_simulation


def test_sharded_mode_source_workspace_is_smaller_than_one_field(monkeypatch):
    devices = jax.devices()
    if not cuda_backend_status().available or len(devices) not in (2, 8):
        pytest.skip("requires two or eight visible CUDA GPUs")
    monkeypatch.setenv("BEAMZ_CUDA_MEMORY_POLICY", "capacity")
    with jax.default_device(jax.devices("cpu")[0]):
        sim = build_simulation(
            Namespace(
                shape=(512, 128, 256),
                steps=32,
                pml=12,
                monitors=2,
                frequencies=101,
                material="binary",
                source="mode",
                monitor_type="mode",
            )
        )
        program = sim.compile(
            num_steps=32,
            backend="cuda_streamed",
            sharding=dict(axis="z", num_devices=len(devices), backend="gpu"),
        )
        state = initial_program_state(program, t=0, current_step=0, monitor_steps=32)
    state = sharding.prepare_state(
        program, state, replicated_fields=(*observe.MONITOR_FIELDS, "t", "current_step")
    )
    coefficients = sharding.place_tree(program, program.coefficients)
    jax.block_until_ready((state, coefficients))
    workspace = []
    for candidate in (program, replace(program, sources=())):
        executable = (
            build_scan(candidate, donate_state=True).lower(state, coefficients).compile()
        )
        workspace.append(executable.memory_analysis().temp_size_in_bytes)
    # The failing 15B lowering introduced two complete local field buffers.
    # Keep the assertion about allocation size, not compiler-specific HLO names.
    local_field_bytes = min(
        getattr(state, name).addressable_shards[0].data.nbytes
        for name in ("ex", "ey", "ez", "hx", "hy", "hz")
    )
    assert workspace[0] - workspace[1] < local_field_bytes
