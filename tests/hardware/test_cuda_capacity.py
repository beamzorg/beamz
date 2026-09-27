"""Capacity scheduling must retain full-state and modal CUDA/JAX parity."""

from argparse import Namespace
from unittest.mock import patch

import jax
import numpy as np
import pytest

from beamz.simulation import sharding
from beamz.simulation.backend import cuda_backend_status
from beamz.simulation.execute import build_scan
from scripts.benchmark_cuda_realistic import build_simulation
from tests.hardware.test_cuda_backends import _assert_state_close, _copy_state

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(
    not STATUS.available, reason=STATUS.reason or "CUDA backend unavailable"
)


@pytest.mark.parametrize("axis", [None, "x", "z"])
@pytest.mark.parametrize("material", ["binary", "smooth"])
def test_donating_capacity_schedule_matches_jax(axis, material, monkeypatch):
    if axis is None and material == "smooth":
        monkeypatch.setenv("BEAMZ_CUDA_FIELD_PADDING", "64x8")
        monkeypatch.setenv("BEAMZ_CUDA_TEMPORAL_STEPS", "2")
    sim = build_simulation(
        Namespace(
            shape=(37, 41, 49),
            steps=48,
            pml=12,
            monitors=2,
            frequencies=3,
            material=material,
            source="mode",
            monitor_type="mode",
        )
    )
    state = sim.initial_state()
    rng = np.random.default_rng(288)
    state = state._replace(
        **{
            name: rng.normal(0, 1e-5, getattr(state, name).shape).astype(np.float32)
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        }
    )
    reference = sim.advance(
        state=_copy_state(state), num_steps=48, backend="jax", performance=False
    ).state
    cfg = None if axis is None else dict(axis=axis, num_devices=1, backend="gpu")
    with patch.object(
        sharding, "_jax_devices_for_config", lambda _: (jax.devices("gpu")[0],)
    ):
        first = sim.advance(
            state=_copy_state(state),
            num_steps=17,
            backend="cuda_streamed",
            sharding=cfg,
            donate_state=True,
            performance=False,
        ).state
        actual = sim.advance(
            state=first,
            num_steps=31,
            backend="cuda_streamed",
            sharding=cfg,
            donate_state=True,
            performance=False,
        ).state
    _assert_state_close(reference, actual)
    assert int(actual.current_step) == 48
    assert np.all(np.asarray(actual.dft_weight_sum) == 48)
    if axis is None:
        program = sim.compile(num_steps=48, backend="cuda_streamed")
        executable = (
            build_scan(program, donate_state=True)
            .lower(actual, program.coefficients)
            .compile()
        )
        memory = executable.memory_analysis()
        field_bytes = sum(
            getattr(state, name).nbytes for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        )
        assert memory.alias_size_in_bytes >= field_bytes
        assert memory.temp_size_in_bytes < 32 * 1024
