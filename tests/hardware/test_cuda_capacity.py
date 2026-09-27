"""Capacity scheduling must retain full-state and modal CUDA/JAX parity."""

from argparse import Namespace
from dataclasses import replace
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
@pytest.mark.parametrize("policy", ["auto", "capacity"])
def test_donating_capacity_schedule_matches_jax(axis, material, policy, monkeypatch):
    monkeypatch.setenv("BEAMZ_CUDA_MEMORY_POLICY", policy)
    if axis is None and material == "binary":
        monkeypatch.setenv("BEAMZ_CUDA_STORAGE_AXES", "201")
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
        if policy == "capacity":
            assert memory.temp_size_in_bytes < 32 * 1024
        else:
            assert memory.temp_size_in_bytes >= field_bytes


@pytest.mark.parametrize("axis", ["none", "x"])
def test_prepared_capacity_fixture_matches_jax(axis):
    from scripts.benchmark_compile_capacity import prepare

    args = Namespace(side=48, steps=48, axis=axis, workload="prepared", backend="jax")
    device = jax.devices("gpu")[0]
    program, state, coefficients = prepare(args, device)
    reference = build_scan(program)(state, coefficients)
    args.backend = "cuda_streamed"
    program, state, coefficients = prepare(args, device)
    program = replace(
        program, config=replace(program.config, cuda_memory_policy="capacity")
    )
    input_snapshot = jax.tree.map(np.array, state)
    actual = build_scan(program)(state, coefficients)
    for before, after in zip(
        jax.tree.leaves(input_snapshot), jax.tree.leaves(state), strict=True
    ):
        np.testing.assert_array_equal(before, after)
    # The sharp pulse accumulates cancellation error in CPML derivatives: at
    # 48 steps the measured FP32 discrepancy is 2.3 ppm of a slab's peak.
    _assert_state_close(reference, actual, dynamic_atol_scale=3e-6)
    # Also check physical fields against each triplet's scale, avoiding the
    # helper's absolute floor for this deliberately small-amplitude fixture.
    for names in (("ex", "ey", "ez"), ("hx", "hy", "hz")):
        scale = max(float(np.max(np.abs(getattr(reference, name)))) for name in names)
        for name in names:
            np.testing.assert_allclose(
                getattr(actual, name),
                getattr(reference, name),
                rtol=0,
                atol=2e-6 * scale,
            )
