"""Real multi-GPU continuation and boundary/source/monitor parity gates."""

from argparse import Namespace

import jax
import numpy as np
import pytest

from beamz.simulation.backend import cuda_backend_status
from scripts.benchmark_cuda_realistic import build_simulation
from tests.hardware.test_cuda_backends import _assert_state_close, _copy_state


@pytest.mark.parametrize("axis", ["x", "z"])
@pytest.mark.parametrize("backend", ["jax", "cuda_streamed"])
@pytest.mark.parametrize("policy", ["auto", "capacity"])
def test_multi_gpu_continuation_matches_unsharded_jax(
    axis, backend, policy, monkeypatch
):
    count = len(jax.devices())
    if not cuda_backend_status().available or count not in (2, 8):
        pytest.skip("requires two or eight visible CUDA GPUs")
    monkeypatch.setenv("BEAMZ_CUDA_MEMORY_POLICY", policy)
    sim = build_simulation(
        Namespace(
            # Keep the boundary slab narrower than each local partition.
            shape=(37, 41, 49) if count == 2 else (137, 41, 145),
            steps=48,
            pml=12,
            monitors=2,
            frequencies=101,
            material="binary",
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
    cfg = dict(axis=axis, num_devices=count, backend="gpu")
    first = sim.advance(
        state=_copy_state(state),
        num_steps=17,
        backend=backend,
        sharding=cfg,
        donate_state=True,
        performance=False,
    ).state
    actual = sim.advance(
        state=first,
        num_steps=31,
        backend=backend,
        sharding=cfg,
        donate_state=True,
        performance=False,
    ).state
    jax.block_until_ready(actual)
    # Use the existing prepared-fixture bound for cancellation in CPML
    # derivatives, while keeping the original bound for every other leaf.
    for name in reference._fields:
        _assert_state_close(
            getattr(reference, name),
            getattr(actual, name),
            dynamic_atol_scale=3e-6 if name.startswith("cpml_psi_") else 1e-6,
        )
    # As in test_prepared_capacity_fixture_matches_jax, physical fields must
    # also satisfy the stricter triplet bound without an absolute 3e-6 floor.
    for names in (("ex", "ey", "ez"), ("hx", "hy", "hz")):
        scale = max(float(np.max(np.abs(getattr(reference, name)))) for name in names)
        for name in names:
            np.testing.assert_allclose(
                np.asarray(getattr(actual, name)),
                np.asarray(getattr(reference, name)),
                rtol=0,
                atol=2e-6 * scale,
            )
    assert int(actual.current_step) == 48
    assert np.all(np.asarray(actual.dft_weight_sum) == 48)
    assert len(actual.ex.sharding.device_set) == count
    # The public API restores canonical, unpadded shapes. If no logical axis
    # divides the mesh size, crop_component currently falls back to replication.
    # Record that limitation rather than confusing it with numerical failure.
    if any(size % count == 0 for size in actual.ex.shape):
        assert not actual.ex.sharding.is_fully_replicated
