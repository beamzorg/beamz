"""Complete-state correctness gate for the modal H100 comparison workload."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from beamz.simulation.backend import cuda_backend_status
from scripts.benchmark_cuda_realistic import build_simulation
from tests.hardware.test_cuda_backends import _assert_state_close, _copy_state

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(not STATUS.available, reason=STATUS.reason or "no CUDA")


@pytest.mark.parametrize("frequencies", [3, 101])
@pytest.mark.parametrize(
    "backend,count",
    [
        ("cuda_streamed", 1),
        ("cuda_streamed", 2),
        ("cuda_streamed", 4),
    ],
)
def test_modal_backend_complete_state(backend, count, frequencies):
    if len(jax.devices()) < count:
        pytest.skip("insufficient GPUs")
    sim = build_simulation(
        SimpleNamespace(
            shape=(37, 49, 65),
            steps=33,
            pml=12,
            monitors=2,
            frequencies=frequencies,
            material="binary",
            source="mode",
            monitor_type="mode",
        )
    )
    state = sim.initial_state()
    rng = np.random.default_rng(20260925)
    state = state._replace(
        **{
            name: rng.normal(size=getattr(state, name).shape).astype(np.float32) * 1e-5
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        }
    )
    reference = sim.advance(state=_copy_state(state), num_steps=33, backend="jax").state
    sharding = (
        None if count == 1 else {"axis": "x", "num_devices": count, "backend": "gpu"}
    )
    first = sim.advance(
        state=_copy_state(state), num_steps=17, backend=backend, sharding=sharding
    ).state
    actual = sim.advance(
        state=first, num_steps=16, backend=backend, sharding=sharding
    ).state
    _assert_state_close(reference, actual)
