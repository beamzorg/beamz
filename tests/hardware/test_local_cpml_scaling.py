"""Large enough supports to exercise both distributed bulk tiles and CPML faces."""

import jax
import pytest

from beamz.simulation.backend import cuda_backend_status
from tests.performance.h100_workloads import H100Workload
from tests.unit.test_cuda_sharded_features import assert_state_close
from tests.unit.test_cuda_sharding import seed_state


@pytest.mark.parametrize("backend", ["cuda_streamed", "jax"])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_distributed_bulk_and_cpml_continuation(backend, axis):
    if not cuda_backend_status().available or len(jax.devices()) < 2:
        pytest.skip("requires two CUDA GPUs and the native component")
    sim = H100Workload(
        name="bulk_cpml",
        shape_zyx=(96, 112, 256),
        timesteps=32,
        resolution=80e-9,
        pml_cells=12,
        cpml=True,
        heterogeneous=True,
        source=True,
        monitor=True,
    ).build()
    state = seed_state(sim)
    reference = sim.advance(
        state=state, num_steps=32, backend="jax", progress=False
    ).state
    cfg = dict(axis=axis, num_devices=2, backend="gpu")
    actual = sim.advance(
        state=state, num_steps=32, backend=backend, sharding=cfg, progress=False
    ).state
    assert_state_close(reference, actual)
    for name in ("ex", "ey", "ez", "hx", "hy", "hz"):
        field = getattr(actual, name)
        assert not field.is_fully_replicated
        assert sum(shard.data.size for shard in field.addressable_shards) == field.size
    first = sim.advance(
        state=state, num_steps=16, backend=backend, sharding=cfg, progress=False
    ).state
    continued = sim.advance(
        state=first, num_steps=16, backend=backend, sharding=cfg, progress=False
    ).state
    assert_state_close(actual, continued)
