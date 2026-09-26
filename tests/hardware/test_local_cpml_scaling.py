"""Large enough supports to exercise both distributed bulk tiles and CPML faces."""

import jax
import pytest

from beamz.simulation.backend import cuda_backend_status
from tests.performance.h100_workloads import H100Workload
from tests.unit.test_cuda_sharded_features import assert_state_close
from tests.unit.test_cuda_sharding import seed_state


@pytest.mark.parametrize("backend", ["cuda_streamed", "jax"])
def test_host_setup_fresh_state_stays_on_host_until_partitioned(backend):
    if not cuda_backend_status().available or len(jax.devices()) < 2:
        pytest.skip("requires two CUDA GPUs and the native component")
    from beamz.simulation.execute import initial_program_state
    from beamz.simulation.model import SimulationState

    cfg = dict(axis="x", num_devices=2, backend="gpu")
    host = jax.devices("cpu")[0]
    with jax.default_device(host):
        sim = H100Workload(
            name="host_initial",
            shape_zyx=(32, 40, 64),
            timesteps=8,
            resolution=80e-9,
            pml_cells=4,
            cpml=True,
            heterogeneous=True,
        ).build()
        program = sim.compile(backend=backend, sharding=cfg)
    for state in (
        SimulationState.initial(program.grid, t=0),
        initial_program_state(program, t=0, current_step=0),
    ):
        for name in ("ex", "ey", "ez", "hx", "hy", "hz"):
            assert getattr(state, name).devices() == {host}
    actual = sim.advance(backend=backend, sharding=cfg, performance=False).state
    reference = sim.advance(backend="jax", performance=False).state
    assert_state_close(reference, actual)


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
    from beamz.simulation import sharding
    from beamz.simulation.execute import runtime_inputs
    from beamz.simulation.observe import MONITOR_FIELDS

    program = sim.compile(num_steps=16, backend=backend, sharding=cfg)
    with jax.transfer_guard_device_to_host("disallow"):
        prepared = sharding.prepare_state(
            program,
            runtime_inputs(program, first, monitor_steps=16),
            replicated_fields=(*MONITOR_FIELDS, "t", "current_step"),
        )
        jax.block_until_ready(prepared)
    del prepared
    continued = sim.advance(
        state=first, num_steps=16, backend=backend, sharding=cfg, progress=False
    ).state
    assert_state_close(actual, continued)
