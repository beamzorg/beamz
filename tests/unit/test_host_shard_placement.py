"""Host setup must not stage global arrays on the default execution device."""

import os
import subprocess
import sys


def test_host_shards_are_placed_directly_and_keep_values(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import jax
import numpy as np
from beamz.simulation import sharding

devices = np.array(jax.devices())
mesh = jax.sharding.Mesh(devices, ('fdtd',))
original = np.arange(5*7*12, dtype=np.float32).reshape(5,7,12)
with jax.default_device(devices[-1]):
    cpu_array = jax.numpy.asarray(original)
for axis in range(3):
    shape = list(original.shape)
    shape[axis] = ((shape[axis]+3)//4)*4
    padded = sharding._pad_high_to_shape(cpu_array, tuple(shape))
    assert isinstance(padded, np.ndarray)
    spec = [None]*3
    spec[axis] = 'fdtd'
    target = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))
    # device_put(global_array, sharding) is precisely the path being replaced.
    put = jax.device_put
    def reject_global(value, *args, **kwargs):
        assert getattr(value, 'shape', None) != padded.shape
        return put(value, *args, **kwargs)
    with patch('jax.device_put', side_effect=reject_global):
        result = sharding._place_array(padded, target)
        result.block_until_ready()
    np.testing.assert_array_equal(result, padded)
    assert len(result.addressable_shards) == 4
    assert sharding._place_array(result, target) is result
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    restored = sharding._place_array(result, replicated)
    np.testing.assert_array_equal(restored, padded)
program = SimpleNamespace(sharding=SimpleNamespace(mesh=mesh,
    layout=SimpleNamespace(enabled=True,axis=2,num_devices=4)))
placed = sharding.place_tree(program, {'coeff':cpu_array})
np.testing.assert_array_equal(placed['coeff'], original)
records = [json.loads(line) for line in Path(__import__('os').environ['BEAMZ_TRACE_PLACEMENT']).read_text().splitlines()]
assert records[-1]['path'] == "['coeff']"
assert records[-1]['shape'] == [5,7,12]
assert len(records[-1]['after']) == 4
""",
        ],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=4",
            BEAMZ_TRACE_PLACEMENT=str(tmp_path / "placement.jsonl"),
        ),
        check=True,
        timeout=120,
    )


def test_donating_native_continuation_preserves_modal_state():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from tests.unit.test_cuda_sharded_features import (
    native_cpu_backend, mode_simulation, assert_state_close,
)
from tests.unit.test_cuda_sharding import seed_state
with native_cpu_backend():
    sim = mode_simulation()
    cfg = dict(axis='x', num_devices=2, backend='cpu')
    expected = sim.advance(num_steps=8, state=seed_state(sim),
        backend='cuda_streamed', sharding=cfg).state
    first = sim.advance(num_steps=4, state=seed_state(sim),
        backend='cuda_streamed', sharding=cfg, donate_state=True).state
    actual = sim.advance(num_steps=4, state=first,
        backend='cuda_streamed', sharding=cfg, donate_state=True).state
    assert_state_close(expected, actual)
    assert int(actual.current_step) == 8
""",
        ],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=2",
        ),
        check=True,
        timeout=180,
    )
