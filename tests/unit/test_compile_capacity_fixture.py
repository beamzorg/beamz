"""The capacity fixture must use true partitioned arrays and crossing pulses."""

import os
import subprocess
import sys


def test_rectangular_two_rank_fixture_matches_jax():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from argparse import Namespace
import jax
import numpy as np
from scripts.benchmark_compile_capacity import prepare
from beamz.simulation.execute import build_scan
from tests.unit.test_cuda_sharded_features import native_cpu_backend
from tests.hardware.test_cuda_backends import _assert_state_close

for axis in ('x', 'z'):
    args = Namespace(side=None, shape=(40,42,46), devices=2, steps=16,
                     axis=axis, workload='prepared', backend='jax')
    with native_cpu_backend():
        p, state, coefficients = prepare(args, jax.devices()[0])
        assert p.sharding.layout.num_devices == 2
        assert len(state.ex.addressable_shards) == 2
        assert not state.ex.sharding.is_fully_replicated
        assert all(s.data.shape[p.sharding.layout.axis] * 2 == state.ex.shape[p.sharding.layout.axis]
                   for s in state.ex.addressable_shards)
        assert all(np.max(np.abs(s.data)) > 0 for s in state.ex.addressable_shards)
        reference = build_scan(p)(state, coefficients)
        args.backend = 'cuda_streamed'
        p, state, coefficients = prepare(args, jax.devices()[0])
        actual = build_scan(p, donate_state=True)(state, coefficients)
        _assert_state_close(reference, actual, dynamic_atol_scale=3e-6)
        # Match the existing prepared-fixture hardware gate, including the
        # stricter field-triplet bound near cancellation zeros.
        for names in (('ex', 'ey', 'ez'), ('hx', 'hy', 'hz')):
            scale = max(float(np.max(np.abs(getattr(reference, n)))) for n in names)
            for name in names:
                np.testing.assert_allclose(
                    np.asarray(getattr(actual, name)),
                    np.asarray(getattr(reference, name)),
                    rtol=0, atol=2e-6 * scale)
        assert int(actual.current_step) == 16
        assert p.coefficients.e_source_x.shape == coefficients.e_source_x.shape
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
