#!/usr/bin/env python3
"""Isolate the distributed CUDA local update on ONE H100; not a scaling result.

The diagnostic overrides only device selection to retain the distributed layout
on a one-device mesh. There is no inter-device communication. Compare identical
shapes/axes/builds to test native local-kernel changes independently of NVLink.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np

from beamz.simulation import sharding
from beamz.simulation.execute import build_scan, runtime_inputs
from tests.performance.h100_workloads import H100Workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", nargs=3, type=int, default=(128, 256, 96))
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x")
    args = parser.parse_args()
    devices = tuple(jax.devices())
    assert len(devices) == 1 and "H100" in devices[0].device_kind
    steps = 16
    sim = H100Workload(
        name="sharded_local",
        shape_zyx=tuple(args.shape),
        timesteps=steps,
        cpml=True,
        heterogeneous=True,
    ).build()
    state = sim.initial_state()
    rng = np.random.default_rng(123)
    state = state._replace(
        **{
            name: rng.normal(size=getattr(state, name).shape).astype(np.float32) * 1e-6
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        }
    )
    reference = sim.advance(state=state, num_steps=steps, backend="jax").state
    with patch.object(sharding, "_jax_devices_for_config", return_value=devices):
        program = sim.compile(
            num_steps=steps,
            backend="cuda_streamed",
            sharding=dict(enabled=True, axis=args.axis, num_devices=1, backend="gpu"),
        )
    assert program.sharding.layout.enabled and program.sharding.layout.num_devices == 1
    prepared = sharding.prepare_state(
        program,
        runtime_inputs(program, state, monitor_steps=steps),
        replicated_fields=(),
    )
    coefficients = sharding.place_tree(program, program.coefficients)
    start = time.perf_counter()
    executable = build_scan(program).lower(prepared, coefficients).compile()
    compile_s = time.perf_counter() - start
    actual = jax.block_until_ready(executable(prepared, coefficients))
    for expected, observed in zip(
        jax.tree.leaves(reference),
        jax.tree.leaves(sharding.crop_state(program, actual)),
        strict=True,
    ):
        expected, observed = np.asarray(expected), np.asarray(observed)
        scale = float(np.max(np.abs(expected), initial=0))
        np.testing.assert_allclose(
            observed, expected, rtol=3e-5, atol=max(1e-12, 1e-6 * scale)
        )
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        actual = jax.block_until_ready(executable(prepared, coefficients))
        samples.append(time.perf_counter() - start)
    import beamz._cuda as native

    result = dict(
        diagnostic="single_device_distributed_local_update_no_communication",
        shape=args.shape,
        axis=args.axis,
        steps=steps,
        samples_s=samples,
        gcups=math.prod(args.shape) * steps / statistics.median(samples) / 1e9,
        compile_s=compile_s,
        complete_state_parity=True,
        native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
