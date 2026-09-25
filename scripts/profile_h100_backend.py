#!/usr/bin/env python3
"""Capture one warmed canonical run; profiler timings are not benchmark samples."""

import argparse
from pathlib import Path

import jax

from beamz.simulation import observe
from beamz.simulation import sharding as placement
from beamz.simulation.execute import build_scan, initial_program_state
from tests.performance.h100_workloads import H100_WORKLOADS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "cuda_streamed"), required=True)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--workload", choices=("realistic_3d", "crossing36"), default="realistic_3d"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hlo", action="store_true", help="Save optimized HLO for kernel attribution"
    )
    args = parser.parse_args()
    shard = (
        None
        if args.devices == 1
        else {"axis": "x", "num_devices": args.devices, "backend": "gpu"}
    )
    if args.workload == "crossing36":
        from benchmark_cosine_h100 import crossing

        sim = crossing(36)
    else:
        sim = H100_WORKLOADS["realistic_3d"].resized(timesteps=32).build()
    program = sim.compile(num_steps=32, backend=args.backend, sharding=shard)
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=32)
    state = placement.prepare_state(
        program, state, replicated_fields=(*observe.MONITOR_FIELDS, "t", "current_step")
    )
    coefficients = placement.place_tree(program, program.coefficients)
    executable = (
        build_scan(program, donate_state=False).lower(state, coefficients).compile()
    )
    if args.hlo:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "optimized-hlo.txt").write_text(executable.as_text())
    for _ in range(3):
        jax.block_until_ready(executable(state, coefficients))
    with jax.profiler.trace(str(args.output), create_perfetto_trace=True):
        jax.block_until_ready(executable(state, coefficients))


if __name__ == "__main__":
    main()
