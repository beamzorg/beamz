#!/usr/bin/env python3
"""Measure real modal preparation through final state/coefficient placement.

Run each case in a fresh process. On CPU, set XLA_FLAGS before starting Python
(e.g. --xla_force_host_platform_device_count=2). This measures preparation only;
it deliberately makes no stepping-throughput or executable-compilation claim.
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import jax

from beamz.simulation.execute import initial_program_state
from beamz.simulation.observe import MONITOR_FIELDS
from beamz.simulation.sharding import place_tree, prepare_state
from scripts.benchmark_cuda_realistic import build_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=3, required=True)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--backend", choices=("jax", "cuda_streamed"), default="jax")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(jax.devices()) < args.devices:
        parser.error("Insufficient visible devices")
    args.steps, args.pml, args.monitors, args.frequencies = 32, 12, 2, 101
    args.material, args.source, args.monitor_type = "binary", "mode", "mode"
    started = time.perf_counter()
    sim = build_simulation(args)
    built = time.perf_counter()
    program = sim.compile(
        num_steps=args.steps,
        backend=args.backend,
        sharding=dict(
            axis=args.axis, num_devices=args.devices, backend=jax.devices()[0].platform
        ),
    )
    planned = time.perf_counter()
    state = initial_program_state(program, t=0, current_step=0)
    state = prepare_state(
        program, state, replicated_fields=(*MONITOR_FIELDS, "t", "current_step")
    )
    coefficients = place_tree(program, program.coefficients)
    jax.block_until_ready((state, coefficients))
    finished = time.perf_counter()
    result = dict(
        shape=args.shape,
        devices=args.devices,
        axis=args.axis,
        backend=args.backend,
        platform=jax.devices()[0].platform,
        pml_cells=12,
        mode_sources=1,
        mode_monitors=2,
        frequencies=101,
        construction_s=built - started,
        plan_s=planned - built,
        state_and_placement_s=finished - planned,
        total_s=finished - started,
        host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        device_memory=[
            device.memory_stats() for device in jax.devices()[: args.devices]
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
