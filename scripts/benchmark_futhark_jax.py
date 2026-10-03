#!/usr/bin/env python3
"""Compare the Futhark and JAX backends on the realistic mode/CPML/DFT workload.

Each (backend, shape) case runs in a fresh process. Setup and compilation are
excluded from the warm GCUPS figures and reported separately, matching
benchmark_cuda_realistic.py. Run with XLA_PYTHON_CLIENT_PREALLOCATE=false so
that Futhark's own device allocations are not starved by XLA's arena.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPTS = Path(__file__).resolve().parent


def run_case(backend: str, shape, steps: int, samples: int, warmups: int, args):
    import jax

    sys.path.insert(0, str(SCRIPTS))
    from benchmark_cuda_realistic import build_simulation

    from beamz.simulation.execute import build_scan, initial_program_state

    spec = SimpleNamespace(
        shape=shape,
        steps=steps,
        pml=args.pml,
        monitors=args.monitors,
        frequencies=args.frequencies,
        material="binary",
        source="mode",
        monitor_type="field",
    )
    start = time.perf_counter()
    sim = build_simulation(spec)
    program = sim.compile(num_steps=steps, backend=backend)
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=steps)
    jax.block_until_ready(state)
    setup_s = time.perf_counter() - start
    start = time.perf_counter()
    executable = (
        build_scan(program, donate_state=False)
        .lower(state, program.coefficients)
        .compile()
    )
    compile_s = time.perf_counter() - start
    # The first Futhark call also compiles its kernels through NVRTC.
    start = time.perf_counter()
    jax.block_until_ready(executable(state, program.coefficients))
    first_call_s = time.perf_counter() - start
    for _ in range(warmups):
        jax.block_until_ready(executable(state, program.coefficients))
    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        jax.block_until_ready(executable(state, program.coefficients))
        timings.append(time.perf_counter() - start)
    median = float(np.median(timings))
    return {
        "backend": backend,
        "device": str(jax.devices()[0].device_kind),
        "shape": list(shape),
        "cells": int(np.prod(shape)),
        "steps": steps,
        "setup_s": setup_s,
        "compile_s": compile_s,
        "first_call_s": first_call_s,
        "samples_s": timings,
        "median_s": median,
        "median_gcups": float(np.prod(shape) * steps / median / 1e9),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["64x96x128", "96x160x256", "128x256x512"],
        help="z x y x x grid shapes",
    )
    parser.add_argument("--backends", nargs="+", default=["jax", "futhark"])
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--pml", type=int, default=12)
    parser.add_argument("--monitors", type=int, default=2)
    parser.add_argument("--frequencies", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        backend, shape = args.case.split(":")
        result = run_case(
            backend,
            tuple(int(v) for v in shape.split("x")),
            args.steps,
            args.samples,
            args.warmups,
            args,
        )
        print("RESULT " + json.dumps(result), flush=True)
        return
    env = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", **os.environ}
    results = []
    for shape in args.shapes:
        for backend in args.backends:
            command = [sys.executable, __file__, "--case", f"{backend}:{shape}"]
            for name in (
                "steps",
                "samples",
                "warmups",
                "pml",
                "monitors",
                "frequencies",
            ):
                command += [f"--{name}", str(getattr(args, name))]
            done = subprocess.run(command, env=env, capture_output=True, text=True)
            lines = [
                line for line in done.stdout.splitlines() if line.startswith("RESULT ")
            ]
            if done.returncode or not lines:
                result = {
                    "backend": backend,
                    "shape": shape,
                    "error": (done.stderr or done.stdout).strip().splitlines()[-1:],
                }
            else:
                result = json.loads(lines[-1][len("RESULT ") :])
            results.append(result)
            print(
                json.dumps({k: v for k, v in result.items() if k != "samples_s"}),
                flush=True,
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
