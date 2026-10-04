#!/usr/bin/env python3
"""Compare FP32 arithmetic with FP32/BF16 CPML storage in JAX and CUDA.

BF16 JAX storage is injected explicitly for this experiment; no default changes.
Warm scan timings exclude compilation and modal postprocessing. Logical cells,
exactly 12 CPML cells, mode excitation and two compact mode monitors are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from beamz.simulation.execute import (
    _compiled_source_launch_powers,
    _decode_monitor_results,
    build_scan,
    initial_program_state,
)
from beamz.simulation.results import SimulationResults, SimulationRun
from scripts.benchmark_cuda_cpml_accuracy import error, snapshot
from scripts.benchmark_cuda_realistic import build_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=3, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--accuracy-steps", type=int, default=256)
    parser.add_argument("--material", choices=("binary", "smooth"), default="binary")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.accuracy_steps < args.steps or args.accuracy_steps % args.steps:
        parser.error("accuracy-steps must be a positive multiple of steps")
    for key, value in {
        "BEAMZ_CUDA_CPML_TEMPORAL": "0",
        "BEAMZ_CUDA_CPML_SPATIAL": "0",
        "BEAMZ_CUDA_TEMPORAL_STEPS": "1",
        "BEAMZ_CUDA_FIELD_PADDING": "none",
        "BEAMZ_CUDA_CPML_SHELL_TILE": "64x4",
    }.items():
        os.environ[key] = value
    os.environ.pop("BEAMZ_CUDA_CPML_CORE_FUSION", None)
    sim = build_simulation(
        SimpleNamespace(
            shape=args.shape,
            steps=args.accuracy_steps,
            pml=12,
            material=args.material,
            source="mode",
            monitors=2,
            frequencies=3,
            monitor_type="mode",
        )
    )
    variants, runners = {}, {}
    for backend in ("jax", "cuda_streamed"):
        for precision in ("fp32", "bf16"):
            name = f"{backend}_{precision}"
            print("Compile", name, flush=True)
            os.environ["BEAMZ_CUDA_CPML_PSI_PRECISION"] = precision
            start = time.perf_counter()
            program = sim.compile(num_steps=args.steps, backend=backend)
            state = initial_program_state(
                program, t=0, current_step=0, monitor_steps=args.accuracy_steps
            )
            dtype = jnp.bfloat16 if precision == "bf16" else jnp.float32
            state = state._replace(
                **{
                    key: tuple(x.astype(dtype) for x in getattr(state, key))
                    for key in ("cpml_psi_h_terms", "cpml_psi_e_terms")
                }
            )
            executable = (
                build_scan(program, donate_state=False)
                .lower(state, program.coefficients)
                .compile()
            )
            jax.block_until_ready(state)
            runners[name] = (program, state, executable)
            variants[name] = dict(
                compile_s=time.perf_counter() - start,
                samples_s=[],
                psi_dtype=str(state.cpml_psi_h_terms[0].dtype),
                field_dtype=str(state.ex.dtype),
                state_bytes=sum(
                    x.size * x.dtype.itemsize for x in jax.tree.leaves(state)
                ),
            )
    names = list(runners)
    orders = [names[i:] + names[:i] for i in range(4)]
    orders += [list(reversed(row)) for row in orders]
    random.Random(20260918).shuffle(orders)
    for _ in range(2):
        for name in names:
            program, state, executable = runners[name]
            jax.block_until_ready(executable(state, program.coefficients))
    for order in orders:
        for name in order:
            program, state, executable = runners[name]
            start = time.perf_counter()
            result = executable(state, program.coefficients)
            jax.block_until_ready(result)
            variants[name]["samples_s"].append(time.perf_counter() - start)
            del result
    for name in names:
        variants[name]["median_gcups"] = (
            np.prod(args.shape)
            * args.steps
            / np.median(variants[name]["samples_s"])
            / 1e9
        )
        print(name, variants[name]["median_gcups"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        shape=args.shape,
        material=args.material,
        steps=args.steps,
        accuracy_steps=args.accuracy_steps,
        cpml_cells=12,
        orders=orders,
        warmups=2,
        variants=variants,
        device=str(jax.devices()[0]),
        jax=jax.__version__,
        note="BF16 auxiliary storage only; FP32 fields, coefficients and arithmetic. "
        "Backward modal power is not isolated absorber reflectivity.",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    references = {}
    for name in names:
        print("Accuracy", name, flush=True)
        program, state, executable = runners[name]
        for _ in range(args.accuracy_steps // args.steps):
            state = executable(state, program.coefficients)
        jax.block_until_ready(state)
        results = SimulationResults.from_run(
            sim,
            runtime_fields=program.grid,
            monitor_results=_decode_monitor_results(sim, program, state),
            store_full_materials=False,
            source_launch_powers=_compiled_source_launch_powers(
                program, len(sim.sources)
            ),
            performance=None,
        )
        current = snapshot(SimulationRun(results=results, state=state))
        backend, precision = name.rsplit("_", 1)
        if precision == "fp32":
            references[backend] = current
        ref = references[backend]
        variants[name]["errors_vs_backend_fp32"] = {
            key: error(ref[key], val) for key, val in current.items()
        }
        variants[name]["mode_power_errors_vs_backend_fp32"] = {
            f"mode_{m}_{direction}": error(
                np.abs(ref[f"mode_{m}"][:, d, :]) ** 2,
                np.abs(current[f"mode_{m}"][:, d, :]) ** 2,
            )
            for m in range(2)
            for d, direction in enumerate(("forward", "backward"))
        }
        if backend != "jax":
            variants[name]["errors_vs_jax_fp32"] = {
                key: error(references["jax"][key], val) for key, val in current.items()
            }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        del state, results, current
    import beamz._cuda as extension

    report["native_sha256"] = hashlib.sha256(
        Path(extension.__file__).read_bytes()
    ).hexdigest()
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted([*root.glob("beamz/**/*.py"), *root.glob("cuda/src/*")]):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    report["source_sha256"] = digest.hexdigest()
    report["benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
