#!/usr/bin/env python3
"""Locate field versus monitor divergence using independent fresh runs.

This is a diagnostic, not a throughput benchmark. Never relax the spectral gate
based on these aggregate errors; use compare_modal_scaling.py for acceptance.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
from benchmark_cuda_realistic import build_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=(40, 48, 96))
    parser.add_argument(
        "--steps", nargs="+", type=int, default=[1, 2, 8, 32, 128, 256, 512, 1024]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for steps in args.steps:
        states = []
        for backend in ("jax", "cuda_streamed"):
            sim = build_simulation(
                SimpleNamespace(
                    shape=args.shape,
                    steps=max(steps, 2),
                    pml=12,
                    monitors=2,
                    frequencies=101,
                    material="binary",
                    source="mode",
                    monitor_type="mode",
                )
            )
            run = sim.advance(
                num_steps=steps, backend=backend, progress=False, performance=False
            )
            states.append(jax.device_get(run.state))
        row = {"steps": steps, "arrays": {}}
        for name in (
            "ex",
            "ey",
            "ez",
            "hx",
            "hy",
            "hz",
            "cpml_psi_e_terms",
            "cpml_psi_h_terms",
            "dft_vec_re",
            "dft_vec_im",
            "dft_weight_sum",
            "t",
        ):
            ref = np.concatenate(
                [
                    np.asarray(x).ravel().astype(np.float64)
                    for x in jax.tree.leaves(getattr(states[0], name))
                ]
            )
            actual = np.concatenate(
                [
                    np.asarray(x).ravel().astype(np.float64)
                    for x in jax.tree.leaves(getattr(states[1], name))
                ]
            )
            delta = actual - ref
            row["arrays"][name] = dict(
                max_abs=float(np.max(np.abs(delta), initial=0)),
                relative_l2=float(
                    np.linalg.norm(delta) / max(np.linalg.norm(ref), 1e-300)
                ),
                exact=bool(np.array_equal(ref, actual)),
                reference_max=float(np.max(np.abs(ref), initial=0)),
            )
        records.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
