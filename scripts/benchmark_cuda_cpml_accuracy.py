#!/usr/bin/env python3
"""Long-pulse CPML precision/schedule comparison, including modal spectra.

This is an accuracy experiment, not a throughput benchmark. Backward modal
power includes source/discretization effects; it is not isolated PML reflectivity.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from scripts.benchmark_cuda_realistic import build_simulation


def error(reference, actual):
    reference, actual = np.asarray(reference), np.asarray(actual)
    assert np.all(np.isfinite(reference)) and np.all(np.isfinite(actual))
    delta = actual - reference
    peak = float(np.max(np.abs(reference), initial=0))
    return {
        "reference_peak": peak,
        "max_absolute_error": float(np.max(np.abs(delta), initial=0)),
        "relative_peak_error": float(np.max(np.abs(delta), initial=0))
        / max(peak, 1e-30),
        "relative_l2_error": float(np.linalg.norm(delta.ravel()))
        / max(float(np.linalg.norm(reference.ravel())), 1e-30),
    }


def snapshot(run):
    fields = {
        name: np.asarray(getattr(run.state, name)).astype(np.float32)
        for name in ("ex", "ey", "ez", "hx", "hy", "hz")
    }
    for family in ("h", "e"):
        for index, value in enumerate(getattr(run.state, f"cpml_psi_{family}_terms")):
            fields[f"psi_{family}_{index}"] = np.asarray(value).astype(np.float32)
    fields["dft"] = np.asarray(run.state.dft_vec_re) + 1j * np.asarray(
        run.state.dft_vec_im
    )
    for index in range(2):
        data = run.results.mode(f"mode_{index}")
        fields[f"mode_{index}"] = np.asarray(data.amps)
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=3, default=(128, 96, 256))
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sim = build_simulation(
        SimpleNamespace(
            shape=args.shape,
            steps=args.steps,
            pml=12,
            material="binary",
            source="mode",
            monitors=2,
            frequencies=3,
            monitor_type="mode",
        )
    )
    checkpoints = sorted({min(512, args.steps), min(2048, args.steps), args.steps})
    reference = {}
    report = {
        "shape": args.shape,
        "steps": args.steps,
        "cpml_cells": 12,
        "checkpoints": checkpoints,
        "variants": {},
        "note": "Backward modal power is not an isolated CPML reflection coefficient.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, precision, paired in (
        ("fp32_existing", "fp32", False),
        ("bf16_existing", "bf16", False),
        ("fp32_pair", "fp32", True),
        ("bf16_pair", "bf16", True),
    ):
        os.environ["BEAMZ_CUDA_CPML_PSI_PRECISION"] = precision
        os.environ["BEAMZ_CUDA_CPML_TEMPORAL"] = "1" if paired else "0"
        os.environ["BEAMZ_CUDA_TEMPORAL_STEPS"] = "2" if paired else "1"
        os.environ["BEAMZ_CUDA_CPML_PAIR_TILE"] = "oriented"
        os.environ.pop("BEAMZ_CUDA_CPML_CORE_FUSION", None)
        state = None
        previous = 0
        values = {}
        print(name, flush=True)
        for checkpoint in checkpoints:
            run = sim.advance(
                state=state,
                num_steps=checkpoint - previous,
                backend="cuda_streamed",
                performance=False,
            )
            jax.block_until_ready(run.state)
            current = snapshot(run)
            if name == "fp32_existing":
                reference[checkpoint] = current
            metrics = {
                key: error(reference[checkpoint][key], value)
                for key, value in current.items()
            }
            modes = {}
            for index in range(2):
                amplitudes = current[f"mode_{index}"]
                power = np.abs(amplitudes) ** 2
                ref_power = np.abs(reference[checkpoint][f"mode_{index}"]) ** 2
                modes[f"mode_{index}"] = {
                    "forward_power": power[:, 0, :].tolist(),
                    "backward_power": power[:, 1, :].tolist(),
                    "forward_power_error": error(ref_power[:, 0, :], power[:, 0, :]),
                    "backward_power_error": error(ref_power[:, 1, :], power[:, 1, :]),
                    "backward_over_forward": (
                        power[:, 1, :] / np.maximum(power[:, 0, :], 1e-30)
                    ).tolist(),
                }
            values[checkpoint] = {
                "errors": metrics,
                "modes": modes,
                "psi_dtype": str(run.state.cpml_psi_h_terms[0].dtype),
            }
            print(
                checkpoint,
                "DFT relative L2",
                metrics["dft"]["relative_l2_error"],
                flush=True,
            )
            state, previous = run.state, checkpoint
            del run, current
        report["variants"][name] = values
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        del state
        gc.collect()
    import beamz._cuda as extension

    report["native_sha256"] = hashlib.sha256(
        Path(extension.__file__).read_bytes()
    ).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
