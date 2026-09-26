#!/usr/bin/env python3
"""Compare propagated modal artifacts with the existing full-state tolerances."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.reference.read_text())
    reference = np.load(args.reference.with_suffix(".npz"))
    rows = []
    for candidate in args.candidates:
        info = json.loads(candidate.read_text())
        for key in ("shape", "steps", "frequencies", "cpml_cells"):
            if info[key] != metadata[key]:
                raise ValueError(f"Mismatched {key}: {candidate}")
        for key in ("workload", "resolution_m", "chunk", "script_sha256"):
            if info.get(key) != metadata.get(key):
                raise ValueError(f"Mismatched {key}: {candidate}")
        if "converged" in metadata and not (
            metadata["converged"] and info.get("converged", False)
        ):
            raise ValueError(
                f"Completion comparison requires converged runs: {candidate}"
            )
        if not info["finite_state"] or not metadata["finite_state"]:
            raise ValueError("Non-finite simulation state")
        actual = np.load(candidate.with_suffix(".npz"))
        if set(reference.files) != set(actual.files):
            raise ValueError(f"Mismatched arrays: {candidate}")
        errors, passed = {}, True
        for name in reference.files:
            ref = reference[name].astype(np.float64)
            value = actual[name].astype(np.float64)
            if ref.shape != value.shape:
                raise ValueError(f"Mismatched shape for {name}: {candidate}")
            scale = float(np.max(np.abs(ref), initial=0))
            close = bool(
                np.allclose(value, ref, rtol=3e-5, atol=max(1e-12, 2e-6 * scale))
            )
            passed &= close
            errors[name] = {
                "passed": close,
                "relative_l2": float(
                    np.linalg.norm(value - ref) / max(np.linalg.norm(ref), 1e-300)
                ),
                "max_abs": float(np.max(np.abs(value - ref), initial=0)),
            }
        rows.append(
            {
                "candidate": candidate.name,
                "backend": info["backend"],
                "devices": info["devices"],
                "passed": passed,
                "arrays": errors,
            }
        )
    result = {
        "reference": args.reference.name,
        "rtol": 3e-5,
        "atol": "max(1e-12, 2e-6 * max(abs(reference array)))",
        "comparisons": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not all(row["passed"] for row in rows):
        raise SystemExit("Modal comparison failed; see output")
    print(f"All {len(rows)} modal comparisons passed")


if __name__ == "__main__":
    main()
