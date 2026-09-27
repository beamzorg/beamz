#!/usr/bin/env python3
"""Compare compilation and steady stepping in fresh, alternating GPU processes.

Example: python scripts/benchmark_memory_policy.py --baseline ../baseline \
    --output /tmp/memory-policy-comparison
Both checkouts must have the same native extension built. No GPU cases overlap.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for repeat in range(args.repeats):
        for backend in ("jax", "cuda_streamed"):
            for side, axis in ((128, "none"), (256, "none"), (128, "x"), (256, "x")):
                variants = ("baseline", "fixed")
                for variant in variants if repeat % 2 == 0 else variants[::-1]:
                    checkout = (
                        args.baseline.resolve() if variant == "baseline" else root
                    )
                    name = f"{backend}-{side}-{axis}-{variant}-{repeat}"
                    path = output / f"{name}.json"
                    environment = {
                        key: value
                        for key, value in os.environ.items()
                        if not key.startswith(("XLA_", "BEAMZ_CUDA_"))
                    }
                    environment.update(
                        PYTHONPATH=str(checkout),
                        XLA_PYTHON_CLIENT_PREALLOCATE="false",
                        XLA_PYTHON_CLIENT_MEM_FRACTION=".90",
                        BEAMZ_CUDA_AUTOTUNE="off",
                        BEAMZ_ENABLE_JAX_PERSISTENT_CACHE="0",
                    )
                    print(name, flush=True)
                    with path.with_suffix(".log").open("w") as log:
                        subprocess.run(
                            [
                                sys.executable,
                                str(root / "scripts/benchmark_compile_capacity.py"),
                                "--backend",
                                backend,
                                "--workload",
                                "modal",
                                "--side",
                                str(side),
                                "--axis",
                                axis,
                                "--steps",
                                "128",
                                "--samples",
                                "9",
                                "--output",
                                str(path),
                            ],
                            cwd=checkout,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=300,
                        )
                    result = json.loads(path.read_text())
                    records.append(
                        dict(
                            name=name,
                            warm_seconds=statistics.median(
                                stage["seconds"]
                                for stage in result["stages"]
                                if stage["stage"].startswith("run_")
                                and stage["stage"] != "run_0"
                            ),
                        )
                    )
    (output / "warm-medians.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
