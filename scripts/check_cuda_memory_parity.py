"""Check whole-process memory and elapsed-time budgets on matched notebook runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def trial(directory):
    results = json.loads((directory / "results.json").read_text())
    execution = json.loads((directory / "execution.json").read_text())
    cold_end = execution["original_cell_count"] + 1
    return dict(
        notebook_sha256=execution["notebook_sha256"],
        grid_shape=results["grid_shape"],
        steps=results["steps"],
        frequencies=results["frequencies"],
        metric_kind=results["metric_kind"],
        host_peak_bytes=results["memory"]["host_maxrss_bytes"],
        gpu_peak_bytes=max(
            row["peak_gpu_process_bytes"]
            for row in results["memory"]["phases"].values()
        ),
        warm_engine_s=statistics.median(row["engine_s"] for row in results["runs"][1:]),
        notebook_s=next(
            row["start_s"]
            for row in execution["cell_starts"]
            if row["index"] == cold_end
        ),
        temporal_active=results.get("temporal_active", False),
    )


def compare(reference, candidate):
    checks = {
        "same_" + key: reference[key] == candidate[key]
        for key in (
            "notebook_sha256",
            "grid_shape",
            "steps",
            "frequencies",
            "metric_kind",
        )
    }
    checks.update(
        host_at_or_below_main=candidate["host_peak_bytes"]
        <= reference["host_peak_bytes"],
        gpu_at_or_below_main=candidate["gpu_peak_bytes"] <= reference["gpu_peak_bytes"],
        warm_engine_faster=candidate["warm_engine_s"] < reference["warm_engine_s"],
        notebook_faster=candidate["notebook_s"] < reference["notebook_s"],
        temporal_active=candidate["temporal_active"],
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        reference=reference,
        candidate=candidate,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(trial(args.reference), trial(args.candidate))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=result["passed"],
                failed=[k for k, v in result["checks"].items() if not v],
            )
        )
    )
    raise SystemExit(0 if result["passed"] else 1)
