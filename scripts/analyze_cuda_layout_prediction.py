"""Replay layout predictions against saved timings; never executes GPU work.

Run with PYTHONPATH=. JAX_PLATFORMS=cpu python scripts/analyze_cuda_layout_prediction.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from beamz.simulation.cuda.tuning import predict_layout


def analyze(root):
    reviews = root / "docs/reviews"
    records = []
    paths = sorted(
        path
        for path in (reviews / "rtx3090-2026-09-18-cyclic-storage").glob("*.json")
        if not path.name.endswith(".telemetry.json")
    )
    paths += [
        reviews / "rtx3090-2026-09-18-autotune" / f"{case}.json"
        for case in ("narrow", "wide", "irregular")
    ]
    for directory in (
        "rtx3090-2026-09-18-origin-main-auto-comparison",
        "rtx3090-2026-09-18-origin-main-auto-long",
    ):
        # Second-process cache hits repeat the same calibration samples.
        paths += sorted((reviews / directory).glob("*branch_auto-1.json"))
    for path in paths:
        data = json.loads(path.read_text())
        shape = data.get("shape") or data["workload"]["shape"]
        if "tuning" in data:
            samples = data["tuning"]["samples_s"]
        else:
            samples = {
                key.removeprefix("one_step_") + "/64x4": values
                for key, values in data["samples_s"].items()
                if key.startswith("one_step_")
            }
        timings = {key: statistics.median(values) for key, values in samples.items()}
        prediction = predict_layout(shape)
        choice = prediction["choice"]
        selected = "".join(map(str, choice["axes"])) + "/" + choice["shell_tile"]
        best = min(timings, key=timings.get)
        records.append(
            dict(
                source=str(path.relative_to(root)),
                shape=shape,
                selected=selected,
                measured_fastest=best,
                selected_time_s=timings[selected],
                slowdown_from_fastest_pct=100 * (timings[selected] / timings[best] - 1),
                throughput_gain_over_canonical_pct=100
                * (timings["012/64x4"] / timings[selected] - 1),
                prediction=prediction,
            )
        )
    return dict(
        note="Retrospective replay, not independent held-out validation or new timings",
        profiles=len(records),
        distinct_shapes=len({tuple(r["shape"]) for r in records}),
        worst_slowdown_from_fastest_pct=max(
            r["slowdown_from_fastest_pct"] for r in records
        ),
        records=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(Path(__file__).resolve().parents[1])
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["records"]:
        print(
            f"{row['source']}: {row['selected']}, "
            f"gain {row['throughput_gain_over_canonical_pct']:.2f}%, "
            f"gap to fastest {row['slowdown_from_fastest_pct']:.2f}%"
        )
    print(
        f"{result['profiles']} profiles, {result['distinct_shapes']} distinct shapes; "
        f"worst gap {result['worst_slowdown_from_fastest_pct']:.3f}%"
    )


if __name__ == "__main__":
    main()
