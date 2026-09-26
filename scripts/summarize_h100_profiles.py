#!/usr/bin/env python3
"""Summarize GPU event durations, not wall-time attribution, from JAX traces."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def category(name):
    if "nccl" in name.lower():
        return "nccl"
    if "UpdateTiled" in name:
        return "hopper_update"
    if "UpdateSharded" in name:
        return "sharded_update"
    if "UpdateCombinedCpmlQueue" in name:
        return "streamed_update"
    if any(
        token in name for token in ("AccumulateDft", "PrepareDftPhases", "GatherDft")
    ):
        return "native_dft"
    if "ApplySingleSourceGroup" in name:
        return "native_sources"
    if "memcpy" in name.lower():
        return "copies"
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    records = []
    for path in sorted(args.directory.rglob("*.trace.json.gz")):
        with gzip.open(path, "rt") as handle:
            events = json.load(handle)["traceEvents"]
        gpu_pids = {
            event["pid"]
            for event in events
            if event.get("name") == "process_name"
            and "GPU" in event.get("args", {}).get("name", "")
        }
        events = [
            event
            for event in events
            if event.get("ph") == "X" and event.get("pid") in gpu_pids
        ]
        if not events:
            raise RuntimeError(f"No GPU events in {path}")
        totals = defaultdict(float)
        kernels = defaultdict(lambda: {"count": 0, "duration_ms": 0.0})
        for event in events:
            name, duration = event["name"], event["dur"] / 1000
            totals[category(name)] += duration
            kernels[name]["count"] += 1
            kernels[name]["duration_ms"] += duration
        total = sum(totals.values())
        records.append(
            {
                "trace": str(path.relative_to(args.directory)),
                "gpu_count": len(gpu_pids),
                "gpu_events": len(events),
                "summed_gpu_event_ms": total,
                "category_ms": dict(totals),
                "category_percent": {k: 100 * v / total for k, v in totals.items()},
                "top_kernels": sorted(
                    ({"name": name, **data} for name, data in kernels.items()),
                    key=lambda row: row["duration_ms"],
                    reverse=True,
                )[:12],
            }
        )
    if not records:
        raise SystemExit("No traces found")
    output = args.directory / "summary.json"
    output.write_text(json.dumps(records, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
