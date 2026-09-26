#!/usr/bin/env python3
"""Read historical backend records (including retired backends); run no solver."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    records = []
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text())
        if (
            isinstance(data, dict)
            and data.get("schema_version") == "beamz.performance/v3"
        ):
            data["file"] = path.name
            records.append(data)
    if not records:
        raise SystemExit("No completed benchmark records found")
    grouped = {}
    for record in records:
        key = (
            record["workload"],
            tuple(record["grid_dimensions"]),
            record.get("shard_axis", "auto"),
        )
        grouped.setdefault(key, {})[(record["backend"], record["device_count"])] = (
            record
        )
    rows = []
    for (workload, shape, axis), group in sorted(grouped.items()):
        for (backend, count), record in sorted(group.items()):
            base = group.get((backend, 1))
            samples = record["warm_runtime_samples_s"]
            speedup = (
                base["median_warm_runtime_s"] / record["median_warm_runtime_s"]
                if base
                else None
            )
            rows.append(
                {
                    "workload": workload,
                    "shape": "x".join(map(str, shape)),
                    "axis": axis,
                    "backend": backend,
                    "gpus": count,
                    "gcups": record["kernel_gcups"],
                    "public_gcups": record["end_to_end_gcups"],
                    "median_s": record["median_warm_runtime_s"],
                    "sample_min_s": min(samples),
                    "sample_max_s": max(samples),
                    "speedup_vs_same_backend_1gpu": speedup,
                    "parallel_efficiency": speedup / count if speedup else None,
                    "compile_s": record["compile_s"],
                    "aggregate_peak_gib": record["peak_memory_bytes"] / 2**30,
                    "file": record["file"],
                }
            )
    with (root / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# H100 backend pilot measurements",
        "",
        "These are fixed-step performance workloads, not optical convergence results.",
        "Complete-state hardware parity is reported separately in the parent directory.",
        "",
        "| Workload / grid (z,y,x) | Backend | GPUs | Warm GCUPS | Public-path GCUPS | Speedup vs own 1 GPU |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        speedup = row["speedup_vs_same_backend_1gpu"]
        lines.append(
            f"| {row['workload']} / {row['shape']} / {row['axis']} | {row['backend']} | {row['gpus']} | "
            f"{row['gcups']:.3f} | {row['public_gcups']:.3f} | "
            + (f"{speedup:.3f}x |" if speedup else "unmeasured |")
        )
    lines += [
        "",
        "## Single-GPU Hopper versus streamed",
        "",
        "Ratios above 1 mean Hopper is faster. Raw samples remain in each JSON record.",
        "",
    ]
    for (workload, shape, axis), group in sorted(grouped.items()):
        hopper, streamed = (
            group.get(("cuda_hopper", 1)),
            group.get(("cuda_streamed", 1)),
        )
        if hopper and streamed:
            ratio = streamed["median_warm_runtime_s"] / hopper["median_warm_runtime_s"]
            lines.append(f"- {workload} / {shape} / {axis}: {ratio:.3f}x")
    lines += [
        "",
        "## Measurement limits",
        "",
        "- Fresh process per configuration; five synchronized warm samples; explicit FP32 CPML.",
        "- The inherited canonical harness preserves input state (no donation).",
        "- Public-path timing includes allocation and placement, but excludes cold geometry/mode setup.",
        "- Peak memory includes preparation; aggregate peaks are not a simultaneous sampled peak.",
        "- Single-GPU and sharded streamed execution use different schedules.",
        "- No 8-GPU result is inferred from 4-GPU measurements.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(root / "REPORT.md")


if __name__ == "__main__":
    main()
