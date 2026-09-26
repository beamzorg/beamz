#!/usr/bin/env python3
"""Summarize capacity evidence without treating reserved memory as live arrays."""

import argparse
import csv
import json
import statistics
from contextlib import suppress
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(source):
    rows = []
    for item in json.loads((source / "summary.json").read_text()):
        if item["status"] != "ok":
            continue
        name = item["name"]
        d = json.loads((source / f"{name}.json").read_text())
        samples = d["warm_runtime_samples_s"]
        shape = d["shape"]
        cells = int(np.prod(shape))
        device = [x["stats"] for x in d["device_memory"]]
        timed = [x["stats"] for x in d.get("timed_memory", d["device_memory"])]
        telemetry = []
        with (source / f"{name}.nvml.csv").open() as f:
            for line in csv.DictReader(f, skipinitialspace=True):
                with suppress(KeyError, ValueError):
                    telemetry.append(float(line["memory.used [MiB]"]))
        rows.append(
            dict(
                name=name,
                series=name.split("-")[0],
                shape="x".join(map(str, shape)),
                resolution_nm=d["resolution_nm"],
                total_cells=cells,
                cells_per_gpu=cells / 8,
                gcups=d["kernel_gcups"],
                sample_cv_pct=100
                * statistics.stdev(samples)
                / statistics.mean(samples),
                gcups_min=cells * d["steps"] / max(samples) / 1e9,
                gcups_max=cells * d["steps"] / min(samples) / 1e9,
                timestep_ms=1000 * statistics.median(samples) / d["steps"],
                peak_live_max_gib=max(x["peak_bytes_in_use"] for x in device) / 2**30,
                timed_live_max_gib=max(x["bytes_in_use"] for x in timed) / 2**30,
                peak_pool_max_gib=max(x.get("peak_pool_bytes", 0) for x in device)
                / 2**30,
                sampled_resident_max_gib=max(telemetry, default=0) / 1024,
                setup_s=d["setup_s"],
                compile_s=d["compile_s"],
                finite=d["final_state_finite"],
                monitor_weight_min=d["monitor_weight_min"],
            )
        )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = summarize(a.source)
    if not rows:
        raise SystemExit("No successful cases")
    with (a.output / "capacity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.output / "capacity.json").write_text(json.dumps(rows, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    labels = dict(
        balanced="Domain growth: cubic shards",
        planar="Domain growth: planar",
        refined="Resolution refinement",
    )
    colors = dict(balanced="#1769aa", planar="#d55e00", refined="#009e73")
    for series in labels:
        group = sorted(
            [r for r in rows if r["series"] == series], key=lambda r: r["cells_per_gpu"]
        )
        if not group:
            continue
        if series == "refined":
            # The 80 nm balanced-512 run is also the identical physical-domain
            # reference for the refinement series; no duplicate measurement.
            group = [r for r in rows if r["name"] == "balanced-512"] + group
        rate = np.array([r["gcups"] for r in group])
        error = np.array(
            [
                [r["gcups"] - r["gcups_min"] for r in group],
                [r["gcups_max"] - r["gcups"] for r in group],
            ]
        )
        for ax, key, scale in [
            (axes[0], "cells_per_gpu", 1e6),
            (axes[1], "peak_live_max_gib", 1),
        ]:
            ax.errorbar(
                [r[key] / scale for r in group],
                rate,
                yerr=error,
                marker="o",
                capsize=3,
                color=colors[series],
                label=labels[series],
            )
        axes[2].plot(
            [r["total_cells"] / 1e9 for r in group],
            [r["timestep_ms"] for r in group],
            marker="o",
            color=colors[series],
            label=labels[series],
        )
    axes[0].set(xlabel="Million physical cells per GPU", ylabel="Aggregate GCUPS")
    axes[1].set(
        xlabel="Maximum per-GPU peak live allocation (GiB)", ylabel="Aggregate GCUPS"
    )
    axes[2].set(
        xlabel="Billion physical cells, all GPUs",
        ylabel="Median time per complete timestep (ms)",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)
    for ax in axes[:2]:
        ax.set_ylim(0, 1.12 * max(r["gcups_max"] for r in rows))
    axes[2].set_ylim(0, 1.12 * max(r["timestep_ms"] for r in rows))
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "8× H100 SXM · streamed CUDA · CPML12 + mode source + 2×101-frequency monitors\nFive synchronized warm samples; whiskers show sample range",
        fontsize=12,
    )
    fig.savefig(a.output / "capacity.png", dpi=170)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
