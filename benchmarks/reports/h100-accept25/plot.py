"""Plot measured public preparation RAM and per-device allocation stages."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
raw = ROOT / "raw"
start = datetime.fromisoformat("2026-09-27T10:09:13.245776+00:00")
points = []
for line in (raw / "host-telemetry.jsonl").read_text().splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    for process in row["processes"]:
        if "public-eight-z-15b-fixed.json" in process["command"]:
            points.append(
                (
                    (datetime.fromisoformat(row["time"]) - start).total_seconds() / 60,
                    int(process["memory"]["VmRSS"].split()[0]) / 2**20,
                    int(process["memory"]["VmHWM"].split()[0]) / 2**20,
                )
            )
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
if points:
    t, rss, peak = zip(*points, strict=True)
    axes[0].plot(t, rss, label="Resident RAM")
    axes[0].plot(t, peak, ls="--", label="Process peak")
    axes[0].axhline(2012999999488 / 2**30, color="gray", ls=":", label="Pod RAM limit")
axes[0].set(
    xlabel="Minutes since public workload started",
    ylabel="Host RAM (GiB)",
    title="15B public preparation (timed out)",
)
axes[0].legend(fontsize=8)
p = raw / "public-eight-z-15b-fixed.json"
kind = "Public"
if not p.exists():
    p = raw / "prepared-eight-z-15b-fixed.json"
    kind = "Prepared"
if p.exists():
    d = json.loads(p.read_text())
    stages = d["stages"]
    for device in range(8):
        axes[1].plot(
            range(len(stages)),
            [s["gpus"][device]["stats"]["bytes_in_use"] / 2**30 for s in stages],
            marker="o",
            label=f"GPU {device}",
        )
    if stages:
        axes[1].axhline(
            stages[-1]["gpus"][0]["stats"]["bytes_limit"] / 2**30,
            color="gray",
            ls=":",
            label="Allocator limit",
        )
        axes[1].set_xticks(
            range(len(stages)), [s["stage"] for s in stages], rotation=35
        )
    axes[1].set_title(f"15B {kind.lower()} CUDA: {d['status']}")
axes[1].set(ylabel="Live JAX allocations per GPU (GiB)")
axes[1].legend(fontsize=7, ncol=3)
for ax in axes:
    ax.grid(alpha=0.2)
fig.savefig(ROOT / "memory.png", dpi=160)
