#!/usr/bin/env python3
"""Plot historical H100 records, including the retired backend; never execute it."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    records = []
    for subdir in ("sweep", "followup"):
        for path in (args.directory / subdir).glob("*.json"):
            data = json.loads(path.read_text())
            if (
                isinstance(data, dict)
                and data.get("schema_version") == "beamz.performance/v3"
            ):
                records.append(data)

    def find(workload, shape, backend, count):
        return next(
            (
                r
                for r in records
                if r["workload"] == workload
                and tuple(r["grid_dimensions"]) == shape
                and r["backend"] == backend
                and r["device_count"] == count
                and r.get("shard_axis", "auto") == "auto"
            ),
            None,
        )

    cases = [
        ("bare_3d", (128, 256, 384), "Bare\n128×256×384"),
        ("realistic_3d", (128, 256, 384), "CPML + DFT\n128×256×384"),
        ("realistic_3d", (256, 256, 256), "CPML + DFT\n256³"),
        ("realistic_3d", (96, 256, 768), "CPML + DFT\n96×256×768"),
        ("modal_cpml12", (128, 256, 384), "Modal waveguide\n128×256×384"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    colors = {"jax": "#546E7A", "cuda_streamed": "#1565C0", "cuda_hopper": "#EF6C00"}
    x = np.arange(len(cases))
    for i, (backend, color) in enumerate(colors.items()):
        values = [find(w, s, backend, 1) for w, s, _ in cases]
        axes[0].bar(
            x + (i - 1) * 0.25,
            [r["kernel_gcups"] if r else np.nan for r in values],
            width=0.24,
            color=color,
            label=backend,
        )
    axes[0].set_xticks(x, [name for _, _, name in cases], fontsize=8)
    axes[0].set_title("Single H100: warm execution")
    axes[0].set_ylabel("GCUPS (higher is better)")
    axes[0].legend(fontsize=8)
    for shape, color in [((128, 256, 384), "#1565C0"), ((512, 512, 512), "#00897B")]:
        for backend, style in [("cuda_streamed", "-"), ("jax", "--")]:
            points = [(n, find("realistic_3d", shape, backend, n)) for n in (1, 2, 4)]
            points = [(n, r["kernel_gcups"]) for n, r in points if r]
            if points:
                label = f"{'×'.join(map(str, shape))} / {backend}"
                axes[1].plot(
                    *zip(*points, strict=True),
                    marker="o",
                    linestyle=style,
                    color=color,
                    label=label,
                )
    axes[1].set_title("CPML + DFT: fixed-domain scaling")
    axes[1].set_xlabel("H100 GPUs on one NVLink node")
    axes[1].set_ylabel("Aggregate GCUPS")
    axes[1].set_xticks([1, 2, 4])
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    fig.suptitle(
        "BeamZ 2f41186 + Hopper coefficient fix · H100 SXM 80 GB · FP32", fontsize=13
    )
    fig.savefig(args.directory / "throughput.png", dpi=180)
    fig.savefig(args.directory / "throughput.svg")


if __name__ == "__main__":
    main()
