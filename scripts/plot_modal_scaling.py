#!/usr/bin/env python3
"""Plot measured local-volume saturation; never extrapolate an eight-GPU result."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["weak"])
    args = parser.parse_args()
    with args.measurements.open() as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["variant"] in args.variants
        ]
    if not rows:
        raise SystemExit("No matching measurements")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, backend, target in zip(
        axes, ("cuda_streamed", "jax"), (150, 75), strict=True
    ):
        groups = sorted(
            {
                (int(r["gpus"]), int(r["frequencies"]))
                for r in rows
                if r["backend"] == backend
            }
        )
        for count, frequencies in groups:
            selected = sorted(
                (
                    r
                    for r in rows
                    if r["backend"] == backend
                    and int(r["gpus"]) == count
                    and int(r["frequencies"]) == frequencies
                ),
                key=lambda r: int(r["material_cells"]),
            )
            axis.plot(
                [int(r["material_cells"]) / count / 1e6 for r in selected],
                [float(r["kernel_gcups"]) / count for r in selected],
                marker="o" if count == 1 else "s",
                linestyle="-" if frequencies == 3 else "--",
                label=f"{count} GPU{'s' if count != 1 else ''}, {frequencies} frequencies",
            )
        axis.axhline(
            target / 8,
            color="0.45",
            linestyle=":",
            linewidth=1,
            label=f"8-GPU target ÷ 8 ({target / 8:g})",
        )
        axis.set(
            title=backend,
            xlabel="Physical cells per GPU (millions)",
            ylabel="Measured stepping GCUPS per GPU",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(
        "12-cell CPML · mode source · two mode monitors\nDotted line is a required rate, not an eight-GPU prediction",
        fontsize=11,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
