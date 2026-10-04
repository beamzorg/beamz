#!/usr/bin/env python3
"""Export inspectable, source-backed scientific plots from cosine trial JSON."""

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
    rows = [json.loads(p.read_text()) for p in args.directory.glob("throughput-*.json")]
    good = sorted(
        [r for r in rows if r.get("status") == "ok"], key=lambda r: r["cells"]
    )
    if not good:
        raise SystemExit("No successful throughput measurements to plot")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    blue = "#2468a0"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    n = np.array([r["cells"] / 1e6 for r in good])
    throughput = np.array([r["gcups"] for r in good])
    ranges = np.array(
        [
            [
                r["cells"] * r["steps"] / max(r["runtime_samples_s"]) / 1e9,
                r["cells"] * r["steps"] / min(r["runtime_samples_s"]) / 1e9,
            ]
            for r in good
        ]
    )
    axes[0].errorbar(
        n,
        throughput,
        yerr=[throughput - ranges[:, 0], ranges[:, 1] - throughput],
        color=blue,
        marker="o",
        capsize=3,
        linewidth=1.5,
    )
    axes[0].set(ylabel="Billion cell updates / second", title="Warm solver throughput")
    axes[1].plot(
        n,
        [r["nvidia_smi_peak_bytes"] / 2**30 for r in good],
        "o-",
        color=blue,
        label="Total GPU memory (sampled)",
    )
    axes[1].plot(
        n,
        [r["memory_stats"].get("peak_bytes_in_use", 0) / 2**30 for r in good],
        "s--",
        color="#555555",
        label="JAX peak bytes in use",
    )
    axes[1].set(ylabel="Peak memory (GiB)", title="Memory during the complete trial")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[2].plot(
        n, [r["median_runtime_s"] * 1e3 / r["steps"] for r in good], "o-", color=blue
    )
    axes[2].set(ylabel="Milliseconds / timestep", title="Warm timestep latency")
    for ax in axes:
        ax.set_xlabel("Grid cells (millions)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    for r, cells, speed in zip(good, n, throughput, strict=True):
        axes[0].annotate(
            f"{r['resolution_nm']:g} nm",
            (cells, speed),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[0].margins(y=0.2)
    fig.suptitle(
        "BeamZ cosine crossing · single H100 SXM · CUDA streamed · FP32", fontsize=14
    )
    fig.supxlabel(
        "Fixed geometry, CPML thickness, mode source and 101-frequency flux monitors. "
        "Error bars: observed min–max across warm repetitions.",
        fontsize=9,
    )
    fig.savefig(args.directory / "performance.png", dpi=180)
    fig.savefig(args.directory / "performance.svg")
    plt.close(fig)
    full = [json.loads(p.read_text()) for p in args.directory.glob("full-*.json")]
    full = sorted(
        [r for r in full if r.get("status") == "ok"],
        key=lambda r: r["resolution_nm"],
        reverse=True,
    )
    if full:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        colors = ["#2468a0", "#b97619", "#7e5488", "#607c41"]
        for i, row in enumerate(full):
            for ax, key in zip(axes, ["transmission_db", "crosstalk_db"], strict=True):
                ax.plot(
                    np.linspace(1260, 1360, 101),
                    row[key],
                    color=colors[i % len(colors)],
                    label=f"{row['resolution_nm']:g} nm",
                )
        for ax, title in zip(
            axes, ["Transmission (dB)", "Crosstalk (dB)"], strict=True
        ):
            ax.set(xlabel="Wavelength (nm)", ylabel=title)
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
            ax.legend()
        fig.suptitle("Cosine crossing · measured spectra after 1 ps", fontsize=14)
        fig.savefig(args.directory / "spectra.png", dpi=180)
        fig.savefig(args.directory / "spectra.svg")


if __name__ == "__main__":
    main()
