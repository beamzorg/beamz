"""Reproduce diagnostic figures from the committed run summaries."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = ["#236192", "#d97722", "#7a629e", "#6b7740"]


def main():
    runs = json.loads((HERE / "runs.json").read_text())
    controls = [
        r for r in runs if r["device"].startswith("straight_") and r["backend"] == "jax"
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8), constrained_layout=True, sharex=True, sharey=True
    )
    for ax, row in zip(axes.flat, controls, strict=False):
        wl = np.asarray(row["wavelengths_um"]) * 1000
        order = np.argsort(wl)
        for idx, (port, power) in enumerate(row["powers"].items()):
            ax.plot(
                wl[order],
                np.asarray(power)[order],
                marker=["o", "s", "^"][idx],
                color=COLORS[idx],
                label=port,
                ms=4,
            )
        ax.axhline(1, color="#444444", ls="--", lw=1)
        ax.axhspan(0.99, 1.01, color="#236192", alpha=0.07)
        ax.set(
            title=row["device"].removeprefix("straight_"),
            xlabel="Wavelength (nm)",
            ylabel="Selected modal transmission",
            ylim=(0.985, 1.015),
        )
        ax.text(
            0.03,
            0.04,
            f"Max |T−1|: {100 * row['max_transmission_error']:.2f}%\nMax reflection: {100 * row['reflection_max']:.2f}%",
            transform=ax.transAxes,
        )
        ax.grid(alpha=0.2)
    axes.flat[0].legend(loc="upper left")
    fig.suptitle(
        "3D straight-guide normalization controls · 6 PPW\nThree output planes; shaded region is the proposed ±1% control target"
    )
    fig.savefig(HERE / "straight_controls.png", dpi=170)
    plt.close(fig)
    exact_controls = [r for r in runs if r["device"].startswith("converter_grid_")]
    if exact_controls:
        fig, axes = plt.subplots(
            1,
            len(exact_controls),
            figsize=(6 * len(exact_controls), 4.5),
            constrained_layout=True,
            squeeze=False,
        )
        for ax, row in zip(axes.flat, exact_controls, strict=True):
            wl = np.asarray(row["wavelengths_um"]) * 1000
            order = np.argsort(wl)
            for port, power in row["powers"].items():
                ax.plot(wl[order], np.asarray(power)[order], "o-", label=port)
            ax.axhspan(0.99, 1.01, color="#236192", alpha=0.08)
            ax.axhline(1, color="#444444", ls="--", lw=1)
            ax.set(
                title=row["device"],
                xlabel="Wavelength (nm)",
                ylabel="Selected transmission / incident power",
            )
            ax.text(
                0.03,
                0.04,
                f"Max |T−1|: {100 * row['max_transmission_error']:.2f}%\nMax reflection: {100 * row['reflection_max']:.2f}%",
                transform=ax.transAxes,
            )
            ax.legend()
            ax.grid(alpha=0.2)
        fig.suptitle("Straight controls on the converter's exact realized grid · 6 PPW")
        fig.savefig(HERE / "converter_grid_controls.png", dpi=170)
        plt.close(fig)
    rings = [r for r in runs if r["device"] == "ring_resonator"]
    if rings:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        for idx, row in enumerate(rings):
            wl = np.asarray(row["wavelengths_um"]) * 1000
            order = np.argsort(wl)
            power = np.asarray(row["powers"]["o2"])
            clock = "fixed clock" if "fixed" in row["run"] else "old clock"
            label = f"{clock} · {row['run_time_ps']:g} ps · {row['backend']}"
            axes[0].plot(
                wl[order],
                power[order],
                label=label,
                color=COLORS[idx % len(COLORS)],
                lw=1.2,
            )
            axes[1].scatter(
                row["run_time_ps"],
                row["termination"]["field_decay"],
                label=label,
                color=COLORS[idx % len(COLORS)],
                s=45,
            )
        axes[0].set(
            xlabel="Wavelength (nm)",
            ylabel="Through power / incident power",
            title="Ring spectra: clock correction and duration",
        )
        axes[0].axhline(1.02, color="#444444", ls="--", lw=1)
        axes[1].axhline(1e-5, color="#444444", ls="--", label="Required decay")
        axes[1].set(
            xlabel="Configured duration (ps)",
            ylabel="Terminal field-energy ratio",
            yscale="log",
            title="Time-convergence evidence",
        )
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        fig.savefig(HERE / "ring_duration.png", dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    main()
