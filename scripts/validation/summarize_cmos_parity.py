"""Compare saved broadband reference-parity runs without rerunning FDTD."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/reviews/cmos-reference-parity"


def main():
    spectra = {}
    old = np.load(
        ROOT / "docs/reviews/rtx3090-dispersion/25nm/calibrated_efficiency.npz"
    )
    spectra["Old materials, 25 nm"] = {key: old[key] for key in old.files}
    notebook = np.load(OUT / "notebook25nm/calibrated_efficiency.npz")
    spectra["Exact materials, 25 nm"] = {key: notebook[key] for key in notebook.files}
    reports = {}
    for label in ["graded-20-5nm", "graded-15-5nm"]:
        folder = OUT / label
        if (
            not (folder / "sensor_spectra.npz").exists()
            or not (folder / "reference_spectra.npz").exists()
        ):
            continue
        device = np.load(folder / "sensor_spectra.npz")
        reference = np.load(folder / "reference_spectra.npz")
        np.testing.assert_allclose(
            device["wavelength_nm"], reference["wavelength_nm"], rtol=1e-12
        )
        incident = reference["incident_plane"]
        channels = {
            "wavelength_nm": device["wavelength_nm"],
            "red": device["red"] / incident,
            "green": (device["green1"] + device["green2"]) / incident,
            "blue": device["blue"] / incident,
        }
        if not all(np.isfinite(v).all() for v in channels.values()):
            raise ValueError("Nonfinite spectrum")
        np.savez(folder / "calibrated_efficiency.npz", **channels)
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for channel, color in (("red", "r"), ("green", "g"), ("blue", "b")):
            ax.plot(
                channels["wavelength_nm"],
                100 * channels[channel],
                color=color,
                label=channel,
            )
        ax.set(
            xlabel="Wavelength (nm)",
            ylabel="Calibrated optical efficiency (%)",
            title=label,
        )
        ax.legend()
        ax.grid(alpha=0.2)
        fig.savefig(folder / "calibrated_efficiency.png", dpi=160)
        plt.close(fig)
        spectra[label] = channels
        reports[label] = json.loads((folder / "sensor_report.json").read_text())
    centers = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for label, data in spectra.items():
        w = data["wavelength_nm"]
        order = np.argsort(w)
        centers[label] = {}
        for ax, (channel, wl) in zip(
            axes, [("red", 650), ("green", 550), ("blue", 450)], strict=True
        ):
            ax.plot(
                w[order],
                100 * data[channel][order],
                label=label,
                linestyle="--" if label.startswith("Old") else "-",
            )
            ax.set(
                xlabel="Wavelength (nm)",
                ylabel="Optical efficiency (%)",
                title=channel,
            )
            ax.grid(alpha=0.2)
            centers[label][channel] = float(
                np.interp(wl, w[order], data[channel][order])
            )
    for ax, channel in zip(axes, ("red", "green", "blue"), strict=True):
        wl, value = {"red": (650, 6), "green": (550, 20), "blue": (450, 6)}[channel]
        ax.scatter(
            [wl],
            [value],
            marker="D",
            facecolors="none",
            edgecolors="black",
            label="Published plot ≈",
        )
        ax.set_ylim(
            0, 110 * max(float(data[channel].max()) for data in spectra.values())
        )
        ax.legend(fontsize=7)
    fig.savefig(OUT / "comparison.png", dpi=160)
    differences = {}
    for a, b in [
        ("Exact materials, 25 nm", "graded-20-5nm"),
        ("graded-20-5nm", "graded-15-5nm"),
    ]:
        if b not in spectra:
            continue
        x, y = spectra[a], spectra[b]
        order = np.argsort(x["wavelength_nm"])
        differences[a + " to " + b] = {
            c: float(
                np.max(
                    abs(
                        np.interp(
                            y["wavelength_nm"], x["wavelength_nm"][order], x[c][order]
                        )
                        - y[c]
                    )
                )
            )
            for c in ("red", "green", "blue")
        }
    summary = {
        "central_efficiencies": centers,
        "published_plot_approximate": {"red": 0.06, "green": 0.20, "blue": 0.06},
        "reference_values_are_visual_estimates": True,
        "max_absolute_spectrum_changes": differences,
        "runs": reports,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
