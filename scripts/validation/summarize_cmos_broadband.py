"""Calibrate saved sensor spectra and compare the completed mesh refinements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/reviews/rtx3090-dispersion"


def main():
    spectra = {}
    reports = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for dx in (50, 25, 20):
        folder = OUTPUT / f"{dx}nm"
        if not (folder / "reference_spectra.npz").exists():
            continue
        with (
            np.load(folder / "sensor_spectra.npz") as device,
            np.load(folder / "reference_spectra.npz") as reference,
        ):
            incident = reference["incident_plane"]
            if np.any(incident <= 0) or not np.isfinite(incident).all():
                raise ValueError("Invalid broadband reference power")
            wl = device["wavelength_nm"]
            channels = {
                "red": device["red"] / incident,
                "green": (device["green1"] + device["green2"]) / incident,
                "blue": device["blue"] / incident,
            }
        spectra[dx] = channels
        reports[dx] = json.loads((folder / "sensor_report.json").read_text())
        reports[dx]["calibration_incident_minmax"] = [
            float(incident.min()),
            float(incident.max()),
        ]
        reports[dx]["peak_efficiency"] = {
            k: float(v.max()) for k, v in channels.items()
        }
        np.savez(folder / "calibrated_efficiency.npz", wavelength_nm=wl, **channels)
        local, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for index, (channel, color) in enumerate(
            (("red", "r"), ("green", "g"), ("blue", "b"))
        ):
            local_values = channels[channel]
            axes[index].plot(wl, local_values, label=f"{dx} nm grid")
            axes[index].set(
                xlabel="Wavelength (nm)", ylabel="Optical efficiency", title=channel
            )
            axes[index].legend()
            axes[index].grid(alpha=0.2)
            ax.plot(wl, local_values, color, label=channel)
        ax.set(
            xlabel="Wavelength (nm)",
            ylabel="Optical efficiency",
            title=f"Broadband RGGB sensor — {dx} nm grid",
        )
        ax.legend()
        ax.grid(alpha=0.2)
        local.savefig(folder / "efficiency.png", dpi=160)
        plt.close(local)
    comparison = {}
    for coarse, fine in ((50, 25), (25, 20)):
        if coarse not in spectra or fine not in spectra:
            continue
        comparison[f"{coarse}_to_{fine}_nm"] = {
            channel: float(
                np.max(abs(spectra[fine][channel] - spectra[coarse][channel]))
            )
            for channel in ("red", "green", "blue")
        }
    result = {
        "runs": reports,
        "max_absolute_efficiency_changes": comparison,
        "interpretation": "Temporal convergence and slab validation do not establish spatial convergence of the full sensor. These mesh comparisons quantify the remaining dependence on resolution.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    fig.savefig(OUTPUT / "mesh_comparison.png", dpi=160)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
