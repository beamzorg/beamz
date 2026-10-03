"""Regenerate the committed spectral evidence from spectra.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).parent
DEVICES = {
    "mmi2x2": ("2×2 MMI", "Cross TE₀ power / incident power"),
    "mode_converter": ("Mode converter", "Converted TE₁ power / incident power"),
    "polarization_splitter_rotator": (
        "Polarization splitter-rotator",
        "Converted TE₀ power / incident power",
    ),
    "ring_resonator": ("Ring resonator", "Normalized through transmission"),
}


def ring_metrics(wavelength_nm: np.ndarray, power: np.ndarray):
    """Apply the pinned supplementary repository's FWHM procedure."""
    dense_wavelength = np.linspace(wavelength_nm.min(), wavelength_nm.max(), 1000)
    dense_power = CubicSpline(wavelength_nm, power)(dense_wavelength)
    half_depth = 0.5 * (1.0 + float(dense_power.min()))
    transitions = np.diff((dense_power < half_depth).astype(np.int8))
    left_edges = np.flatnonzero(transitions == 1) + 1
    right_edges = np.flatnonzero(transitions == -1) + 1
    left = next(int(edge) for edge in left_edges if np.any(right_edges > edge))
    right = int(right_edges[right_edges > left][0])
    fwhm_nm = float(dense_wavelength[right] - dense_wavelength[left])
    center_nm = float(0.5 * (dense_wavelength[left] + dense_wavelength[right]))
    return (
        dense_wavelength,
        dense_power,
        half_depth,
        left,
        right,
        fwhm_nm,
        center_nm / fwhm_nm,
    )


def main():
    payload = json.loads((HERE / "spectra.json").read_text())
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for axis, (key, (title, ylabel)) in zip(axes.flat, DEVICES.items(), strict=True):
        spectrum = payload["devices"][key]
        wavelength = np.asarray(spectrum["wavelength_nm"])
        power = np.asarray(spectrum["power"])
        axis.plot(wavelength, power, color="#1f77b4", marker="o", ms=4, lw=2)
        axis.set_title(title, fontsize=18)
        axis.set_xlabel("Wavelength (nm)", fontsize=13)
        axis.set_ylabel(ylabel, fontsize=13)
        axis.tick_params(labelsize=11)
        if key != "ring_resonator":
            axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.suptitle(
        "Additional passive-SOI transmission spectra · 6 PPW · JAX", fontsize=24
    )
    fig.text(
        0.5,
        0.015,
        "Fresh-process runs after the #244 material-ownership fix. Ring transmission is normalized as in the published analysis.",
        ha="center",
        fontsize=12,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.savefig(HERE / "transmission_spectra.png", dpi=180)
    plt.close(fig)

    ring = payload["devices"]["ring_resonator"]
    wavelength = np.asarray(ring["wavelength_nm"])
    power = np.asarray(ring["power"])
    dense_wavelength, dense_power, half_depth, left, right, fwhm_nm, q = ring_metrics(
        wavelength, power
    )
    figure = plt.figure(figsize=(16, 7))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.7, 1.0))
    spectrum_axis = figure.add_subplot(grid[:, 0])
    fwhm_axis = figure.add_subplot(grid[0, 1])
    q_axis = figure.add_subplot(grid[1, 1])
    spectrum_axis.plot(
        wavelength,
        power,
        "o",
        color="#1f77b4",
        ms=5,
        label="BeamZ monitor samples · 0.2 nm",
    )
    spectrum_axis.plot(
        dense_wavelength,
        dense_power,
        color="#c07a00",
        lw=2,
        label="Cubic spline · 1,000 points",
    )
    spectrum_axis.axhline(
        half_depth, color="#555", ls="--", label=f"Half depth = {half_depth:.3f}"
    )
    spectrum_axis.axvspan(
        dense_wavelength[left],
        dense_wavelength[right],
        color="#b83b78",
        alpha=0.2,
        label=f"First-dip FWHM = {fwhm_nm:.3f} nm",
    )
    spectrum_axis.set(
        title="BeamZ first-resonance extraction",
        xlabel="Wavelength (nm)",
        ylabel="Normalized through transmission",
        xlim=(wavelength.min(), wavelength.max()),
    )
    spectrum_axis.legend(fontsize=10)
    fwhm_axis.bar(["BeamZ"], [fwhm_nm], color="#1f77b4")
    fwhm_axis.axhspan(
        0.84, 0.90, color="#c07a00", alpha=0.45, label="Published 6 PPW range"
    )
    fwhm_axis.set(title="FWHM at 6 PPW", ylabel="FWHM (nm)", ylim=(0, 4.2))
    fwhm_axis.text(0, fwhm_nm + 0.08, f"{fwhm_nm:.3f} nm", ha="center")
    fwhm_axis.legend(fontsize=9)
    q_axis.bar(["BeamZ"], [q], color="#1f77b4")
    q_axis.axhspan(
        1756.5, 1839.4, color="#c07a00", alpha=0.45, label="Published 6 PPW range"
    )
    q_axis.set(title="Q at 6 PPW", ylabel="Q", ylim=(0, 2000))
    q_axis.text(0, q + 40, f"{q:.1f}", ha="center")
    q_axis.legend(fontsize=9)
    figure.suptitle(
        "Ring-resonator analysis using the published repository method", fontsize=23
    )
    figure.text(
        0.5,
        0.015,
        "BeamZ reached the 6.40 ps time limit with field-decay ratio 0.050719; spectral metrics are characterization and fail the published comparison.",
        ha="center",
        fontsize=11,
        color="dimgray",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))
    figure.savefig(HERE / "ring_analysis.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
