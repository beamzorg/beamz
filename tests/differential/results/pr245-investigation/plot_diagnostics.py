"""Plot committed evidence only; no new FDTD measurements or fitted references."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = HERE.parent
EVIDENCE = RESULTS / "rtx4060ti-2026-09-15"
measurements = json.loads((EVIDENCE / "measurements.json").read_text())
spectra = json.loads((EVIDENCE / "spectra.json").read_text())["devices"]
with (RESULTS / "rtx3090-2026-09-16/summary.csv").open() as f:
    previous = list(csv.DictReader(f))
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
names = [
    "crossing",
    "directional_coupler",
    "mmi2x2",
    "mode_converter",
    "polarization_splitter_rotator",
]
labels = [
    "Crossing · through TE0",
    "Directional coupler · cross TE0",
    "MMI · cross TE0",
    "Mode converter · TE1",
    "PSR · converted TE0",
]
for ax, name, label in zip(axes.flat, names, labels, strict=False):
    protocol = json.loads(
        (ROOT / f"tests/differential/cases/passive_soi_{name}.json").read_text()
    )["geometry"]["simulation"]
    key = next(k for k in protocol if k.startswith("published_converged_"))
    reference = protocol[key]
    series = protocol[reference["series_key"]]
    ppw = sorted(int(k) for k in series if k.isdigit())
    for solver, color in [("lumerical", "#4477aa"), ("tidy3d", "#ee7733")]:
        ax.plot(
            ppw,
            [series[str(p)][solver] for p in ppw],
            "o-",
            color=color,
            label=solver.title(),
        )
    runs = (
        [r for r in previous if r["device"] == name]
        if name in names[:2]
        else [r for r in measurements["runs"] if r["device"] == name]
    )
    ax.plot(
        [int(r["ppw"]) for r in runs],
        [float(r["beamz_power"]) for r in runs],
        "D--",
        color="#222222",
        label="BeamZ measured",
        zorder=5,
    )
    ax.axvspan(reference["convergence_start_ppw"], 25, color="#228833", alpha=0.08)
    ax.set(
        title=label,
        xlabel="Cells per wavelength (nominal PPW)",
        ylabel="Power / incident power",
        xticks=ppw,
    )
    ax.grid(alpha=0.2)
    if name == "crossing":
        ax.set_ylim(0.93, 0.98)
    else:
        ax.set_ylim(0, 1.04)
axes.flat[0].legend(fontsize=9)
ax = axes.flat[5]
for i, name in enumerate(names[2:]):
    r = next(r for r in measurements["runs"] if r["device"] == name)
    ax.plot([r["lower"], r["upper"]], [i, i], lw=12, alpha=0.2, color="#aa3377")
    ax.scatter([r["lumerical"], r["tidy3d"]], [i, i], marker="|", s=220, c="#4477aa")
    ax.scatter(r["beamz_power"], i, marker="D", c="black", zorder=4)
ax.axvline(0, color="gray", ls=":")
ax.axvline(1, color="gray", ls=":")
ax.set(
    title="Why a 6-PPW “pass” is weak evidence",
    xlabel="Power / incident power",
    yticks=range(3),
    yticklabels=["MMI", "Converter", "PSR"],
    ylim=(-0.6, 2.6),
)
ax.text(
    0.04,
    0.97,
    "Pink: current acceptance interval\nBlue ticks: published 6-PPW values\nBlack: BeamZ",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
)
fig.suptitle(
    "PR #245 audit — published resolution sweeps vs committed BeamZ measurements\nGreen shading: manifest’s reference convergence region; lines only connect available points",
    fontsize=14,
)
fig.savefig(HERE / "resolution_audit.png", dpi=170)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
metadata = measurements["validation_reports"]["mode_converter"]["metrics"][0][
    "metadata"
]
w = np.asarray(metadata["wavelengths_um"]) * 1000
order = np.argsort(w)
ax = axes[0, 0]
for key, label in [
    ("selected_output_spectrum", "Sum of selected outputs"),
    ("conversion_spectrum", "Converted TE1"),
]:
    ax.plot(w[order], np.asarray(metadata[key])[order], "o-", label=label)
ax.axhline(1, color="gray", ls=":", label="Unity")
ax.axhline(1.02, color="#cc3311", ls="--", label="Current bound 1.02")
ax.set(
    title="Converter: output bound fails at both band edges",
    xlabel="Wavelength (nm)",
    ylabel="Power / incident power",
)
ax.legend(fontsize=9)
ring = spectra["ring_resonator"]
x = np.asarray(ring["wavelength_nm"])
y = np.asarray(ring["power"])
dense = np.linspace(x.min(), x.max(), 1000)
yd = CubicSpline(x, y)(dense)
half = (1 + yd.min()) / 2
transitions = np.diff((yd < half).astype(int))
left = np.flatnonzero(transitions == 1)[0] + 1
right = next(i + 1 for i in np.flatnonzero(transitions == -1) if i + 1 > left)
ax = axes[0, 1]
ax.plot(dense, yd, lw=1, label="Cubic interpolation")
ax.plot(x, y, ".", ms=3, label="101 retained samples")
ax.axhline(half, ls="--", color="gray", label="Global half depth")
ax.axvspan(
    dense[left],
    dense[right],
    alpha=0.2,
    color="#cc3311",
    label="Reported FWHM interval",
)
ax.set(
    title="Ring: unconverged spectrum; Q is provisional",
    xlabel="Wavelength (nm)",
    ylabel="Through power / band maximum",
)
ax.legend(fontsize=8)
ax = axes[1, 0]
mask = (x > 1542) & (x < 1547)
peaks, _ = find_peaks(y[mask])
peak_wavelengths = x[mask][peaks]
ripple_spacing = float(np.median(np.diff(peak_wavelengths)))
window_scale = (1550e-9) ** 2 / (299792458 * 6.4e-12) * 1e9
ax.plot(x[mask], y[mask], "o-")
ax.plot(peak_wavelengths, y[mask][peaks], "v", color="#cc3311")
ax.set(
    title="Ring ripple: consistent with a finite-time contribution",
    xlabel="Wavelength (nm)",
    ylabel="Normalized through power",
)
ax.text(
    0.04,
    0.97,
    f"Peak spacing in this window: {ripple_spacing:.3f} nm\nλ²/(cT), T = 6.4 ps: {window_scale:.3f} nm\nA diagnostic clue, not proof of cause",
    va="top",
    transform=ax.transAxes,
    fontsize=9,
)
ax = axes[1, 1]
for i, name in enumerate(names[2:] + ["ring_resonator"]):
    value = measurements["run_metadata"][name]["termination_field_decay"]
    ax.bar(i, value, color="#cc3311" if name == "ring_resonator" else "#4477aa")
    ax.text(i, value * 1.5, f"{value:.2g}", ha="center", fontsize=9)
ax.axhline(1e-5, color="black", ls="--", label="Required field decay")
ax.set(
    yscale="log",
    ylim=(1e-10, 1),
    xticks=range(4),
    xticklabels=["MMI", "Converter", "PSR", "Ring"],
    ylabel="Terminal field-energy ratio",
    title="Only ring misses the recorded decay criterion",
)
ax.legend(fontsize=9)
for ax in axes.flat:
    ax.grid(alpha=0.2)
fig.suptitle("PR #245 failure diagnostics — existing 6-PPW runs only", fontsize=14)
fig.savefig(HERE / "failure_diagnostics.png", dpi=170)
plt.close(fig)
metrics = {
    "ring_local_ripple_peak_wavelengths_nm": peak_wavelengths.tolist(),
    "ring_local_ripple_median_spacing_nm": ripple_spacing,
    "ring_6p4ps_window_scale_nm_at_1550nm": window_scale,
    "ring_decay_threshold_multiple": 0.05071902926309659 / 1e-5,
    "ring_recomputed_fwhm_nm": float(dense[right] - dense[left]),
    "ring_fwhm_crossings_nm": [float(dense[left]), float(dense[right])],
    "converter_over_bound_wavelengths_nm": w[
        np.asarray(metadata["selected_output_spectrum"]) > 1.02
    ].tolist(),
}
(HERE / "derived_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
