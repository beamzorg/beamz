"""Rebuild passive RGB fits and diagnostic plots from checked-in optical data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import beamz as bz

root = Path(__file__).resolve().parents[2]
data = root / "examples/data/cmos_rgb"
output = data / "experimental_lorentz_fit"
output.mkdir(exist_ok=True)
models, reports = {}, {}
fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
for col, name in enumerate(("red", "green", "blue")):
    wl, n, k = np.loadtxt(data / f"{name}_eps.csv", delimiter=",").T
    k = k + 0.01
    model, report = bz.fit_nk(
        wl * 1e-6, n, k, num_poles=9, weights=(0.1, 1.9), max_nfev=3000
    )
    models[name] = model.to_spec()
    reports[name] = report
    nk = np.sqrt(model.eps_model(bz.LIGHT_SPEED / (wl * 1e-6)))
    for row, target, fit in ((0, n, nk.real), (1, k, nk.imag)):
        axes[row, col].plot(wl * 1000, target, "k--", label="target")
        axes[row, col].plot(wl * 1000, fit, color=name, label="passive fit")
        axes[row, col].set(
            xlabel="Wavelength (nm)", ylabel="n" if row == 0 else "k", title=name
        )
        axes[row, col].legend()
    print(name, report, flush=True)
(output / "filters.json").write_text(json.dumps(models, indent=2) + "\n")
(output / "fit_report.json").write_text(json.dumps(reports, indent=2) + "\n")
fig.savefig(output / "filter_fits.png", dpi=160)
