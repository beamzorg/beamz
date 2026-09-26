"""Import exact passive RGB models from the saved public Tidy3D simulation.

The website embeds a gzip/base64 HDF5 simulation in its 3D viewer. The decoded
JSON snapshot is checked in, so this import is offline and deterministic.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import beamz as bz

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "examples/data/cmos_rgb"


def main():
    simulation = json.loads((DATA / "tidy3d_reference_simulation.json").read_text())
    materials = {}
    for structure, key in (
        ("len blue", "SiO2_Palik_LowLoss"),
        ("metal shield", "Al_Rakic1995"),
        ("anti-reflection", "SiN_Horiba"),
        ("silicon layer", "aSi_Horiba"),
    ):
        medium = next(
            s["medium"] for s in simulation["structures"] if s["name"] == structure
        )
        model = bz.PoleResidue(
            medium["eps_inf"],
            [(complex(**a), complex(**c)) for a, c in medium["poles"]],
            frequency_range=medium["frequency_range"],
        )
        materials[key] = model.to_spec()
    (DATA / "materials.json").write_text(json.dumps(materials, indent=2) + "\n")
    models, reports = {}, {}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for col, (channel, structure) in enumerate(
        (("red", "red filter"), ("green", "green1 filter"), ("blue", "blue filter"))
    ):
        medium = next(
            s["medium"] for s in simulation["structures"] if s["name"] == structure
        )
        model = bz.PoleResidue(
            medium["eps_inf"],
            [(complex(**a), complex(**c)) for a, c in medium["poles"]],
            frequency_range=(bz.LIGHT_SPEED / 700e-9, bz.LIGHT_SPEED / 400e-9),
        )
        wl, n, k = np.loadtxt(DATA / f"{channel}_eps.csv", delimiter=",").T
        k = k + 0.01
        nk = np.sqrt(model.eps_model(bz.LIGHT_SPEED / (wl * 1e-6)))
        passband = k < 0.011
        if np.max(nk.imag[passband]) > 0.02:
            raise ValueError(
                f"{channel}: reference passband loss exceeds acceptance limit"
            )
        models[channel] = model.to_spec()
        reports[channel] = {
            "source": "exact embedded reference model",
            "rms_n": float(np.sqrt(np.mean((nk.real - n) ** 2))),
            "rms_k": float(np.sqrt(np.mean((nk.imag - k) ** 2))),
            "passband_k_max": float(nk.imag[passband].max()),
        }
        for row, target, values in ((0, n, nk.real), (1, k, nk.imag)):
            ax = axes[row, col]
            ax.plot(wl * 1000, target, "k--", label="CSV target")
            ax.plot(wl * 1000, values, color=channel, label="Reference model")
            ax.set(
                xlabel="Wavelength (nm)", ylabel="n" if row == 0 else "k", title=channel
            )
            ax.legend()
    (DATA / "filters.json").write_text(json.dumps(models, indent=2) + "\n")
    (DATA / "fit_report.json").write_text(json.dumps(reports, indent=2) + "\n")
    fig.savefig(DATA / "filter_fits.png", dpi=160)


if __name__ == "__main__":
    main()
