"""RTX/CPU broadband slab checks against analytic Fresnel thin-film optics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax
import matplotlib.pyplot as plt
import numpy as np

import beamz as bz

ROOT = Path(__file__).resolve().parents[2]


def analytic_slab(epsilon, wavelength, thickness):
    n = np.sqrt(epsilon)
    r = (1 - n) / (1 + n)
    phase = np.exp(2j * np.pi * n * thickness / wavelength)
    reflection = r * (1 - phase**2) / (1 - r * r * phase**2)
    transmission = (1 - r * r) * phase / (1 - r * r * phase**2)
    return abs(reflection) ** 2, abs(transmission) ** 2


def run_slab(material, dx, wavelengths, thickness, courant=0.7):
    # Narrow periodic cross section is sufficient for a normal plane wave.
    size = (4 * dx, 4 * dx, 2.4e-6)
    f = bz.LIGHT_SPEED / wavelengths
    pulse = bz.GaussianPulse(
        float((f.min() + f.max()) / 2), float((f.max() - f.min()) * 0.9)
    )
    monitors = [
        bz.FluxMonitor(center=(0, 0, z), size=(size[0], size[1], 0), freqs=f, name=name)
        for name, z in [("input", 0.45e-6), ("output", -0.45e-6)]
    ]
    source = bz.PlaneWaveSource(
        center=(0, 0, 0.7e-6),
        size=(size[0], size[1], 0),
        source_time=pulse,
        direction="-z",
        power=1.0,
    )
    design = bz.Design(background=bz.Material(1.0))
    if material is not None:
        design += bz.Box(
            center=(0, 0, 0), size=(size[0], size[1], thickness), material=material
        )
    sim = bz.Simulation(
        design=design,
        size=size,
        sources=[source],
        monitors=monitors,
        boundaries=[
            bz.Periodic(axes=("x", "y")),
            bz.PML(edges=("front", "back"), thickness=0.3e-6),
        ],
        grid_spec=bz.GridSpec.uniform(dx, courant=courant),
        run_time=90e-15,
    )
    t = time.perf_counter()
    result = sim.run(backend="jax", progress=False)
    return {
        name: np.asarray(result[name].flux) for name in ("input", "output")
    }, time.perf_counter() - t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/reviews/rtx3090-dispersion/slabs")
    parser.add_argument("--dx-nm", nargs="+", type=float, default=[10, 5, 2.5])
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    wavelength = np.linspace(400e-9, 700e-9, 31)
    specs = json.loads((ROOT / "examples/data/cmos_rgb/materials.json").read_text())
    materials = {
        "aluminum": (bz.PoleResidue.from_spec(specs["Al_Rakic1995"]), 40e-9),
        "silicon": (bz.PoleResidue.from_spec(specs["aSi_Horiba"]), 100e-9),
    }
    report = {"device": str(jax.devices()), "runs": []}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for col, (name, (material, thickness)) in enumerate(materials.items()):
        R, T = analytic_slab(
            material.eps_model(bz.LIGHT_SPEED / wavelength), wavelength, thickness
        )
        axes[0, col].plot(wavelength * 1e9, R, "k--", label="analytic")
        axes[1, col].plot(wavelength * 1e9, T, "k--", label="analytic")
        for dx_nm in args.dx_nm:
            reference, t0 = run_slab(None, dx_nm * 1e-9, wavelength, thickness)
            device, t1 = run_slab(material, dx_nm * 1e-9, wavelength, thickness)
            incident = -reference["input"]
            measured_R = 1 + device["input"] / incident
            measured_T = -device["output"] / (-reference["output"])
            record = {
                "material": name,
                "dx_nm": dx_nm,
                "R_max_abs_error": float(np.max(abs(measured_R - R))),
                "T_max_abs_error": float(np.max(abs(measured_T - T))),
                "min_absorption": float(np.min(1 - measured_R - measured_T)),
                "seconds": t0 + t1,
            }
            report["runs"].append(record)
            print(record, flush=True)
            np.savez(
                output / f"{name}_{dx_nm:g}nm.npz",
                wavelength=wavelength,
                R=measured_R,
                T=measured_T,
                R_exact=R,
                T_exact=T,
            )
            for row, y in enumerate((measured_R, measured_T)):
                axes[row, col].plot(wavelength * 1e9, y, label=f"{dx_nm:g} nm")
                axes[row, col].set(
                    xlabel="Wavelength (nm)",
                    ylabel="R" if row == 0 else "T",
                    title=name,
                )
                axes[row, col].legend()
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            fig.savefig(output / "slabs.png", dpi=160)
            jax.clear_caches()
    if any(
        r["R_max_abs_error"] > 0.025 or r["T_max_abs_error"] > 0.025
        for r in report["runs"]
        if r["dx_nm"] == min(args.dx_nm)
    ):
        raise SystemExit(
            "Slab validation exceeds 2.5 percentage points at finest resolution"
        )


if __name__ == "__main__":
    main()
