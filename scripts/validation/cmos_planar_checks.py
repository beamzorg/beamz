"""Broadband GPU validation of reference filter films and the detector stack."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax
import matplotlib.pyplot as plt
import numpy as np

import beamz as bz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
from cmos_rgb_sensor import load_materials, t_ar  # noqa: E402


def analytic_stack(layers, wavelengths):
    """Normal-incidence transfer matrix, air on both exterior sides, e^-iwt."""
    f = bz.LIGHT_SPEED / wavelengths
    matrix = np.broadcast_to(np.eye(2, dtype=complex), (len(f), 2, 2)).copy()
    for material, thickness in layers:
        n = np.sqrt(material.eps_model(f))
        phase = 2 * np.pi * n * thickness / wavelengths
        layer = np.zeros_like(matrix)
        layer[:, 0, 0] = layer[:, 1, 1] = np.cos(phase)
        layer[:, 0, 1] = -1j * np.sin(phase) / n
        layer[:, 1, 0] = -1j * n * np.sin(phase)
        matrix = matrix @ layer
    denominator = matrix[:, 0, 0] + matrix[:, 0, 1] + matrix[:, 1, 0] + matrix[:, 1, 1]
    r = (
        matrix[:, 0, 0] + matrix[:, 0, 1] - matrix[:, 1, 0] - matrix[:, 1, 1]
    ) / denominator
    t = 2 / denominator
    return abs(r) ** 2, abs(t) ** 2


def run(layers, dx, wavelengths, absorber=0.35e-6):
    thickness = sum(d for _, d in layers)
    # Keep monitor and source locations identical in device and reference.
    size = (4 * dx, 4 * dx, 3.2e-6)
    design = bz.Design(background=bz.Material(1))
    z = 0.5e-6
    for material, d in layers:
        if material is not None:
            design += bz.Box(
                center=(0, 0, z - d / 2), size=(*size[:2], d), material=material
            )
        z -= d
    freqs = bz.LIGHT_SPEED / wavelengths
    sim = bz.Simulation(
        design=design,
        size=size,
        sources=[
            bz.PlaneWaveSource(
                center=(0, 0, 1e-6),
                size=(*size[:2], 0),
                direction="-z",
                source_time=bz.GaussianPulse(
                    float(np.mean([freqs.min(), freqs.max()])), 2e14
                ),
            )
        ],
        monitors=[
            bz.FluxMonitor(
                center=(0, 0, zc), size=(*size[:2], 0), freqs=freqs, name=name
            )
            for name, zc in [("input", 0.75e-6), ("output", 0.3e-6 - thickness)]
        ],
        boundaries=[
            bz.Periodic(axes=("x", "y")),
            bz.Absorber(edges=("front", "back"), thickness=absorber),
        ],
        grid_spec=bz.GridSpec.uniform(dx, courant=0.7),
        run_time=160e-15,
    )
    result = sim.run(backend="jax", progress=False)
    return {name: np.asarray(result[name].flux) for name in ("input", "output")}


def main():
    output = ROOT / "docs/reviews/cmos-reference-parity/planar"
    output.mkdir(parents=True, exist_ok=True)
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("GPU required")
    mats = load_materials()
    wl = np.linspace(400e-9, 700e-9, 61)
    cases = {c: [(mats[c], 1e-6)] for c in ("red", "green", "blue")}
    cases["detector"] = [
        (mats["silica"], 0.3e-6),
        (mats["sin"], t_ar),
        (mats["silicon"], 0.1e-6),
    ]
    report = {"device": str(jax.devices()), "runs": []}
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    for col, (name, layers) in enumerate(cases.items()):
        exact_r, exact_t = analytic_stack(layers, wl)
        for row, values in enumerate((exact_r, exact_t)):
            axes[row, col].plot(wl * 1e9, values, "k--", label="Transfer matrix")
        for dx_nm in (10, 5):
            reference = run([(None, d) for _, d in layers], dx_nm * 1e-9, wl)
            device = run(layers, dx_nm * 1e-9, wl)
            R = 1 - device["input"] / reference["input"]
            T = device["output"] / reference["output"]
            record = {
                "case": name,
                "dx_nm": dx_nm,
                "max_R_error": float(np.max(abs(R - exact_r))),
                "max_T_error": float(np.max(abs(T - exact_t))),
                "min_absorption": float(np.min(1 - R - T)),
            }
            print(record, flush=True)
            report["runs"].append(record)
            np.savez(
                output / f"{name}-{dx_nm}nm.npz",
                wavelength_nm=wl * 1e9,
                R=R,
                T=T,
                R_exact=exact_r,
                T_exact=exact_t,
            )
            for row, values in enumerate((R, T)):
                axes[row, col].plot(wl * 1e9, values, label=f"{dx_nm} nm")
                axes[row, col].set(
                    title=name,
                    xlabel="Wavelength (nm)",
                    ylabel="R" if row == 0 else "T",
                )
                axes[row, col].legend()
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            fig.savefig(output / "planar.png", dpi=150)
            jax.clear_caches()
    assert all(
        r["max_R_error"] < 0.025
        and r["max_T_error"] < 0.025
        and r["min_absorption"] > -0.005
        for r in report["runs"]
        if r["dx_nm"] == 5
    )


if __name__ == "__main__":
    main()
