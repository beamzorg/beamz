"""Audit silicon-interface flux interpolation and absorber sensitivity on GPU."""

import json
import sys
from pathlib import Path

import jax
import numpy as np

import beamz as bz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
from cmos_rgb_sensor import load_materials  # noqa: E402


def run(dx, thickness, material):
    size = (4 * dx, 4 * dx, 3.2e-6)
    design = bz.Design(background=bz.Material(1))
    if material is not None:
        design += bz.Box(
            center=(0, 0, -0.8e-6), size=(*size[:2], 1.6e-6), material=material
        )
    wavelengths = np.linspace(400e-9, 700e-9, 31)
    f = bz.LIGHT_SPEED / wavelengths
    monitors = [
        bz.FluxMonitor(center=(0, 0, z), size=(*size[:2], 0), freqs=f, name=name)
        for name, z in [("input", 0.5e-6), ("interface", 0), ("inside", -3 * dx)]
    ]
    sim = bz.Simulation(
        design=design,
        size=size,
        sources=[
            bz.PlaneWaveSource(
                center=(0, 0, 0.85e-6),
                size=(*size[:2], 0),
                source_time=bz.GaussianPulse(5.75e14, 2e14),
                direction="-z",
            )
        ],
        monitors=monitors,
        boundaries=[
            bz.Periodic(axes=("x", "y")),
            bz.Absorber(edges=("front", "back"), thickness=thickness),
        ],
        grid_spec=bz.GridSpec.uniform(dx, courant=0.7),
        run_time=130e-15,
    )
    r = sim.run(backend="jax", progress=False)
    return {name: np.asarray(r[name].flux) for name in ("input", "interface", "inside")}


def main():
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("GPU required")
    output = ROOT / "docs/reviews/cmos-reference-parity/interfaces"
    output.mkdir(parents=True, exist_ok=True)
    wl = np.linspace(400e-9, 700e-9, 31)
    material = load_materials()["silicon"]
    n = np.sqrt(material.eps_model(bz.LIGHT_SPEED / wl))
    R = abs((1 - n) / (1 + n)) ** 2
    T = 4 * n.real / abs(1 + n) ** 2
    records = []
    spectra = {}
    for dx_nm, absorber_nm in [(5, 250), (2.5, 250), (2.5, 500)]:
        dx = dx_nm * 1e-9
        ref = run(dx, absorber_nm * 1e-9, None)
        dev = run(dx, absorber_nm * 1e-9, material)
        measured_R = 1 - dev["input"] / ref["input"]
        measured_T = dev["interface"] / ref["interface"]
        inside = dev["inside"] / ref["inside"]
        exact_inside = T * np.exp(-4 * np.pi * n.imag * 3 * dx / wl)
        record = {
            "dx_nm": dx_nm,
            "absorber_nm": absorber_nm,
            "max_R_error": float(np.max(abs(measured_R - R))),
            "max_interface_T_error": float(np.max(abs(measured_T - T))),
            "max_inside_T_error": float(np.max(abs(inside - exact_inside))),
            "max_interface_balance_error": float(
                np.max(abs(measured_R + measured_T - 1))
            ),
        }
        print(record, flush=True)
        records.append(record)
        spectra[(dx_nm, absorber_nm)] = measured_T
        np.savez(
            output / f"{dx_nm:g}nm-absorber{absorber_nm}.npz",
            wavelength_nm=wl * 1e9,
            R=measured_R,
            T=measured_T,
            T_inside=inside,
            R_exact=R,
            T_exact=T,
            T_inside_exact=exact_inside,
        )
        (output / "report.json").write_text(
            json.dumps({"runs": records}, indent=2) + "\n"
        )
        jax.clear_caches()
    sensitivity = float(np.max(abs(spectra[(2.5, 250)] - spectra[(2.5, 500)])))
    (output / "report.json").write_text(
        json.dumps(
            {
                "device": str(jax.devices()),
                "runs": records,
                "absorber_max_T_change": sensitivity,
            },
            indent=2,
        )
        + "\n"
    )
    assert max(r["max_interface_T_error"] for r in records if r["dx_nm"] == 2.5) < 0.025
    assert sensitivity < 0.005


if __name__ == "__main__":
    main()
