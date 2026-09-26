"""Run the broadband sensor on the local JAX GPU and save review artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
from cmos_rgb_sensor import make_sim, pixel_signs, sensor_grid  # noqa: E402

import beamz as bz  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx-nm", type=float, default=25)
    parser.add_argument("--runtime-fs", type=float, default=300)
    parser.add_argument("--output", default="docs/reviews/cmos-reference-parity")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--graded", action="store_true")
    parser.add_argument("--metal-nm", type=float, default=5)
    parser.add_argument("--absorber-nm", type=float, default=400)
    parser.add_argument("--num-freqs", type=int)
    parser.add_argument("--no-fields", action="store_true")
    args = parser.parse_args()
    label = (
        f"graded-{args.dx_nm:g}-{args.metal_nm:g}nm"
        if args.graded
        else f"{args.dx_nm:g}nm"
    )
    output = ROOT / args.output / label
    output.mkdir(parents=True, exist_ok=True)
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("This runner requires JAX GPU; no GPU device found.")
    sim = make_sim(
        dx=args.dx_nm * 1e-9,
        run_time=args.runtime_fs * 1e-15,
        reference=args.reference,
        fields=not args.reference and not args.no_fields,
        num_freqs=args.num_freqs,
        grid_spec=sensor_grid(lateral_nm=args.dx_nm, metal_nm=args.metal_nm)
        if args.graded
        else None,
        absorber_thickness=args.absorber_nm * 1e-9,
    )
    report = {
        "device": str(jax.devices()),
        "dx_nm": args.dx_nm,
        "runtime_fs": args.runtime_fs,
        "grid": list(sim.grid.shape),
        "steps": int(sim.num_steps),
        "dt": float(sim.dt),
        "reference": args.reference,
        "graded": args.graded,
        "metal_nm": args.metal_nm if args.graded else None,
        "absorber_nm": args.absorber_nm,
        "material_checksums": {
            name: hashlib.sha256(
                (ROOT / "examples/data/cmos_rgb" / name).read_bytes()
            ).hexdigest()
            for name in ("materials.json", "filters.json")
        },
    }
    print(report, flush=True)
    start = time.perf_counter()
    print("Compiling/rasterizing", flush=True)
    program = sim.compile(backend="jax")
    report["setup_seconds"] = time.perf_counter() - start
    report["polarization_bytes"] = sum(
        np.prod(r.shape) * 8 for r in program.dispersion.regions
    )
    print("Setup finished", report, flush=True)
    half = int(sim.num_steps // 2)
    start = time.perf_counter()
    first = sim.advance(num_steps=half, backend="jax", performance=False)
    report["first_half_seconds"] = time.perf_counter() - start
    print("First half finished", report["first_half_seconds"], flush=True)
    start = time.perf_counter()
    final = sim.advance(
        state=first.state,
        num_steps=sim.num_steps - half,
        backend="jax",
        performance=False,
        donate_state=True,
    )
    report["second_half_seconds"] = time.perf_counter() - start
    result = final.results
    report["finite_fields"] = all(
        bool(np.isfinite(np.asarray(getattr(final.state, c))).all())
        for c in ("ex", "ey", "ez", "hx", "hy", "hz")
    )
    report["finite_polarization"] = all(
        bool(np.isfinite(np.asarray(q)).all()) for q in final.state.polarization
    )
    wl = bz.LIGHT_SPEED / np.asarray(result["incident_plane"].dft_frequencies) * 1e9
    raw = {pixel: -np.asarray(result[f"flux_si_{pixel}"].flux) for pixel in pixel_signs}
    first_raw = {
        pixel: -np.asarray(first.results[f"flux_si_{pixel}"].flux)
        for pixel in pixel_signs
    }
    report["temporal_max_absolute_change"] = {
        pixel: float(np.max(abs(raw[pixel] - first_raw[pixel])))
        for pixel in pixel_signs
    }
    report["launched_power"] = float(result.launched_power())
    report["peak_field"] = max(
        float(np.max(abs(np.asarray(getattr(final.state, c)))))
        for c in ("ex", "ey", "ez")
    )
    name = "reference" if args.reference else "sensor"
    np.savez(
        output / f"{name}_spectra.npz",
        wavelength_nm=wl,
        incident_plane=-np.asarray(result["incident_plane"].flux),
        **raw,
    )
    report["polarization_bytes"] = int(report["polarization_bytes"])
    (output / f"{name}_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["finite_fields"] or not report["finite_polarization"]:
        raise RuntimeError(
            "Simulation became nonfinite; inspect material/timestep stability."
        )
    if not args.reference:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for channel, color in [("red", "r"), ("green", "g"), ("blue", "b")]:
            efficiency = (
                raw["green1"] + raw["green2"] if channel == "green" else raw[channel]
            )
            ax.plot(wl, efficiency / report["launched_power"], color, label=channel)
        ax.set(
            xlabel="Wavelength (nm)",
            ylabel="Optical efficiency",
            title=f"Broadband RGGB sensor — {args.dx_nm:g} nm grid",
        )
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(output / "efficiency.png", dpi=160)
        plt.close(fig)
        for monitor in (
            () if args.no_fields else ("field_xy_silicon", "field_yz_BG", "field_yz_GR")
        ):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, wavelength in zip(axes, (650, 550, 450), strict=True):
                result.plot_field(
                    monitor,
                    "E",
                    frequency=bz.LIGHT_SPEED / (wavelength * 1e-9),
                    val="abs^2",
                    ax=ax,
                    show=False,
                )
                ax.set_title(f"{wavelength} nm")
            fig.savefig(output / f"{monitor}.png", dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    main()
