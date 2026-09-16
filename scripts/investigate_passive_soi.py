"""Run one auditable passive-SOI experiment per process.

Example: python -m scripts.investigate_passive_soi ring_resonator --run-time-ps 12.8
Raw artifacts are retained under validation-artifacts by default.
"""

# ruff: noqa: E402

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
from dataclasses import asdict
from pathlib import Path

# Keep small repeated port-mode solves from oversubscribing CPU cores.
# Explicit caller settings take precedence; configure before NumPy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax
import numpy as np

from beamz import LIGHT_SPEED, AutoTermination, µm
from beamz.analysis import s_parameters
from tests.differential.passive_soi.common import (
    load_passive_soi_case,
    reference_absorber_warning_scope,
)
from tests.differential.passive_soi.experiments import (
    ExperimentOptions,
    power_comparison,
)
from tests.differential.passive_soi.four_port import (
    _save_four_port_artifacts,
    build_four_port_simulation,
)
from tests.differential.passive_soi.mode_conversion import (
    build_mode_conversion_simulation,
)
from tests.differential.passive_soi.ring_resonator import (
    build_ring_resonator_simulation,
    extract_ring_resonances,
)
from tests.differential.passive_soi.straight_control import (
    build_converter_grid_control,
    build_straight_control,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "device",
        choices=[
            "mmi2x2",
            "mode_converter",
            "polarization_splitter_rotator",
            "ring_resonator",
            "straight_te0",
            "straight_wide_te0",
            "straight_te1",
            "straight_tm0",
            "converter_grid_te0",
            "converter_grid_te1",
        ],
    )
    parser.add_argument("--ppw", type=int, default=6)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--port-extension-policy",
        choices=("reference", "through_boundary"),
        default="reference",
    )
    parser.add_argument("--backend", choices=("jax", "cuda_streamed"), default="jax")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--grid-from",
        type=Path,
        help="Freeze the grid from an earlier monitor_data.npz",
    )
    parser.add_argument("--run-time-ps", type=float)
    parser.add_argument("--monitor-offset-um", type=float)
    parser.add_argument("--boundary-thickness-um", type=float, default=1.0)
    parser.add_argument("--transverse-span-um", type=float)
    parser.add_argument("--source-profiles", type=int)
    parser.add_argument("--wavelength-step-nm", type=float)
    parser.add_argument("--exact-center", action="store_true")
    parser.add_argument("--smoothing", default="farjadpour_diagonal")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "output directory already exists; choose a new path to preserve evidence"
        )
    options = ExperimentOptions(
        **{k: getattr(args, k) for k in ExperimentOptions.__dataclass_fields__}
    )
    if args.device.startswith("converter_grid_"):
        simulation, ports, outputs, frequencies = build_converter_grid_control(
            args.device.removeprefix("converter_grid_"),
            resolution_ppw=args.ppw,
            options=options,
        )
    elif args.device.startswith("straight_"):
        simulation, ports, outputs, frequencies = build_straight_control(
            args.device.removeprefix("straight_"),
            resolution_ppw=args.ppw,
            options=options,
        )
    elif args.device == "ring_resonator":
        simulation, ports, frequencies = build_ring_resonator_simulation(
            resolution_ppw=args.ppw, diagnostics=True, options=options
        )
        outputs = ("o1", "o2")
    elif args.device == "mmi2x2":
        simulation, ports, frequencies = build_four_port_simulation(
            load_passive_soi_case(args.device),
            resolution_ppw=args.ppw,
            diagnostics=True,
            options=options,
        )
        outputs = ("o3", "o4")
    else:
        simulation, ports, outputs, frequencies = build_mode_conversion_simulation(
            args.device, resolution_ppw=args.ppw, diagnostics=True, options=options
        )
    frozen_grid = None
    if args.grid_from is not None:
        from beamz import Simulation
        from beamz.design.discretization import build_material_grid
        from beamz.design.grid import RectilinearGrid

        if simulation.design is None:
            parser.error("--grid-from requires a design-backed device")
        with np.load(args.grid_from) as raw:
            grid = RectilinearGrid(
                *(raw[f"grid_{axis}_boundaries_m"] for axis in "xyz")
            )
        for old, new in zip(simulation.grid.edges, grid.edges, strict=True):
            if not np.allclose(old[[0, -1]], new[[0, -1]], rtol=0, atol=1e-12):
                parser.error("frozen grid domain does not match the requested device")
        material = build_material_grid(
            simulation.design, grid, quality="balanced", smoothing=options.smoothing
        )
        simulation = Simulation(
            material_grid=material,
            sources=simulation.sources,
            monitors=simulation.monitors,
            boundaries=simulation.boundaries,
            run_time=simulation.run_time,
        )
        frozen_grid = {
            "path": str(args.grid_from.resolve()),
            "sha256": hashlib.sha256(args.grid_from.read_bytes()).hexdigest(),
        }
    args.output.mkdir(parents=True)
    diff = subprocess.check_output(["git", "diff", "HEAD"])
    (args.output / "working-tree.patch").write_bytes(diff)
    config = {
        "device": args.device,
        "ppw": args.ppw,
        "options": asdict(options),
        "frozen_grid": frozen_grid,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_patch_sha256": hashlib.sha256(diff).hexdigest(),
        "grid_shape": list(simulation.grid.shape),
        "cells": int(np.prod(simulation.grid.shape)),
        "run_time_ps": simulation.run_time * 1e12,
        "backend": args.backend,
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config), flush=True)
    with reference_absorber_warning_scope():
        result = simulation.run(
            backend=args.backend,
            progress=args.progress,
            termination=AutoTermination(
                field_decay=1e-5, monitor_change=None, consecutive_checks=1
            ),
        )
    print(
        json.dumps(
            {"stage": "simulation_complete", "termination": asdict(result.termination)}
        ),
        flush=True,
    )
    scattering = s_parameters(
        result,
        source_port="o1",
        ports=ports,
        output_ports=outputs,
        frequencies=frequencies,
        min_incident_db=-45,
    )
    if not np.all(scattering.diagnostics["valid_mask"]):
        raise ValueError("invalid incident signal in retained spectrum")
    _save_four_port_artifacts(
        args.output,
        simulation,
        result,
        scattering,
        execution_backend=args.backend,
        options=options,
    )
    wavelengths = LIGHT_SPEED / np.asarray(frequencies) / µm
    powers = {key[0]: np.abs(value) ** 2 for key, value in scattering.s_matrix.items()}
    summary = {
        **config,
        "wavelengths_um": wavelengths.tolist(),
        "powers": {k: v.tolist() for k, v in powers.items()},
        "termination": asdict(result.termination),
        "performance": asdict(result.performance),
    }
    if args.device.startswith(("straight_", "converter_grid_")):
        stack = np.stack(list(powers.values()))
        summary["max_transmission_error"] = float(np.max(np.abs(stack - 1)))
        summary["max_monitor_power_spread"] = float(np.max(np.ptp(stack, axis=0)))
    elif args.device == "ring_resonator":
        resonances, fsr, fwhm, q, extinction, normalized = extract_ring_resonances(
            wavelengths, powers["o2"]
        )
        summary["ring"] = {
            "resonances_um": resonances,
            "fsr_nm": fsr,
            "fwhm_nm": fwhm,
            "q": q,
            "extinction_db": extinction,
            "field_decay_passed": result.termination.field_decay <= 1e-5,
            "output_power_passed": float(np.max(sum(powers.values()))) <= 1.02,
            "time_convergence_established": False,
        }
        summary["selected_output_max"] = float(np.max(sum(powers.values())))
    else:
        channel = "o3" if args.device == "mmi2x2" else "conversion"
        # Four-port protocol identifies the cross port explicitly.
        if args.device == "mmi2x2":
            channel = load_passive_soi_case(args.device).geometry["simulation"][
                "cross_port"
            ]
        center = int(np.argmin(np.abs(wavelengths - 1.55)))
        summary["reference"] = power_comparison(
            load_passive_soi_case(args.device),
            "cross" if args.device == "mmi2x2" else "conversion",
            args.ppw,
            float(powers[channel][center]),
        )
        summary["selected_output_max"] = float(np.max(sum(powers.values())))
        if options.port_extension_policy != "reference":
            summary["reference"].update(
                reference_eligible=False,
                reference_agreement=None,
                protocol_limitation="Port extensions differ from the pinned paper setup",
            )
    device = jax.devices()[0]
    memory = device.memory_stats() or {}
    summary["environment"] = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "device": device.device_kind,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    summary["peak_host_rss_mib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
    summary["peak_jax_device_bytes_in_use"] = memory.get("peak_bytes_in_use")
    summary["device_memory_scope"] = (
        "JAX allocator only; excludes native CUDA allocations"
    )
    summary["monitor_data_sha256"] = hashlib.sha256(
        (args.output / "monitor_data.npz").read_bytes()
    ).hexdigest()
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("powers", "wavelengths_um")},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
