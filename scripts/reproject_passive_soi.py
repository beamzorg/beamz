"""Reanalyze retained DFT fields without rerunning or changing the FDTD simulation."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from beamz import Simulation
from beamz.analysis import s_parameters
from beamz.analysis.data import AnalysisData
from beamz.design.discretization import build_material_grid
from beamz.design.grid import RectilinearGrid
from beamz.devices.monitors.monitors import ModeMonitor
from beamz.simulation.results import SimulationMetadata, material_region_for_monitor
from scripts.investigate_passive_soi import build_experiment
from tests.differential.passive_soi.experiments import ExperimentOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve prior analysis")
    source = json.loads((args.input / "summary.json").read_text())
    raw_path = args.input / "monitor_data.npz"
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if raw_hash != source["monitor_data_sha256"]:
        raise ValueError("retained monitor data does not match its recorded hash")
    raw = np.load(raw_path)
    simulation, ports, outputs, frequencies = build_experiment(
        source["device"], source["ppw"], ExperimentOptions(**source["options"])
    )
    grid = RectilinearGrid(*(raw[f"grid_{axis}_boundaries_m"] for axis in "xyz"))
    if any(
        not np.array_equal(old, new)
        for old, new in zip(simulation.grid.edges, grid.edges, strict=True)
    ):
        if simulation.design is None:
            raise ValueError(
                "a pre-rasterized control must reproduce the recorded grid exactly"
            )
        material = build_material_grid(
            simulation.design,
            grid,
            quality="balanced",
            smoothing=source["options"]["smoothing"],
        )
        simulation = Simulation(
            material_grid=material,
            sources=simulation.sources,
            monitors=simulation.monitors,
            boundaries=simulation.boundaries,
            run_time=simulation.run_time,
        )
    np.testing.assert_array_equal(frequencies, raw["frequencies_hz"])
    program = simulation.compile(num_steps=1, backend="jax")
    metadata = SimulationMetadata.from_simulation(
        simulation, runtime_fields=program.grid
    )
    inputs = {}
    for monitor in simulation.monitors:
        if not isinstance(monitor, ModeMonitor):
            continue
        fields = {
            c: raw[f"{monitor.name}__{c}"] for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        }
        port = next(p for p in ports if p.monitor_name == monitor.name)
        fields["flux"] = raw[f"flux_{port.name}__monitor_flux"]
        inputs[monitor.name] = AnalysisData(
            metadata,
            fields,
            material_region_for_monitor(
                simulation, monitor, runtime_fields=program.grid
            ),
            frequencies,
            monitor,
        )

    scattering = s_parameters(
        inputs,
        source_port="o1",
        ports=ports,
        output_ports=outputs,
        frequencies=frequencies,
        min_incident_db=-45,
    )
    if not np.all(scattering.diagnostics["valid_mask"]):
        raise ValueError("invalid incident signal in retained spectrum")
    powers = {key[0]: np.abs(v) ** 2 for key, v in scattering.s_matrix.items()}
    arrays = {
        f"S_{key[0]}_{key[1]}": np.asarray(v) for key, v in scattering.s_matrix.items()
    }
    for name, wave in scattering.diagnostics["waves"].items():
        for key in (
            "a_plus",
            "a_minus",
            "P_plus",
            "P_minus",
            "mode_neff",
            "projection_residual",
            "condition_number",
            "projected_signed_power",
        ):
            arrays[f"diagnostic_{name}__{key}"] = np.asarray(wave[key])
    for name, flux in scattering.diagnostics["monitor_flux_checks"].items():
        for key in (
            "monitor_flux",
            "P_modal_sum",
            "P_modal_net",
            "P_selected",
            "P_rejected",
            "P_selected_modal_net",
        ):
            arrays[f"flux_{name}__{key}"] = np.asarray(flux[key])
    for name, key in (
        ("valid_mask", "valid_mask"),
        ("incident_power", "P_in"),
        ("guided_output_power", "P_guided_out"),
        ("power_sum", "power_sum"),
        ("loss_estimate", "loss_est"),
    ):
        arrays[name] = np.asarray(scattering.diagnostics[key])
    arrays["frequencies_hz"] = np.asarray(frequencies)
    for axis in "xyz":
        arrays[f"grid_{axis}_boundaries_m"] = raw[f"grid_{axis}_boundaries_m"]
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite projection diagnostics")
    summary = {
        "field_run": str(args.input.resolve()),
        "field_data_sha256": raw_hash,
        "field_simulation_commit": source["commit"],
        "field_simulation_patch_sha256": source["working_tree_patch_sha256"],
        "analysis_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": source["device"],
        "ppw": source["ppw"],
        "options": source["options"],
        "wavelengths_um": source["wavelengths_um"],
        "previous_powers": source["powers"],
        "powers": {key: value.tolist() for key, value in powers.items()},
    }
    if source["device"].startswith(("straight_", "converter_grid_")):
        stack = np.stack(list(powers.values()))
        summary["max_transmission_error"] = float(np.max(np.abs(stack - 1)))
        summary["max_monitor_power_spread"] = float(np.max(np.ptp(stack, axis=0)))
    else:
        summary["selected_output_max"] = float(np.max(sum(powers.values())))
    args.output.mkdir(parents=True)
    patch = subprocess.check_output(["git", "diff", "HEAD"])
    (args.output / "analysis.patch").write_bytes(patch)
    summary["analysis_patch_sha256"] = hashlib.sha256(patch).hexdigest()
    np.savez_compressed(args.output / "reprojected_diagnostics.npz", **arrays)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("previous_powers", "powers", "wavelengths_um")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
