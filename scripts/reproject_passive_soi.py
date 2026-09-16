"""Reanalyze retained DFT fields without rerunning or changing the FDTD simulation."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from beamz import Simulation
from beamz.analysis import mode_projection, s_parameters
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
    parser.add_argument("--verify-previous-unsampled-basis", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve prior analysis")
    source = json.loads((args.input / "summary.json").read_text())
    raw_path = args.input / "monitor_data.npz"
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

    def project():
        return s_parameters(
            inputs,
            source_port="o1",
            ports=ports,
            output_ports=outputs,
            frequencies=frequencies,
            min_incident_db=-45,
        )

    baseline_error = None
    if args.verify_previous_unsampled_basis:
        original = mode_projection._normal_mode_sampling_factors_3d
        try:
            mode_projection._normal_mode_sampling_factors_3d = (
                lambda sim, monitor, components, **kw: {c: 1.0 for c in components}
            )
            previous = project()
        finally:
            mode_projection._normal_mode_sampling_factors_3d = original
        baseline_error = max(
            float(np.max(np.abs(np.abs(v) ** 2 - source["powers"][key[0]])))
            for key, v in previous.s_matrix.items()
        )
        if baseline_error > 1e-8:
            raise ValueError(
                f"reconstructed previous analysis differs by {baseline_error}; do not attribute this solely to sampling"
            )
    corrected = project()
    powers = {key[0]: np.abs(v) ** 2 for key, v in corrected.s_matrix.items()}
    arrays = {
        f"S_{key[0]}_{key[1]}": np.asarray(v) for key, v in corrected.s_matrix.items()
    }
    for name, wave in corrected.diagnostics["waves"].items():
        for key in (
            "a_plus",
            "a_minus",
            "P_plus",
            "P_minus",
            "mode_neff",
            "mode_wave_number",
            "projection_residual",
            "condition_number",
        ):
            arrays[f"diagnostic_{name}__{key}"] = np.asarray(wave[key])
    arrays["incident_power"] = np.asarray(corrected.diagnostics["P_in"])
    arrays["frequencies_hz"] = np.asarray(frequencies)
    summary = {
        "field_run": str(args.input.resolve()),
        "field_data_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "field_simulation_commit": source["commit"],
        "analysis_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": source["device"],
        "ppw": source["ppw"],
        "options": source["options"],
        "wavelengths_um": source["wavelengths_um"],
        "previous_analysis_max_reconstruction_error": baseline_error,
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
