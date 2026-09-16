"""Size realized meshes; extrapolate host RSS without allocating FDTD fields.

The RSS extrapolation is a planning estimate, not a measured requirement or
confidence bound. Native CUDA memory is not included in JAX allocator statistics.
"""

import json
from pathlib import Path

import numpy as np

from scripts.investigate_passive_soi import build_experiment
from tests.differential.passive_soi.experiments import ExperimentOptions

HERE = Path(__file__).resolve().parent


def main():
    runs = json.loads((HERE / "runs.json").read_text())
    measured = [
        r for r in runs if r["device"] == "mmi2x2" and r["run"].startswith("mmi-fixed-")
    ]
    slope, intercept = np.polyfit(
        [r["cells"] / 1e6 for r in measured],
        [r["peak_host_rss_mib"] / 1024 for r in measured],
        1,
    )
    rows = []
    for device, ppws in [
        ("mmi2x2", (20, 25)),
        ("mode_converter", (15, 20)),
        ("polarization_splitter_rotator", (15, 20, 25)),
        ("ring_resonator", (10, 15, 20)),
    ]:
        for ppw in ppws:
            simulation, _, _, _ = build_experiment(device, ppw, ExperimentOptions())
            cells = int(np.prod(simulation.grid.shape))
            rows.append(
                {
                    "device": device,
                    "ppw": ppw,
                    "realized_grid_shape": list(simulation.grid.shape),
                    "cells": cells,
                    "estimated_process_host_rss_gib": float(
                        intercept + slope * cells / 1e6
                    ),
                }
            )
    report = {
        "method": "Linear host-RSS fit to three measured MMI runs versus actual cell count; extrapolated across devices, with no uncertainty model",
        "available_machine_host_ram_gib": 30,
        "available_machine_gpu_vram_gib": 24,
        "limitations": [
            "Host RSS estimates exclude the OS and other processes",
            "Different devices and solver/postprocessing paths can use different memory per cell",
            "GPU peak is not inferred from JAX-only allocator statistics",
            "No high-resolution FDTD fields were allocated for these estimates",
        ],
        "measured_mmi_runs": [
            {
                "run": r["run"],
                "cells": r["cells"],
                "peak_process_rss_gib": r["peak_host_rss_mib"] / 1024,
            }
            for r in measured
        ],
        "estimated_runs": rows,
    }
    (HERE / "resource_estimates.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
