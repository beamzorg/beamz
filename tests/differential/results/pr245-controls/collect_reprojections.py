"""Export analysis-only reruns while preserving their original FDTD provenance."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from tests.differential.passive_soi.common import load_passive_soi_case
from tests.differential.passive_soi.experiments import power_comparison
from tests.differential.passive_soi.ring_resonator import extract_ring_resonances

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.runs:
        analysis = json.loads((path / "summary.json").read_text())
        field = json.loads((Path(analysis["field_run"]) / "summary.json").read_text())
        raw = np.load(path / "reprojected_diagnostics.npz")
        shutil.copyfile(path / "reprojected_diagnostics.npz", HERE / f"{path.name}.npz")
        row = {
            **analysis,
            "run": path.name,
            "field_run_name": Path(analysis["field_run"]).name,
            "termination": field["termination"],
            "run_time_ps": field["run_time_ps"],
            "field_backend": field["backend"],
            "grid_shape": field["grid_shape"],
        }
        row["input_reflection_max"] = float(
            np.max(raw["diagnostic_o1__P_plus"] / raw["incident_power"])
        )
        device = row["device"]
        if device == "ring_resonator":
            res, fsr, fwhm, q, extinction, _ = extract_ring_resonances(
                np.asarray(row["wavelengths_um"]), np.asarray(row["powers"]["o2"])
            )
            row["ring"] = {
                "resonances_um": res,
                "fsr_nm": fsr,
                "fwhm_nm": fwhm,
                "q": q,
                "extinction_db": extinction,
                "time_convergence_established": False,
            }
        elif device in ("mmi2x2", "mode_converter", "polarization_splitter_rotator"):
            case = load_passive_soi_case(device)
            channel = (
                case.geometry["simulation"]["cross_port"]
                if device == "mmi2x2"
                else "conversion"
            )
            center = int(np.argmin(np.abs(np.asarray(row["wavelengths_um"]) - 1.55)))
            row["power_1550nm"] = row["powers"][channel][center]
            row["reference"] = power_comparison(
                case,
                "cross" if device == "mmi2x2" else "conversion",
                row["ppw"],
                row["power_1550nm"],
            )
            if row["options"].get("port_extension_policy", "reference") != "reference":
                row["reference"].update(
                    reference_eligible=False,
                    reference_agreement=None,
                    protocol_limitation="Modified port-extension geometry",
                )
        rows.append(row)
    (HERE / "reprojections.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
