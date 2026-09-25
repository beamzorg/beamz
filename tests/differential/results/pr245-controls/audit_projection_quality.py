"""Expose field-reconstruction residuals without treating them as power errors."""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    rows = json.loads((HERE / "reprojections.json").read_text())
    rows += [
        r
        for r in json.loads((HERE / "runs.json").read_text())
        if r["run"] == "converter-grid-te0-10ppw-sampling-fixed"
        or r["device"] == "ring_bus"
    ]
    report = {
        "definition": "Field-vector reconstruction norm divided by measured field-vector norm; not a power-fraction error bound.",
        "limitations": [
            "Unretained polarization/modes and weak signals can produce large relative residuals.",
            "Condition number near one does not establish completeness of the retained basis.",
            "Residuals alone do not identify the physical source of unrepresented fields.",
        ],
        "runs": [],
    }
    for row in rows:
        arrays = np.load(HERE / f"{row['run']}.npz")
        wavelengths = np.asarray(row["wavelengths_um"])
        center = int(np.argmin(abs(wavelengths - 1.55)))
        ports = {}
        for key in arrays.files:
            if not key.endswith("__projection_residual"):
                continue
            port = key.split("__")[0].removeprefix("diagnostic_")
            residual = arrays[key]
            index = int(np.argmax(residual))
            power = row["powers"].get(port)
            ports[port] = {
                "maximum_residual": float(residual[index]),
                "wavelength_at_maximum_nm": float(wavelengths[index] * 1000),
                "residual_at_1550nm": float(residual[center]),
                "selected_output_power_at_maximum": None
                if power is None
                else power[index],
                "maximum_condition_number": float(
                    arrays[f"diagnostic_{port}__condition_number"].max()
                ),
            }
        report["runs"].append({"run": row["run"], "ports": ports})
    (HERE / "projection_quality.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
