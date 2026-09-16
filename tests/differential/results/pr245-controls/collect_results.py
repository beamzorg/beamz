"""Export compact, inspectable diagnostics from explicitly selected local runs."""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    rows = []
    for path in args.runs:
        summary = json.loads((path / "summary.json").read_text())
        raw = np.load(path / "monitor_data.npz")
        compact = {
            key: raw[key]
            for key in raw.files
            if key.startswith(("diagnostic_", "flux_", "grid_", "S_"))
            or key in ("frequencies_hz", "incident_power", "valid_mask")
        }
        np.savez_compressed(HERE / f"{path.name}.npz", **compact)
        metadata = json.loads((path / "run_metadata.json").read_text())
        row = {
            "run": path.name,
            "raw_artifact_path": str(path.resolve()),
            **summary,
            "monitor_centers_m": metadata["monitor_centers_m"],
            "time_step_s": float(np.diff(raw["source_time_s"][:2])[0]),
        }
        if summary["device"].startswith(("straight_", "converter_grid_")):
            incident = raw["incident_power"]
            row["reflection_max"] = float(
                np.max(raw["diagnostic_o1__P_plus"] / incident)
            )
            row["max_modal_flux_relative_error"] = float(
                max(
                    np.max(
                        np.abs(
                            raw[f"diagnostic_{port}__projected_signed_power"]
                            - raw[f"flux_{port}__monitor_flux"]
                        )
                        / incident
                    )
                    for port in ("o1", "near", "middle", "far")
                )
            )
        rows.append(row)
    (HERE / "runs.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
