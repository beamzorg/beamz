"""Estimate normal interpolation bias without fitting or changing measured powers.

A single propagating mode is assumed, with beta = 2*pi*neff/lambda. This model
omits vector impedance errors, discrete dispersion, radiation, and attenuation.
It is a diagnostic prediction, not a correction to the simulation or acceptance.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    rows = json.loads((HERE / "runs.json").read_text())
    report = {}
    for row in rows:
        if not row["device"].startswith("converter_grid_"):
            continue
        raw = np.load(HERE / f"{row['run']}.npz")
        edges = raw["grid_x_boundaries_m"]
        centers = (edges[:-1] + edges[1:]) / 2
        factors = {}
        for port in ("o1", "near", "middle", "far"):
            x = row["monitor_centers_m"][port][0]
            beta = (
                2
                * np.pi
                * raw["frequencies_hz"]
                / 299792458
                * raw[f"diagnostic_{port}__mode_neff"].real
            )
            transfers = []
            for coords in (edges, centers):
                k = np.searchsorted(coords, x)
                a, b = coords[k - 1 : k + 1]
                weight = (x - a) / (b - a)
                transfers.append(
                    (1 - weight) * np.exp(1j * beta * (a - x))
                    + weight * np.exp(1j * beta * (b - x))
                )
            factors[port] = np.abs((transfers[0] + transfers[1]) / 2) ** 2
        report[row["run"]] = {}
        for port in ("near", "middle", "far"):
            prediction = factors[port] / factors["o1"]
            measured = np.array(row["powers"][port])
            report[row["run"]][port] = {
                "predicted_transmission_ratio": prediction.tolist(),
                "measured_transmission_ratio": measured.tolist(),
                "maximum_absolute_prediction_error": float(
                    np.max(np.abs(prediction - measured))
                ),
            }
    (HERE / "plane_sampling_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
