"""Report measured convergence without treating a single passing point as a sweep."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    runs = json.loads((HERE / "runs.json").read_text())
    report = {"ring_duration_pairs": [], "mesh_sweeps": {}}
    rings = sorted(
        (r for r in runs if r["run"].startswith("ring-fixed-")),
        key=lambda r: r["run_time_ps"],
    )
    for first, second in zip(rings, rings[1:], strict=False):
        assert first["grid_shape"] == second["grid_shape"]
        assert first["wavelengths_um"] == second["wavelengths_um"]
        assert {k: v for k, v in first["options"].items() if k != "run_time_ps"} == {
            k: v for k, v in second["options"].items() if k != "run_time_ps"
        }
        a, b = first["ring"], second["ring"]
        same_count = len(a["resonances_um"]) == len(b["resonances_um"])
        drift = (
            float(
                np.max(np.abs(np.array(a["resonances_um"]) - b["resonances_um"])) * 1000
            )
            if same_count
            else None
        )
        width_change = abs(b["fwhm_nm"] / a["fwhm_nm"] - 1)
        q_change = abs(b["q"] / a["q"] - 1)
        endpoints_valid = all(
            r["termination"]["field_decay"] <= 1e-5 and r["selected_output_max"] <= 1.02
            for r in (first, second)
        )
        report["ring_duration_pairs"].append(
            {
                "runs": [first["run"], second["run"]],
                "backends": [first["backend"], second["backend"]],
                "relative_fwhm_change": width_change,
                "relative_q_change": q_change,
                "maximum_resonance_drift_nm": drift,
                "both_endpoints_physically_valid": endpoints_valid,
                "proposed_time_convergence_passed": endpoints_valid
                and same_count
                and drift < 0.02
                and max(width_change, q_change) < 0.01,
            }
        )
    devices = ["mmi2x2", "mode_converter", "polarization_splitter_rotator"]
    for device in devices:
        rows = sorted(
            (
                r
                for r in runs
                if r["device"] == device
                and "fixed" in r["run"]
                and "ports-fixed" not in r["run"]
            ),
            key=lambda r: r["ppw"],
        )
        if not rows:
            continue
        case = json.loads(
            (HERE.parents[1] / "cases" / f"passive_soi_{device}.json").read_text()
        )
        protocol = case["geometry"]["simulation"]
        observable = "cross" if device == "mmi2x2" else "conversion"
        config = protocol[f"published_converged_{observable}_power_1550nm_span20nm"]
        series = protocol[config["series_key"]]
        channel = protocol["cross_port"] if device == "mmi2x2" else "conversion"
        values = [
            r["powers"][channel][
                int(np.argmin(np.abs(np.array(r["wavelengths_um"]) - 1.55)))
            ]
            for r in rows
        ]
        report["mesh_sweeps"][device] = [
            {
                "run": r["run"],
                "ppw": r["ppw"],
                "power_1550nm": v,
                "selected_output_max": r["selected_output_max"],
                "field_decay": r["termination"]["field_decay"],
                "reference": r["reference"],
            }
            for r, v in zip(rows, values, strict=True)
        ]
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        ppws = sorted(int(k) for k in series if k.isdigit())
        for solver, color in (("lumerical", "#888888"), ("tidy3d", "#d97722")):
            ax.plot(
                ppws,
                [series[str(p)][solver] for p in ppws],
                "o--",
                color=color,
                label=f"Published {solver}",
            )
        ax.plot(
            [r["ppw"] for r in rows],
            values,
            "o-",
            color="#236192",
            label="BeamZ · corrected clock",
        )
        lower, upper = rows[0]["reference"]["converged_range"]
        ax.axhspan(
            lower,
            upper,
            alpha=0.1,
            color="#236192",
            label="Converged reference envelope + digitization allowance",
        )
        ax.set(
            xlabel="Nominal points per wavelength (realized grids differ)",
            ylabel=f"{observable.capitalize()} power / incident power",
            title=f"{device}: mesh-refinement evidence",
            ylim=(0, 1.05),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.savefig(HERE / f"{device}_mesh.png", dpi=170)
        plt.close(fig)
    (HERE / "sweep_analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
