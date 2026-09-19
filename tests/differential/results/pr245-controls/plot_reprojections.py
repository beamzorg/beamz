"""Plot the current modal analysis, keeping field-run and analysis provenance separate."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = ["#236192", "#d97722", "#7a629e", "#6b7740"]


def main():
    rows = json.loads((HERE / "reprojections.json").read_text())
    field_runs = json.loads((HERE / "runs.json").read_text())
    refined_controls = [
        r for r in field_runs if r["run"] == "converter-grid-te0-10ppw-sampling-fixed"
    ]
    if refined_controls:
        row = refined_controls[0]
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        wl = np.asarray(row["wavelengths_um"]) * 1000
        order = np.argsort(wl)
        for port, power in row["powers"].items():
            ax.plot(wl[order], np.asarray(power)[order], "o-", label=port)
        ax.axhspan(0.99, 1.01, color=COLORS[0], alpha=0.08)
        ax.axhline(1, color="#444444", ls="--", lw=1)
        ax.set(
            title="Fresh TE0 control · exact converter grid · 10 PPW\nCorrected clock and modal sampling",
            xlabel="Wavelength (nm)",
            ylabel="Transmission / incident power",
            ylim=(0.988, 1.012),
        )
        ax.legend()
        ax.grid(alpha=0.2)
        fig.savefig(HERE / "refined_control.png", dpi=170)
        plt.close(fig)
    controls = [
        r for r in rows if r["device"].startswith("converter_grid_") and r["ppw"] == 6
    ]
    if controls:
        fig, axes = plt.subplots(
            len(controls),
            2,
            figsize=(12, 4 * len(controls)),
            constrained_layout=True,
            squeeze=False,
            sharey=True,
        )
        for pair, row in zip(axes, controls, strict=True):
            wl = np.asarray(row["wavelengths_um"]) * 1000
            order = np.argsort(wl)
            for ax, key, label in zip(
                pair,
                ("previous_powers", "powers"),
                ("Previous basis", "Basis sampled like the monitor"),
                strict=True,
            ):
                for idx, (port, value) in enumerate(row[key].items()):
                    ax.plot(
                        wl[order],
                        np.asarray(value)[order],
                        "o-",
                        color=COLORS[idx],
                        label=port,
                        ms=3,
                    )
                ax.axhspan(0.99, 1.01, color=COLORS[0], alpha=0.08)
                ax.axhline(1, color="#444444", ls="--", lw=1)
                ax.set(
                    title=f"{row['device']} · {label}",
                    xlabel="Wavelength (nm)",
                    ylabel="Transmission / incident power",
                )
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
        fig.suptitle(
            "Same retained FDTD fields · 6 PPW · exact converter grid\nShading: unchanged proposed ±1% calibration target"
        )
        fig.savefig(HERE / "normal_sampling_controls.png", dpi=170)
        plt.close(fig)
    devices = ("mmi2x2", "mode_converter", "polarization_splitter_rotator")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, device in zip(axes, devices, strict=True):
        matching = sorted(
            (
                r
                for r in rows
                if r["device"] == device
                and r["options"].get("port_extension_policy", "reference")
                == "reference"
            ),
            key=lambda r: r["ppw"],
        )
        if not matching:
            continue
        case = json.loads(
            (HERE.parents[1] / "cases" / f"passive_soi_{device}.json").read_text()
        )
        protocol = case["geometry"]["simulation"]
        observable = "cross" if device == "mmi2x2" else "conversion"
        config = protocol[f"published_converged_{observable}_power_1550nm_span20nm"]
        series = protocol[config["series_key"]]
        ppws = sorted(int(k) for k in series if k.isdigit())
        for solver, color in (("lumerical", "#888888"), ("tidy3d", COLORS[1])):
            ax.plot(
                ppws,
                [series[str(p)][solver] for p in ppws],
                "o--",
                color=color,
                label=f"Published {solver}",
                ms=4,
            )
        ax.plot(
            [r["ppw"] for r in matching],
            [r["power_1550nm"] for r in matching],
            "o-",
            color=COLORS[0],
            label="BeamZ · both numerical fixes",
        )
        ax.axhspan(
            *matching[0]["reference"]["converged_range"], color=COLORS[0], alpha=0.1
        )
        ax.set(
            title=device,
            xlabel="Nominal PPW (realized grids differ)",
            ylabel=f"{observable.capitalize()} power / incident power",
            ylim=(0, 1.05),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Reference port geometry · exact 1550 nm · latest modal analysis\nShading: independently declared converged reference envelope"
    )
    fig.savefig(HERE / "updated_mesh_sweeps.png", dpi=170)
    plt.close(fig)
    rings = sorted(
        (r for r in rows if r["device"] == "ring_resonator"),
        key=lambda r: r["run_time_ps"],
    )
    duration_pairs = []
    for first, second in zip(rings, rings[1:], strict=False):
        assert first["grid_shape"] == second["grid_shape"]
        assert first["wavelengths_um"] == second["wavelengths_um"]
        first_options = {"port_extension_policy": "reference", **first["options"]}
        second_options = {"port_extension_policy": "reference", **second["options"]}
        assert {k: v for k, v in first_options.items() if k != "run_time_ps"} == {
            k: v for k, v in second_options.items() if k != "run_time_ps"
        }
        a, b = first["ring"], second["ring"]
        same_count = len(a["resonances_um"]) == len(b["resonances_um"])
        drift = (
            float(np.max(np.abs(np.asarray(a["resonances_um"]) - b["resonances_um"])))
            * 1000
            if same_count
            else None
        )
        width_change = abs(b["fwhm_nm"] / a["fwhm_nm"] - 1)
        q_change = abs(b["q"] / a["q"] - 1)
        endpoints_valid = all(
            r["termination"]["field_decay"] <= 1e-5 and r["selected_output_max"] <= 1.02
            for r in (first, second)
        )
        actual_times = [r["termination"]["time"] for r in (first, second)]
        # Same grid/dt: integer steps avoid float32 timestamp roundoff at the
        # one-step allowance needed when a configured cap is rounded upward.
        duration_doubled = (
            second["termination"]["steps"] + 1 >= 2 * first["termination"]["steps"]
        )
        duration_pairs.append(
            {
                "runs": [first["run"], second["run"]],
                "actual_durations_ps": [t * 1e12 for t in actual_times],
                "actual_duration_doubled": duration_doubled,
                "fixed_source_prefix_invariance_verified": False,
                "relative_fwhm_change": width_change,
                "relative_q_change": q_change,
                "maximum_resonance_drift_nm": drift,
                "both_endpoints_physically_valid": endpoints_valid,
                "proposed_time_convergence_passed": endpoints_valid
                and duration_doubled
                and same_count
                and drift < 0.02
                and max(width_change, q_change) < 0.01,
            }
        )
    (HERE / "updated_duration_analysis.json").write_text(
        json.dumps(duration_pairs, indent=2, allow_nan=False) + "\n"
    )
    if rings:
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)
        for idx, row in enumerate(rings):
            wl = np.asarray(row["wavelengths_um"]) * 1000
            order = np.argsort(wl)
            actual_ps = row["termination"]["time"] * 1e12
            label = f"{actual_ps:.2f} ps (cap {row['run_time_ps']:g}) · {row['field_backend']}"
            axes[0].plot(
                wl[order],
                np.asarray(row["powers"]["o2"])[order],
                color=COLORS[idx % len(COLORS)],
                label=label,
            )
            axes[1].plot(
                wl[order],
                np.asarray(row["powers"]["o1"])[order],
                color=COLORS[idx % len(COLORS)],
                label=label,
            )
            axes[2].scatter(
                actual_ps,
                row["termination"]["field_decay"],
                color=COLORS[idx % len(COLORS)],
                label=label,
            )
        axes[0].axhline(1.02, color="#444444", ls="--", lw=1)
        axes[0].set(
            xlabel="Wavelength (nm)",
            ylabel="Through power / incident power",
            title="Corrected clock and modal sampling",
        )
        axes[1].set(
            xlabel="Wavelength (nm)",
            ylabel="Reflected power / incident power",
            title="Resonant reflection remains unresolved",
        )
        for control in (r for r in field_runs if r["run"] == "ring-bus-exact6p4"):
            wl = np.asarray(control["wavelengths_um"]) * 1000
            order = np.argsort(wl)
            data = np.load(HERE / f"{control['run']}.npz")
            reflection = data["diagnostic_o1__P_plus"] / data["incident_power"]
            axes[0].plot(
                wl[order],
                np.asarray(control["powers"]["o2"])[order],
                color="#777777",
                ls=":",
                label="Empty bus · same ring grid",
            )
            axes[1].plot(
                wl[order],
                reflection[order],
                color="#777777",
                ls=":",
                label="Empty bus · same ring grid",
            )
        axes[2].axhline(1e-5, color="#444444", ls="--", label="Required decay")
        axes[2].set(
            xlabel="Actual simulated duration (ps)",
            ylabel="Terminal field-energy ratio",
            yscale="log",
            title="Time-convergence evidence",
        )
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.savefig(HERE / "updated_ring_duration.png", dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    main()
