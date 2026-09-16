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
    if rings:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for idx, row in enumerate(rings):
            wl = np.asarray(row["wavelengths_um"]) * 1000
            order = np.argsort(wl)
            label = f"{row['run_time_ps']:g} ps · {row['field_backend']}"
            axes[0].plot(
                wl[order],
                np.asarray(row["powers"]["o2"])[order],
                color=COLORS[idx % len(COLORS)],
                label=label,
            )
            axes[1].scatter(
                row["run_time_ps"],
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
        axes[1].axhline(1e-5, color="#444444", ls="--", label="Required decay")
        axes[1].set(
            xlabel="Configured duration (ps)",
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
