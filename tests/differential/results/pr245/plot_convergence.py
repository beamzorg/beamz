"""Regenerate the three-device mesh comparison from retained results and manifests."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DEVICES = {
    "mmi2x2": ("MMI cross TE0", "cross"),
    "mode_converter": ("Mode converter TE1", "conversion"),
    "polarization_splitter_rotator": (
        "Polarization splitter-rotator TE0",
        "conversion",
    ),
}


def main():
    records = json.loads((HERE / "results.json").read_text())["runs"]
    data = {
        "wavelength_nm": 1550,
        "source_bandwidth_nm": 20,
        "power_unit": "percent of incident power",
        "note": "Retained RTX3090 fields reprojected without the extra modal normal-interpolation correction. Published samples include figure digitizations. Equal nominal PPW does not imply equal realized meshes. These sweeps do not establish complete mesh convergence.",
        "devices": {},
    }
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True
    )
    for ax, (device, (title, observable)) in zip(axes, DEVICES.items(), strict=True):
        case = json.loads(
            (HERE.parents[1] / "cases" / f"passive_soi_{device}.json").read_text()
        )
        protocol = case["geometry"]["simulation"]
        reference = protocol[f"published_converged_{observable}_power_1550nm_span20nm"]
        series = protocol[reference["series_key"]]
        published = [
            {
                "ppw": ppw,
                "lumerical": 100 * series[str(ppw)]["lumerical"],
                "tidy3d": 100 * series[str(ppw)]["tidy3d"],
            }
            for ppw in sorted(int(key) for key in series if key.isdigit())
        ]
        measured = [
            {
                "run_id": row["run_id"],
                "ppw": row["ppw"],
                "power": 100 * row["target_mode_power"],
            }
            for row in sorted(records, key=lambda row: row["ppw"])
            if row["device"] == device
        ]
        data["devices"][device] = {
            "target_mode": title,
            "published_value_basis": series["value_basis"],
            "published": published,
            "beamz": measured,
        }
        for solver, label, color in (
            ("lumerical", "Published Lumerical", "#777777"),
            ("tidy3d", "Published Tidy3D", "#d97722"),
        ):
            ax.plot(
                [row["ppw"] for row in published],
                [row[solver] for row in published],
                "o--",
                color=color,
                label=label,
                markersize=4,
            )
        ax.plot(
            [row["ppw"] for row in measured],
            [row["power"] for row in measured],
            "o-",
            color="#236192",
            label="BeamZ RTX3090 (reprojected)",
        )
        ax.set(
            title=title,
            xlabel="Nominal cells per wavelength",
            xticks=[6, 10, 15, 20, 25],
            ylim=(0, 105),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Target-mode power / incident power (%)")
    fig.suptitle(
        "Three-device mesh refinement · 1550 nm · 20 nm source bandwidth\n"
        "Pinned geometry; realized meshes differ between engines",
        fontsize=14,
    )
    fig.savefig(HERE / "convergence.png", dpi=170)
    plt.close(fig)
    (HERE / "convergence.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
