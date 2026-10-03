"""Separate the port-termination intervention from reproduction of the paper setup."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()
    source = "reprojections.json" if args.latest else "runs.json"
    runs = {r["run"]: r for r in json.loads((HERE / source).read_text())}
    variants = {
        "converter-fixed-6ppw": "Reference stubs · corrected clock",
        "converter-ports-fixed-6ppw": "Extended stubs · automatic remesh",
        "converter-ports-fixed-frozen6": "Extended stubs · original frozen mesh",
    }
    if args.latest:
        variants = {
            "reprojected-converter-fixed-6ppw": "Reference stubs · both numerical fixes",
            "reprojected-converter-ports-fixed-frozen6": "Extended stubs · same mesh and fixes",
        }
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    metrics = {}
    for name, label in variants.items():
        if name not in runs:
            continue
        row = runs[name]
        raw = np.load(HERE / f"{name}.npz")
        incident = raw["incident_power"]
        wl = np.asarray(row["wavelengths_um"]) * 1000
        order = np.argsort(wl)
        selected = sum(np.asarray(v) for v in row["powers"].values())
        reflection = raw["diagnostic_o1__P_plus"] / incident
        output_backward = raw["diagnostic_o4__P_plus"] / incident
        values = [selected, row["powers"]["conversion"], reflection, output_backward]
        for ax, value in zip(axes.flat, values, strict=True):
            ax.plot(wl[order], np.asarray(value)[order], label=label)
        metrics[name] = {
            "selected_output_max": float(selected.max()),
            "input_reflection_max": float(reflection.max()),
            "o4_backward_power_max": float(output_backward.max()),
            "selected_outputs_plus_input_reflection_max": float(
                (selected + reflection).max()
            ),
            "physical_variant_matches_pinned_port_extensions": name
            in ("converter-fixed-6ppw", "reprojected-converter-fixed-6ppw"),
        }
    for ax, title in zip(
        axes.flat,
        [
            "Selected forward outputs",
            "Converted TE1 power",
            "Reflected power at input",
            "Backward-wave power at narrow output",
        ],
        strict=True,
    ):
        ax.set(title=title, xlabel="Wavelength (nm)", ylabel="Power / incident power")
        ax.grid(alpha=0.2)
    axes[0, 0].axhline(
        1.02, color="#444444", ls="--", lw=1, label="Existing selected-output bound"
    )
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Converter · 6 PPW · corrected clock in every run\nExtended-port variants change the pinned paper geometry"
    )
    prefix = "updated_" if args.latest else ""
    fig.savefig(HERE / f"{prefix}converter_port_intervention.png", dpi=170)
    (HERE / f"{prefix}converter_port_analysis.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
