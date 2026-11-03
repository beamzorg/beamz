"""Recompute saved modal-source regression metrics and figures.

Run this file with NumPy and Matplotlib installed. It reads the adjacent main/
and branch/ arrays, execution summaries and launch diagnostics. Source revisions
and the notebook's validation scope are recorded in comparison.json.
"""

# Select a headless backend before importing plotting helpers.
# ruff: noqa: E402
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

root = Path(__file__).resolve().parent
main = np.load(root / "main/arrays.npz")
branch = np.load(root / "branch/arrays.npz")
metadata = {
    label: json.loads((root / label / "summary.json").read_text())
    for label in ["main", "branch"]
}
for key in [
    "cuda_sha256",
    "raster_sha256",
    "notebook_code_sha256",
    "backend",
    "grid_shape_xyz",
    "steps",
    "nfreqs",
    "broadband_profiles",
    "python_version",
    "jax_version",
    "numpy_version",
    "device",
]:
    assert metadata["main"][key] == metadata["branch"][key], key
for key in [
    "freqs",
    "profile_freqs",
    "time",
    "grid_edges_x",
    "grid_edges_y",
    "grid_edges_z",
    "preview_edges_0",
    "preview_edges_1",
]:
    np.testing.assert_array_equal(main[key], branch[key])
wave = 299792458 / main["freqs"] / 1e-6
index = len(wave) // 2
results = {}
for case in ["single", "broadband", "junction"]:
    a, b = main[case + "_field_ey"], branch[case + "_field_ey"]
    overlap = np.vdot(a, b)
    phase = overlap / abs(overlap)
    result = {"main": {}, "branch": {}}
    for label, data in [("main", main), ("branch", branch)]:
        flux = data[case + "_flux"]
        forward = data[case + "_power_forward"]
        backward = data[case + "_power_backward"]
        result[label] = {
            "central_flux_w": float(flux[index]),
            "central_forward_power_w": forward[index].tolist(),
            "central_modal_flux_w": float(data[case + "_modal_flux"][index]),
            "maximum_mode0_flux_relative_difference": float(
                np.max(abs(forward[:, 0] - flux) / abs(flux))
            ),
            "maximum_backward_mode0_power_w": float(np.max(backward[:, 0])),
            "native_runtime_s": metadata[label]["performance"][case]["runtime_s"],
        }
    result["max_absolute_flux_change_w"] = float(
        np.max(abs(branch[case + "_flux"] - main[case + "_flux"]))
    )
    result["max_relative_raw_flux_change"] = float(
        np.max(abs(branch[case + "_raw_flux"] / main[case + "_raw_flux"] - 1))
    )
    result["max_relative_flux_change"] = float(
        np.max(abs(branch[case + "_flux"] / main[case + "_flux"] - 1))
    )
    result["max_absolute_mode0_power_change_w"] = float(
        np.max(
            abs(
                branch[case + "_power_forward"][:, 0]
                - main[case + "_power_forward"][:, 0]
            )
        )
    )
    result["max_absolute_mode_power_change_w"] = float(
        np.max(abs(branch[case + "_power_forward"] - main[case + "_power_forward"]))
    )
    result["phase_aligned_field_relative_l2_change"] = float(
        np.linalg.norm(b - phase * a) / np.linalg.norm(a)
    )
    result["intensity_relative_l2_change"] = float(
        np.linalg.norm(abs(b) ** 2 - abs(a) ** 2) / np.linalg.norm(abs(a) ** 2)
    )
    results[case] = result
summary = {
    "main_revision": metadata["main"]["revision"],
    "branch_revision": metadata["branch"]["revision"],
    "all_notebook_assertions_passed": {
        k: v["all_cells_passed"] for k, v in metadata.items()
    },
    "elapsed_seconds": {k: v["elapsed_seconds"] for k, v in metadata.items()},
    "controlled_inputs_match": True,
    "effective_indices": {
        "main": main["neffs"].real.tolist(),
        "branch": branch["neffs"].real.tolist(),
        "max_absolute_change": float(np.max(abs(main["neffs"] - branch["neffs"]))),
    },
    "cases": results,
    "notes": [
        "Both notebooks ran with default settings: 17 frequency bins and seven broadband source profiles.",
        "Each run uses the same Python environment, native CUDA/raster binaries and GPU; the notebook executable statements are identical.",
        "Field differences are reported after aligning one global phase; intensity comparisons need no phase alignment.",
        "Higher-mode indices can change character or ordering when the basis changes. Junction mode-0 power does not represent all transmitted power.",
        "Runtime values are individual runs, not a performance benchmark.",
    ],
}
# Only the y direction is a physical mirror symmetry; the vertical air/substrate stack is asymmetric.
for label, data in [("main", main), ("branch", branch)]:
    ey = abs(data["preview_e_fields"][0, 0, 1])
    summary.setdefault("preview_y_mirror_error", {})[label] = float(
        np.linalg.norm(ey - ey[:, ::-1]) / np.linalg.norm(ey)
    )
for label in ["main", "branch"]:
    diagnostic = root / label / "launch_diagnostics.json"
    if diagnostic.exists():
        launch = json.loads(diagnostic.read_text())[0]
        summary.setdefault("compiled_launch", {})[label] = {
            k: v for k, v in launch.items() if k != "refinement_validation"
        }
if set(summary.get("compiled_launch", {})) == {"main", "branch"}:
    summary["compiled_launch"]["absolute_phase_index_change"] = abs(
        summary["compiled_launch"]["main"]["launch_phase_neff"]
        - summary["compiled_launch"]["branch"]["launch_phase_neff"]
    )
    summary["compiled_launch"]["interpretation"] = (
        "Raw preview indices change with corrected sampling. Both versions accept Yee launch refinement and converge to almost identical propagation constants."
    )
for label in ["main", "branch"]:
    summary["cases"]["junction"][label].pop("maximum_mode0_flux_relative_difference")
summary["cases"]["junction"]["interpretation"] = (
    "Multimode junction: fundamental-mode power alone is not expected to equal total flux."
)
summary["scope"] = (
    "Three full default notebook simulations on one geometry family; these checks do not establish accuracy for every device or continuum convergence."
)
(root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
blue = "#2166ac"
orange = "#d47b17"
grey = "#555555"
# Direct flux versus modal reconstruction, with identical encodings in each case.
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for ax, case, title in zip(
    axes,
    ["single", "broadband"],
    ["Single-profile source", "Seven-profile broadband source"],
    strict=True,
):
    for label, data, color, style in [
        ("main", main, grey, "--"),
        ("branch", branch, blue, "-"),
    ]:
        ax.plot(
            wave,
            data[case + "_flux"],
            color=color,
            ls=style,
            lw=2,
            label=f"{label}: flux",
        )
        ax.plot(
            wave,
            data[case + "_power_forward"][:, 0],
            color=color,
            ls=":",
            marker="o" if label == "branch" else None,
            markersize=3,
            label=f"{label}: mode 0",
        )
    ax.axhline(1, color="#aaaaaa", lw=0.8)
    ax.set(xlabel="Wavelength (µm)", ylabel="Power (W)", title=title)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(root / "straight-guide-spectra.png", dpi=180)
# Absolute branch-minus-main changes expose small differences hidden by overlaid curves.
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for case, color in [("single", blue), ("broadband", orange)]:
    axes[0].plot(
        wave,
        1e6 * (branch[case + "_flux"] - main[case + "_flux"]),
        color=color,
        lw=2,
        label=case,
    )
    for label, data, style in [("main", main, "--"), ("branch", branch, "-")]:
        error = 100 * abs(
            data[case + "_power_forward"][:, 0] / data[case + "_flux"] - 1
        )
        axes[1].plot(wave, error, color=color, ls=style, lw=2, label=f"{case}: {label}")
axes[0].axhline(0, color="#888888", lw=0.8)
axes[0].set(
    xlabel="Wavelength (µm)",
    ylabel="Branch − main flux (µW)",
    title="Straight-guide flux changes",
)
axes[1].set(
    xlabel="Wavelength (µm)",
    ylabel="Mode-0 / flux difference (%)",
    title="Modal power consistency",
)
for ax in axes:
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(root / "straight-guide-differences.png", dpi=180)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for label, data, style in [("main", main, "--"), ("branch", branch, "-")]:
    axes[0].plot(
        wave,
        data["junction_flux"],
        color=grey,
        ls=style,
        lw=2,
        label=f"{label}: total flux",
    )
    axes[0].plot(
        wave,
        data["junction_power_forward"][:, 0],
        color=blue,
        ls=style,
        lw=2,
        label=f"{label}: mode 0",
    )
    for mode, color in [(1, orange), (2, "#8671a9")]:
        axes[1].plot(
            wave,
            data["junction_plotted_power"][:, mode],
            color=color,
            ls=style,
            lw=2,
            label=f"{label}: mode {mode}",
        )
axes[0].set(
    xlabel="Wavelength (µm)", ylabel="Power (W)", title="Widened-waveguide junction"
)
axes[1].set(
    xlabel="Wavelength (µm)",
    ylabel="Normalized forward modal power (W)",
    title="Additional sampled modes",
)
for ax in axes:
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(root / "junction-spectra.png", dpi=180)
fig, axes = plt.subplots(
    1, 2, figsize=(9, 3.6), layout="constrained", sharex=True, sharey=True
)
for ax, label, data in zip(axes, ["main", "branch"], [main, branch], strict=True):
    ey = abs(data["preview_e_fields"][0, 0, 1])
    y = data["preview_edges_1"] / 1e-6
    z = data["preview_edges_0"] / 1e-6
    mesh = ax.pcolormesh(
        y, z, ey / ey.max(), shading="flat", cmap="magma", vmin=0, vmax=1
    )
    ax.add_patch(
        Rectangle((-0.225, 0), 0.45, 0.22, fill=False, ec="#bfbfbf", ls="--", lw=1)
    )
    ax.set(
        xlabel="y (µm)",
        ylabel="z (µm)",
        xlim=(-0.7, 0.7),
        ylim=(-0.45, 0.6),
        title=f"{label}: input mode |Ey| / max",
    )
fig.colorbar(mesh, ax=axes, label="Normalized |Ey|", shrink=0.85)
fig.savefig(root / "input-mode-comparison.png", dpi=180)
print(json.dumps({k: v for k, v in summary.items() if k != "notes"}, indent=2))
