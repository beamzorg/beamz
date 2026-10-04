"""Execute the full modal or cosine-crossing tutorial and export numerical results.

Run once per checkout in a fresh process with that checkout on PYTHONPATH.
Only backend selection is changed; a final cell exports arrays for comparison.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("jax", "cuda_streamed"), default="cuda_streamed"
    )
    parser.add_argument(
        "--notebook",
        choices=("modal_sources_monitors", "cosine_waveguide_crossing"),
        default="modal_sources_monitors",
    )
    parser.add_argument(
        "--expected-device", help="Require this substring in the JAX GPU name"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.pop("BEAMZ_DOCS_TEST", None)
    os.environ.update(
        PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        PYTHONPATH=str(root),
        MPLBACKEND="module://matplotlib_inline.backend_inline",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        BEAMZ_EXECUTION_BACKEND=args.backend,
    )
    path = root / "examples/notebooks" / (args.notebook + ".ipynb")
    notebook = nbformat.read(path, as_version=4)
    if args.notebook == "modal_sources_monitors":
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.source = cell.source.replace(
                    ".run(progress=", f'.run(backend="{args.backend}", progress='
                )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            """
import json
import hashlib
import jax
import beamz._cuda as extension
assert not test_mode
assert any(device.platform == "gpu" for device in jax.devices())
if EXPECTED_DEVICE:
    assert EXPECTED_DEVICE in jax.devices()[0].device_kind
assert Path(bz.__file__).resolve().parent.parent == Path(ROOT)
assert Path(extension.__file__).resolve().parent.parent == Path(ROOT)
arrays = {}
for label, data in (
    ("single", sim_data_single), ("broadband", sim_data_bb),
    ("junction", sim_data_jct_bb),
):
    raw = data.renormalize(None)
    arrays[label + "_flux"] = np.asarray(raw["flux"].flux)
    arrays[label + "_amps"] = np.asarray(raw.mode("mode").amps)
    arrays[label + "_mode_flux"] = np.asarray(raw.mode("mode").flux)
    arrays[label + "_ey"] = np.asarray(raw["field"].dft_fields["Ey"])
    arrays[label + "_normalized_flux"] = np.asarray(data["flux"].flux)
    arrays[label + "_normalized_amps"] = np.asarray(data.mode("mode").amps)
    arrays[label + "_normalized_ey"] = np.asarray(data["field"].dft_fields["Ey"])
for name in (
    "flux_single", "flux_single_pulse", "mode_power_f_single", "mode_power_b_single",
    "mode_power_f_single_pulse", "flux_bb", "mode_power_f_bb", "mode_power_b_bb",
    "net_guided_single", "unresolved_single", "net_guided_bb", "unresolved_bb",
):
    arrays["plotted_" + name] = np.asarray(globals()[name])
arrays["plotted_junction_mode_power"] = np.abs(amps_jct_bb.values) ** 2
arrays["neffs"] = np.asarray(modes.neffs)
arrays["profile_freqs"] = np.asarray(profile_freqs)
arrays["freqs"] = np.asarray(freqs)
for name, value in arrays.items():
    assert np.isfinite(value).all(), name
np.savez_compressed(Path(OUTPUT) / "arrays.npz", **arrays)
metadata = {
    "backend": sim_single.compile(backend=BACKEND).config.backend,
    "grid_shape": list(sim0.grid.shape), "steps": sim0.num_steps,
    "nfreqs": nfreqs, "broadband_profiles": broadband_profile_count,
    "extension_version": extension.__version__,
    "extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
    "jax_version": jax.__version__, "python": sys.executable,
    "devices": [device.device_kind for device in jax.devices()],
    "metric_kind": sim0.grid.metric_kind,
    "arrays": {name: list(value.shape) for name, value in arrays.items()},
    "performance": {
        label: None if data.performance is None else {
            "runtime_s": data.performance.runtime_s, "gcups": data.performance.gcups,
            "steps": data.performance.steps, "cells": data.performance.cells,
        }
        for label, data in (("single", sim_data_single), ("broadband", sim_data_bb), ("junction", sim_data_jct_bb))
    },
}
(Path(OUTPUT) / "results.json").write_text(json.dumps(metadata, indent=2))
print(metadata)
""".replace("ROOT", repr(str(root)))
            .replace("OUTPUT", repr(str(output)))
            .replace("EXPECTED_DEVICE", repr(args.expected_device))
            .replace("BACKEND", repr(args.backend))
        )
    )
    if args.notebook == "cosine_waveguide_crossing":
        notebook.cells.pop()  # Replace the modal-specific export cell.
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.source = cell.source.replace(
                    'backend = "cuda_streamed"', f'backend = "{args.backend}"'
                )
        notebook.cells.append(
            nbformat.v4.new_code_cell(
                f"""
import json, hashlib, jax
import beamz._cuda as extension
assert not test_mode
assert Path(bz.__file__).resolve().parent.parent == Path({str(root)!r})
assert any(device.platform == "gpu" for device in jax.devices())
assert Path(extension.__file__).resolve().parent.parent == Path({str(root)!r})
if {args.expected_device!r}:
    assert {args.expected_device!r} in jax.devices()[0].device_kind
raw = sim_data.renormalize(None)
arrays = {{
    "raw_flux_through": np.asarray(raw["flux_through"].flux),
    "raw_flux_cross": np.asarray(raw["flux_cross"].flux),
    "through": np.asarray(T_through), "cross": np.asarray(T_cross),
    "neffs": np.asarray(modes.neffs), "freqs": np.asarray(freqs),
    "launched_power": np.asarray(source_power),
}}
arrays.update({{"raw_field_" + k: np.asarray(v) for k,v in raw["field"].dft_fields.items()}})
for name,value in arrays.items():
    assert np.isfinite(value).all(), name
np.savez_compressed(Path({str(output)!r}) / "arrays.npz", **arrays)
metadata = {{
    "backend": backend, "grid_shape": list(sim.grid.shape), "steps": sim.num_steps,
    "nfreqs": len(freqs), "jax_version": jax.__version__,
    "extension_version": extension.__version__,
    "extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
    "devices": [d.device_kind for d in jax.devices()],
    "metric_kind": sim.grid.metric_kind,
    "performance": None if sim_data.performance is None else {{
        "runtime_s": sim_data.performance.runtime_s, "gcups": sim_data.performance.gcups,
    }},
}}
(Path({str(output)!r}) / "results.json").write_text(json.dumps(metadata, indent=2))
print(metadata)
"""
            )
        )
    start = time.monotonic()
    client = NotebookClient(
        notebook,
        timeout=3600,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
        on_cell_start=lambda cell, cell_index: print(
            f"cell {cell_index}: {time.monotonic() - start:.1f}s", flush=True
        ),
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output / (args.notebook + ".ipynb"))
        (output / "provenance.json").write_text(
            json.dumps(
                {
                    "root": str(root),
                    "commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=root, text=True
                    ).strip(),
                    "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "elapsed_s": time.monotonic() - start,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
