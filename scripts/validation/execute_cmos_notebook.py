"""Execute the checked-in notebook with the current Python environment on a GPU."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]


class LocalKernelManager(KernelManager):
    @property
    def kernel_spec(self):
        return KernelSpec(
            argv=[
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ],
            display_name="BeamZ local Python",
            language="python",
        )


def main():
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.pop("BEAMZ_DOCS_TEST", None)
    os.environ["JAX_PLATFORMS"] = "cuda"
    notebook_path = ROOT / "examples/notebooks/cmos_rgb_sensor.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1200,
        kernel_manager_class=LocalKernelManager,
        resources={"metadata": {"path": str(ROOT)}},
        on_cell_start=lambda cell_index, **_: print(
            f"Executing cell {cell_index}", flush=True
        ),
    )
    start = time.perf_counter()
    try:
        client.execute()
    finally:
        nbformat.write(notebook, notebook_path)
    report = {
        "notebook": str(notebook_path.relative_to(ROOT)),
        "backend": "jax-cuda",
        "material_checksums": {
            name: hashlib.sha256(
                (ROOT / "examples/data/cmos_rgb" / name).read_bytes()
            ).hexdigest()
            for name in ("materials.json", "filters.json")
        },
        "seconds": time.perf_counter() - start,
        "code_cells_executed": sum(
            c.cell_type == "code" and c.execution_count is not None
            for c in notebook.cells
        ),
        "errors": sum(
            o.output_type == "error"
            for c in notebook.cells
            if c.cell_type == "code"
            for o in c.outputs
        ),
    }
    output = ROOT / "docs/reviews/cmos-reference-parity/notebook_execution.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
