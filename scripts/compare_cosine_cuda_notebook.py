"""Execute an instrumented full cosine notebook against a pinned frontend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=("main", "bounded"), required=True)
    p.add_argument("--module", type=Path)
    p.add_argument("--notebook", type=Path)
    p.add_argument(
        "--native-memory-policy",
        choices=("preserve", "host-reset"),
        default="host-reset",
    )
    p.add_argument("--warm-runs", type=int, default=3)
    a = p.parse_args()
    root = a.root.resolve()
    output = a.output.resolve()
    output.mkdir(exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    os.environ.pop("BEAMZ_DOCS_TEST", None)
    os.environ.update(
        PYTHONPATH=str(root) + os.pathsep + str(repo),
        MPLBACKEND="module://matplotlib_inline.backend_inline",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        BEAMZ_EXECUTION_BACKEND="cuda_streamed",
        PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
    )
    os.environ["BEAMZ_NATIVE_HOST_RESET"] = str(
        int(a.variant == "bounded" and a.native_memory_policy == "host-reset")
    )
    if a.variant == "bounded" and a.native_memory_policy == "host-reset":
        os.environ["BEAMZ_SETUP_DEVICE"] = "cpu"
    os.environ["BEAMZ_TRACE_PREPARATION"] = str(output / "frontend-preparation.jsonl")
    original = (
        a.notebook.resolve()
        if a.notebook
        else root / "examples/notebooks/cosine_waveguide_crossing.ipynb"
    )
    notebook = nbformat.read(original, as_version=4)
    bootstrap = f"""import sys,time\nfrom pathlib import Path\nsys.path.insert(0,{str(root)!r})\nsys.path.insert(1,{str(repo)!r})\nfrom scripts.cuda_notebook_runtime import Trial\n_trial=Trial({a.variant!r},{str(output)!r},{str(a.module.resolve()) if a.module else None!r})\n"""
    notebook.cells.insert(0, nbformat.v4.new_code_cell(bootstrap))
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            '_trial.export(globals())\nprint("Full original notebook completed")'
        )
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell(f"""for _repeat in range({a.warm_runs}):
    _repeated=sim.run(progress=True,backend=backend)
    for _name,_expected in sim_data.renormalize(None)["field"].dft_fields.items():
        np.testing.assert_array_equal(_repeated.renormalize(None)["field"].dft_fields[_name],_expected)
    del _repeated
_summary=_trial.export(globals(),finish=True)
print({{"notebook_wall_s":_summary["notebook_wall_s"],"runs":len(_summary["runs"]),"peak_gpu_process_bytes":_summary["memory"]["peak_gpu_process_bytes"]}})
""")
    )
    start = time.perf_counter()
    timings = []

    def begin(cell, cell_index, **kwargs):
        timings.append(dict(index=cell_index, start_s=time.perf_counter() - start))
        print(
            json.dumps(dict(cell=cell_index, elapsed_s=time.perf_counter() - start)),
            flush=True,
        )

    client = NotebookClient(
        notebook,
        timeout=3600,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
        on_cell_start=begin,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output / "cosine_waveguide_crossing.ipynb")
        (output / "execution.json").write_text(
            json.dumps(
                dict(
                    elapsed_with_repeats_s=time.perf_counter() - start,
                    root=str(root),
                    variant=a.variant,
                    native_memory_policy=(
                        a.native_memory_policy if a.variant == "bounded" else None
                    ),
                    notebook_path=str(original),
                    original_cell_count=len(notebook.cells) - 3,
                    notebook_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                    cell_starts=timings,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
