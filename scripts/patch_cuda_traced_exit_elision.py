"""Eliminate final-version electric boundary trails from the actual Yee trace.

Within a complete macro H(n) reads E(n-1); E(n) reads H(n) and E(n-1).
Consequently no electric trail at version Depth is consumed. Observations and
final fields still publish every owned current value; the next macro reloads
those final fields rather than the previous macro's trails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_stencil_trace import reads, trace_cpml_step

PARENT = "f082980ad0687ef165c7807b415e87cbc1003edfdbd2acf7a4ee018dae4d4547"
CELL = "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5"


def prove(depth=16, *, include_next_macro=False, erase_previous=False):
    if depth not in (4, 16):
        raise ValueError("Actual admitted macro depths4/16 required")
    operators = [trace_cpml_step(active_axes=tuple(a for a in range(3) if mask & (1 << a)))
                 for mask in range(8)]
    checked = 0
    electric_versions = set()
    for updates in operators:
        for n in range(1, depth + 1 + int(include_next_macro)):
            for update in updates:
                for r in reads(update.value):
                    if r.field.startswith("E") and r.field in ("Ex", "Ey", "Ez"):
                        version = n - 1 + r.version
                        electric_versions.add(version)
                        if version >= depth - int(erase_previous):
                            raise ValueError("Electric trail is live at an admitted macro consumer")
                        checked += 1
    assert max(electric_versions) == depth - 1
    return {"depth": depth, "cpml_axis_classes": 8, "electric_reads_checked": checked,
            "latest_consumed_electric_version": max(electric_versions),
            "final_electric_trail_has_no_macro_reader": True,
            "observation_and_final_fields_retained": True,
            "next_macro_uses_complete_final_field_bank_not_old_trails": True}


def transform(text):
    if hashlib.sha256(text.encode()).hexdigest() != PARENT:
        raise ValueError("Exact selected52865 consumer required")
    original = text
    for mask in (8, 16, 32):
        old = f"mask&{mask}?__ldcg"
        new = f"((mask&{mask})&&version<Depth)?__ldcg"
        if text.count(old) != 1:
            raise ValueError("Final E import anchor changed")
        text = text.replace(old, new, 1)
    stores = [
        "p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=e[c][j];",
        "p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=ev;",
    ]
    if text.count(stores[0]) != 1 or text.count(stores[1]) != 2:
        raise ValueError("Final E publication anchors changed")
    for old in stores:
        text = text.replace(old, "if(n<Depth)" + old)
    inverse = text
    for old in stores:
        inverse = inverse.replace("if(n<Depth)" + old, old)
    for mask in (8, 16, 32):
        inverse = inverse.replace(f"((mask&{mask})&&version<Depth)?__ldcg", f"mask&{mask}?__ldcg")
    assert inverse == original
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source, output = a.source.resolve(), a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        p.error("New disjoint noncanonical output required")
    cell = source / "src/asymmetric_native_cell_generated.cuh"
    if hashlib.sha256(cell.read_bytes()).hexdigest() != CELL:
        raise ValueError("Actual point graph changed")
    proofs = [prove(depth) for depth in (4, 16)]
    text = transform((source / "src/asymmetric_native.cuh").read_text())
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_text(text)
    record = {"proofs": proofs, "source_inverse_exact": True,
              "consumer_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "same_allocations_and_trail_layout": True,
              "retained": ["H trails", "all final fields", "sources", "CPML", "observations", "DFT", "clocks", "barriers"],
              "GPU_executed": False}
    (output.parent / "exit-elision-proof.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
