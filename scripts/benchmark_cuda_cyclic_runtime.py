#!/usr/bin/env python3
"""Native large worker plus independent inverse-world readback comparisons.

All CLI arguments are those of benchmark_cuda_large_runtime.py. --fixture is
the immutable internally rotated fixture; its bound world fixture provides
the independent world references. Timed solve calls are unmodified. Host
conversion and the additional world checks occur outside those solve calls.
The ordinary identical-fixture comparator is intentionally unchanged.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import benchmark_cuda_large_runtime as base
from scripts.cuda_cyclic_fixture import transform_leaf
from scripts.cuda_fixture_store import ArrayReader, compare_arrays, drop_pages
from scripts.export_fdtdx_comparison import file_digest
from scripts.prove_cuda_cyclic_axis_equivariance import AxisMap

ORIGINAL_NATIVE = base.native
ORIGINAL_WORKER = Path(base.__file__).resolve()


def native(args, fixture, store, inputs, chunks):
    if args.method != "moving":
        raise ValueError("The cyclic adapter admits moving and its ordinary in-place tail only")
    layout = fixture.get("cyclic_layout", {})
    if layout.get("schema") != "beamz.cyclic-storage-layout.v1":
        raise ValueError("Bound internal storage-layout fixture required")
    world_path = Path(layout["world_fixture"])
    if file_digest(world_path) != layout["world_fixture_sha256"]:
        raise ValueError("World fixture changed after layout transformation")
    world = json.loads(world_path.read_text())
    if world.get("data_format") != "npy_directory_v1":
        raise ValueError("Bounded-memory world comparisons require NPY storage")
    if file_digest(world_path.parent / world["data_file"]) != world["data_sha256"]:
        raise ValueError("World reference manifest differs")
    world_store = ArrayReader(world_path.parent / world["data_directory"])
    counts = world_store[world["inputs"][117]["key"]]
    inverse = AxisMap(tuple(layout["xyz_map"])).inverse()
    by_path = {}
    for internal, original in zip(fixture["state_leaves"], world["state_leaves"], strict=True):
        if internal["name"] != original["name"]:
            raise ValueError("World leaf identity changed")
        entry = store.metadata[internal["reference_key"]]
        by_path[str((store.directory / entry["file"]).resolve())] = (internal, original)
    world_checks = []
    world_check_seconds = []

    def check(actual, expected, *pos, **kw):
        first = compare_arrays(actual, expected, *pos, **kw)
        filename = getattr(expected, "filename", None)
        if filename is None or str(Path(filename).resolve()) not in by_path:
            raise ValueError("Unexpected unbound comparison in native cyclic worker")
        internal, original = by_path[str(Path(filename).resolve())]
        started = time.perf_counter()
        restored = transform_leaf(actual, internal["mapping"], inverse, counts)
        world_expected = world_store[original["reference_key"]]
        row = compare_arrays(restored, world_expected, *pos, **kw)
        row["name"] = original["name"]
        world_checks.append(row)
        world_check_seconds.append(time.perf_counter() - started)
        drop_pages(world_expected)
        if not row["bitwise_exact"]:
            first["passed"] = first["exact"] = first["bitwise_exact"] = False
        return first

    old_compare = base.compare_arrays
    base.compare_arrays = check
    try:
        result = ORIGINAL_NATIVE(args, fixture, store, inputs, chunks)
    finally:
        base.compare_arrays = old_compare
    names = [leaf["name"] for leaf in world["state_leaves"]]

    def group(rows):
        return {"passed": len(rows) == 30 and [x["name"] for x in rows] == names and all(x["passed"] and x["bitwise_exact"] for x in rows),
                "exact": len(rows) == 30 and all(x["exact"] for x in rows),
                "bitwise_exact": len(rows) == 30 and all(x["bitwise_exact"] for x in rows),
                "leaves": rows}

    result["world_parity"] = group(world_checks[:30])
    result["module"] = str(Path(args.module).resolve())
    result["world_repeat_parity"] = group(world_checks[30:60])
    result["world_fixture"] = str(world_path)
    result["world_fixture_sha256"] = file_digest(world_path)
    result["world_data_manifest_sha256"] = world["data_sha256"]
    result["cyclic_layout"] = layout
    result["world_inverse_and_comparison_s"] = sum(world_check_seconds)
    result["world_inverse_and_comparison_outside_timed_solve"] = True
    result["underlying_worker_sha256"] = file_digest(ORIGINAL_WORKER)
    result["cyclic_transform_worker_sha256"] = file_digest(Path(__file__).with_name("cuda_cyclic_fixture.py"))
    if not result["world_parity"]["passed"] or not result["world_repeat_parity"]["passed"]:
        result["status"] = "failed_world_inverse_parity"
    return result


def main():
    base.native = native
    # base.main records __file__ as the launched worker identity.
    base.__file__ = __file__
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
