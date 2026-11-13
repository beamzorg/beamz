#!/usr/bin/env python3
"""Derive rolling-D4 operational queues while preserving all physical data.

The independent reference remains the original full 30-leaf CUDA/JAX oracle.
Only static bounds/masks/first/final row queues change; arrays use mmap NPY
storage with absolute aliases to avoid copying large existing fixture stores.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cuda_fixture_store import ArrayReader, ArrayWriter
from scripts.cuda_rolling4_plan import build_rolling4_plan
from scripts.export_fdtdx_comparison import file_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Output fixture must be new")
    original = json.loads(a.fixture.read_text())
    f = copy.deepcopy(original)
    directory = a.output.parent / (a.output.stem + ".arrays")
    writer = ArrayWriter(directory)
    if original.get("data_format") == "npy_directory_v1":
        store = ArrayReader(a.fixture.parent / original["data_directory"])
        for key, item in store.metadata.items():
            entry = dict(item)
            entry["file"] = str((store.directory / item["file"]).resolve())
            writer.arrays[key] = entry
    else:
        store = np.load(a.fixture.parent / original["data_file"], allow_pickle=False)
        for key in store.files:
            writer.put(key, store[key])
    shapes = [tuple(x["shape"]) for x in f["inputs"][:6]]
    groups = []
    for g in range(9):
        i = 86 + 3 * g
        groups.append(
            (tuple(f["inputs"][i]["shape"]), store[f["inputs"][i + 2]["key"]])
        )
    thickness = f["attributes"]["boundary_code"] >> 8
    # All six terms in each family must actually use the declared symmetric slab.
    for index in [12, 43]:
        descriptor = np.asarray(store[f["inputs"][index]["key"]])
        if descriptor.shape != (6, 5) or not np.all(descriptor[:, 2:4] == thickness):
            raise ValueError("Uniform symmetric CPML metadata required")
    plan = build_rolling4_plan(
        shapes, thickness, groups, store[f["inputs"][113]["key"]]
    )
    for index, value in [
        (129, plan.bounds),
        (130, plan.core_masks),
        (132, plan.first_cells),
        (133, plan.edge_cells),
    ]:
        key = f["inputs"][index]["key"]
        entry = writer.put(key, value)
        f["inputs"][index].update(
            shape=entry["shape"], dtype=entry["dtype"], sha256=entry["sha256"]
        )
    manifest = writer.finish()
    f["data_format"] = "npy_directory_v1"
    f["data_directory"] = directory.name
    f["data_file"] = str(manifest.relative_to(a.output.parent))
    f["data_sha256"] = file_digest(manifest)
    f["attributes"]["rolling_plan_depth"] = 4
    f["attribute_dtypes"]["rolling_plan_depth"] = "<i4"
    f["rolling4_plan"] = plan.summary()
    f["derivation"] = {
        "base_fixture": str(a.fixture.resolve()),
        "base_fixture_sha256": file_digest(a.fixture),
        "planner_sha256": file_digest(
            Path(__file__).with_name("cuda_rolling4_plan.py")
        ),
        "changed_input_indices": [129, 130, 132, 133],
        "unchanged_physical_input_and_reference_arrays": True,
        "same_independent_30leaf_reference": True,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(f, indent=2) + "\n")
    print(json.dumps({"output": str(a.output), **plan.summary()}, indent=2))


if __name__ == "__main__":
    main()
