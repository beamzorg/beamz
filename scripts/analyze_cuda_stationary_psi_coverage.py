"""CPU native tile-class and active CPML traffic model for a frozen fixture.

Counts actual valid component/recurrence updates, including ragged ownership and
CPML12. Exact logical payload predictions are not measured cache/DRAM traffic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_fixture_store import ArrayReader
from scripts.cuda_native_geometry_ir import derive_geometry, verify_component_shapes
from scripts.cuda_space_time_ir import FIELDS


def cache_axis(flags):
    if not 0 <= flags < 64:
        raise ValueError("Known six-bit immutable tile facts required")
    for axis in range(3):
        if flags in (2 << axis, 32 | (2 << axis)):
            return axis
    return -1


def volume(lo, hi, shape, axis=None):
    lower, upper = np.maximum(lo, 0), np.minimum(hi, shape)
    lengths = np.maximum(upper - lower, 0)
    if axis is not None:
        size = shape[axis]
        lengths[:, axis] = (np.maximum(np.minimum(upper[:, axis], 12) - lower[:, axis], 0)
                            + np.maximum(upper[:, axis] - np.maximum(lower[:, axis], size - 12), 0))
    return lengths.prod(axis=1)


def analyze(path, depth=16, owned=(5, 15, 31)):
    path = Path(path)
    fixture = json.loads(path.read_text())
    data = path.parent / fixture["data_file"]
    assert hashlib.sha256(data.read_bytes()).hexdigest() == fixture["data_sha256"]
    store = ArrayReader(path.parent / fixture["data_directory"])
    entries = {e["index"]: e for e in fixture["inputs"]}
    shapes = tuple(tuple(entries[c]["shape"]) for c in range(6))
    minimum = verify_component_shapes(shapes)
    counts = tuple(math.ceil((max(s[a] for s in shapes) + depth) / owned[a]) for a in range(3))
    tiles = np.indices(counts, dtype=np.int64).reshape(3, -1).T
    origins = tiles * owned
    low, high = origins - depth, origins + np.asarray(owned) - 2
    flags = np.zeros(len(tiles), dtype=np.uint8)
    for a in range(3):
        flags[(low[:, a] < 12) | (high[:, a] >= minimum[a] - 12)] |= 2 << a
        flags[(low[:, a] <= 0) | (high[:, a] + 1 >= minimum[a])] |= 32
    consumed = []
    for group in range(3, 9):
        coefficient = entries[86 + 3 * group]
        if not coefficient["shape"][0]:
            continue
        entry = entries[88 + 3 * group]
        store.verify(entry["key"])
        consumed.append({"input": entry["index"], "sha256": entry["sha256"]})
        for start in store[entry["key"]]:
            hit = np.all((low < start + coefficient["shape"][1:]) & (high >= start), axis=1)
            flags[hit] |= 16
    entry = entries[113]
    store.verify(entry["key"])
    consumed.append({"input": 113, "sha256": entry["sha256"]})
    indices = store[entry["key"]]
    for component in range(6):
        native = (component + 3) % 6
        shape = shapes[native]
        logical = np.unique(indices[:, component].reshape(-1))
        logical = logical[(logical >= 0) & (logical < math.prod(shape))].astype(np.int64)
        points = np.stack((logical // (shape[1] * shape[2]), logical // shape[2] % shape[1], logical % shape[2]), axis=1)
        for n in range(1, depth + 1):
            cell = (points + n) // owned
            ids = (cell[:, 0] * counts[1] + cell[:, 1]) * counts[2] + cell[:, 2]
            flags[ids] |= 1
    component_updates = np.zeros(len(tiles), dtype=np.int64)
    psi_updates = np.zeros_like(component_updates)
    cache_updates = np.zeros_like(component_updates)
    cache_loads = np.zeros_like(component_updates)
    cache_stores = np.zeros_like(component_updates)
    eligible_axis = np.asarray([cache_axis(f) for f in range(64)], dtype=np.int8)[flags]
    terms = derive_geometry()["cpml_terms"]
    for n in range(1, depth + 1):
        lo, hi = origins - n, origins + np.asarray(owned) - n
        for shape in shapes:
            component_updates += volume(lo, hi, shape)
        for term in terms:
            axis, shape = term["axis_zyx"], shapes[FIELDS.index(term["output"])]
            active = volume(lo, hi, shape, axis)
            psi_updates += active
            selected = eligible_axis == axis
            cache_updates[selected] += active[selected]
            incoming = active if n == 1 else active - volume(lo + 1, hi, shape, axis)
            outgoing = active if n == depth else active - volume(lo, hi - 1, shape, axis)
            cache_loads[selected] += incoming[selected]
            cache_stores[selected] += outgoing[selected]
    histogram = np.bincount(flags, minlength=64).tolist()
    by_flag = [{"flags": f, "tiles": int(np.sum(flags == f)),
                "valid_component_updates": int(component_updates[flags == f].sum()),
                "active_psi_updates": int(psi_updates[flags == f].sum()),
                "cached_psi_updates": int(cache_updates[flags == f].sum()),
                "predicted_cached_psi_loads": int(cache_loads[flags == f].sum()),
                "predicted_cached_psi_stores": int(cache_stores[flags == f].sum())}
               for f in range(64) if histogram[f]]
    eligible = eligible_axis >= 0
    total_psi, covered_psi = int(psi_updates.sum()), int(cache_updates.sum())
    assert int(component_updates.sum()) == depth * sum(math.prod(shape) for shape in shapes)
    expected_psi = depth * sum(24 * math.prod(size for a, size in enumerate(shapes[FIELDS.index(t["output"])])
                                             if a != t["axis_zyx"]) for t in terms)
    assert total_psi == expected_psi
    direct_bytes = covered_psi * 8
    cached_bytes = int(cache_loads.sum() + cache_stores.sum()) * 4
    groups = {
        "bulk": flags == 0,
        "general-no-IO": (flags != 0) & ((flags & 17) == 0),
        "IO-no-CPML-or-boundary": ((flags & 17) != 0) & ((flags & 46) == 0),
        "IO-with-CPML-or-boundary": ((flags & 17) != 0) & ((flags & 46) != 0),
    }
    return {"fixture": str(path), "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "data_sha256": fixture["data_sha256"], "consumed_arrays": consumed,
            "depth": depth, "owned_zyx": owned, "counts_zyx": counts,
            "flag_histogram": histogram, "by_flag": by_flag,
            "tile_counts": {k: int(v.sum()) for k, v in groups.items()},
            "valid_component_updates_per_macro": {k: int(component_updates[v].sum()) for k, v in groups.items()},
            "all_valid_component_updates": int(component_updates.sum()),
            "eligible_noIO_single_axis_tiles": int(eligible.sum()),
            "eligible_valid_component_updates": int(component_updates[eligible].sum()),
            "eligible_component_update_fraction": float(component_updates[eligible].sum() / component_updates.sum()),
            "all_active_psi_updates": total_psi, "eligible_active_psi_updates": covered_psi,
            "eligible_active_psi_fraction": covered_psi / total_psi,
            "eligible_direct_psi_payload_bytes": direct_bytes,
            "eligible_cached_psi_payload_bytes": cached_bytes,
            "predicted_all_psi_payload_fraction_saved": (direct_bytes - cached_bytes) / (8 * total_psi),
            "region_class": "flags in {2,4,8,34,36,40}; no source/observer, exactly one CPML axis; boundary32 permitted",
            "limits": "Exact logical counts under lazy first-owned load/high-or-final store; not DRAM/cache/timing measurements"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--reference-histogram", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.fixture) != len(args.label):
        parser.error("Supply one label per fixture")
    results = [analyze(path) | {"label": label} for path, label in zip(args.fixture, args.label, strict=True)]
    if args.reference_histogram:
        reference = {r["fixture"]: r for r in json.loads(args.reference_histogram.read_text())}
        for result in results:
            old = reference[result["label"]]
            for key in ("flag_histogram", "tile_counts", "valid_component_updates_per_macro"):
                assert result[key] == old[key], f"Existing native geometry cross-check failed: {key}"
            result["existing_reference_crosscheck"] = "all flags/classes/valid updates exact"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
