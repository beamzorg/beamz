"""Compare world-equivalent cyclic storage runs without relaxing the comparator.

Requires the candidate's actual inverse-world first/repeat checks and audits
captured bytes through the explicit proper cyclic storage map. The ordinary
identical-fixture comparator remains unmodified and rejects these fixtures.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_cyclic_fixture import input_mapping, leaf_mapping, transform_input, transform_leaf
from scripts.cuda_fixture_store import ArrayReader, digest_array, drop_pages
from scripts.cuda_research_compare import digest, read_run
from scripts.prove_cuda_cyclic_axis_equivariance import AxisMap


def audit_layout(path):
    started = time.monotonic()
    path = Path(path).resolve()
    internal = json.loads(path.read_text())
    layout = internal.get("cyclic_layout", {})
    if layout.get("schema") != "beamz.cyclic-storage-layout.v1":
        raise ValueError("Typed cyclic storage layout required")
    rotation = AxisMap(tuple(layout["xyz_map"]))
    if rotation.odd:
        raise ValueError("Proper cyclic storage orientation required")
    world_path = Path(layout["world_fixture"])
    if digest(world_path) != layout["world_fixture_sha256"]:
        raise ValueError("World fixture changed")
    world = json.loads(world_path.read_text())
    for fixture, origin in ((world, world_path), (internal, path)):
        if fixture.get("data_format") != "npy_directory_v1" or digest(origin.parent / fixture["data_file"]) != fixture["data_sha256"]:
            raise ValueError("Immutable NPY source manifests required")
    world_store = ArrayReader(world_path.parent / world["data_directory"])
    internal_store = ArrayReader(path.parent / internal["data_directory"])
    shapes = [tuple(i["shape"]) for i in world["inputs"][:6]]
    counts = world_store[world["inputs"][117]["key"]]
    attrs = dict(world["attributes"])
    for name in ("coincident_source_group_mask", "disjoint_source_group_mask"):
        mask = attrs.get(name, 0)
        attrs[name] = sum(1 << rotation.source_group(g) for g in range(9) if mask & (1 << g))
    logical = rotation.vector(tuple(attrs.get("logical_" + a, 0) for a in "zyx"))
    attrs.update({"logical_" + a: n for a, n in zip("zyx", logical, strict=True)})
    if attrs != internal["attributes"]:
        raise ValueError("Cyclic transform changed a physical or execution attribute")
    records = []
    for item in world["inputs"]:
        target = internal["inputs"][input_mapping(item["index"], rotation)]
        before = world_store[item["key"]]
        after = internal_store[target["key"]]
        if digest_array(before) != item["sha256"] or digest_array(after) != target["sha256"]:
            raise ValueError("Input bytes no longer match captured fixture hashes")
        expected = transform_input(item["index"], before, rotation, shapes, counts)
        if digest_array(expected) != digest_array(after):
            raise ValueError(f"Cyclic input mapping differs:{item['name']}")
        records.append({"world_input": item["index"], "internal_input": target["index"], "passed": True})
        del expected
        drop_pages(before); drop_pages(after)
    leaves = []
    for original, transformed in zip(world["state_leaves"], internal["state_leaves"], strict=True):
        if original["name"] != transformed["name"] or leaf_mapping(original["mapping"], rotation) != transformed["mapping"]:
            raise ValueError("Logical world leaf mapping differs")
        for label in ("initial_key", "reference_key"):
            before = world_store[original[label]]
            after = internal_store[transformed[label]]
            world_store.verify(original[label]); internal_store.verify(transformed[label])
            expected = transform_leaf(before, original["mapping"], rotation, counts)
            if digest_array(expected) != digest_array(after):
                raise ValueError(f"Independent world reference changed:{original['name']}/{label}")
            del expected
            drop_pages(before); drop_pages(after)
        leaves.append({"name": original["name"], "initial_and_reference_mapping_exact": True})
    if len(records) != 134 or len(leaves) != 30:
        raise ValueError("Complete134 input and30 world state audit required")
    return {"schema": "beamz.cyclic-world-byte-audit.v1", "passed": True,
            "world_fixture": str(world_path), "world_fixture_sha256": digest(world_path),
            "internal_fixture": str(path), "internal_fixture_sha256": digest(path),
            "inputs": records, "state_leaves": leaves, "elapsed_s": time.monotonic() - started}


def check_world_parity(report, world):
    expected = [(x["name"], x["shape"], x["dtype"]) for x in world["state_leaves"]]
    for key in ("world_parity", "world_repeat_parity"):
        result = report.get(key, {})
        rows = result.get("leaves", [])
        if (not all(result.get(k) for k in ("passed", "exact", "bitwise_exact"))
                or len(rows) != 30 or any(not r.get("bitwise_exact") or not r.get("passed") for r in rows)
                or [(x["name"], x["shape"], x["dtype"]) for x in rows] != expected):
            raise ValueError("Actual first/repeat world all30 readback evidence is required")


def compare(baselines, candidates, expected_control, expected_candidate):
    if not baselines or len(baselines) != len(candidates):
        raise ValueError("Equal nonzero paired process counts required")
    audits, pairs = {}, []
    for baseline_path, candidate_path in zip(baselines, candidates, strict=True):
        ca, ta, pa = read_run(baseline_path, expected_module_sha256=expected_control)
        cb, tb, pb = read_run(candidate_path, expected_module_sha256=expected_candidate)
        a, b = json.loads(Path(baseline_path).read_text()), json.loads(Path(candidate_path).read_text())
        if a.get("method") != "moving" or b.get("method") != "moving" or a.get("moving_depth") != b.get("moving_depth"):
            raise ValueError("Matched native moving execution required")
        path = Path(b["fixture"]).resolve()
        if str(path) not in audits:
            audits[str(path)] = audit_layout(path)
        audit = audits[str(path)]
        world = json.loads(Path(audit["world_fixture"]).read_text())
        if (audit["world_fixture_sha256"] != a["fixture_sha256"] or b.get("world_fixture_sha256") != a["fixture_sha256"]
                or ca["ownership"] != cb["ownership"] or ca["chunks"] != cb["chunks"]):
            raise ValueError("World physics, timing chunks or ownership differ")
        check_world_parity(b, world)
        pairs.append({"baseline": pa, "candidate": pb, "baseline_s": ta, "candidate_s": tb, "speedup": ta / tb})
    ratios = [p["speedup"] for p in pairs]
    return {"schema": "beamz.cyclic-world-runtime-comparison.v1", "scope": "Same independent world workload and exact ordered FP32 outputs; different internal storage coordinates",
            "strict_identical_fixture_comparator_unmodified": True,
            "actual_world_first_repeat_all30_verified": True, "audits": list(audits.values()),
            "expected_control_sha256": expected_control, "expected_candidate_sha256": expected_candidate,
            "fresh_process_pairs": len(pairs), "median_paired_speedup": statistics.median(ratios),
            "observed_range": [min(ratios), max(ratios)], "range_is_confidence_interval": False,
            "setup_and_inverse_conversion_excluded_from_warm_solve_metric": True, "pairs": pairs,
            "producer_sha256": digest(Path(__file__))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, nargs="+", required=True)
    p.add_argument("--candidate", type=Path, nargs="+", required=True)
    p.add_argument("--expected-control-sha256", required=True)
    p.add_argument("--expected-candidate-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = compare(a.baseline, a.candidate, a.expected_control_sha256, a.expected_candidate_sha256)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("fresh_process_pairs", "median_paired_speedup", "observed_range")}))


if __name__ == "__main__":
    main()
