"""Compare complete native research runs only under an identical physics contract.

Launch tile/segment flags may differ. Data, physics attributes, dtype, clocks,
timesteps, references and preserving/owned policy must agree. Field-only probes
are intentionally not accepted as complete-engine results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCH_BITS = sum(1 << bit for bit in range(20, 27))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rolling4_identity(packet, manifest):
    """Compare physical bytes while admitting four explicitly operational arrays."""
    if packet.get("data_format") != "npy_directory_v1":
        raise ValueError("Rolling4 comparison requires NPY manifests for reference hashes")
    arrays = json.loads(manifest.read_text())["arrays"]
    inputs = packet["inputs"]
    if len(inputs) != 134 or [x["index"] for x in inputs] != list(range(134)):
        raise ValueError("Rolling4 comparison requires the complete Cartesian CPML packet")
    expected = {129: "plan_bounds", 130: "plan_core_masks",
                132: "plan_first_cells", 133: "plan_edge_cells"}
    if any(inputs[i]["name"] != name for i, name in expected.items()):
        raise ValueError("Operational array indices do not match the prepared ABI")

    def signature(key):
        value = arrays[key]
        return {k: value[k] for k in ("shape", "dtype", "sha256")}

    physical = []
    for i, item in enumerate(inputs):
        value = signature(item["key"])
        if value != {k: item[k] for k in ("shape", "dtype", "sha256")}:
            raise ValueError(f"Manifest/input identity mismatch: {item['name']}")
        if i not in expected:
            physical.append({"index": i, "name": item["name"], **value})
    return {"inputs": physical,
            "state_bytes": [{"name": leaf["name"],
                             "initial": signature(leaf["initial_key"]),
                             "reference": signature(leaf["reference_key"])}
                            for leaf in packet["state_leaves"]]}


def read_run(path, *, allow_rolling4_plan=False, expected_module_sha256=None):
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if expected_module_sha256 is not None:
        if report.get("module_sha256") != expected_module_sha256:
            raise ValueError(f"Control module SHA mismatch: {path}")
        module = report.get("module")
        if not module:
            raise ValueError(f"Missing control module path: {path}")
        module = Path(module)
        if not module.is_absolute():
            module = ROOT / module
        if not module.is_file() or digest(module) != expected_module_sha256:
            raise ValueError(f"Control binary changed or missing: {module}")
    if report.get("status") != "ok" or not report.get("host_inputs_unchanged"):
        raise ValueError(f"Incomplete run or modified inputs: {path}")
    for key in ("parity", "repeat_parity"):
        check = report.get(key, {})
        if not all(check.get(k) for k in ("passed", "exact", "bitwise_exact")):
            raise ValueError(f"Missing exact complete-state acceptance: {path}/{key}")
        leaves = check.get("leaves", [])
        if len(leaves) != 30 or not all(
            leaf.get("bitwise_exact") and leaf.get("passed") for leaf in leaves
        ):
            raise ValueError(f"Expected 30 accepted leaves: {path}/{key}")
    fixture = Path(report["fixture"])
    if not fixture.is_absolute():
        fixture = ROOT / fixture
    if digest(fixture) != report["fixture_sha256"]:
        raise ValueError(f"Fixture changed after timing: {fixture}")
    packet = json.loads(fixture.read_text())
    expected = [(x["name"], x["shape"], x["dtype"]) for x in packet["state_leaves"]]
    for key in ("parity", "repeat_parity"):
        actual = [(x["name"], x["shape"], x["dtype"])
                  for x in report[key]["leaves"]]
        if actual != expected:
            raise ValueError(f"Accepted leaves do not describe the packet state: {path}")
    manifest = fixture.parent / packet["data_file"]
    if digest(manifest) != packet["data_sha256"]:
        raise ValueError(f"Input manifest changed: {manifest}")
    attributes = dict(packet["attributes"])
    attributes["cuda_flags"] &= ~LAUNCH_BITS
    identity = packet["data_sha256"]
    if allow_rolling4_plan:
        if attributes.get("rolling_plan_depth", 2) not in (2, 4):
            raise ValueError("Unsupported rolling plan depth")
        attributes.pop("rolling_plan_depth", None)
        identity = rolling4_identity(packet, manifest)
    if sum(report["chunks"]) != attributes["nsteps"]:
        raise ValueError("Timed schedule does not cover the declared simulation")
    seconds = report["median_wall_s"]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Expected a positive finite timing")
    contract = {
        "attributes": attributes,
        "packet_identity": identity,
        "ownership": report["ownership"],
        "chunks": report["chunks"],
        "state": [{k: leaf[k] for k in ("name", "shape", "dtype", "reference_key")}
                  for leaf in packet["state_leaves"]],
    }
    return contract, seconds, {"path": str(path), "sha256": digest(path),
                               "fixture": str(fixture),
                               "module_sha256": report.get("module_sha256")}


def compare(baseline, candidate, *, allow_rolling4_plan=False,
            expected_control_sha256=None):
    if expected_control_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_control_sha256):
            raise ValueError("Expected a lowercase 64-digit control module SHA256")
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("Use equal nonzero counts of paired fresh processes")
    rows, contract = [], None
    for a, b in zip(baseline, candidate, strict=True):
        ca, ta, pa = read_run(a, allow_rolling4_plan=allow_rolling4_plan,
                              expected_module_sha256=expected_control_sha256)
        cb, tb, pb = read_run(b, allow_rolling4_plan=allow_rolling4_plan)
        if ca != cb or (contract is not None and ca != contract):
            raise ValueError(f"Physics/ownership contract mismatch: {a} versus {b}")
        contract = ca
        rows.append({"baseline": pa, "candidate": pb, "baseline_s": ta,
                     "candidate_s": tb, "speedup": ta / tb})
    ratios = [r["speedup"] for r in rows]
    return {"schema": "beamz.complete-research-comparison.v1",
            "scope": "Complete identical physics; exact all30 outputs and repeat state",
            "allowed_operational_differences": [129, 130, 132, 133]
            if allow_rolling4_plan else [],
            "fresh_process_pairs": len(rows), "contract": contract,
            "expected_control_sha256": expected_control_sha256,
            "control_identity_verified": expected_control_sha256 is not None,
            "median_paired_speedup": statistics.median(ratios),
            "observed_range": [min(ratios), max(ratios)],
            "range_is_confidence_interval": False, "pairs": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, nargs="+", required=True)
    p.add_argument("--candidate", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-rolling4-plan", action="store_true",
                   help="Admit only four operational plan arrays and rolling_plan_depth; hash all physical inputs and initial/reference leaves")
    p.add_argument("--expected-control-sha256",
                   help="Require every baseline report and its actual binary to match this exact control module")
    a = p.parse_args()
    result = compare(a.baseline, a.candidate, allow_rolling4_plan=a.allow_rolling4_plan,
                     expected_control_sha256=a.expected_control_sha256)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("fresh_process_pairs",
                     "median_paired_speedup", "observed_range")}))


if __name__ == "__main__":
    main()
