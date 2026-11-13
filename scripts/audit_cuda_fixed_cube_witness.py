"""Derive a compact, source-bound AOT path witness from existing CUPTI data.

Reads existing profile/activity/build artifacts only; never imports a CUDA
module, modifies raw traces, builds code, or executes GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def witness(profile, build, output):
    profile, build, output = Path(profile), Path(build), Path(output)
    data = json.loads(profile.read_text())
    trace = Path(data["trace"])
    manifest_path = build / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    module = Path(data["module"])
    if sha(module) != data["module_sha256"] or manifest["module_sha256"] != data["module_sha256"]:
        raise ValueError("Actual profiled/frozen/build module identities differ")
    actual_source_hashes = {name: sha(build / "source" / name) for name in manifest["source_hashes"]}
    if actual_source_hashes != manifest["source_hashes"]:
        raise ValueError("An actual compiled-source manifest member changed")
    rows, collectors = [], []
    for line in trace.read_text().splitlines():
        row = json.loads(line)
        if row.get("kind") == "kernel":
            rows.append(row)
        elif row.get("kind") == "collector":
            collectors.append(row)
    if not collectors or any(c.get("dropped_records") != 0 for c in collectors):
        raise ValueError("Missing clean collector coverage witness")
    if data["activity"].get("dropped_records") != 0:
        raise ValueError("Profile summary reports dropped records")
    fixed = [r for r in rows if "AsymmetricNoIOFixedCube" in r["name"]]
    if not fixed:
        raise ValueError("No actual new-path kernel executions")
    fixed.sort(key=lambda r: (r["start_ns"], r["graph_node_id"]))
    byname = defaultdict(list)
    for row in fixed:
        byname[row["name"]].append(row)
    kernels = []
    for name, values in sorted(byname.items()):
        m = re.search(r"FixedCubeILi(\d+)ELi(\d+)E", name)
        if m is None:
            raise ValueError("Unrecognized exact AOT depth/shape symbol")
        resources = {tuple(r["block"]) + (r["registers"], r["dynamic_shared_bytes"], r["local_bytes_per_thread"]) for r in values}
        if resources != {(1024, 1, 1, 64, 55204, 0)}:
            raise ValueError("Actual fixed path resource contract differs")
        kernels.append({"name": name, "depth": int(m[1]), "cube_N": int(m[2]),
                        "launches": len(values), "CTAs": sum(math.prod(r["grid"]) for r in values),
                        "block": [1024, 1, 1], "registers": 64, "dynamic_shared_bytes": 55204,
                        "local_bytes_per_thread": 0})
    old_uniform = [r for r in rows if "AsymmetricNoIOUniform1024" in r["name"]]
    generic = [r for r in rows if re.search(r"10AsymmetricILi\d+ELi1ELb0ELb1ELi1E", r["name"])]
    macro = int(data["profile_timing"]["moving_macro_count"])
    ranks = int(data["session_metadata"]["moving_rank_count"])
    certified = int(data["session_metadata"]["moving_uniform_noio_tiles"])
    expected_ctas = macro * certified
    observed_ctas = sum(k["CTAs"] for k in kernels)
    if old_uniform or observed_ctas != expected_ctas:
        raise ValueError("Eligible certified NoIO execution was not fully replaced")
    build_resources = json.loads((build / "ptxas-resources.json").read_text())
    if len(build_resources) != 66 or any(any(r[k] for k in ("stack", "spill_loads", "spill_stores")) for r in build_resources):
        raise ValueError("Actual66-entry zero-stack/spill build gate differs")
    new_resources = [r for r in build_resources if "AsymmetricNoIOFixedCube" in r["name"]]
    if len(new_resources) != 4:
        raise ValueError("Expected four actual shape/depth AOT entries")
    indices = sorted({0, ranks // 2, ranks - 1})
    selected = [{"matching_launch_index": i, "grid": fixed[i]["grid"], "graph_id": fixed[i]["graph_id"],
                 "graph_node_id": fixed[i]["graph_node_id"], "name": fixed[i]["name"]} for i in indices if i < len(fixed)]
    result = {"schema": "beamz.actual-common-shape-AOT-kernel-witness.v1",
              "module_sha256": data["module_sha256"], "fixture": data["fixture"], "fixture_sha256": data["fixture_sha256"],
              "raw_profile": str(profile), "raw_profile_sha256": sha(profile), "raw_activity": str(trace), "raw_activity_sha256": sha(trace),
              "manifest_sha256": sha(manifest_path), "source_consumer_sha256": actual_source_hashes["src/asymmetric_native.cuh"],
              "source_host_capture_sha256": actual_source_hashes["src/native_extension.cc"],
              "canonical_FP_graph_sha256": actual_source_hashes["src/asymmetric_native_cell_generated.cuh"],
              "all_manifest_source_hashes_verified": True,
              "collector_dropped_records": 0, "kernel_records": len(rows), "unique_kernel_names": len(Counter(r["name"] for r in rows)),
              "actual_AOT_kernels": kernels, "actual_macro_count": macro, "rank_count": ranks,
              "certified_NoIO_tiles_per_macro": certified, "expected_certified_CTAs": expected_ctas,
              "observed_fixed_CTAs": observed_ctas, "eligible_certified_CTA_coverage_exact": observed_ctas == expected_ctas,
              "old_cached_NoIO_uniform_launches": len(old_uniform),
              "preserved_noncertified_generic_NoIO_launches": len(generic),
              "generic_NoIO_note": "Generic512-thread Policy1 is the original noncertified subrange; its presence is expected and does not mean eligible cached NoIO fell back",
              "all66_compiled_entries_zero_stack_spills": True, "four_new_compiled_entries": new_resources,
              "actual_profile_all30_bitwise": data["parity"].get("all30_bitwise") is True,
              "host_inputs_unchanged": data.get("host_inputs_unchanged") is True,
              "observed_matching_launch_windows_sorted_by_start": selected,
              "ordering_note": "Activity JSONL callback order is not launch order; matching windows above use actual start_ns order. Recheck indices if fixture/source configuration changes.",
              "coverage_limit": "Only listed shape/depth combinations actually executed in this profile; compiled entries do not establish runtime/sanitizer coverage",
              "timing_scope": "Witness only; no duration share or speed claim derived from this instrumented run",
              "producer_sha256": sha(__file__),
              "raw_artifacts_modified": False, "GPU_executed_by_this_audit": False}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(witness(a.profile, a.build, a.output), indent=2))
