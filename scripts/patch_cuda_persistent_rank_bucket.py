"""Host-plan-only rank-bucket Morton ready-queue priority for locality mode.

This is an ordering heuristic. It does not bound the span of in-flight CUDA
tile ranks or add rank joins. RAW/WAR edges, trail slots and rank mode remain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PARENT = {
    "native_persistent.h": "00b85f62ad6128a8d391128a6cc1f6d2459daf9e8fa45a91fff7c3877d44160f",
    "native_persistent.cuh": "2e3408b1d514a8b0d90620cbb495199a438e4e5f729b1f7c6830dc1f3c2174b2",
    "native_extension.cc": "e863084f0680d25981585fd4bd0f5ce1a1dd2bde8dd2e507f6ec5895352fef27",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transformed(text, window):
    if window not in (2, 4):
        raise ValueError("Reviewed rank-bucket windows2/4 only")
    if sha(text.encode()) != PARENT["native_persistent.h"]:
        raise ValueError("Exact reviewed persistent parent header required")
    old = text
    edits = []

    def change(a, b):
        nonlocal text
        if text.count(a) != 1:
            raise ValueError("One exact ready-priority anchor required:" + a)
        text = text.replace(a, b, 1)
        edits.append((a, b))

    change("#include <utility>", "#include <utility>\n#include <tuple>")
    change("  using Ready=std::pair<uint64_t,int32_t>;std::priority_queue<Ready,std::vector<Ready>,std::greater<Ready>>ready;",
           f"  // Locality means rank-bucket Morton(window={window}) in this experimental module.\n  // This orders available host-plan jobs; it never bounds in-flight rank span.\n  using Ready=std::tuple<int,uint64_t,int32_t>;std::priority_queue<Ready,std::vector<Ready>,std::greater<Ready>>ready;")
    change("ready.push({BeamzPersistentMorton(t,p),int32_t(t)})", f"ready.push({{BeamzPersistentTileRank(t,p)/{window},BeamzPersistentMorton(t,p),int32_t(t)}})")
    change("const int t=ready.top().second;", "const int t=std::get<2>(ready.top());")
    change("ready.push({BeamzPersistentMorton(child,p),child})", f"ready.push({{BeamzPersistentTileRank(child,p)/{window},BeamzPersistentMorton(child,p),child}})")
    inverse = text
    for a, b in reversed(edits):
        if inverse.count(b) != 1:
            raise ValueError("Exact priority inverse became ambiguous")
        inverse = inverse.replace(b, a, 1)
    if inverse != old:
        raise ValueError("Host ready-priority exact inverse failed")
    return text


def patch_copy(source, output, window):
    source, output = Path(source).resolve(), Path(output).resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        raise ValueError("New disjoint private source required")
    before = {n: (source / "src" / n).read_bytes() for n in PARENT}
    if any(sha(before[n]) != expected for n, expected in PARENT.items()):
        raise ValueError("Exact P5 g4 floor/global source and original host binding required")
    changed = transformed(before["native_persistent.h"].decode(), window)
    shutil.copytree(source, output)
    (output / "src/native_persistent.h").write_text(changed)
    proof = {"schema": "beamz.persistent-rank-bucket-Morton.v1", "window": window,
             "variant": f"P5 g4 floor/global; locality=rank-bucket Morton window{window}",
             "priority_tuple": [f"rank//{window}", "MortonXYZ", "tileID"],
             "rank_mode_unchanged": True, "RAW_WAR_validation_and_slot_mapping_unchanged": True,
             "GPU_kernel_source_unchanged": True, "host_Session_source_unchanged_before_parent_overlay": True,
             "no_inflight_rank_span_bound_claim": True, "exact_source_inverse": True,
             "parent_hashes": PARENT, "result_header_sha256": sha(changed.encode()),
             "producer_sha256": sha(Path(__file__).read_bytes()), "GPU_executed": False}
    (output.parent / "rank-bucket-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--window", required=True, type=int, choices=(2, 4))
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output, a.window), indent=2))
