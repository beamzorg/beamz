"""Compose exact six guarded CG trail imports with source-plane+NoIO-axis.

This is an orthogonal cache-policy experiment; no source/axis gains are assumed
to multiply. The hidden source-plane selector remains available for a causal
same-CUDA control that retains only the axis+CG changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENTS = {
    "asymmetric_native.cuh": "0b4fffc3a51d15835bd27bcf388398020246dcd2dfe6a135704af1095bf2042b",
    "native_extension.cc": "e863084f0680d25981585fd4bd0f5ce1a1dd2bde8dd2e507f6ec5895352fef27",
    "native_moving.h": "d3d1abaca2ca066b75295df35ec26da2c1df53c75bdb9f140b067fea09288b3c",
    "native_moving.cc": "44e47b882864f2338764592a5911e2fe9f37ad9c49862dc62ed27d93e099bfd1",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
}
ORIGINAL = """ return {mask&1?point[0*SurfacePoints]:0.f,mask&2?point[1*SurfacePoints]:0.f,
         mask&4?point[2*SurfacePoints]:0.f,mask&8?point[3*SurfacePoints]:0.f,
         mask&16?point[4*SurfacePoints]:0.f,mask&32?point[5*SurfacePoints]:0.f};"""
REPLACEMENT = ORIGINAL
for _f in range(6):
    REPLACEMENT = REPLACEMENT.replace(f"point[{_f}*SurfacePoints]", f"__ldcg(point+{_f}*SurfacePoints)")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transformed(parent):
    for n, expected in PARENTS.items():
        if sha(parent[n]) != expected:
            raise ValueError(f"Exact source-plane-axis parent and planner required: {n}")
    old = parent["asymmetric_native.cuh"].decode()
    if old.count(ORIGINAL) != 1:
        raise ValueError("Expected one exact guarded ImportPacket return")
    new = old.replace(ORIGINAL, REPLACEMENT, 1)
    if new.replace(REPLACEMENT, ORIGINAL, 1) != old:
        raise ValueError("Exact cache-policy source inverse failed")
    files = dict(parent)
    files["asymmetric_native.cuh"] = new.encode()
    return files


def patch_copy(source, output):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents or output in source.parents
            or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint noncanonical output required")
    parent = {n: (source / "src" / n).read_bytes() for n in PARENTS}
    files = transformed(parent)
    shutil.copytree(source, output)
    for n, data in files.items():
        (output / "src" / n).write_bytes(data)
    proof = {"parent_hashes": PARENTS, "parent_reference": "source-plane-axis-v1 plusmandatorya939control",
             "result_hashes": {n: sha(d) for n, d in files.items()}, "producer_sha256": sha(Path(__file__).read_bytes()),
             "exact_source_inverse": True, "changed_expressions": 6,
             "guards_addresses_load_order_fp_graph_and_all_other_source_bytes_identical": True,
             "source_geometry_guard_planner_binding": PARENTS["native_moving.cc"],
             "memory_order_admission": "lower-rank trail producers and all prior ring consumers retire at existing rank/event joins; same-stream/event launch ordering unchanged",
             "source_plane_axis_and_host_key_selectors_unchanged": True,
             "causal_control": "CXX-only PlanePair selectorOff retains same62CUDAentries and axis+CG",
             "expected_moving_kernels": 62, "gpu_executed": False,
             "requires": ["zero-local exact-contextresource", "strictall30physicsandcache/fallbackgates", "matchedwholeworkloadscreen;noassumedmultiplicationofisolatedgains"]}
    (output.parent / "source-plane-axis-cg-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output)))


if __name__ == "__main__":
    main()
