"""Isolate the cache policy of lower-rank immutable trail imports.

__ldcg requests L2 caching for those imported float words. It preserves actual
support guards, addresses, payload bits, producer order and FP arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from patch_cuda_v12_coincident_source_geometry import PARENT_HASHES, PARENT_MODULE

ORIGINAL = """ return {mask&1?point[0*SurfacePoints]:0.f,mask&2?point[1*SurfacePoints]:0.f,
         mask&4?point[2*SurfacePoints]:0.f,mask&8?point[3*SurfacePoints]:0.f,
         mask&16?point[4*SurfacePoints]:0.f,mask&32?point[5*SurfacePoints]:0.f};"""
REPLACEMENT = ORIGINAL
for _field in range(6):
    REPLACEMENT = REPLACEMENT.replace(f"point[{_field}*SurfacePoints]", f"__ldcg(point+{_field}*SurfacePoints)")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transformed(parent):
    for name, expected in PARENT_HASHES.items():
        if sha(parent[name]) != expected:
            raise ValueError(f"Exact a939 v12 parent required: {name}")
    original = parent["asymmetric_native.cuh"].decode()
    if original.count(ORIGINAL) != 1:
        raise ValueError("Expected exact six guarded trail imports")
    result = original.replace(ORIGINAL, REPLACEMENT, 1)
    if result.replace(REPLACEMENT, ORIGINAL, 1) != original:
        raise ValueError("Exact source inverse failed")
    files = dict(parent)
    files["asymmetric_native.cuh"] = result.encode()
    return files


def patch_copy(source, output):
    source, output = source.resolve(), output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (output.exists() or source == output or source in output.parents
            or output in source.parents or output == canonical or canonical in output.parents):
        raise ValueError("New disjoint noncanonical source output required")
    parent = {name: (source / "src" / name).read_bytes() for name in PARENT_HASHES}
    files = transformed(parent)
    shutil.copytree(source, output)
    for name, data in files.items():
        (output / "src" / name).write_bytes(data)
    proof = {
        "parent_module_sha256": PARENT_MODULE,
        "parent_hashes": PARENT_HASHES,
        "result_hashes": {name: sha(data) for name, data in files.items()},
        "patcher_sha256": sha(Path(__file__).read_bytes()),
        "exact_source_inverse": True,
        "changed_load_expressions": 6,
        "guard_addresses_FP_order_host_schedule_and_allocation_byte_identical": True,
        "scope": "ImportPacket only: six same-address guarded float reads use __ldcg; no readonly __ldg or async ordering change",
        "gpu_executed": False,
    }
    (output.parent / "trail-load-cache-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output)))
