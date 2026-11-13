"""Apply the verified paired source field-storage body to certified NoIO tiles.

This experiment isolates its smaller private field state and directional H
exchange from source arithmetic. It does not yet add private/shared psi caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from patch_cuda_v12_coincident_source_geometry import PARENT_HASHES, PARENT_MODULE

ORIGINAL = """template<int Depth>
__global__ __launch_bounds__(1024,1) void AsymmetricNoIOUniform1024(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 // Host compact range contains only certified NoIO IDs.
 UniformBody<Depth,1,false,true,1,true>(p,tiles);
}

"""
REPLACEMENT = """template<int Depth>
__global__ __launch_bounds__(1024,1) void AsymmetricNoIOUniform1024(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 // Same certified NoIO prefix; paired storage changes no cell arithmetic.
 SourcePendingPairedBody<Depth,1,false,true,1,true,1024,-1>(p,tiles);
}

"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transformed(parent):
    for name, expected in PARENT_HASHES.items():
        if sha(parent[name]) != expected:
            raise ValueError(f"Exact a939 v12 parent required: {name}")
    original = parent["asymmetric_native.cuh"].decode()
    if original.count(ORIGINAL) != 1:
        raise ValueError("Expected one original NoIO wrapper")
    result = original.replace(ORIGINAL, "", 1)
    anchor = "inline bool MakeSource("
    if result.count(anchor) != 1:
        raise ValueError("Expected exact post-body placement")
    result = result.replace(anchor, REPLACEMENT + anchor, 1)
    start, end = result.index("inline cudaError_t LaunchNoIO("), result.index("inline cudaError_t LaunchObservers(")
    launch = result[start:end]
    if launch.count("SharedBytes") != 2:
        raise ValueError("Expected two NoIO shared allocation sites")
    result = result[:start] + launch.replace("SharedBytes", "SourcePendingSharedBytes") + result[end:]
    # Exact inverse: proves that no field/CPML operation or other body changes.
    inverse = result.replace(REPLACEMENT, "", 1)
    start, end = inverse.index("inline cudaError_t LaunchNoIO("), inverse.index("inline cudaError_t LaunchObservers(")
    inverse = inverse[:start] + inverse[start:end].replace("SourcePendingSharedBytes", "SharedBytes") + inverse[end:]
    observer = "template<int Depth>\n__global__ __launch_bounds__(1024,1) void AsymmetricObserverUniform1024("
    inverse = inverse.replace(observer, ORIGINAL + observer, 1)
    if inverse != original:
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
        "paired_body_byte_identical": True,
        "admission_key_host_plan_and_cell_arithmetic_byte_identical": True,
        "scope": "NoIO certified prefix only, same1024 threads, paired contiguousZ field storage20floats instead24, same canonical scalar cell; no psi caching",
        "shared_bytes": 58680,
        "barriers_per_step": 5,
        "gpu_executed": False,
    }
    (output.parent / "noio-paired-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output)))
