"""Hoist certified coincident source geometry without changing FP operations.

Only the SourceMask20 callback used by the paired v12 source kernels changes.
The admitted program already checks that all starts in each nonempty group are
coincident. The general callback keeps its original source-by-source geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT_MODULE = "a9394c3031d6635a875d673d50e7e09123807b7d9d2fe353c49fa05063b5941b"
PARENT_HASHES = {
    "asymmetric_native.cuh": "3b4039564fad23a88c8e801768a93b1ced302643d640886095e0f0ad1fc29aaf",
    "native_extension.cc": "ec7f48eb42ce5a6c5136e88ea2542c809d786762e26fe9bae8f8a2c211787f45",
    "asymmetric_native_cell_generated.cuh": "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5",
    "native_moving.h": "917fe9043362071cd1cac5ba503571a72de3d34b51b5e3c68eeea1fa35dbd7a9",
}
ORIGINAL = """    for(int source=0;source<s.count;++source){const int sz=z-s.starts[3*source],sy=y-s.starts[3*source+1],sx=x-s.starts[3*source+2];
     if(sz<0||sy<0||sx<0||sz>=s.nz||sy>=s.ny||sx>=s.nx)continue;
     int time=*s.current_step+p.macro_step+version-1;time=max(0,min(time,s.ntime-1));
     const int offset=((source*s.nz+sz)*s.ny+sy)*s.nx+sx;
     // Initial admission is coincident multi-source: match io.cu split rounds.
     value=__fadd_rn(value,__fmul_rn(__ldg(s.coefficients+offset),__ldg(s.waveforms+source*s.ntime+time)));
    }"""
COINCIDENT = """    if(s.count>0){
     // PrepareMoving validates every start against start[0]; use that fact only
     // in the SourceMask20 specialization. Empty groups never dereference it.
     const int sz=z-s.starts[0],sy=y-s.starts[1],sx=x-s.starts[2];
     if(sz>=0&&sy>=0&&sx>=0&&sz<s.nz&&sy<s.ny&&sx<s.nx){
      int time=*s.current_step+p.macro_step+version-1;time=max(0,min(time,s.ntime-1));
      for(int source=0;source<s.count;++source){
       const int offset=((source*s.nz+sz)*s.ny+sy)*s.nx+sx;
       // Preserve each product and ordered RN addition; no source reduction.
       value=__fadd_rn(value,__fmul_rn(__ldg(s.coefficients+offset),__ldg(s.waveforms+source*s.ntime+time)));
      }
     }
    }"""
REPLACEMENT = "    if constexpr(SourceMask==20){\n" + COINCIDENT + "\n    }else{\n" + ORIGINAL + "\n    }"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compose_text(text):
    if text.count(ORIGINAL) != 1:
        raise ValueError("Expected one exact original source callback")
    result = text.replace(ORIGINAL, REPLACEMENT, 1)
    if result.replace(REPLACEMENT, ORIGINAL, 1) != text:
        raise ValueError("Exact inverse failed")
    return result


def transformed(parent):
    for name, expected in PARENT_HASHES.items():
        if sha(parent[name]) != expected:
            raise ValueError(f"Exact a939 v12 parent required: {name}")
    text = parent["asymmetric_native.cuh"].decode()
    result = compose_text(text)
    boundary = "// Private source-only storage schedule; source-bound"
    if result[result.index(boundary):] != text[text.index(boundary):]:
        raise ValueError("Source storage schedule or host launches changed")
    admission = parent["native_extension.cc"].decode()
    if 'if(starts[3*s+a]!=starts[a])throw std::invalid_argument("moving coincident source starts differ")' not in admission:
        raise ValueError("Missing actual per-source coincident admission")
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
        "changed_consumer_regions": 1,
        "canonical_scalar_math_and_storage_and_host_admission_byte_identical": True,
        "scope": "SourceMask20 only: hoist coincident integer coordinates, bounds and time; unchanged per-source offset/product/add ordering",
        "gpu_executed": False,
    }
    (output.parent / "source-geometry-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output)))


if __name__ == "__main__":
    main()
