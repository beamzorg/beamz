"""Copy-only NoIO support lowering from the actual native-boundary certificate.

CPML remains active. Only certified NoIO1024 tiles whose immutable host bit32
is clear bypass component support and native-wall checks in their body.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_cuda_shared_profiles import PARENT


def sha(value):
    return hashlib.sha256(value).hexdigest()


def transformed(files):
    for name, expected in PARENT.items():
        if sha(files[name]) != expected:
            raise ValueError("Exact selected52865 source required:" + name)
    host = files["native_moving.cc"].decode()
    certificate = "if(low[a]<=0||high[a]+1>=minimum[a])flags|=kNativeBoundary;"
    if host.count(certificate) != 1 or "low[a]=origin[a]-depth;high[a]=origin[a]+p.owned[a]-2;" not in host:
        raise ValueError("Actual native-boundary envelope changed")
    original = files["asymmetric_native.cuh"].decode()
    text, edits = original, []

    def change(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError("One exact support-fact anchor required:" + old)
        text = text.replace(old, new, 1)
        edits.append((old, new))

    change("int SourceMask=-1,bool PlanePair=false,int ActiveAxes=-1> struct Environment", "int SourceMask=-1,bool PlanePair=false,int ActiveAxes=-1,bool SafeNoIO=false> struct Environment")
    change(" static constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);", " static constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4)||SafeNoIO;\n static_assert(!SafeNoIO||(Policy==1&&!Fast&&UniformE&&IsotropicH&&Layout==1),\"Host-certified interior NoIO only\");")
    change("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody", "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool SafeNoIO=false>\n__device__ __forceinline__ void UniformBody")
    change(" constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);\n extern __shared__ float frame[];size_t*bases=", " constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4)||SafeNoIO;\n extern __shared__ float frame[];size_t*bases=")
    change(" Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes>env{p,flags};", " Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,SafeNoIO>env{p,flags};")
    start = text.index("__global__ __launch_bounds__(1024,1) void AsymmetricNoIOUniform1024(")
    end = text.index("template<int Depth>", start)
    before = text[start:end]
    selector = " // Original full certified NoIO grid; selector uniform before any barrier.\n"
    safe = " // Immutable host bit32 proves every owned/read/import support; CPML stays enabled.\n if(!(p.flags[tiles[blockIdx.x]]&32)){\n  switch(p.flags[tiles[blockIdx.x]]&14){\n"
    for flag, axis in ((2, 1), (4, 2), (8, 4)):
        safe += f"   case {flag}:UniformBody<Depth,1,false,true,1,true,{axis},true>(p,tiles);return;\n"
    safe += "   default:UniformBody<Depth,1,false,true,1,true,-1,true>(p,tiles);return;\n  }\n }\n"
    change(before, before.replace(selector, selector + safe))
    inverse = text
    for old, new in reversed(edits):
        if inverse.count(new) != 1:
            raise ValueError("Exact inverse anchor became ambiguous")
        inverse = inverse.replace(new, old, 1)
    if inverse != original or text.count("__syncthreads()") != original.count("__syncthreads()"):
        raise ValueError("Support-fact exact inverse or barrier audit failed")
    result = dict(files)
    result["asymmetric_native.cuh"] = text.encode()
    return result, {"schema": "beamz.noio-interior-support-fact.v1", "exact_source_inverse": True,
                    "selector": "existing certified NoIOUniform1024 AND immutable bit32 clear",
                    "CPML_kept_active": True, "startup_Initial_Contains_checks_kept": True,
                    "all_original_nonselected_templates_default_false": True,
                    "all_barrier_sites_identical": True, "shared_bytes_unchanged": 51748,
                    "expected_moving_kernels": 62, "GPU_executed": False,
                    "resource_claim": "No new source-level storage; actual ptxas resource gate required"}


def patch_copy(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        raise ValueError("New disjoint private source required")
    before = {name: (source / "src" / name).read_bytes() for name in PARENT}
    after, proof = transformed(before)
    shutil.copytree(source, output)
    for name, value in after.items():
        (output / "src" / name).write_bytes(value)
    proof.update(parent_hashes=PARENT, result_hashes={name: sha(value) for name, value in after.items()},
                 producer_sha256=sha(Path(__file__).read_bytes()))
    (output.parent / "support-fact-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output), indent=2))
