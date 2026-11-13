"""Roll Y/Z packet imports and low-X retirement in the reviewed NoIO512 mux.

Main-only experiment remains untouched. Startup and all shared/trail/final
publication loops retain their original static unrolling.
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


PARENT_CONSUMER = "18185af85f286b263bad94e0af8a69db7817a65803cfadce6093c99f4113d81d"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def retire_helper(text):
    begin = text.index("template<int T,int Slots>\n__device__ __forceinline__ void RetireRegisterLow(")
    end = text.index("template<int Depth,int Layout", begin)
    original = text[begin:end]
    result = original.replace("void RetireRegisterLow(", "void NoIOMux512RetireLow(", 1)
    result = result.replace(" if(sx==hx&&sy!=hy){", " static_assert(T==512&&Slots==8,\"Exact NoIO512 ghost bank\");\n if(sx==hx&&sy!=hy){", 1)
    result = result.replace("  #pragma unroll\n  for(int j=0;j<Slots;++j)", "  #pragma unroll 1\n  for(int j=0;j<Slots;++j)", 1)
    old = "h[c][j]=stage[c*LowXPoints+key];e[c][j]=stage[(c+3)*LowXPoints+key];"
    new = "NoIOUpdate8(h[c],j,stage[c*LowXPoints+key]);NoIOUpdate8(e[c],j,stage[(c+3)*LowXPoints+key]);"
    if result.count(old) != 1:
        raise ValueError("Exact low-X retirement assignments required")
    return result.replace(old, new, 1)


def transformed(text):
    if sha(text.encode()) != PARENT_CONSUMER:
        raise ValueError("Exact reviewed main-only512 mux required")
    original = text
    start = text.index("__device__ __forceinline__ void NoIOMux512Body(")
    end = text.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>", start)
    body = text[start:end]
    prefix = "  #pragma unroll\n  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;if(zs==hz||sy==hy){const int dz=(zs-hz)&7;"
    if body.count(prefix) != 1:
        raise ValueError("One exact Y/Z ghost packet loop required")
    patched = body.replace(prefix, prefix.replace("#pragma unroll", "#pragma unroll 1"), 1)
    old = "   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;"
    new = "   NoIOUpdate8(h[0],j,v.hx);NoIOUpdate8(h[1],j,v.hy);NoIOUpdate8(h[2],j,v.hz);NoIOUpdate8(e[0],j,v.ex);NoIOUpdate8(e[1],j,v.ey);NoIOUpdate8(e[2],j,v.ez);"
    if patched.count(old) != 1:
        raise ValueError("One exact Y/Z ghost assignment list required")
    patched = patched.replace(old, new, 1)
    call = "RetireRegisterLow<T,Slots>(low_stage,h,e,n,sy,sx);"
    if patched.count(call) != 1:
        raise ValueError("One exact low-X private retirement required")
    patched = patched.replace(call, call.replace("RetireRegisterLow", "NoIOMux512RetireLow"), 1)
    helper = retire_helper(original)
    # Helpers8 appear before this body template, so the new retirement helper
    # can call the same reviewed constant-index eight-word update function.
    template = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n" + body
    if text.count(template) != 1:
        raise ValueError("Exact cloned body template required")
    changed = helper + template.replace(body, patched, 1)
    text = text.replace(template, changed, 1)
    if text.replace(changed, template, 1) != original:
        raise ValueError("Ghost rolling exact inverse failed")
    if patched.count("__syncthreads()") != body.count("__syncthreads()"):
        raise ValueError("Ghost rolling changed barrier graph")
    return text, {"schema": "beamz.noio512-ghost-mux.v1", "exact_source_inverse": True,
                  "parent_consumer_sha256": PARENT_CONSUMER, "result_consumer_sha256": sha(text.encode()),
                  "rolled_loops": "main H/E points plus Y/Z ghost packet imports and low-X private retirement",
                  "startup_and_shared_trail_final_publication_loops_unmodified": True,
                  "CPML_canonical_FP_and_barrier_graph_unmodified": True,
                  "threads": 512, "slots": 8, "shared_bytes_unchanged": 51748,
                  "expected_moving_kernels": 62, "GPU_executed": False,
                  "resource_claim": "None; require actual zero-stack/spill build gate"}


def patch_copy(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        raise ValueError("New disjoint private source required")
    text, proof = transformed((source / "src/asymmetric_native.cuh").read_text())
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_text(text)
    proof["producer_sha256"] = sha(Path(__file__).read_bytes())
    (output.parent / "mux512-ghost-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output), indent=2))
