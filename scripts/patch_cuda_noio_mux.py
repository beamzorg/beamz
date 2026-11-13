"""Roll only NoIO1024 point loops using explicit four-scalar register muxes.

This copy-only variant tests code size against extra selection instructions.
Actual register allocation, stack, instruction-cache behavior and speed remain
build/measurement questions; source-level scalar indexing does not prove them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_cuda_shared_profiles import PARENT


HELPERS = r'''
// Explicit scalar indices keep the four private words visible to register SSA.
__device__ __forceinline__ float NoIOSelect4(const float(&values)[4],int index){
 return index==0?values[0]:index==1?values[1]:index==2?values[2]:values[3];
}
__device__ __forceinline__ void NoIOUpdate4(float(&values)[4],int index,float value){
 if(index==0)values[0]=value;
 if(index==1)values[1]=value;
 if(index==2)values[2]=value;
 if(index==3)values[3]=value;
}
'''


def sha(value):
    return hashlib.sha256(value).hexdigest()


def original_body(text):
    begin = text.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody")
    end = text.index("{", begin) + 1
    level = 1
    while level:
        level += (text[end] == "{") - (text[end] == "}")
        end += 1
    if text[end:end+2] != "\n\n":
        raise ValueError("Original body trailing separator changed")
    return text[begin:end+2]


def point_blocks(body):
    h_begin = body.index("  {const int step=*phase_step,hz=")
    h_end = body.index("  const int hz=(-n-1)", h_begin)
    e_begin = body.index("  {const int step=*phase_step,ehz=")
    e_end = body.index("  __syncthreads(); // B3", e_begin)
    return (body[h_begin:h_end], body[e_begin:e_end])


def mux_body(text):
    old = original_body(text)
    body = old.replace("void UniformBody(", "void NoIOMuxBody(", 1)
    declaration = " constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);"
    body = body.replace(declaration, declaration + "\n static_assert(T==1024&&Slots==4&&ZStep==2&&!Fast&&Layout==1&&IsotropicH&&Policy==1&&UniformE,\"Certified NoIO four-word register banks only\");", 1)
    originals, transformed_blocks = point_blocks(body), []
    for phase, block in enumerate(originals):
        marker = "   #pragma unroll\n   for(int j=0;j<Slots;++j)"
        if block.count(marker) != 1:
            raise ValueError("One exact point j loop required per phase")
        block = block.replace(marker, marker.replace("#pragma unroll", "#pragma unroll 1"), 1)
        destination = "h" if phase == 0 else "e"
        assignment = f"}}{destination}[c][j]=value;}}"
        if block.count(assignment) != 1:
            raise ValueError("One exact canonical point update assignment required")
        block = block.replace(assignment, f"}}NoIOUpdate4({destination}[c],j,value);}}", 1)
        block = re.sub(r"([he])\[([^\]]+)\]\[j\]", r"NoIOSelect4(\1[\2],j)", block)
        if re.search(r"[he]\[[^\]]+\]\[j\]", block):
            raise ValueError("Dynamic private point index remains")
        transformed_blocks.append(block)
    for before, after in zip(originals, transformed_blocks, strict=True):
        if body.count(before) != 1:
            raise ValueError("Point block replacement became ambiguous")
        body = body.replace(before, after, 1)
    # Exact inverse ensures every import, private/shared publication, source,
    # CPML, boundary predicate and barrier remains the original source.
    inverse = body
    for before, after in zip(originals, transformed_blocks, strict=True):
        inverse = inverse.replace(after, before, 1)
    inverse = inverse.replace(declaration + "\n static_assert(T==1024&&Slots==4&&ZStep==2&&!Fast&&Layout==1&&IsotropicH&&Policy==1&&UniformE,\"Certified NoIO four-word register banks only\");", declaration, 1)
    inverse = inverse.replace("void NoIOMuxBody(", "void UniformBody(", 1)
    if inverse != old:
        raise ValueError("Mux body exact inverse failed")
    return body


def transformed(files):
    for name, expected in PARENT.items():
        if sha(files[name]) != expected:
            raise ValueError("Exact selected52865 source required:" + name)
    original = files["asymmetric_native.cuh"].decode()
    clone = mux_body(original)
    insertion = HELPERS + "\n" + clone
    marker = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__ __launch_bounds__"
    if original.count(marker) != 1:
        raise ValueError("Exact original-body following marker required")
    text = original.replace(marker, insertion + marker, 1)
    begin = text.index("__global__ __launch_bounds__(1024,1) void AsymmetricNoIOUniform1024(")
    end = text.index("template<int Depth>", begin)
    wrapper = text[begin:end]
    selected = wrapper.replace("UniformBody<", "NoIOMuxBody<")
    if wrapper.count("UniformBody<") != 4:
        raise ValueError("Original four certified NoIO axis classes changed")
    text = text.replace(wrapper, selected, 1)
    if text.replace(selected, wrapper, 1).replace(insertion, "", 1) != original:
        raise ValueError("Whole source exact inverse failed")
    result = dict(files)
    result["asymmetric_native.cuh"] = text.encode()
    return result, {"schema": "beamz.noio-four-scalar-point-mux.v1", "exact_source_inverse": True,
                    "original_UniformBody_byte_identical": True,
                    "selector": "only original certified AsymmetricNoIOUniform1024 wrapper, all four axis classes",
                    "rolled_loops": "only main H and E per-point j; c remains unrolled",
                    "array_indices_in_helpers": [0, 1, 2, 3],
                    "all_init_import_publication_loops_retained": True,
                    "canonical_FP_CPML_and_predicates_unchanged": True,
                    "cloned_body_barriers": clone.count("__syncthreads()"),
                    "original_body_barriers": original_body(original).count("__syncthreads()"),
                    "shared_bytes_unchanged": 51748, "expected_moving_kernels": 62,
                    "GPU_executed": False, "resource_claim": "None: actual compiler may still allocate local memory; require zero-stack/spill exact build gate",
                    "clone_sha256": sha(clone.encode()), "helpers_sha256": sha(HELPERS.encode())}


def patch_copy(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        raise ValueError("New disjoint private source required")
    before = {n: (source / "src" / n).read_bytes() for n in PARENT}
    after, proof = transformed(before)
    shutil.copytree(source, output)
    for name, value in after.items():
        (output / "src" / name).write_bytes(value)
    proof.update(parent_hashes=PARENT, result_hashes={n: sha(v) for n, v in after.items()}, producer_sha256=sha(Path(__file__).read_bytes()))
    (output.parent / "mux-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output), indent=2))
