"""Copy-only NoIO512 rolled point loops with eight explicit scalar words.

Keeps the existing Uniform512 ZStep1/Slots8 ownership and static ancillary
loops. Tests extra per-thread register budget against half as many CTA warps.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_cuda_noio_mux import PARENT, mux_body, sha


HELPERS = r'''
// Eight statically indexed words; actual ptxas allocation remains a build gate.
__device__ __forceinline__ float NoIOSelect8(const float(&values)[8],int index){
 return index==0?values[0]:index==1?values[1]:index==2?values[2]:index==3?values[3]:index==4?values[4]:index==5?values[5]:index==6?values[6]:values[7];
}
__device__ __forceinline__ void NoIOUpdate8(float(&values)[8],int index,float value){
 if(index==0)values[0]=value;
 if(index==1)values[1]=value;
 if(index==2)values[2]=value;
 if(index==3)values[3]=value;
 if(index==4)values[4]=value;
 if(index==5)values[5]=value;
 if(index==6)values[6]=value;
 if(index==7)values[7]=value;
}
'''


def body512(original):
    text = mux_body(original)
    old = "constexpr int T=(Fast||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512,Slots=PAD/T,ZStep=T/512;"
    new = "constexpr int T=512,Slots=PAD/T,ZStep=T/512;"
    if text.count(old) != 1:
        raise ValueError("Original uniform ownership expression changed")
    text = text.replace(old, new, 1)
    text = text.replace("T==1024&&Slots==4&&ZStep==2", "T==512&&Slots==8&&ZStep==1", 1)
    text = text.replace("Certified NoIO four-word register banks only", "Certified NoIO eight-word register banks only", 1)
    text = text.replace("NoIOSelect4(", "NoIOSelect8(").replace("NoIOUpdate4(", "NoIOUpdate8(")
    text = text.replace("void NoIOMuxBody(", "void NoIOMux512Body(", 1)
    if "NoIOSelect4" in text or "NoIOUpdate4" in text:
        raise ValueError("Four-word helper remains")
    inverse = text.replace("void NoIOMux512Body(", "void NoIOMuxBody(", 1)
    inverse = inverse.replace("NoIOSelect8(", "NoIOSelect4(").replace("NoIOUpdate8(", "NoIOUpdate4(")
    inverse = inverse.replace("Certified NoIO eight-word register banks only", "Certified NoIO four-word register banks only", 1)
    inverse = inverse.replace("T==512&&Slots==8&&ZStep==1", "T==1024&&Slots==4&&ZStep==2", 1).replace(new, old, 1)
    if inverse != mux_body(original):
        raise ValueError("Exact eight-word ownership inverse failed")
    return text


def transformed(files):
    for name, expected in PARENT.items():
        if sha(files[name]) != expected:
            raise ValueError("Exact selected52865 source required:" + name)
    original = files["asymmetric_native.cuh"].decode()
    clone = body512(original)
    insertion = HELPERS + "\n" + clone
    marker = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__ __launch_bounds__"
    if original.count(marker) != 1:
        raise ValueError("One original global marker required")
    text = original.replace(marker, insertion + marker, 1)
    begin = text.index("__global__ __launch_bounds__(1024,1) void AsymmetricNoIOUniform1024(")
    end = text.index("template<int Depth>", begin)
    wrapper = text[begin:end]
    selected = wrapper.replace("UniformBody<", "NoIOMux512Body<").replace("__launch_bounds__(1024,1)", "__launch_bounds__(512,1)", 1)
    if wrapper.count("UniformBody<") != 4:
        raise ValueError("Original NoIO axis classes changed")
    text = text.replace(wrapper, selected, 1)
    launch = "AsymmetricNoIOUniform1024<Depth><<<uniform_count,1024,SharedBytes,stream>>>(p,ids);"
    launch512 = launch.replace("uniform_count,1024", "uniform_count,512")
    if text.count(launch) != 1:
        raise ValueError("Original certified NoIO launcher changed")
    text = text.replace(launch, launch512, 1)
    if text.replace(launch512, launch, 1).replace(selected, wrapper, 1).replace(insertion, "", 1) != original:
        raise ValueError("Whole-source inverse failed")
    result = dict(files)
    result["asymmetric_native.cuh"] = text.encode()
    return result, {"schema": "beamz.noio512-eight-scalar-point-mux.v1", "exact_source_inverse": True,
                    "selector": "only original certified NoIO wrapper; historical symbol name retains1024 but actual launch and launch_bounds are512",
                    "threads": 512, "slots": 8, "ZStep": 1, "zbase": "threadIdx.x/512==0",
                    "private_field_words_per_thread": 48,
                    "rolled_loops": "only main H/E j loops, component loops unrolled",
                    "original_UniformBody_byte_identical": True,
                    "all_init_import_publication_loops_retained_unrolled": True,
                    "canonical_FP_CPML_and_predicates_unchanged": True,
                    "shared_bytes_unchanged": 51748, "expected_moving_kernels": 62,
                    "GPU_executed": False,
                    "resource_claim": "No assumed register allocation:512threads permit at most128regs/thread on64K-register SM; zero-stack/spill build gate still required",
                    "performance_hypothesis": "smaller rolled compute with extra selection work and16 rather than32 warps; whole wrapper text is not active instruction working set",
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
    (output.parent / "mux512-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output), indent=2))
