"""Actual512/564 fixed methods and staged independent CPML profile bits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_cuda_noio_fixed_cubes_cached import PARENT, WRAPPER, clone_body, helper, original_body, transformed
from scripts.prove_cuda_noio_fixed512 import HARNESS as ORIGINAL_HARNESS
from scripts.validate_cuda_cpml_point_context import method


MAIN = r'''
template<int N>void check_selector(){Parameters p{};
 for(auto dims:std::array<std::array<int,3>,9>{{{N,N,N},{N-1,N,N},{N,N+1,N},{N,N,N-1},{512,512,512},{564,564,564},{32,32,32},{N,N-1,N+1},{N+1,N,N}}}){
  for(int a=0;a<3;++a)p.physical_n[a]=dims[a];const bool expected=dims==std::array<int,3>{N,N,N};errors+=FixedNoIOCubeEligible<N>(p)!=expected;++selector_checks;
  if(!expected){Environment<1,false,true,1,true>old{p,127};float cache[864]{};FixedCubeCachedEnvironment<N>fixed(p,127,cache);for(int slot=0;slot<12;++slot)shape_negative+=old.Packed(slot,N-1,N-1,N-1)!=fixed.Packed(slot,N-1,N-1,N-1);for(int field=0;field<6;++field)shape_negative+=old.Valid(field,N,N,N)!=fixed.Valid(field,N,N,N);}
 }
}
int main(){run<512,-1>();run<512,1>();run<512,2>();run<512,4>();run<564,-1>();run<564,1>();run<564,2>();run<564,4>();check_selector<512>();check_selector<564>();
 errors+=!wrong_phase_profile||!metadata_dispatch_negative||!shape_negative;
 std::printf("{\"Valid_checks\":%zu,\"LocalActive_checks\":%zu,\"Packed_PsiOffset_checks\":%zu,\"independent_profile_bit_checks\":%zu,\"physics_parameter_bit_checks\":%zu,\"ordered_Difference_bit_checks\":%zu,\"state_load_store_bit_checks\":%zu,\"signedzero_checks\":%zu,\"cross_phase_profile_negatives\":%zu,\"base_dispatch_metadata_fault_negatives\":%zu,\"wrong_shape_negatives\":%zu,\"actual_shape_selector_checks\":%zu,\"staged_independent_profile_bit_checks\":%zu,\"actual_cache_pointer_route_checks\":%zu,\"errors\":%zu}\n",valid_checks,active_checks,address_checks,profile_checks,physics_checks,difference_checks,state_checks,zero_checks,wrong_phase_profile,metadata_dispatch_negative,shape_negative,selector_checks,stage_checks,cache_routes,errors);return errors!=0;
}
'''


def harness():
    text = ORIGINAL_HARNESS[:ORIGINAL_HARNESS.index("int main(){")]
    text = text.replace("FixedNoIOEnvironment", "FixedCubeCachedEnvironment")
    text = text.replace("template<int Axes>void run(){", "template<int N,int Axes>void run(){")
    for literal, expression in ((543, "N+31"), (513, "N+1"), (512, "N"), (511, "N-1"), (501, "N-11"), (500, "N-12"), (499, "N-13")):
        text = re.sub(r"\b" + str(literal) + r"\b", expression, text)
    text = text.replace("#define __device__", "struct Thread{int x;};Thread threadIdx;\n#define __host__\n#define __device__", 1)
    text = text.replace("int ActiveAxes=-1>struct Environment", "int ActiveAxes=-1,bool CachedProfiles=false>struct Environment", 1)
    text = text.replace("const Parameters&p;uint8_t flags;", "const Parameters&p;uint8_t flags;const float*profiles=nullptr;", 1)
    text = text.replace("@METHODS@", "@CACHE_HELPERS@\n@METHODS@", 1)
    # Cache helpers belong at namespace scope, before the real base methods.
    text = text.replace("@CACHE_HELPERS@\n", "", 1)
    text = text.replace("template<int Layout,bool Fast", "@CACHE_HELPERS@\ntemplate<int Layout,bool Fast", 1)
    text = text.replace("size_t wrong_phase_profile=", "size_t stage_checks=0,cache_routes=0;\nsize_t wrong_phase_profile=", 1)
    marker = " Environment<1,false,true,1,true,-1,false,Axes> old{p,127};FixedCubeCachedEnvironment<N,Axes> fixed(p,127);"
    staged = """ float cached[864];for(float&v:cached)v=std::bit_cast<float>(0x7fc0deadU);
 for(int tid=0;tid<1024;++tid){threadIdx.x=tid;StageIndependentCpmlProfiles(p,cached);}
 for(int slot=0;slot<12;++slot)for(int kind=2;kind<5;++kind)for(int packed=0;packed<24;++packed){const auto&psi=p.phase[slot/6].psi[slot%6];const float*values=kind==2?psi.a:kind==3?psi.b:psi.inv;errors+=bits(cached[IndependentProfileCacheIndex(slot,kind,packed)])!=bits(values[packed]);++stage_checks;}
 Environment<1,false,true,1,true,-1,false,Axes> old{p,127};FixedCubeCachedEnvironment<N,Axes> fixed(p,127,cached);"""
    if text.count(marker) != 1:
        raise ValueError("Reviewed dual-env construction changed")
    text = text.replace(marker, staged, 1)
    # Prove the actual fixed coefficient uses its captured cache pointer, not
    # the still-independent global input. This is CPU fault injection only.
    marker = " // Physics values are never baked:"
    fault = """ for(int slot=0;slot<12;++slot)for(int kind=2;kind<5;++kind){const int axis=beamz::cuda::yee::CpmlAxis(slot%6),z=axis==0?0:1,y=axis==1?0:1,x=axis==2?0:1,index=IndependentProfileCacheIndex(slot,kind,0);const float saved=cached[index];const uint32_t modified=bits(saved)^0x80000001u;cached[index]=std::bit_cast<float>(modified);errors+=bits(fixed.Coefficient(kind,slot,z,y,x))!=modified;errors+=bits(old.Coefficient(kind,slot,z,y,x))!=bits(saved);cached[index]=saved;++cache_routes;}
"""
    text = text.replace(marker, fault + marker, 1)
    return text + MAIN


def validate(source, output):
    source, output = Path(source), Path(output)
    files = {n: (source / "src" / n).read_bytes() for n in PARENT}
    before = dict(files)
    before["asymmetric_native.cuh"] = (Path(__file__).resolve().parents[1] / ".cache/cuda-trace-five-hour-2026-11-13-1059/source-uniform-profiles-v1/source/src/asymmetric_native.cuh").read_bytes()
    expected, patch = transformed(before)
    if files != expected:
        raise ValueError("Actual cached fixed-cube source/capture ABI differs")
    original = before["asymmetric_native.cuh"].decode()
    methods = "\n".join(method(original, n) for n in ("bool Valid(", "int Packed(", "bool LocalActive(", "int PsiOffset(", "float LoadLocal(", "void StoreLocal(", "float Coefficient(", "float Difference("))
    def free_function(name):
        at = original.index(name)
        start = original.rfind("\n", 0, at) + 1
        end = original.index("{", at) + 1
        level = 1
        while level:
            level += (original[end] == "{") - (original[end] == "}")
            end += 1
        return original[start:end]
    cache = "\n".join(free_function(n) for n in ("int IndependentProfileCacheIndex(", "float ReadIndependentProfileCache(", "void StageIndependentCpmlProfiles("))
    selector = WRAPPER[WRAPPER.index("template<int N>inline bool FixedNoIOCubeEligible("):]
    program = harness().replace("@METHODS@", methods).replace("@FIXED@", helper(original)).replace("@SELECTOR@", selector).replace("@CACHE_HELPERS@", cache)
    output.mkdir(parents=True, exist_ok=True)
    cpp, binary = output / "fixed-cubes.cc", output / "fixed-cubes"
    cpp.write_text(program)
    command = ["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-fno-fast-math", "-fsanitize=undefined", "-fno-sanitize-recover=undefined", str(cpp), "-o", str(binary)]
    compilation = subprocess.run(command, capture_output=True, text=True)
    (output / "compile.log").write_text(compilation.stdout + compilation.stderr)
    compilation.check_returncode()
    run = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    calls = lambda text: re.findall(r"AdvanceNativeCell<[^>]+>", text)
    if calls(clone_body(original)) != calls(original_body(original)):
        raise AssertionError("Canonical ordered FP call sequence changed")
    fp = lambda text: dict(Counter(re.findall(r"\b(__f(?:add|sub|mul|maf)_rn|ScaleYeeDifference)\s*\(", text)))
    if fp(method(original, "float Difference(")) != fp(method(helper(original), "float Difference(")):
        raise AssertionError("Fixed Difference FP operation sites changed")
    result = {"schema": "beamz.actual-cached-fixed-cubes-proof.v1", "shapes": [512, 564], "actual_CXX": json.loads(run.stdout),
              "source_patch": patch, "actual_consumer_sha256": hashlib.sha256(files["asymmetric_native.cuh"]).hexdigest(),
              "oracle_sha256": hashlib.sha256(program.encode()).hexdigest(), "command": command,
              "unchanged_canonical_FP_sites": fp(files["asymmetric_native_cell_generated.cuh"].decode()),
              "ordered_canonical_calls": calls(clone_body(original)), "GPU_executed": False,
              "scope": "Actual512/564 fixed methods versus original best cached-capable methods under UBSan; actual864-profile staging/read helpers, unequal per-phase/term/kind bits, cache-pointer route fault injection, canonical addresses/state/bounds, four axis classes, shape rejection, unchanged private host capture; common-shape AOT only, no GPU resource/speed claim"}
    (output / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = validate(a.source, a.output)
    print(json.dumps({"actual_CXX": result["actual_CXX"], "consumer_sha256": result["actual_consumer_sha256"]}, indent=2))
