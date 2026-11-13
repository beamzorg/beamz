"""Common-shape AOT NoIO geometry plus independent shared CPML profiles.

Adds512/564 depth4/16 entries to the exact source-uniform/profile-cache best.
Other shapes use its original cached NoIO entry. This is not arbitrary-shape JIT.
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

from scripts.patch_cuda_noio_fixed512 import derive_strides, helper as fixed_helper
from scripts.patch_cuda_shared_profiles import PARENT as ORIGINAL_PARENT, constructor_witness
from scripts.patch_cuda_native_cpml_strides import ADMISSION


PARENT = dict(ORIGINAL_PARENT, **{
    "asymmetric_native.cuh": "713bdfc88315ede5d4d13f1e0ec9374403a6cf113a1a94188a7bfae94be0ec9a",
    "native_extension.cc": "e7d20ac35dc6e6cbacca1c15d5d80d7987e0b9050a9ecd4cf87f20510312bd09",
})


def sha(data):
    return hashlib.sha256(data).hexdigest()


def helper(original):
    text = fixed_helper(original).replace("FixedNoIOEnvironment", "FixedCubeCachedEnvironment")
    text = text.replace("1,true,-1,false,ActiveAxes>", "1,true,-1,false,ActiveAxes,true>")
    text = text.replace("using Base::p;using Base::flags;", "using Base::p;using Base::flags;using Base::profiles;")
    text = text.replace('static_assert(N==512,"This experimental module admits exact512 geometry only");', 'static_assert(N==512||N==564,"Common-shape AOT512/564 only");')
    text = text.replace("(const Parameters&physics,uint8_t facts):Base{physics,facts}{}", "(const Parameters&physics,uint8_t facts,const float*captured_profiles):Base{physics,facts,captured_profiles}{}")
    old = "  if(kind>=2){const auto&psi=p.phase[index/6].psi[index%6];const int packed=Packed(index,z,y,x);return kind==2?__ldg(psi.a+packed):kind==3?__ldg(psi.b+packed):__ldg(psi.inv+packed);}"
    new = "  if(kind>=2)return ReadIndependentProfileCache(profiles,index,kind,Packed(index,z,y,x));"
    if text.count(old) != 1:
        raise ValueError("Reviewed fixed CPML coefficient override changed")
    return text.replace(old, new, 1)


def original_body(text):
    begin = text.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool CachedProfiles=false>\n__device__ __forceinline__ void UniformBody")
    end = text.index("{", begin) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[begin:end+2]


def clone_body(original):
    old = original_body(original)
    prefix = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool CachedProfiles=false>\n__device__ __forceinline__ void UniformBody(const Parameters&p,const int32_t*tiles){"
    new_prefix = "template<int Depth,int N,int ActiveAxes=-1>\n__device__ __forceinline__ void FixedCubeCachedBody(const Parameters&p,const int32_t*tiles){\n constexpr int Layout=1,Policy=1;constexpr bool Fast=false,IsotropicH=true,UniformE=true,CachedProfiles=true;"
    text = old.replace(prefix, new_prefix, 1)
    env = "Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,CachedProfiles>env{p,flags,profiles};"
    fixed = "FixedCubeCachedEnvironment<N,ActiveAxes>env{p,flags,profiles};"
    if text.count(env) != 1:
        raise ValueError("Original cached environment construction changed")
    text = text.replace(env, fixed, 1)
    if text.replace(fixed, env, 1).replace(new_prefix, prefix, 1) != old:
        raise ValueError("Fixed cached body inverse failed")
    return text


WRAPPER = r'''
template<int Depth,int N>
__global__ __launch_bounds__(1024,1) void AsymmetricNoIOFixedCube(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 switch(p.flags[tiles[blockIdx.x]]&14){
  case 2:FixedCubeCachedBody<Depth,N,1>(p,tiles);return;
  case 4:FixedCubeCachedBody<Depth,N,2>(p,tiles);return;
  case 8:FixedCubeCachedBody<Depth,N,4>(p,tiles);return;
  default:FixedCubeCachedBody<Depth,N>(p,tiles);return;
 }
}
template<int N>inline bool FixedNoIOCubeEligible(const Parameters&p){
 static_assert(N==512||N==564,"Common-shape AOT512/564 only");
 return p.physical_n[0]==N&&p.physical_n[1]==N&&p.physical_n[2]==N;
}
'''


def transformed(files):
    for name, expected in PARENT.items():
        if sha(files[name]) != expected:
            raise ValueError("Exact current source-uniform/profile-cache best required:" + name)
    if any(anchor not in files["native_extension.cc"].decode() for anchor in ADMISSION):
        raise ValueError("Actual canonical field/slab admission changed")
    capture = constructor_witness(files)
    original = files["asymmetric_native.cuh"].decode()
    body, fixed = clone_body(original), helper(original)
    declaration = "constexpr size_t ProfileCachedSharedBytes=SharedBytes+864ULL*sizeof(float);"
    fixed_declaration = declaration + '\nconstexpr size_t FixedCubeSharedBytes=SharedBytes+864ULL*sizeof(float);\nstatic_assert(FixedCubeSharedBytes==55204&&FixedCubeSharedBytes==ProfileCachedSharedBytes,"Same independent864-profile allocation");'
    if original.count(declaration) != 1:
        raise ValueError("Original appended shared profile allocation changed")
    text = original.replace(declaration, fixed_declaration, 1)
    marker = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__ __launch_bounds__"
    insertion = fixed + "\n" + body + "\n" + WRAPPER + "\n"
    if text.count(marker) != 1:
        raise ValueError("Original global insertion marker changed")
    text = text.replace(marker, insertion + marker, 1)
    anchor = "  const auto kernel=AsymmetricNoIOUniform1024<Depth>;\n"
    checks = "  // Common-shape AOT only; all original capture/admission certificates still apply.\n"
    for n in (512, 564):
        checks += f"""  if(FixedNoIOCubeEligible<{n}>(p)){{
   const auto fixed_kernel=AsymmetricNoIOFixedCube<Depth,{n}>;
   status=cudaFuncSetAttribute(fixed_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,FixedCubeSharedBytes);if(status!=cudaSuccess)return status;
   AsymmetricNoIOFixedCube<Depth,{n}><<<uniform_count,1024,FixedCubeSharedBytes,stream>>>(p,ids);
   return cudaGetLastError();
  }}
"""
    if text.count(anchor) != 1:
        raise ValueError("Original cached NoIO fallback launch changed")
    text = text.replace(anchor, checks + anchor, 1)
    inverse = text.replace(checks + anchor, anchor, 1).replace(insertion, "", 1).replace(fixed_declaration, declaration, 1)
    if inverse != original or body.count("__syncthreads()") != original_body(original).count("__syncthreads()"):
        raise ValueError("Fixed cached whole source/barrier inverse failed")
    stage = body.index("if constexpr(CachedProfiles)StageIndependentCpmlProfiles(p,profiles);")
    if not stage < body.index("__syncthreads();", stage) < body.index("FixedCubeCachedEnvironment<N,ActiveAxes>env"):
        raise ValueError("Actual staging is not before existing startup barrier and first read")
    result = dict(files)
    result["asymmetric_native.cuh"] = text.encode()
    return result, {"schema": "beamz.common-shape-AOT-shared-profiles.v1", "exact_source_inverse": True,
                    "shapes": [512, 564], "expected_moving_kernels": 66, "all_original62_entries_retained": True,
                    "new_kernel_name": "AsymmetricNoIOFixedCube<Depth,N>", "AOT_common_shapes_not_arbitrary_shape_JIT": True,
                    "fixed_dynamic_shared_bytes": 55204, "independent_profile_words": 864,
                    "staging_before_existing_barrier_no_new_barrier": True,
                    "physics_values_and_profile_identities_not_baked": True,
                    "best_source_uniform_host_capture_unchanged": True,
                    "other_shapes_keep_original_cached_NoIO_fallback": True,
                    "private_capture_profile_immutability_binding": capture,
                    "geometry": [derive_strides(n) for n in (512, 564)],
                    "GPU_executed": False}


def patch_copy(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        raise ValueError("New disjoint private source required")
    before = {n: (source / "src" / n).read_bytes() for n in PARENT}
    after, proof = transformed(before)
    shutil.copytree(source, output)
    for name, data in after.items():
        (output / "src" / name).write_bytes(data)
    proof.update(parent_hashes=PARENT, result_hashes={n: sha(v) for n, v in after.items()}, producer_sha256=sha(Path(__file__).read_bytes()))
    (output.parent / "fixed-cubes-cached-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output), indent=2))
