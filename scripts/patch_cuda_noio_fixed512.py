"""Exact-shape NoIO CUDA entries with compiler-known canonical512 geometry.

All62 old entries and bodies remain. New entries use the same ordered field
graph and independent CPML storage/profiles; physics values stay parameters.
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

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA, derive_geometry
from scripts.patch_cuda_native_cpml_strides import ADMISSION, AXES, NEW as CANONICAL_OFFSET
from scripts.patch_cuda_noio_mux import original_body
from scripts.patch_cuda_shared_profiles import PARENT
from scripts.validate_cuda_cpml_point_context import method


def sha(data):
    return hashlib.sha256(data).hexdigest()


def derive_strides(n=512):
    geometry = derive_geometry()
    rows = []
    for phase in range(2):
        for term in range(6):
            axis = AXES[term]
            target = phase * 3 + term // 2
            shape = [n + d for d in SUPPORT_DELTA[target]]
            shape[axis] = 24
            ny = 24 if axis == 1 else n + (term // 2 != 1 if phase else term // 2 == 1)
            nx = 24 if axis == 2 else n + (term // 2 != 0 if phase else term // 2 == 0)
            if [shape[1], shape[2]] != [ny, nx]:
                raise ValueError("Canonical packed target strides disagree with typed supports")
            rows.append({"slot": phase * 6 + term, "target": target, "axis": axis,
                         "packed_shape": shape, "row": nx, "plane": ny * nx})
    return {"N": n, "slots": rows, "geometry_sha256": geometry["geometry_sha256"]}


def helper(original):
    derive_strides()
    methods = {}
    for key in ("bool Valid(", "int Packed(", "bool LocalActive(", "int PsiOffset(", "float LoadLocal(", "void StoreLocal(", "float Difference("):
        methods[key] = method(original, key)
    methods["bool Valid("] = methods["bool Valid("].replace("const int nz=p.physical_n[0],ny=p.physical_n[1],nx=p.physical_n[2];", "constexpr int nz=N,ny=N,nx=N;")
    for key in ("int Packed(", "bool LocalActive("):
        methods[key] = methods[key].replace("p.physical_n[axis]+family", "N+family")
    a = methods["int PsiOffset("].index("{")
    offset = CANONICAL_OFFSET.replace("p.physical_n[1]", "N").replace("p.physical_n[2]", "N")
    methods["int PsiOffset("] = methods["int PsiOffset("][:a+1] + "\n" + offset + "\n }"
    methods["float Difference("] = methods["float Difference("].replace("coordinate==p.physical_n[axis]", "coordinate==N")
    coefficient = r''' __device__ __forceinline__ float Coefficient(int kind,int index,int z,int y,int x)const{
  if(kind>=2){const auto&psi=p.phase[index/6].psi[index%6];const int packed=Packed(index,z,y,x);return kind==2?__ldg(psi.a+packed):kind==3?__ldg(psi.b+packed):__ldg(psi.inv+packed);}
  return Base::Coefficient(kind,index,z,y,x);
 }
'''
    text = r'''
// Only geometry is specialized; every phase/term retains distinct psi/profile pointers.
template<int N,int ActiveAxes=-1>struct FixedNoIOEnvironment:Environment<1,false,true,1,true,-1,false,ActiveAxes>{
 using Base=Environment<1,false,true,1,true,-1,false,ActiveAxes>;
 using Base::p;using Base::flags;
 static constexpr bool HasCpml=Base::HasCpml,SupportSafe=Base::SupportSafe;
 static_assert(N==512,"This experimental module admits exact512 geometry only");
 __device__ __forceinline__ FixedNoIOEnvironment(const Parameters&physics,uint8_t facts):Base{physics,facts}{}
'''
    text += "\n".join(methods.values()) + "\n" + coefficient + "};\n"
    if "p.physical_n" in text or "psi.row" in text or "psi.plane" in text:
        raise ValueError("Runtime geometry leaked into Fixed environment")
    return text


def clone_body(original):
    old = original_body(original)
    prefix = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody(const Parameters&p,const int32_t*tiles){"
    new_prefix = "template<int Depth,int N,int ActiveAxes=-1>\n__device__ __forceinline__ void FixedNoIOBody(const Parameters&p,const int32_t*tiles){\n constexpr int Layout=1,Policy=1;constexpr bool Fast=false,IsotropicH=true,UniformE=true;"
    text = old.replace(prefix, new_prefix, 1)
    env = "Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes>env{p,flags};"
    fixed = "FixedNoIOEnvironment<N,ActiveAxes>env{p,flags};"
    if text.count(env) != 1:
        raise ValueError("One exact original NoIO environment required")
    text = text.replace(env, fixed, 1)
    if text.replace(fixed, env, 1).replace(new_prefix, prefix, 1) != old:
        raise ValueError("Fixed NoIO body exact inverse failed")
    return text


WRAPPER = r'''
template<int Depth>
__global__ __launch_bounds__(1024,1) void AsymmetricNoIOFixed512(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 switch(p.flags[tiles[blockIdx.x]]&14){
  case 2:FixedNoIOBody<Depth,512,1>(p,tiles);return;
  case 4:FixedNoIOBody<Depth,512,2>(p,tiles);return;
  case 8:FixedNoIOBody<Depth,512,4>(p,tiles);return;
  default:FixedNoIOBody<Depth,512>(p,tiles);return;
 }
}
inline bool FixedNoIO512Eligible(const Parameters&p){
 return p.physical_n[0]==512&&p.physical_n[1]==512&&p.physical_n[2]==512;
}
'''


def transformed(files):
    for name, expected in PARENT.items():
        if sha(files[name]) != expected:
            raise ValueError("Exact selected52865 source required:" + name)
    host = files["native_extension.cc"].decode()
    if any(anchor not in host for anchor in ADMISSION):
        raise ValueError("Canonical actual target/slab admission changed")
    original = files["asymmetric_native.cuh"].decode()
    generated, body = helper(original), clone_body(original)
    marker = "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__ __launch_bounds__"
    if original.count(marker) != 1:
        raise ValueError("Original global insertion marker changed")
    insertion = generated + "\n" + body + "\n" + WRAPPER + "\n"
    text = original.replace(marker, insertion + marker, 1)
    anchor = "  const auto kernel=AsymmetricNoIOUniform1024<Depth>;\n"
    new = """  // Existing Layout1/isotropic/canonical slab/UniformE certificates still gate this branch.
  if(FixedNoIO512Eligible(p)){
   const auto fixed_kernel=AsymmetricNoIOFixed512<Depth>;
   status=cudaFuncSetAttribute(fixed_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
   AsymmetricNoIOFixed512<Depth><<<uniform_count,1024,SharedBytes,stream>>>(p,ids);
   return cudaGetLastError();
  }
""" + anchor
    if text.count(anchor) != 1:
        raise ValueError("Original NoIO uniform launch anchor changed")
    text = text.replace(anchor, new, 1)
    if text.replace(new, anchor, 1).replace(insertion, "", 1) != original:
        raise ValueError("Fixed-shape whole-source inverse failed")
    if body.count("__syncthreads()") != original_body(original).count("__syncthreads()"):
        raise ValueError("Fixed body barrier graph differs")
    result = dict(files)
    result["asymmetric_native.cuh"] = text.encode()
    return result, {"schema": "beamz.exact512-NoIO-geometry.v1", "exact_source_inverse": True,
                    "selector": "existing canonical Layout1 + lossless isotropicH + certified UniformE NoIO subrange AND physical_n exactly(512,512,512)",
                    "new_kernel_name": "AsymmetricNoIOFixed512", "expected_moving_kernels": 64,
                    "all_original62_entries_and_UniformBody_retained": True,
                    "shape_baked_only": True, "physics_parameters_and_all_profile_identities_retained": True,
                    "Initial_Final_import_and_window_geometry_kept_original": True,
                    "overrides": ["Valid", "Packed", "LocalActive", "PsiOffset", "LoadLocal", "StoreLocal", "Coefficient profiles", "Difference"],
                    "Load_Store_overrides_required_for_static_derived_PsiOffset_dispatch": True,
                    "ordered_field_FP_calls_and_barriers_unchanged": True,
                    "shared_bytes_unchanged": 51748, "geometry": derive_strides(),
                    "helper_sha256": sha(generated.encode()), "body_sha256": sha(body.encode()),
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
    (output.parent / "fixed512-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(patch_copy(a.source, a.output), indent=2))
