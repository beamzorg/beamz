"""Actual fixed-geometry C++ methods against original NoIO methods under UBSan."""

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

from scripts.patch_cuda_noio_fixed512 import PARENT, WRAPPER, clone_body, derive_strides, helper, transformed
from scripts.patch_cuda_noio_mux import original_body
from scripts.validate_cuda_cpml_point_context import method


HARNESS = r'''
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define __device__
#define __forceinline__ inline
float __ldg(const float*p){return*p;}
int32_t __ldg(const int32_t*p){return*p;}
float __fsub_rn(float a,float b){volatile float value=a-b;return value;}
float __fmul_rn(float a,float b){volatile float value=a*b;return value;}
namespace beamz::cuda::yee{
int CpmlAxis(int t){const int axes[6]={1,0,0,2,2,1};return axes[t];}
float ScaleYeeDifference(float difference,float inverse){return __fmul_rn(difference,inverse);}
float PackedMaterialSource(const float*,int,const uint8_t*,int){std::abort();}
}
struct Psi{const float*input,*a,*b,*inv;float*output;int row,plane;};
struct Phase{Psi psi[6];const float*decay[3],*source[3],*tables[3];const uint8_t*codes[3];int table_sizes[3];float inv;};
struct Parameters{Phase phase[2];int physical_n[3];float h_source,common_e[3];};
template<int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5),bool UniformE=false,int SourceMask=-1,bool PlanePair=false,int ActiveAxes=-1>struct Environment{
 static constexpr bool HasCpml=Policy==1||Policy==5,SupportSafe=Fast||(Policy>=2&&Policy<=4);
 const Parameters&p;uint8_t flags;
@METHODS@
};
@FIXED@
@SELECTOR@
size_t valid_checks=0,active_checks=0,address_checks=0,profile_checks=0,physics_checks=0,difference_checks=0,state_checks=0,zero_checks=0;
size_t wrong_phase_profile=0,metadata_dispatch_negative=0,shape_negative=0,selector_checks=0,errors=0;
uint32_t bits(float v){return std::bit_cast<uint32_t>(v);}
template<int Axes>void run(){
 constexpr int dz[6]={0,0,1,1,1,0},dy[6]={0,1,0,1,0,1},dx[6]={1,0,0,0,1,1};
 constexpr int coordinate[13]={-17,-1,0,1,11,12,499,500,501,511,512,513,543};
 Parameters p{};for(int a=0;a<3;++a)p.physical_n[a]=512;
 p.h_source=std::bit_cast<float>(0x3d123457u);p.common_e[0]=0.f;p.common_e[1]=-0.f;p.common_e[2]=std::bit_cast<float>(0x3f765431u);p.phase[0].inv=0.25f;p.phase[1].inv=std::bit_cast<float>(0x3f234567u);
 std::array<std::array<float,24>,36>profiles;
 for(int slot=0;slot<12;++slot){const int field=(slot/6)*3+(slot%6)/2,axis=beamz::cuda::yee::CpmlAxis(slot%6);int shape[3]={512+dz[field],512+dy[field],512+dx[field]};shape[axis]=24;
  auto&psi=p.phase[slot/6].psi[slot%6];psi.row=shape[2];psi.plane=shape[1]*shape[2];
  for(int kind=0;kind<3;++kind)for(int k=0;k<24;++k){uint32_t value=0x3f000001u+(slot*3+kind)*1000+k;if(k==0)value=(slot*3+kind)&1?0x80000000u:0;profiles[slot*3+kind][k]=std::bit_cast<float>(value);}
  psi.a=profiles[slot*3].data();psi.b=profiles[slot*3+1].data();psi.inv=profiles[slot*3+2].data();
 }
 Environment<1,false,true,1,true,-1,false,Axes> old{p,127};FixedNoIOEnvironment<512,Axes> fixed(p,127);
 static_assert(sizeof(fixed)==sizeof(old),"Geometry specialization adds no stored context");
 for(int z:coordinate)for(int y:coordinate)for(int x:coordinate){
  for(int field=0;field<6;++field){errors+=old.Valid(field,z,y,x)!=fixed.Valid(field,z,y,x);++valid_checks;}
  for(int slot=0;slot<12;++slot){errors+=old.Packed(slot,z,y,x)!=fixed.Packed(slot,z,y,x);errors+=old.PsiOffset(slot,z,y,x)!=fixed.PsiOffset(slot,z,y,x);++address_checks;
   errors+=old.LocalActive(slot,z,y,x)!=fixed.LocalActive(slot,z,y,x);++active_checks;
   const int packed=old.Packed(slot,z,y,x);if(packed>=0&&packed<24){for(int kind=2;kind<5;++kind){const uint32_t want=bits(old.Coefficient(kind,slot,z,y,x));errors+=want!=bits(fixed.Coefficient(kind,slot,z,y,x));++profile_checks;
    const int other=(1-slot/6)*6+slot%6;const auto&psi=p.phase[other/6].psi[other%6];const float*wrong=kind==2?psi.a:kind==3?psi.b:psi.inv;wrong_phase_profile+=want!=bits(wrong[packed]);if(packed==0)++zero_checks;
   }}
  }
  for(int kind=0;kind<2;++kind)for(int field=0;field<6;++field){errors+=bits(old.Coefficient(kind,field,z,y,x))!=bits(fixed.Coefficient(kind,field,z,y,x));++physics_checks;}
 }
 for(int flags=0;flags<128;++flags){old.flags=fixed.flags=uint8_t(flags);for(int slot=0;slot<12;++slot)for(int c:coordinate){const int axis=beamz::cuda::yee::CpmlAxis(slot%6),z=axis==0?c:1,y=axis==1?c:1,x=axis==2?c:1;errors+=old.LocalActive(slot,z,y,x)!=fixed.LocalActive(slot,z,y,x);++active_checks;}}
 const uint32_t values[]={0u,0x80000000u,1u,0x80000001u,0x3f800001u,0xbf800001u,0x00800001u,0x7e800000u};
 for(int phase=0;phase<2;++phase)for(int axis=0;axis<3;++axis)for(int c:coordinate)for(uint32_t a:values)for(uint32_t b:values){const int z=axis==0?c:1,y=axis==1?c:1,x=axis==2?c:1;
  const float center=std::bit_cast<float>(a),adjacent=std::bit_cast<float>(b);errors+=bits(old.Difference(phase,0,axis,z,y,x,center,adjacent))!=bits(fixed.Difference(phase,0,axis,z,y,x,center,adjacent));++difference_checks;
 }
 // Only current term state is allocated: bounded memory, distinct term/profile
 // identities remain represented and every term is exercised independently.
 old.flags=fixed.flags=127;
 for(int slot=0;slot<12;++slot){auto&psi=p.phase[slot/6].psi[slot%6];const int field=(slot/6)*3+(slot%6)/2,axis=beamz::cuda::yee::CpmlAxis(slot%6);int shape[3]={512+dz[field],512+dy[field],512+dx[field]};shape[axis]=24;
  std::vector<float>input(size_t(shape[0])*shape[1]*shape[2]),output(input.size());psi.input=input.data();psi.output=output.data();
  for(int c:{0,1,11,12,499,500,511,512})for(int a:{0,1,511,512})for(int b:{0,1,511,512}){const int z=axis==0?c:a,y=axis==1?c:axis==0?a:b,x=axis==2?c:b;
   if(!old.Valid(field,z,y,x)||!old.LocalActive(slot,z,y,x))continue;
   const int offset=old.PsiOffset(slot,z,y,x);const uint32_t value=uint32_t(offset*2654435761u)^uint32_t((slot+1)*0x5319u);input[offset]=std::bit_cast<float>(value);
   errors+=bits(old.LoadLocal(slot,z,y,x))!=bits(fixed.LoadLocal(slot,z,y,x));++state_checks;
   const uint32_t store=(slot&1)?0x80000000u:0u;fixed.StoreLocal(slot,z,y,x,std::bit_cast<float>(store));errors+=bits(output[offset])!=store;++state_checks;++zero_checks;
   // Fault injection proves Fixed Load/Store statically use Fixed PsiOffset,
   // not a base method. Poisoned descriptors are NOT admitted GPU inputs.
   const int row=psi.row,plane=psi.plane;psi.row=1;psi.plane=2;
   errors+=fixed.PsiOffset(slot,z,y,x)!=offset;errors+=bits(fixed.LoadLocal(slot,z,y,x))!=value;
   metadata_dispatch_negative+=old.PsiOffset(slot,z,y,x)!=offset;
   output[offset]=1.f;fixed.StoreLocal(slot,z,y,x,std::bit_cast<float>(store));errors+=bits(output[offset])!=store;
   psi.row=row;psi.plane=plane;
  }
 }
 // Physics values are never baked: changed common-E/H/inverse values must flow.
 for(uint32_t value:values){p.h_source=std::bit_cast<float>(value);for(float&v:p.common_e)v=std::bit_cast<float>(value);for(auto&phase:p.phase)phase.inv=std::bit_cast<float>(value);
  for(int field=0;field<6;++field){errors+=bits(old.Coefficient(1,field,1,1,1))!=bits(fixed.Coefficient(1,field,1,1,1));++physics_checks;}
  for(int phase=0;phase<2;++phase){errors+=bits(old.Difference(phase,0,0,1,1,1,1.25f,-0.5f))!=bits(fixed.Difference(phase,0,0,1,1,1,1.25f,-0.5f));++difference_checks;}
 }
}
int main(){run<-1>();run<1>();run<2>();run<4>();
 Parameters p{};for(auto dims:std::array<std::array<int,3>,8>{{{512,512,512},{511,512,512},{512,513,512},{512,512,511},{564,564,564},{32,32,32},{512,511,513},{513,512,512}}}){
  for(int a=0;a<3;++a)p.physical_n[a]=dims[a];const bool expected=dims==std::array<int,3>{512,512,512};errors+=FixedNoIO512Eligible(p)!=expected;++selector_checks;
  if(!expected){Environment<1,false,true,1,true>old{p,127};FixedNoIOEnvironment<512>fixed(p,127);for(int slot=0;slot<12;++slot)shape_negative+=old.Packed(slot,511,511,511)!=fixed.Packed(slot,511,511,511);for(int field=0;field<6;++field)shape_negative+=old.Valid(field,512,512,512)!=fixed.Valid(field,512,512,512);}
 }
 errors+=!wrong_phase_profile||!metadata_dispatch_negative||!shape_negative;
 std::printf("{\"Valid_checks\":%zu,\"LocalActive_checks\":%zu,\"Packed_PsiOffset_checks\":%zu,\"independent_profile_bit_checks\":%zu,\"physics_parameter_bit_checks\":%zu,\"ordered_Difference_bit_checks\":%zu,\"state_load_store_bit_checks\":%zu,\"signedzero_checks\":%zu,\"cross_phase_profile_negatives\":%zu,\"base_dispatch_metadata_fault_negatives\":%zu,\"wrong_shape_negatives\":%zu,\"actual_exact512_selector_checks\":%zu,\"errors\":%zu}\n",valid_checks,active_checks,address_checks,profile_checks,physics_checks,difference_checks,state_checks,zero_checks,wrong_phase_profile,metadata_dispatch_negative,shape_negative,selector_checks,errors);return errors!=0;
}
'''


def validate(source, output):
    source, output = Path(source), Path(output)
    files = {n: (source / "src" / n).read_bytes() for n in PARENT}
    before = dict(files)
    before["asymmetric_native.cuh"] = (Path(__file__).resolve().parents[1] / ".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src/asymmetric_native.cuh").read_bytes()
    expected, patch = transformed(before)
    if files != expected:
        raise ValueError("Actual fixed512 source/ABI differs from bound exact transformation")
    original = before["asymmetric_native.cuh"].decode()
    fixed = helper(original)
    names = ("bool Valid(", "int Packed(", "bool LocalActive(", "int PsiOffset(", "float LoadLocal(", "void StoreLocal(", "float Coefficient(", "float Difference(")
    methods = "\n".join(method(original, n) for n in names)
    selector = WRAPPER[WRAPPER.index("inline bool FixedNoIO512Eligible("):]
    program = HARNESS.replace("@METHODS@", methods).replace("@FIXED@", fixed).replace("@SELECTOR@", selector)
    output.mkdir(parents=True, exist_ok=True)
    cpp, binary = output / "fixed512.cc", output / "fixed512"
    cpp.write_text(program)
    command = ["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-fno-fast-math", "-fsanitize=undefined", "-fno-sanitize-recover=undefined", str(cpp), "-o", str(binary)]
    compilation = subprocess.run(command, capture_output=True, text=True)
    (output / "compile.log").write_text(compilation.stdout + compilation.stderr)
    compilation.check_returncode()
    run = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    body, oldbody = clone_body(original), original_body(original)
    calls = lambda t: re.findall(r"AdvanceNativeCell<[^>]+>", t)
    if calls(body) != calls(oldbody):
        raise AssertionError("Ordered canonical FP call list differs")
    fp_sites = lambda text: dict(Counter(re.findall(r"\b(__f(?:add|sub|mul|maf)_rn|ScaleYeeDifference)\s*\(", text)))
    original_difference = method(original, "float Difference(")
    fixed_difference = method(fixed, "float Difference(")
    if fp_sites(original_difference) != fp_sites(fixed_difference):
        raise AssertionError("Fixed Difference changes the ordered FP operation sites")
    result = {"schema": "beamz.actual-fixed512-method-proof.v1", "actual_CXX": json.loads(run.stdout),
              "source_patch": patch, "actual_consumer_sha256": hashlib.sha256(files["asymmetric_native.cuh"]).hexdigest(),
              "actual_environment_sha256": hashlib.sha256(fixed.encode()).hexdigest(),
              "ordered_canonical_cell_calls": calls(body), "geometry": derive_strides(),
              "unchanged_canonical_FP_operation_sites": fp_sites(files["asymmetric_native_cell_generated.cuh"].decode()),
              "original_and_fixed_Difference_FP_operation_sites": fp_sites(original_difference),
              "oracle_sha256": hashlib.sha256(program.encode()).hexdigest(), "command": command,
              "GPU_executed": False,
              "scope": "Actual original and fixed C++ methods under UBSan, canonical512 bounds/packed strides, four axis classes, all128 low7 flags, unequal independent phase/term profiles, state bits, ordered RN difference structure, mutable physics parameters; poisoned descriptors are negative fault injection only, never admitted inputs; actual512 selector rejects other shapes; no GPU resource or speed claim"}
    (output / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(validate(a.source, a.output), indent=2))
