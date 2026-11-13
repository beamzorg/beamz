"""Actual C++ point-context offsets and state/profile bits vs original methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_cuda_cpml_point_context import helper
from scripts.patch_cuda_shared_profiles import ORIGINAL_COEFFICIENT


def method(text, name):
    begin = text.index(" __device__ __forceinline__ " + name)
    brace = text.index("{", begin)
    level = 1
    end = brace + 1
    while level:
        level += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[begin:end]


HARNESS = r'''
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>
#define __device__
#define __forceinline__ inline
namespace beamz::cuda::yee {inline int CpmlAxis(int term){const int axes[6]={1,0,0,2,2,1};return axes[term];}}
struct Psi {const float *input,*a,*b,*inv;float*output;int row,plane;};
struct PhaseData {Psi psi[6];};
struct Parameters {PhaseData phase[2];int physical_n[3];};
float __ldg(const float*p){return *p;}
template<int Axis>struct OriginalEnvironment{
 const Parameters&p;uint8_t flags=uint8_t(2u<<Axis);
 static constexpr bool HasCpml=true;
 static constexpr int ActiveAxes=1<<Axis;
@METHODS@
 float Coefficient(int kind,int index,int z,int y,int x)const{
@COEFFICIENT@
  return std::bit_cast<float>(uint32_t(0x3f000001u+kind*1024+index));
 }
};
@CONTEXT@
size_t address_checks=0,active_checks=0,profile_checks=0,state_checks=0,zero_checks=0,errors=0,negative_stride=0,negative_profile=0;
template<int Phase,int Axis>void run(){
 constexpr int dz[6]={0,0,1,1,1,0},dy[6]={0,1,0,1,0,1},dx[6]={1,0,0,0,1,1};
 for(int n:{24,25,31,32,35,64,136,272,512,564,680,1289}){
  Parameters p{};for(int a=0;a<3;++a)p.physical_n[a]=n;
  std::array<std::array<float,24>,36> profiles;
  std::array<std::vector<float>,12> values,outputs;
  for(int slot=0;slot<12;++slot){int field=(slot/6)*3+(slot%6)/2,axis=beamz::cuda::yee::CpmlAxis(slot%6),shape[3]={n+dz[field],n+dy[field],n+dx[field]};shape[axis]=24;
   auto&psi=p.phase[slot/6].psi[slot%6];psi.row=shape[2];psi.plane=shape[1]*shape[2];
   if(n<=35){const size_t words=size_t(shape[0])*shape[1]*shape[2];values[slot].resize(words);outputs[slot].resize(words);
    for(size_t i=0;i<words;++i){uint32_t bits=uint32_t(i*2654435761u)^uint32_t((slot+1)*0x5319u);if(i==0)bits=slot&1?0x80000000u:0u;values[slot][i]=std::bit_cast<float>(bits);outputs[slot][i]=std::bit_cast<float>(0x7fc01111u);}
    psi.input=values[slot].data();psi.output=outputs[slot].data();
   }
   for(int kind=0;kind<3;++kind){int id=slot*3+kind;for(int k=0;k<24;++k){uint32_t bits=0x3f000001u+id*1000+k;if(k==0)bits=id&1?0x80000000u:0u;profiles[id][k]=std::bit_cast<float>(bits);}}
   psi.a=profiles[slot*3].data();psi.b=profiles[slot*3+1].data();psi.inv=profiles[slot*3+2].data();
  }
  OriginalEnvironment<Axis> old{p};
  for(int coordinate:{-2,-1,0,1,11,12,n-13,n-12,n-11,n-1,n,n+1,n+2})
  for(int a:{-1,0,1,n-1,n,n+1})for(int b:{-1,0,1,n-1,n,n+1}){
   int point[3]={a,b,coordinate};if constexpr(Axis==0){point[0]=coordinate;point[1]=a;point[2]=b;}else if constexpr(Axis==1){point[0]=a;point[1]=coordinate;point[2]=b;}
   int z=point[0],y=point[1],x=point[2];
   CubicCpmlPointContext<Phase,Axis,OriginalEnvironment<Axis>>ctx(old,z,y,x);
   for(int slot=0;slot<12;++slot){
    errors+=ctx.LocalActive(slot,z,y,x)!=old.LocalActive(slot,z,y,x);++active_checks;
    if(ctx.Eligible(slot)){
     int want=old.PsiOffset(slot,z,y,x),actual=ctx.Offset(slot);errors+=want!=actual;++address_checks;
     int swapped=ctx.point_base+(slot%6!=ctx.ExtraTerm?ctx.point_delta:0);negative_stride+=want!=swapped;
    }
    int packed=old.Packed(slot,z,y,x);
    if(packed>=0&&packed<24){
     for(int kind=2;kind<5;++kind){uint32_t want=std::bit_cast<uint32_t>(old.Coefficient(kind,slot,z,y,x));uint32_t actual=std::bit_cast<uint32_t>(ctx.Coefficient(kind,slot,z,y,x));errors+=want!=actual;++profile_checks;if(packed==0)++zero_checks;
      if(ctx.Eligible(slot)){int other=(1-Phase)*6+slot%6;const auto&psi=p.phase[other/6].psi[other%6];const float*wrong=kind==2?psi.a:kind==3?psi.b:psi.inv;negative_profile+=want!=std::bit_cast<uint32_t>(wrong[packed]);}
     }
    }
    int field=(slot/6)*3+(slot%6)/2,shape[3]={n+dz[field],n+dy[field],n+dx[field]};
    bool valid=z>=0&&y>=0&&x>=0&&z<shape[0]&&y<shape[1]&&x<shape[2];
    if(n<=35&&valid&&old.LocalActive(slot,z,y,x)){
     errors+=std::bit_cast<uint32_t>(old.LoadLocal(slot,z,y,x))!=std::bit_cast<uint32_t>(ctx.LoadLocal(slot,z,y,x));++state_checks;
     uint32_t bits=(slot&1)?0x80000000u:0u;float v=std::bit_cast<float>(bits);ctx.StoreLocal(slot,z,y,x,v);
     int offset=old.PsiOffset(slot,z,y,x);errors+=std::bit_cast<uint32_t>(p.phase[slot/6].psi[slot%6].output[offset])!=bits;++state_checks;++zero_checks;
    }
   }
   for(int kind=0;kind<2;++kind)for(int field=0;field<6;++field){errors+=std::bit_cast<uint32_t>(ctx.Coefficient(kind,field,z,y,x))!=std::bit_cast<uint32_t>(old.Coefficient(kind,field,z,y,x));}
  }
 }
}
int main(){run<0,0>();run<0,1>();run<0,2>();run<1,0>();run<1,1>();run<1,2>();
 errors+=negative_stride==0||negative_profile==0;
 std::printf("{\"address_checks\":%zu,\"active_guard_checks\":%zu,\"profile_bit_checks\":%zu,\"state_load_store_checks\":%zu,\"signed_zero_checks\":%zu,\"wrong_extra_term_counterexamples\":%zu,\"cross_phase_profile_counterexamples\":%zu,\"errors\":%zu}\n",address_checks,active_checks,profile_checks,state_checks,zero_checks,negative_stride,negative_profile,errors);
 return errors!=0;
}
'''


def validate(source, output):
    source, output = Path(source), Path(output)
    consumer = (source / "src/asymmetric_native.cuh").read_text()
    context = helper()
    if consumer.count(context) != 1:
        raise ValueError("Actual trace-derived context differs")
    methods = "\n".join(method(consumer, name) for name in (
        "int Packed(", "bool LocalActive(", "int PsiOffset(", "float LoadLocal(", "void StoreLocal("))
    program = HARNESS.replace("@METHODS@", methods).replace("@COEFFICIENT@", ORIGINAL_COEFFICIENT).replace("@CONTEXT@", context)
    output.mkdir(parents=True, exist_ok=True)
    cpp, binary = output / "point-context.cc", output / "point-context"
    cpp.write_text(program)
    command = ["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-fno-fast-math", "-fsanitize=undefined", "-fno-sanitize-recover=undefined", str(cpp), "-o", str(binary)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    run = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    result = json.loads(run.stdout)
    result.update(actual_consumer_sha256=hashlib.sha256(consumer.encode()).hexdigest(),
                  context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                  oracle_sha256=hashlib.sha256(program.encode()).hexdigest(), command=command,
                  scope="Actual original Environment methods vs actual new Phase/Axis context; all6classes, N24..1289, outside/support/CPML boundaries, profile identities and state bits; UBSan integer bounds; no CUDA claim")
    (output / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(validate(a.source, a.output), indent=2))


if __name__ == "__main__":
    main()
