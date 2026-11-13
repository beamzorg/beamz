"""Actual C++ point predicates and coherent-load route oracle; no cache emulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_cuda_persistent_mixed_psi import BOUND, HELPER, MIXED_LOAD, PARENTS, SITE_ASSIGNMENTS, transformed
from scripts.validate_cuda_cpml_point_context import method


HARNESS = r'''
#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>
#define __device__
#define __forceinline__ inline
constexpr int CZ=8,CY=16;
namespace beamz::cuda::yee{int CpmlAxis(int term){const int axes[6]={1,0,0,2,2,1};return axes[term];}}
struct Psi{const float*input;int row,plane;};struct Phase{Psi psi[6];};struct Parameters{Phase phase[2];int physical_n[3];};
size_t cg=0,ca=0;const float*last_address=nullptr;
float __ldcg(const float*p){++cg;last_address=p;return*p;}
float __ldca(const float*p){++ca;last_address=p;return*p;}
@HELPER@
struct Base{const Parameters&p;uint8_t flags;
@PACKED@
@OFFSET@
};
struct ActualEnvironment:Base{ActualEnvironment(const Parameters&p,uint8_t facts):Base{p,facts}{}
@LOAD@
};
struct Facts{uint8_t flags;};
uint8_t paired_h(uint8_t flags,int n,int first,int zs,int ys,int sx){
 volatile int window[2]={first,16};const int hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31;Facts env{flags};@PH@return env.flags;
}
uint8_t paired_e0(uint8_t flags,int n,int first,int zs,int ys,int sx){
 volatile int window[2]={first,16};const int hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31,dz=(zs-hz)&7,dy=(ys-hy)&15,dx=(sx-hx)&31;Facts env{flags};@PE@return env.flags;
}
uint8_t paired_e1(uint8_t flags,int n,int first,int zs,int ys,int sx){
 volatile int window[2]={first,16};const int hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31,dz=(zs-hz)&7,dy=(ys-hy)&15,dx=(sx-hx)&31;Facts env{flags};@PE@return env.flags;
}
template<bool Fast>uint8_t uniform_h(uint8_t flags,int n,int first,int zs,int sy,int sx){
 volatile int window[2]={first,16};const int hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31,dy=(sy-hy)&15,dx=(sx-hx)&31;Facts env{flags};@UH@return env.flags;
}
template<bool Fast>uint8_t uniform_e(uint8_t flags,int n,int first,int zs,int sy,int sx){
 volatile int window[2]={first,16};const int ehz=(-n-1)&7,ehy=(-n-1)&15,ehx=(-n-1)&31;Facts env{flags};@UE@return env.flags;
}
int main(){size_t point_checks=0,continued=0,fresh=0,flags_checks=0,route_checks=0,prior=0,errors=0;
 for(int depth:{4,16})for(int first=1;first<=depth;++first)for(int n=first;n<=depth;++n)
 for(int zs=0;zs<8;++zs)for(int ys=0;ys<16;++ys)for(int sx=0;sx<32;++sx){
  const int hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31,dz=(zs-hz)&7,dy=(ys-hy)&15,dx=(sx-hx)&31;
  if(!dz||!dy||!dx)continue;
  const bool isfresh=n==first||dz==1||dy==1||dx==1;const uint8_t flags=uint8_t((zs*17+ys*7+sx)&255),want=uint8_t((flags&127u)|(isfresh?128u:0u));
  for(uint8_t actual:{@POINT_VALUES@}){errors+=actual!=want;++point_checks;}
  if(isfresh)++fresh;else{++continued;const int pn=n-1,pdz=(zs-((-pn-1)&7))&7,pdy=(ys-((-pn-1)&15))&15,pdx=(sx-((-pn-1)&31))&31;
   errors+=n<=first||!pdz||!pdy||!pdx||dz!=pdz+1||dy!=pdy+1||dx!=pdx+1;
   errors+=-n-1+dz!=-pn-1+pdz||-n-1+dy!=-pn-1+pdy||-n-1+dx!=-pn-1+pdx;++prior;
  }
 }
 for(int facts=0;facts<256;++facts)for(int n:{1,2,7,16})for(int first:{1,2})for(int delta:{1,2,7}){
  const uint8_t got=PersistentMixedPsiFlags(uint8_t(facts),n,first,delta,2,3);
  errors+=(got&127)!=(facts&127);errors+=bool(got&128)!=(n==first||delta==1);++flags_checks;
  // Fast bulk never mutates even incoming provenance; LoadLocal is dead.
  @FAST_CHECKS@
 }
 Parameters p{};for(int a=0;a<3;++a)p.physical_n[a]=32;std::vector<std::vector<float>>state(12);
 for(int slot=0;slot<12;++slot){auto&psi=p.phase[slot/6].psi[slot%6];psi.row=35+slot%3;psi.plane=(35+slot%2)*psi.row;state[slot].resize(36*psi.plane);
  for(size_t k=0;k<state[slot].size();++k){uint32_t bits=uint32_t(k*2654435761u)^uint32_t((slot+1)*0x5319u);if(k==0)bits=slot&1?0x80000000u:0;state[slot][k]=std::bit_cast<float>(bits);}psi.input=state[slot].data();
 }
 for(int facts=0;facts<256;++facts)for(int slot=0;slot<12;++slot)for(int coordinate:{0,11,20,31}){
  const int axis=beamz::cuda::yee::CpmlAxis(slot%6),z=axis==0?coordinate:1,y=axis==1?coordinate:1,x=axis==2?coordinate:1;ActualEnvironment env(p,uint8_t(facts));
  const auto&psi=p.phase[slot/6].psi[slot%6];const int packed=coordinate<12?coordinate:24+coordinate-(32+slot/6);
  const int offset=(axis==0?packed:z)*psi.plane+(axis==1?packed:y)*psi.row+(axis==2?packed:x);const size_t beforecg=cg,beforeca=ca;
  const float got=env.LoadLocal(slot,z,y,x);errors+=std::bit_cast<uint32_t>(got)!=std::bit_cast<uint32_t>(psi.input[offset]);errors+=last_address!=psi.input+offset;
  errors+=cg-beforecg!=size_t(bool(facts&128))||ca-beforeca!=size_t(!(facts&128));++route_checks;
 }
 std::printf("{\"actual_point_site_checks\":%zu,\"fresh_owned_points\":%zu,\"continued_owned_points\":%zu,\"same_point_prior_store_checks\":%zu,\"all_low7_fact_checks\":%zu,\"actual_LoadLocal_route_and_address_bit_checks\":%zu,\"CG_route_checks\":%zu,\"CA_route_checks\":%zu,\"errors\":%zu}\n",point_checks,fresh,continued,prior,flags_checks,route_checks,cg,ca,errors);return errors!=0;
}
'''


def validate(parent, source, output, variant):
    parent, source, output = Path(parent), Path(source), Path(output)
    old = (parent / "src/native_persistent.cuh").read_text()
    new = (source / "src/native_persistent.cuh").read_text()
    if new != transformed(old, variant):
        raise ValueError("Actual mixed-psi consumer differs from bound point-site lowering")
    for name, expected in BOUND.items():
        if hashlib.sha256((source / "src" / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Actual bound graph or environment changed:" + name)
    for name, assignment in SITE_ASSIGNMENTS.items():
        want = 2 if name == "paired_E0_E1" else 1
        if name.startswith("uniform") and variant == "p5":
            want = 0
        if new.count(assignment) != want:
            raise ValueError("Actual point assignment missing or duplicated:" + name)
    environment = (source / "src/asymmetric_native.cuh").read_text()
    load = method(new, "float LoadLocal(")
    if MIXED_LOAD not in load or "__ldg(" in load or ".nc" in load:
        raise ValueError("Mutable psi load route is not explicit coherent CG/CA")
    program = HARNESS.replace("@HELPER@", HELPER).replace("@PACKED@", method(environment, "int Packed(")).replace("@OFFSET@", method(environment, "int PsiOffset(")).replace("@LOAD@", load)
    for token, name in (("@PH@", "paired_H"), ("@PE@", "paired_E0_E1"), ("@UH@", "uniform_H"), ("@UE@", "uniform_E")):
        program = program.replace(token, SITE_ASSIGNMENTS[name])
    values = ["paired_h(flags,n,first,zs,ys,sx)", "paired_e0(flags,n,first,zs,ys,sx)", "paired_e1(flags,n,first,zs,ys,sx)"]
    if variant == "specialized":
        values += ["uniform_h<false>(flags,n,first,zs,ys,sx)", "uniform_e<false>(flags,n,first,zs,ys,sx)"]
    program = program.replace("@POINT_VALUES@", ",".join(values)).replace("@FAST_CHECKS@", "errors+=uniform_h<true>(uint8_t(facts),n,first,3,5,7)!=facts;errors+=uniform_e<true>(uint8_t(facts),n,first,3,5,7)!=facts;" if variant == "specialized" else "")
    output.mkdir(parents=True, exist_ok=True)
    cpp, binary = output / "point-routes.cc", output / "point-routes"
    cpp.write_text(program)
    command = ["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-fno-fast-math", "-fsanitize=undefined", "-fno-sanitize-recover=undefined", str(cpp), "-o", str(binary)]
    compilation = subprocess.run(command, capture_output=True, text=True)
    (output / "compile.log").write_text(compilation.stdout + compilation.stderr)
    compilation.check_returncode()
    run = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    report = {"schema": "beamz.mixed-psi-actual-point-route-proof.v1", "variant": variant,
              "actual_CXX": json.loads(run.stdout), "actual_consumer_sha256": hashlib.sha256(new.encode()).hexdigest(),
              "actual_load_method_sha256": hashlib.sha256(load.encode()).hexdigest(),
              "oracle_sha256": hashlib.sha256(program.encode()).hexdigest(), "command": command,
              "actual_point_assignment_sites_tested": len(values),
              "scope": "Actual emitted point-assignment expressions and LoadLocal/address method compiled on host under UBSan; low7 invariants, specialized Fast branch unchanged when present, all first versions and ring heads, prior same physical point, CG/CA address/bit routing; CPU intrinsics are route shims and do not emulate GPU coherence",
              "remaining_GPU_gates": "real cache handoff stress, ptxas resources, coherent PTX/store audit, all30 first/replay parity and sanitizers",
              "GPU_executed": False}
    (output / "proof.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--variant", required=True, choices=tuple(PARENTS))
    a = p.parse_args()
    print(json.dumps(validate(a.parent, a.source, a.output, a.variant), indent=2))
