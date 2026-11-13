"""Compile actual ghost loops and prove the full Y/Z + low-X owner partition."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_cuda_noio_mux512 import HELPERS
from scripts.patch_cuda_noio_mux512_ghost import retire_helper, transformed
from scripts.prove_cuda_noio_mux512 import token_case


HARNESS = r'''
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstddef>
#define __device__
#define __forceinline__ inline
constexpr int LowXPoints=105,BY=15;
struct Thread{int x;};Thread threadIdx;
struct Parameters{};struct FieldPacket{float hx,hy,hz,ex,ey,ez;};
uint32_t packet_bits(int field,int version,int dz,int dy,int dx){return uint32_t(field*0x5319)^uint32_t(version*0x100000)^uint32_t(dz*0x10000)^uint32_t(dy*0x100)^uint32_t(dx);}
template<int Depth,bool Safe>FieldPacket ImportPacket(const Parameters&,const size_t*,int version,int,int,int,int dz,int dy,int dx){
 return {std::bit_cast<float>(packet_bits(0,version,dz,dy,dx)),std::bit_cast<float>(packet_bits(1,version,dz,dy,dx)),std::bit_cast<float>(packet_bits(2,version,dz,dy,dx)),std::bit_cast<float>(packet_bits(3,version,dz,dy,dx)),std::bit_cast<float>(packet_bits(4,version,dz,dy,dx)),std::bit_cast<float>(packet_bits(5,version,dz,dy,dx))};
}
@HELPERS@
@RETIRE@
template<int Depth>void ActualYZ(const Parameters&p,const size_t*bases,int n,int sy,int sx,float(&h)[3][8],float(&e)[3][8]){
 constexpr int Slots=8,ZStep=1;constexpr bool SupportSafe=false;
 const int zbase=0,hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31,dy=(sy-hy)&15,dx=(sx-hx)&31,oz=64,oy=64,ox=64;
@YZ@
}
int main(){size_t checks=0,retired=0,yz=0,intersection=0,negative=0,errors=0;
 float stage[6*LowXPoints];for(int k=0;k<6*LowXPoints;++k){uint32_t bits=uint32_t(k)*2654435761u;if(k==0)bits=0;if(k==1)bits=0x80000000u;stage[k]=std::bit_cast<float>(bits);}
 Parameters p;size_t bases[8]{};
 for(int n=1;n<=32;++n)for(int tid=0;tid<512;++tid){threadIdx.x=tid;const int sy=(tid/32)&15,sx=tid&31,hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31;
  float h[3][8],e[3][8];for(int c=0;c<3;++c)for(int j=0;j<8;++j){h[c][j]=std::bit_cast<float>(uint32_t(0x3f000001u+c*100+j));e[c][j]=std::bit_cast<float>(uint32_t(0x3e000001u+c*100+j));}
  ActualYZ<16>(p,bases,n,sy,sx,h,e);
  NoIOMux512RetireLow<512,8>(stage,h,e,n,sy,sx);
  for(int j=0;j<8;++j){const bool isyz=j==hz||sy==hy,isx=sx==hx&&sy!=hy&&j!=hz;
   errors+=isyz&&isx;intersection+=isyz&&(sx==hx);yz+=isyz;retired+=isx;
   const int dz=(j-hz)&7,dy=(sy-hy)&15,dx=(sx-hx)&31,key=(dz-1)*15+dy-1;
   for(int c=0;c<3;++c)for(int phase=0;phase<2;++phase){
    uint32_t want=phase?0x3e000001u+c*100+j:0x3f000001u+c*100+j;
    if(isyz)want=packet_bits(phase*3+c,n,dz,dy,dx);
    else if(isx)want=std::bit_cast<uint32_t>(stage[(phase*3+c)*LowXPoints+key]);
    const uint32_t actual=std::bit_cast<uint32_t>(phase?e[c][j]:h[c][j]);errors+=want!=actual;++checks;
    if(isyz)negative+=want!=packet_bits(phase*3+c,n,((j-((-n-2)&7))&7),dy,dx);
   }
  }
 }
 errors+=retired!=32*105||yz!=32*736||!negative;
 std::printf("{\"actual_ghost_and_unselected_word_bit_checks\":%zu,\"low_X_points\":%zu,\"YZ_points\":%zu,\"YZ_X_intersection_kept_by_YZ\":%zu,\"wrong_head_counterexamples\":%zu,\"errors\":%zu}\n",checks,retired,yz,intersection,negative,errors);return errors!=0;
}
'''


def validate(parent, source, output, versions=range(1, 33)):
    parent, source, output = Path(parent), Path(source), Path(output)
    original = (parent / "src/asymmetric_native.cuh").read_text()
    expected, patch = transformed(original)
    actual = (source / "src/asymmetric_native.cuh").read_text()
    if actual != expected:
        raise ValueError("Actual emitted ghost source differs")
    helper = retire_helper(original)
    begin = actual.index("  #pragma unroll 1\n  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;if(zs==hz||sy==hy)")
    end = actual.index("  StageRegisterLow<", begin)
    yz = actual[begin:end]
    if actual.count(HELPERS) != 1 or actual.count(helper) != 1:
        raise ValueError("Actual eight-word helpers or retirement differs")
    program = HARNESS.replace("@HELPERS@", HELPERS).replace("@RETIRE@", helper).replace("@YZ@", yz)
    output.mkdir(parents=True, exist_ok=True)
    cpp, binary = output / "ghost.cc", output / "ghost"
    cpp.write_text(program)
    command = ["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-fno-fast-math", "-fsanitize=undefined", "-fno-sanitize-recover=undefined", str(cpp), "-o", str(binary)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    bits = json.loads(subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout)
    cases = [token_case(n, seed) for n in versions for seed in (13, 37)]
    report = {"schema": "beamz.noio512-rolled-ghost-proof.v1", "actual_CXX_ghost_bits": bits,
              "field_token_cases": len(cases), "traced_field_reads": sum(x["actual_traced_field_read_checks"] for x in cases),
              "source_patch": patch, "actual_consumer_sha256": hashlib.sha256(actual.encode()).hexdigest(),
              "actual_ghost_code_sha256": hashlib.sha256((helper + yz).encode()).hexdigest(),
              "oracle_cpp_sha256": hashlib.sha256(program.encode()).hexdigest(), "command": command,
              "GPU_executed": False,
              "scope": "Actual emitted YZ import loop and low-X retirement compile against independent token-valued packet stub; all32 ring heads/all512 owners/eight slots/six fields, exact ghost partition736+105=841, unchanged owned words; full canonical field read tokens; no CUDA resource/speed claim"}
    (output / "proof.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    r = validate(a.parent, a.source, a.output)
    print(json.dumps({"actual_CXX_ghost_bits": r["actual_CXX_ghost_bits"], "field_token_cases": r["field_token_cases"], "traced_field_reads": r["traced_field_reads"]}))
