"""Execute the actual original/hoisted CUDA callback bodies as scalar C++."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from patch_cuda_v12_coincident_source_geometry import COINCIDENT, ORIGINAL, transformed

HARNESS = r"""
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>
using std::min;using std::max;
float __fadd_rn(float a,float b){volatile float v=a+b;return v;}
float __fmul_rn(float a,float b){volatile float v=a*b;return v;}
float __ldg(const float*p){return *p;}
struct Source {int count,nz,ny,nx,ntime;const int*starts;const int*current_step;const float*coefficients;const float*waveforms;};
struct Params{int macro_step;};
float reference(const Source&s,const Params&p,int version,int z,int y,int x,float value){
@ORIGINAL@
return value;}
float candidate(const Source&s,const Params&p,int version,int z,int y,int x,float value){
@COINCIDENT@
return value;}
uint32_t random_word(){static uint32_t q=0x91edba13;q^=q<<13;q^=q>>17;q^=q<<5;return q;}
float random_float(){uint32_t w=random_word();w=(w&0x807fffffu)|(((w>>23)%180+20)<<23);return std::bit_cast<float>(w);}
int main(){
 size_t checks=0,errors=0,negative=0,empty=0;
 for(int count:{0,2,3,7})for(int trial=0;trial<128;++trial){
  const int nz=1+random_word()%4,ny=1+random_word()%4,nx=1+random_word()%4,nt=1+random_word()%17;
  int step=int(random_word()%35)-8;Params p{int(random_word()%23)};
  int version=1+random_word()%17;std::vector<int> starts(count*3);
  int origin[3]={int(random_word()%7)-3,int(random_word()%7)-3,int(random_word()%7)-3};
  for(int i=0;i<count;++i)for(int a=0;a<3;++a)starts[3*i+a]=origin[a];
  std::vector<float> coefficients(count*nz*ny*nx),waveforms(count*nt);
  for(auto&f:coefficients)f=random_float();for(auto&f:waveforms)f=random_float();
  if(!coefficients.empty()){coefficients[0]=std::bit_cast<float>(uint32_t(trial&1?0x80000000:0));waveforms[0]=std::bit_cast<float>(uint32_t(trial&2?0x80000000:0));}
  Source s{count,nz,ny,nx,nt,count?starts.data():nullptr,&step,count?coefficients.data():nullptr,count?waveforms.data():nullptr};
  for(int z=origin[0]-1;z<=origin[0]+nz;++z)for(int y=origin[1]-1;y<=origin[1]+ny;++y)for(int x=origin[2]-1;x<=origin[2]+nx;++x){
   const float value=random_float();const float a=reference(s,p,version,z,y,x,value),b=candidate(s,p,version,z,y,x,value);
   ++checks;empty+=count==0;errors+=std::bit_cast<uint32_t>(a)!=std::bit_cast<uint32_t>(b);
  }
  // This optimization requires the real coincident-start admission: ensure an
  // illegal changed start has a detectable numerical witness.
  if(count){for(auto&f:coefficients)f=1.f;for(auto&f:waveforms)f=1.f;starts[3]+=nz+3;
   const float a=reference(s,p,version,origin[0],origin[1],origin[2],0.f),b=candidate(s,p,version,origin[0],origin[1],origin[2],0.f);
   negative+=std::bit_cast<uint32_t>(a)!=std::bit_cast<uint32_t>(b);
  }
 }
 std::printf("{\"checks\":%zu,\"errors\":%zu,\"noncoincident_negative_witnesses\":%zu,\"empty_null_pointer_checks\":%zu}\n",checks,errors,negative,empty);
 return errors||negative!=384||empty==0;
}
"""


def validate(source, output):
    from patch_cuda_v12_coincident_source_geometry import PARENT_HASHES

    parent = {name: (source / "src" / name).read_bytes() for name in PARENT_HASHES}
    files = transformed(parent)
    if files["asymmetric_native.cuh"].count(COINCIDENT.encode()) != 1:
        raise ValueError("Actual transformed callback mismatch")
    output.mkdir(parents=True, exist_ok=False)
    cpp, binary = output / "source_geometry.cc", output / "source_geometry"
    cpp.write_text(HARNESS.replace("@ORIGINAL@", ORIGINAL).replace("@COINCIDENT@", COINCIDENT))
    subprocess.run(["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-fno-fast-math", str(cpp), "-o", str(binary)], check=True)
    result = json.loads(subprocess.check_output([str(binary)], text=True))
    result["actual_callback_cxx_sha256"] = hashlib.sha256(cpp.read_bytes()).hexdigest()
    result["scope"] = "CPU source-level exact FP32 order/geometry oracle; not a GPU correctness or performance claim"
    result["parent_hashes"] = PARENT_HASHES
    (output / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(validate(a.source.resolve(), a.output.resolve())))
