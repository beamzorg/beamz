"""CPU execution of actual source-plane predicate and FP callback bodies."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

try:
    from scripts.patch_cuda_v12_source_plane_pair import ORIGINAL, PLANE
except ModuleNotFoundError:
    from patch_cuda_v12_source_plane_pair import ORIGINAL, PLANE

HARNESS = r'''
#include "native_moving.h"
#include <algorithm>
#include <bit>
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
template<bool PlanePair=true>float candidate(const Source&s,const Params&p,int version,int z,int y,int x,float value){
@PLANE@
return value;}
uint32_t random_word(){static uint32_t q=0x91edba13;q^=q<<13;q^=q>>17;q^=q<<5;return q;}
float random_float(){uint32_t w=random_word();w=(w&0x807fffffu)|(((w>>23)%170+20)<<23);return std::bit_cast<float>(w);}
int main(){size_t checks=0,errors=0,facts=0,rejected=0,bounds=0;
 BeamzSourceGroupLaunch groups[9]{};int minimum[3]={64,96,128};
 int s5[6]={12,12,44,12,12,44},s7[6]={12,12,45,12,12,45};const int32_t*starts[9]{};starts[5]=s5;starts[7]=s7;
 for(int g:{5,7}){auto&c=groups[g].coefficients;c.rank=4;c.element_type=kBeamzF32;c.dims[0]=2;c.dims[1]=40;c.dims[2]=72;c.dims[3]=1;}
 errors+=!BeamzNativeSourcePlanePairFact(groups,9,starts,minimum,20);++facts;
 auto reject=[&](){rejected+=!BeamzNativeSourcePlanePairFact(groups,9,starts,minimum,20);++facts;};
 for(int g:{5,7})for(int a=0;a<4;++a){++groups[g].coefficients.dims[a];reject();--groups[g].coefficients.dims[a];}
 for(int*g:{s5,s7})for(int a:{0,1,3,4,5}){--g[a];reject();++g[a];}
 s5[2]=s5[5]=-1;reject();s5[2]=s5[5]=128;reject();s5[2]=s5[5]=44;
 for(int g:{5,7}){groups[g].coefficients.rank=3;reject();groups[g].coefficients.rank=4;}
 errors+=BeamzNativeSourcePlanePairFact(groups,9,starts,minimum,4);++facts;
 errors+=BeamzNativeSourcePlanePairFact(groups,8,starts,minimum,20);++facts;
 errors+=rejected!=22;
 // Policy3/4 comes from actual immutable trajectory-axis flags, not a point
 // test. Check the full owned interval extrema for each native axis.
 for(int n:{35,64,136,272,512})for(int depth:{4,16})for(int owned:{7,15,31})
 for(int q=0;q<(n+depth+owned)/owned;++q){int origin=q*owned,low=origin-depth,high=origin+owned-2;
  for(int f:{16,17}){int flags=f;if(low<12||high>=n-12)flags|=2;if(low<=0||high+1>=n)flags|=32;
   int policy=beamz::cuda::moving::Policy(uint8_t(flags));
   if(policy==3||policy==4){errors+=low<12||high>n-13;
    for(int version:{1,depth})for(int rel:{1,owned}){const int p=origin-version+rel-1;errors+=p<12||p>n-13;++bounds;}}
  }
 }
 for(int nz:{32,35,64,136})for(int ny:{32,49,96})for(int trial=0;trial<32;++trial){
  int count=2,sz=nz-24,sy=ny-24,nt=1+random_word()%17,step=int(random_word()%35)-8;
  Params p{int(random_word()%23)};int version=1+random_word()%16;
  int sx=23+int(random_word()%13),ss[6]={12,12,sx,12,12,sx};
  std::vector<float>coeff(2*sz*sy),wave(2*nt);for(auto&f:coeff)f=random_float();for(auto&f:wave)f=random_float();
  coeff[0]=std::bit_cast<float>(uint32_t(trial&1?0x80000000:0));wave[0]=std::bit_cast<float>(uint32_t(trial&2?0x80000000:0));
  Source s{count,sz,sy,1,nt,ss,&step,coeff.data(),wave.data()};
  for(int sample=0;sample<128;++sample){int z=12+random_word()%(nz-24),y=12+random_word()%(ny-24),x=sx+int(random_word()%5)-2;
   const float initial=random_float(),a=reference(s,p,version,z,y,x,initial),b=candidate(s,p,version,z,y,x,initial),c=candidate<false>(s,p,version,z,y,x,initial);
   errors+=std::bit_cast<uint32_t>(a)!=std::bit_cast<uint32_t>(b);errors+=std::bit_cast<uint32_t>(a)!=std::bit_cast<uint32_t>(c);++checks;
  }
 }
 std::printf("{\"fp_cases\":%zu,\"errors\":%zu,\"fact_cases\":%zu,\"rejected_topology_mutations\":%zu,\"owned_bounds_checks\":%zu}\n",checks,errors,facts,rejected,bounds);
 return errors||!bounds;
}
'''


def validate(source, output):
    c = (source / "src/asymmetric_native.cuh").read_text()
    if c.count(PLANE) != 1:
        raise ValueError("Actual emitted plane callback differs")
    output.mkdir(parents=True, exist_ok=True)
    cpp, binary = output / "source-plane.cc", output / "source-plane"
    text = HARNESS.replace("@ORIGINAL@", ORIGINAL).replace("@PLANE@", PLANE)
    cpp.write_text(text)
    command = ["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-fno-fast-math", "-I" + str(source / "src"), str(cpp), "-o", str(binary)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    run = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    result = json.loads(run.stdout)
    result.update({"actual_consumer_sha256": hashlib.sha256(c.encode()).hexdigest(),
                   "actual_header_sha256": hashlib.sha256((source / "src/native_moving.h").read_bytes()).hexdigest(),
                   "oracle_cxx_sha256": hashlib.sha256(text.encode()).hexdigest(), "command": command,
                   "scope": "Actual emitted host topology predicate, actual parent/new FP callback order, actual native Policy classifier; no CUDA execution"})
    (output / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(validate(a.source, a.output)))


if __name__ == "__main__":
    main()
