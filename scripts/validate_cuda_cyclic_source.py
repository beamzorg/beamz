"""Execute the actual emitted mask10 topology and FP source callback on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_cuda_v12_coincident_source_geometry import ORIGINAL
from scripts.patch_cuda_cyclic_source import FACT10, PLANE10


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
template<bool PlanePair=true,int SourceMask=10>float candidate(const Source&s,const Params&p,int version,int z,int y,int x,float value){
@CALLBACK@
return value;}
uint32_t random_word(){static uint32_t q=0x671ca571;q^=q<<13;q^=q>>17;q^=q<<5;return q;}
float random_float(){uint32_t w=random_word();w=(w&0x807fffffu)|(((w>>23)%170+20)<<23);return std::bit_cast<float>(w);}
int main(){size_t checks=0,errors=0,facts=0,rejected=0;
 BeamzSourceGroupLaunch groups[9]{};int minimum[3]={128,64,96};
 int s4[6]={44,12,12,44,12,12},s6[6]={45,12,12,45,12,12};const int32_t*starts[9]{};starts[4]=s4;starts[6]=s6;
 for(int g:{4,6}){auto&c=groups[g].coefficients;c.rank=4;c.element_type=kBeamzF32;c.dims[0]=2;c.dims[1]=1;c.dims[2]=40;c.dims[3]=72;}
 errors+=!BeamzNativeSourcePlanePairFact10(groups,9,starts,minimum,10);++facts;
 auto reject=[&](){rejected+=!BeamzNativeSourcePlanePairFact10(groups,9,starts,minimum,10);++facts;};
 for(int g:{4,6})for(int a=0;a<4;++a){++groups[g].coefficients.dims[a];reject();--groups[g].coefficients.dims[a];}
 for(int*g:{s4,s6})for(int a:{1,2,3,4,5}){--g[a];reject();++g[a];}
 s4[0]=s4[3]=-1;reject();s4[0]=s4[3]=128;reject();s4[0]=s4[3]=44;
 for(int g:{4,6}){groups[g].coefficients.rank=3;reject();groups[g].coefficients.rank=4;}
 for(int g:{4,6}){groups[g].coefficients.element_type=kBeamzS32;reject();groups[g].coefficients.element_type=kBeamzF32;}
 errors+=BeamzNativeSourcePlanePairFact10(groups,9,starts,minimum,20);++facts;
 errors+=BeamzNativeSourcePlanePairFact10(groups,8,starts,minimum,10);++facts;
 errors+=rejected!=24;
 for(int ny:{32,35,64,136})for(int nx:{32,49,96})for(int trial=0;trial<32;++trial){
  int count=2,sy=ny-24,sx=nx-24,nt=1+random_word()%17,step=int(random_word()%35)-8;
  Params p{int(random_word()%23)};int version=1+random_word()%16;
  int sz=23+int(random_word()%13),ss[6]={sz,12,12,sz,12,12};
  std::vector<float>coeff(2*sy*sx),wave(2*nt);for(auto&f:coeff)f=random_float();for(auto&f:wave)f=random_float();
  coeff[0]=std::bit_cast<float>(uint32_t(trial&1?0x80000000:0));wave[0]=std::bit_cast<float>(uint32_t(trial&2?0x80000000:0));
  Source s{count,1,sy,sx,nt,ss,&step,coeff.data(),wave.data()};
  for(int sample=0;sample<128;++sample){int z=sz+int(random_word()%5)-2,y=12+random_word()%(ny-24),x=12+random_word()%(nx-24);
   const float initial=random_float(),a=reference(s,p,version,z,y,x,initial),b=candidate(s,p,version,z,y,x,initial),c=candidate<false>(s,p,version,z,y,x,initial);
   errors+=std::bit_cast<uint32_t>(a)!=std::bit_cast<uint32_t>(b);errors+=std::bit_cast<uint32_t>(a)!=std::bit_cast<uint32_t>(c);++checks;
  }
 }
 std::printf("{\"fp_cases\":%zu,\"errors\":%zu,\"fact_cases\":%zu,\"rejected_topology_mutations\":%zu}\n",checks,errors,facts,rejected);
 return errors;
}
'''


def callback(text):
    start = text.index("    if constexpr(PlanePair){")
    end = text.index("\n   }\n   if constexpr(HasObserve)", start)
    result = text[start:end]
    if result.count(PLANE10) != 1 or result.count(ORIGINAL) != 1:
        raise ValueError("Actual ordered callback binding differs")
    return result


def validate(source, output):
    source, output = Path(source), Path(output)
    header = (source / "src/native_moving.h").read_text()
    if header.count(FACT10) != 1:
        raise ValueError("Actual captured topology predicate differs")
    consumer = (source / "src/asymmetric_native.cuh").read_text()
    program = HARNESS.replace("@ORIGINAL@", ORIGINAL).replace("@CALLBACK@", callback(consumer))
    output.mkdir(parents=True, exist_ok=True)
    cpp, binary = output / "cyclic-source.cc", output / "cyclic-source"
    cpp.write_text(program)
    command = ["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-fno-fast-math", "-I" + str(source / "src"), str(cpp), "-o", str(binary)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    run = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    result = json.loads(run.stdout)
    result.update(actual_header_sha256=hashlib.sha256(header.encode()).hexdigest(),
                  actual_consumer_sha256=hashlib.sha256(consumer.encode()).hexdigest(),
                  oracle_sha256=hashlib.sha256(program.encode()).hexdigest(), command=command,
                  scope="Actual emitted host topology and scalar RN callbacks; CUDA execution remains a separate gate")
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
