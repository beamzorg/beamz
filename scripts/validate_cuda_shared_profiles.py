"""CPU bit-pattern execution of actual staged-profile helpers and index use."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_cuda_shared_profiles import HELPERS, NEW_COEFFICIENT, PARENT, constructor_witness


HARNESS = r'''
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <random>
#define __device__
#define __host__
#define __forceinline__ inline
struct Psi {const float *a,*b,*inv;};
struct Phase {Psi psi[6];};
struct Parameters {Phase phase[2];};
struct Thread {int x;} threadIdx;
int writes[864];
float __ldg(const float*p){return *p;}
@HELPERS@
int main(){size_t checks=0,errors=0,stage_checks=0,negative_alias_witnesses=0,negative_kind_witnesses=0,zero_checks=0;
 std::mt19937 rng(829771u);
 for(int trial=0;trial<128;++trial){
  std::array<std::array<float,24>,36> data;
  for(int p=0;p<36;++p)for(int k=0;k<24;++k){
   uint32_t bits=uint32_t(rng());
   if(k==0)bits=((p+trial)&1)?0x80000000u:0u;
   else if(k==1)bits=(uint32_t(100+p)<<23)|(uint32_t(k*17+trial)&0x7fffffu);
   data[p][k]=std::bit_cast<float>(bits);
  }
  Parameters packet{};
  for(int slot=0;slot<12;++slot){auto&psi=packet.phase[slot/6].psi[slot%6];psi={data[slot*3].data(),data[slot*3+1].data(),data[slot*3+2].data()};}
  // Deliberately alias selected read-only inputs as well as unequal ones.
  if(trial%3==0){packet.phase[1].psi[4].b=packet.phase[0].psi[1].inv;}
  std::array<float,864> shared;
  for(int i=0;i<864;++i){shared[i]=std::bit_cast<float>(0x7fc01234u);writes[i]=0;}
  std::array<int,1024> threads;for(int i=0;i<1024;++i)threads[i]=i;std::shuffle(threads.begin(),threads.end(),rng);
  for(int tid:threads){threadIdx.x=tid;StageIndependentCpmlProfiles(packet,shared.data());}
  for(int i=0;i<864;++i){errors+=writes[i]!=1;++stage_checks;}
  for(int slot=0;slot<12;++slot)for(int kind=2;kind<5;++kind)for(int packed=0;packed<24;++packed){
   const auto&psi=packet.phase[slot/6].psi[slot%6];const float*original=kind==2?psi.a:kind==3?psi.b:psi.inv;
   uint32_t want=std::bit_cast<uint32_t>(original[packed]);
   uint32_t actual=std::bit_cast<uint32_t>(ReadIndependentProfileCache(shared.data(),slot,kind,packed));
   errors+=want!=actual;++checks;
   if(packed==0){errors+=want!=actual;++zero_checks;}
   if(slot>=6)negative_alias_witnesses+=want!=std::bit_cast<uint32_t>(ReadIndependentProfileCache(shared.data(),slot%6,kind,packed));
   int swapped=kind==2?3:kind==3?2:4;
   negative_kind_witnesses+=want!=std::bit_cast<uint32_t>(ReadIndependentProfileCache(shared.data(),slot,swapped,packed));
  }
 }
 errors+=negative_alias_witnesses==0||negative_kind_witnesses==0;
 std::printf("{\"profile_bit_checks\":%zu,\"single_writer_checks\":%zu,\"signed_zero_checks\":%zu,\"cross_family_alias_counterexamples\":%zu,\"swapped_kind_counterexamples\":%zu,\"errors\":%zu}\n",checks,stage_checks,zero_checks,negative_alias_witnesses,negative_kind_witnesses,errors);
 return errors!=0;
}
'''


def validate(source, output):
    source, output = Path(source), Path(output)
    c = (source / "src/asymmetric_native.cuh").read_text()
    if c.count(HELPERS) != 1 or c.count(NEW_COEFFICIENT) != 1:
        raise ValueError("Actual independent-profile staging or coefficient route changed")
    # Bind exact helper statements; only a write counter is added for the CPU
    # diagnostic. No profile-address arithmetic or data operations are changed.
    helpers = HELPERS.replace("profiles[i]=__ldg(input+packed);", "profiles[i]=__ldg(input+packed);++writes[i];")
    if helpers == HELPERS:
        raise ValueError("CPU single-writer instrumentation anchor missing")
    program = HARNESS.replace("@HELPERS@", helpers)
    output.mkdir(parents=True, exist_ok=True)
    cpp, binary = output / "shared-profiles.cc", output / "shared-profiles"
    cpp.write_text(program)
    command = ["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-fno-fast-math", str(cpp), "-o", str(binary)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    run = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    result = json.loads(run.stdout)
    witness = constructor_witness({n: (source / "src" / n).read_bytes() for n in PARENT})
    result.update(actual_consumer_sha256=hashlib.sha256(c.encode()).hexdigest(),
                  oracle_sha256=hashlib.sha256(program.encode()).hexdigest(), command=command,
                  immutable_profile_admission=witness,
                  scope="Actual emitted staging/read helpers and coefficient route; CPU bit copies across independent H/E/term/kind profiles; no CUDA resource/numerical/timing claim")
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
