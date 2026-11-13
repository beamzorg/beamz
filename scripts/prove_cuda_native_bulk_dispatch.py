"""Compile the actual planner and bulk partition against independent queue checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_cuda_native_bulk_uniform_dispatch import HOST_FUNCTION, once

CPP = r"""
#include "native_moving.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <set>
#include <stdexcept>
int main(){
 using namespace beamz::cuda::moving;
 size_t cases=0,tiles=0,neighbors=0,zero_uniform=0,zero_fallback=0,mixed=0;
 const std::array<int,3> shapes[]={{33,33,33},{64,96,128},{33,47,65},{136,136,136},{35,63,97},{29,29,29}};
 for(auto shape:shapes)for(int depth:{4,16})for(bool policies:{false,true})for(bool enabled:{false,true})for(int pattern=0;pattern<3;++pattern){
  BeamzBuffer fields[6]{};const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
  for(int c=0;c<6;++c){fields[c].rank=3;fields[c].element_type=kBeamzF32;for(int a=0;a<3;++a)fields[c].dims[a]=shape[a]+add[c][a];}
  BeamzSourceGroupLaunch groups[9]{};const int32_t*starts[9]{};
  int32_t source_starts[6]={shape[0]/2,shape[1]/2,shape[2]/2,shape[0]/2,shape[1]/2,shape[2]/2};
  groups[3].coefficients.rank=4;groups[3].coefficients.dims[0]=2;for(int a=1;a<4;++a)groups[3].coefficients.dims[a]=1;starts[3]=source_starts;
  std::vector<std::array<int,3>>observed{{0,0,0},{shape[0],shape[1]-1,shape[2]-1},{shape[0]/2,shape[1]/2,shape[2]/2}};
  auto p=BeamzBuildNativeMovingPlan(fields,depth,observed,groups,starts,12,policies);
  for(size_t tile=0;tile<p.tile_flags.size();++tile){const bool admit=pattern==1||(pattern==2&&tile%3==1);
   if((Policy(p.tile_flags[tile])==0&&admit)||(Policy(p.tile_flags[tile])==1&&(tile&1)))p.tile_flags[tile]|=64;
  }
  const auto old=p;BeamzCompactNativeMovingUniformBulk(p,enabled);
  assert(p.uniform_bulk_counts.size()==old.bulk_counts.size());
  assert(p.rank_offsets==old.rank_offsets&&p.policy_offsets==old.policy_offsets&&p.bulk_counts==old.bulk_counts&&p.tile_flags==old.tile_flags);
  assert(p.max_frontier==old.max_frontier&&p.trail_floats==old.trail_floats&&p.metadata_bytes==old.metadata_bytes);
  std::set<int> seen;
  for(size_t rank=0;rank<p.bulk_counts.size();++rank){
   const int rb=p.rank_offsets[rank],re=p.rank_offsets[rank+1],a=rb,b=rb+p.bulk_counts[rank];std::vector<int>expected;int count=0;
   if(enabled){for(int i=a;i<b;++i)if(old.tile_flags[old.tile_ids[i]]&64){expected.push_back(old.tile_ids[i]);++count;}for(int i=a;i<b;++i)if(!(old.tile_flags[old.tile_ids[i]]&64))expected.push_back(old.tile_ids[i]);}
   else expected.assign(old.tile_ids.begin()+a,old.tile_ids.begin()+b);
   assert(std::equal(expected.begin(),expected.end(),p.tile_ids.begin()+a));assert(p.uniform_bulk_counts[rank]==count);
   assert(std::equal(old.tile_ids.begin()+b,old.tile_ids.begin()+re,p.tile_ids.begin()+b));
   zero_uniform+=count==0;zero_fallback+=count==b-a;mixed+=count>0&&count<b-a;
   for(int i=a;i<b;++i)if(enabled)assert(bool(p.tile_flags[p.tile_ids[i]]&64)==(i<a+count));
   for(int i=rb;i<re;++i){const int tile=p.tile_ids[i];assert(seen.insert(tile).second&&p.slots[tile]==i-rb);
    const int z=tile/(p.counts[1]*p.counts[2]),y=(tile/p.counts[2])%p.counts[1],x=tile%p.counts[2];assert(z+y+x==int(rank));
    for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(z<dz||y<dy||x<dx)continue;
     const int producer=tile-dz*p.counts[1]*p.counts[2]-dy*p.counts[2]-dx,pr=int(rank)-dz-dy-dx,slot=p.slots[producer];
     assert(p.tile_ids[p.rank_offsets[pr]+slot]==producer);assert((pr&3)!=(rank&3));++neighbors;
    }
   }
  }
  assert(seen.size()==p.tile_ids.size());const auto applied=p;BeamzCompactNativeMovingUniformBulk(p,enabled);
  assert(p.tile_ids==applied.tile_ids&&p.slots==applied.slots&&p.uniform_bulk_counts==applied.uniform_bulk_counts);
  ++cases;tiles+=p.tile_ids.size();
 }
 assert(cases==144&&zero_uniform&&zero_fallback&&mixed);
 BeamzBuffer fields[6]{};for(int c=0;c<6;++c){fields[c].rank=3;fields[c].element_type=kBeamzF32;for(int a=0;a<3;++a)fields[c].dims[a]=136;}
 auto valid=BeamzBuildNativeMovingPlan(fields,4,{},nullptr,nullptr,12,true);int negative_checks=0;
 for(size_t rank=0;rank<valid.bulk_counts.size();++rank)if(valid.bulk_counts[rank]){
  auto bad=valid;bad.tile_flags[bad.tile_ids[bad.rank_offsets[rank]]]|=1;
  try{BeamzCompactNativeMovingUniformBulk(bad,true);assert(false);}catch(const std::invalid_argument&){++negative_checks;}
  bad=valid;bad.bulk_counts[rank]=bad.rank_offsets[rank+1]-bad.rank_offsets[rank]+1;
  try{BeamzCompactNativeMovingUniformBulk(bad,false);assert(false);}catch(const std::invalid_argument&){++negative_checks;}
  break;
 }
 assert(negative_checks==2);
 std::cout<<"{\"cases\":"<<cases<<",\"tiles\":"<<tiles<<",\"producer_links\":"<<neighbors<<",\"zero_uniform_ranges\":"<<zero_uniform<<",\"zero_fallback_ranges\":"<<zero_fallback<<",\"mixed_ranges\":"<<mixed<<",\"negative_checks\":"<<negative_checks<<",\"pass\":true}\n";
}
"""


def prove(source: Path, output: Path):
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is required for the actual planner proof")
    output.mkdir(parents=True, exist_ok=True)
    src = source / "src"
    header = (src / "native_moving.h").read_text()
    header = once(header, "  std::vector<int32_t> bulk_counts,policy_offsets;",
                  "  std::vector<int32_t> bulk_counts,policy_offsets,uniform_bulk_counts;")
    header += "\nvoid BeamzCompactNativeMovingUniformBulk(BeamzNativeMovingPlan&,bool);\n"
    host = (src / "native_moving.cc").read_text() + HOST_FUNCTION
    (output / "native_moving.h").write_text(header)
    (output / "host.cc").write_text(host)
    (output / "proof.cc").write_text(CPP)
    command = [compiler, "-std=c++17", "-O2", "-I", str(output), "-I", str(src),
               str(output / "proof.cc"), str(output / "host.cc"), "-o", str(output / "proof")]
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = subprocess.run([str(output / "proof")], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    report["compile_command"] = command
    report["input_sha256"] = {name: hashlib.sha256((src/name).read_bytes()).hexdigest()
                              for name in ["native_moving.h", "native_moving.cc"]}
    report["cpp_sha256"] = hashlib.sha256(CPP.encode()).hexdigest()
    report["helper_sha256"] = hashlib.sha256(HOST_FUNCTION.encode()).hexdigest()
    (output / "proof.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prove(args.source.resolve(), args.output.resolve())))


if __name__ == "__main__":
    main()
