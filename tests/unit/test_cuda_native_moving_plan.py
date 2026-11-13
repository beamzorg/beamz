"""Compile the actual host planner and independently check its tile facts."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_compiled_moving_plan_coverage_sources_observations_and_bounds(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "proof.cc"
    source.write_text(r"""
#include "native_moving.h"
#include <cassert>
#include <set>
#include <stdexcept>
int main(){
  using namespace beamz::cuda::moving;
  for(int n:{33,64,136})for(int depth:{4,8,16,32})for(bool partition:{false,true}){
    BeamzBuffer f[6]{};const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
    for(int c=0;c<6;++c){f[c].rank=3;f[c].element_type=kBeamzF32;for(int a=0;a<3;++a)f[c].dims[a]=n+add[c][a];}
    BeamzSourceGroupLaunch groups[9]{};int32_t source[3]={n/2,n/2,n/2};const int32_t*starts[9]{};
    groups[3].coefficients.rank=4;groups[3].coefficients.dims[0]=1;
    for(int a=1;a<4;++a)groups[3].coefficients.dims[a]=1;starts[3]=source;
    std::vector<std::array<int,3>>obs{{0,0,0},{n,n-1,n-1},{n/2,n/2,n/2}};
    auto p=BeamzBuildNativeMovingPlan(f,depth,obs,groups,starts,12,partition);
    std::set<int>seen;
    for(size_t r=0;r<p.bulk_counts.size();++r){int begin=p.rank_offsets[r],end=p.rank_offsets[r+1];
      assert(end-begin<=p.max_frontier);
      if(partition){
        assert(p.policy_offsets.size()==p.bulk_counts.size()*7);
        assert(p.policy_offsets[7*r]==begin&&p.policy_offsets[7*r+6]==end);
        for(int policy=0;policy<6;++policy){
          int first=p.policy_offsets[7*r+policy],last=p.policy_offsets[7*r+policy+1];
          assert(first<=last);for(int i=first;i<last;++i)assert(Policy(p.tile_flags[p.tile_ids[i]])==policy);
        }
      }else assert(p.policy_offsets.empty());

      for(int i=begin;i<end;++i){int tile=p.tile_ids[i],x=tile%p.counts[2],y=(tile/p.counts[2])%p.counts[1],z=tile/(p.counts[2]*p.counts[1]);
        assert(z+y+x==int(r));assert(p.slots[tile]==i-begin);assert(seen.insert(tile).second);
        assert((i-begin<p.bulk_counts[r])==(p.tile_flags[tile]==0));
        if(!(p.tile_flags[tile]&kNativeBoundary)){
          int coords[3]={z,y,x};for(int a=0;a<3;++a){assert(coords[a]*p.owned[a]-depth-1>=0);assert(coords[a]*p.owned[a]+p.owned[a]-1<n);}
        }
      }
    }
    assert(seen.size()==p.tile_ids.size());
    for(int v=1;v<=depth;++v){int srcowner[3];for(int a=0;a<3;++a)srcowner[a]=(source[a]+v)/p.owned[a];
      int tile=(srcowner[0]*p.counts[1]+srcowner[1])*p.counts[2]+srcowner[2];assert(p.tile_flags[tile]&kMayHaveSources);
      for(auto point:obs){int owner[3];for(int a=0;a<3;++a)owner[a]=(point[a]+v)/p.owned[a];
        tile=(owner[0]*p.counts[1]+owner[1])*p.counts[2]+owner[2];assert(p.tile_flags[tile]&kMayObserve);}
    }
    assert(p.surface_points==645);assert(p.trail_floats==size_t(4)*p.max_frontier*depth*6*645);
    bool rejected=false;try{BeamzBuildNativeMovingPlan(f,3,{},nullptr,nullptr,12);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
  }
}
""")
    binary = tmp_path / "proof"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(ROOT / "cuda/src"),
            str(source),
            str(ROOT / "cuda/src/native_moving.cc"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
