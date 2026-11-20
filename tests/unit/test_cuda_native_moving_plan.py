"""Compile the actual host planner and independently check its tile facts."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.cuda_source_layout import CUDA_INCLUDE_FLAGS

ROOT = Path(__file__).resolve().parents[2]


def test_compiled_moving_plan_coverage_sources_observations_and_bounds(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "proof.cc"
    kernel = (ROOT / "cuda/src/temporal/kernel.cuh").read_text()
    surface = re.search(r"int Surface\(.*?\n}\n", kernel, re.S).group()
    geometry = re.search(r"constexpr int BZ=.*?;", kernel).group()
    source.write_text(
        r"""
#include "native_macro_slab.h"
#include "rolling.cuh"
#include "native_moving_chunks.inc"
#include <cassert>
#include <numeric>
#include <set>
#include <stdexcept>
#define SURFACE_IMPLEMENTATION
static_assert(BZ==beamz::cuda::moving::kOwnedZ&&BY==beamz::cuda::moving::kOwnedY&&BX==beamz::cuda::moving::kOwnedX);

void CheckChunkAlignment() {
  std::vector<int> long_schedule;
  for (int i = 0; i < 20; ++i) long_schedule.insert(long_schedule.end(), {256, 207});
  const std::vector<int> schedules[] = {
      long_schedule, std::vector<int>(33, 1), {3, 5}, {511, 47}, {17}};
  for (int depth : {4, 16}) {
    for (const auto& requested : schedules) {
      const auto actual = BeamzAlignNativeMovingChunks(requested, depth);
      const int total = std::accumulate(requested.begin(), requested.end(), 0);
      assert(std::accumulate(actual.begin(), actual.end(), 0) == total);
      assert(actual.back() % depth == total % depth);
      for (size_t i = 0; i < actual.size(); ++i) {
        assert(actual[i] > 0 && actual[i] <= 256);
        if (i + 1 < actual.size()) assert(actual[i] % depth == 0);
      }
    }
  }
  const std::pair<int, std::vector<int>> invalid[] = {
      {3, {16}}, {16, {}}, {16, {1, 0}}, {4, {-1}}, {16, {INT32_MAX, 1}}};
  for (const auto& [depth, requested] : invalid) {
    try {
      BeamzAlignNativeMovingChunks(requested, depth);
      assert(false && "accepted invalid chunk schedule");
    } catch (const std::invalid_argument&) {}
  }
}

void CheckImmutableHaloViews() {
  namespace roll = beamz::cuda::rolling;
  const std::array<int,3> shape{159,81,93};
  const auto full = roll::MakePlan(shape,64);
  auto trace = [](const roll::Band& band,int z,int y,int x) {
    size_t start=band.offset/sizeof(roll::TileTrace);
    for (bool shell : {false,true}) {
      const auto& t=shell?band.shell:band.core;
      for (int k=0;k<t.count;++k) {
        const auto& box=t.boxes[k];
        if (z<box.lo[0] || z>=box.hi[0] || y<box.lo[1] || y>=box.hi[1] ||
            x<box.lo[2] || x>=box.hi[2]) continue;
        const int iz=(z-box.lo[0])/roll::ZPlanes, iy=(y-box.lo[1])/(shell?16:24);
        return int(start)+t.first[k]+(iz*t.ny[k]+iy)*t.nx[k]+(x-box.lo[2])/24;
      }
      start+=t.size();
    }
    return -1;
  };
  for (int axis : {0,1,2}) for (int low : {-1,47}) for (int high : {-1,112}) {
    const auto plan=roll::MakePlan(shape,64,{axis,low,high},32);
    assert(plan.mask_bytes==full.mask_bytes);
    for (int elapsed=0;elapsed<32;elapsed+=4) {
      const int halo=32-elapsed-4+1;
      for (int z=0;z<=shape[0];++z) for (int y : {0,15,16,40,65,81})
        for (int x : {0,16,40,78,93}) {
          const size_t b=size_t(z)/64;
          const int expected=trace(full.bands[b],z,y,x);
          assert(expected>=0);
          const auto [parts,count]=plan.View(b,elapsed);
          int actual=-1;
          for(size_t k=0;k<count;++k) if(int key=trace(parts[k],z,y,x);key>=0){
            assert(actual<0);actual=key;
          }
          const bool keep=axis!=0 || ((low<0 || z>=low-halo) && (high<0 || z<high+halo));
          assert(actual==(keep?expected:-1));
        }
    }
  }
}

int main(){
 CheckChunkAlignment();
 CheckImmutableHaloViews();
  std::set<int> surface;
  for(int z=0;z<BZ;++z)for(int y=0;y<BY;++y)for(int x=0;x<BX;++x){
    const int slot=Surface(z,y,x);
    if(z==BZ-1||y==BY-1||x==BX-1){assert(slot>=0&&slot<beamz::cuda::moving::kTrailSurfacePoints);assert(surface.insert(slot).second);}
    else assert(slot==-1);
  }
  assert(surface.size()==735);assert(Surface(BZ-1,1,0)%32==0);
  assert(Surface(1,BY-1,0)%32==0);assert(Surface(2,0,BX-1)%32==0);
  using namespace beamz::cuda::moving;
  for(int n:{33,64,136})for(int depth:{4,8,12,16,20,24,28,32})for(bool partition:{false,true}){
    const int slab_steps=kSlabMacros*depth;
    assert((BeamzAlignNativeMovingChunks({slab_steps-1,2*slab_steps+4},depth,slab_steps)==
            std::vector<int>{slab_steps,slab_steps,slab_steps,3}));
    BeamzBuffer f[6]{};const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
    for(int c=0;c<6;++c){f[c].rank=3;f[c].element_type=kBeamzF32;for(int a=0;a<3;++a)f[c].dims[a]=n+add[c][a];}
    BeamzSourceGroupLaunch groups[9]{};int32_t source[3]={n/2,n/2,n/2};const int32_t*starts[9]{};
    groups[3].coefficients.rank=4;groups[3].coefficients.dims[0]=1;
    for(int a=1;a<4;++a)groups[3].coefficients.dims[a]=1;starts[3]=source;
    std::vector<std::array<int,3>>obs{
        {0,0,0},{n,n-1,n-1},{n/2,n/2,n/2},{6,14,30},{13,29,31}};
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
        // A six-face CPML12 certificate also encloses every imported neighbor on
        // the two axes omitted from a single-axis tile's support checks.
        const int coords[3]={z,y,x},axes=(p.tile_flags[tile]>>1)&7;
        for(int a=0;a<3;++a){
          const bool single_axis=axes==1||axes==2||axes==4;
          if(!(p.tile_flags[tile]&kNativeBoundary)||(single_axis&&!(axes&(1<<a)))){
            assert(coords[a]*p.owned[a]-depth-1>=0);
            assert(coords[a]*p.owned[a]+p.owned[a]-1<n);
          }
        }
      }
    }
    assert(seen.size()==p.tile_ids.size());
    std::set<int> observed_tiles;
    for(int v=1;v<=depth;++v){int srcowner[3];for(int a=0;a<3;++a)srcowner[a]=(source[a]+v)/p.owned[a];
      int tile=(srcowner[0]*p.counts[1]+srcowner[1])*p.counts[2]+srcowner[2];assert(p.tile_flags[tile]&kMayHaveSources);
      for(auto point:obs){int owner[3];for(int a=0;a<3;++a)owner[a]=(point[a]+v)/p.owned[a];
        tile=(owner[0]*p.counts[1]+owner[1])*p.counts[2]+owner[2];observed_tiles.insert(tile);}
    }
    for(int tile:p.tile_ids)assert(bool(p.tile_flags[tile]&kMayObserve)==bool(observed_tiles.count(tile)));
    // A single shared generation stores full faces. Twelve edge owners fit
    // in the same row, with four generations live across the diagonal front.
    assert(p.surface_points==768);
    const size_t face_rows=p.max_frontier+(4*p.max_frontier+11)/12;
    assert(p.trail_floats==face_rows*depth*6*768);
    if(partition){
      for(int tile:p.tile_ids)if(tile%3==1)p.tile_flags[tile]|=64;
      BeamzCompactNativeMovingUniformTiles(p,true);
      // Independently order tasks by output level, policy, certification and
      // physical index. Each bank reuse must follow its previous final sample.
      for(int banks:{2,4,8})for(int sms:{0,82}) {
        constexpr int macros=7;
        const auto slab=BeamzBuildNativeMovingMacroSlab(p,macros,banks,true,sms);
        const auto estimate=BeamzBuildNativeMovingMacroSlab(p,macros,banks,false,sms);
        assert(estimate.tile_ids.empty() && estimate.slots.empty());
        assert(estimate.device_bytes()==slab.tile_ids.size()*sizeof(int32_t));
        if(n==136 && depth==4 && banks==2 && sms==0){
          for(int axis=0;axis<3;++axis){
            const std::array<int,3> cut={axis,n/3,2*n/3};
            const auto partial=BeamzBuildNativeMovingMacroSlab(p,macros,banks,true,sms,cut);
            const auto admission=BeamzBuildNativeMovingMacroSlab(p,macros,banks,false,sms,cut);
            assert(admission.tile_ids.empty() && admission.slots.empty());
            assert(partial.tile_ids.size()<partial.slots.size());
            assert(admission.device_bytes()==partial.tile_ids.size()*sizeof(int32_t));
            assert(admission.max_frontier==partial.max_frontier);
          }
        }
        assert(slab.scheduling_sms==sms && estimate.macro_level_stride==slab.macro_level_stride);
        const auto fixed=BeamzBuildNativeMovingMacroSlab(p,macros,banks,false,0);
        assert(slab.observation_slots<=fixed.observation_slots);
        assert(slab.macro_level_stride>=fixed.macro_level_stride);
        const double work=32.*p.tile_ids.size();
        assert(work+sms*double(slab.level_count)<=1.015*(work+sms*double(fixed.level_count)));
        assert(estimate.max_frontier==slab.max_frontier && estimate.observation_slots==slab.observation_slots);
        int first=int(p.bulk_counts.size()),last=-1;
        std::vector<std::array<int,5>> tasks;
        for(size_t rank=0;rank<p.bulk_counts.size();++rank){
          const int* offsets=&p.policy_offsets[rank*7];
          for(int policy=0;policy<6;++policy)for(int i=offsets[policy];i<offsets[policy+1];++i){
            const int tile=p.tile_ids[i];
            if(p.tile_flags[tile]&kMayObserve){first=std::min(first,int(rank));last=int(rank);}
            const int prefix=policy==1?p.uniform_noio_counts[rank]:policy==2?p.uniform_observer_counts[rank]:0;
            for(int m=0;m<macros;++m)tasks.push_back({int(rank)+m*slab.macro_level_stride,policy,
                int(i>=offsets[policy]+prefix),i,int(m*p.tile_ids.size())+tile});
          }
        }
        std::sort(tasks.begin(),tasks.end());
        int current=-1,begin=0;
        for(size_t i=0;i<tasks.size();++i){
          const auto [level,policy,bucket,order,encoded]=tasks[i];
          if(level!=current){begin=i;current=level;}
          assert(slab.tile_ids[i]==encoded && slab.slots[encoded]==int(i)-begin);
          const int* offsets=&slab.policy_offsets[level*7];
          assert(int(i)>=offsets[policy] && int(i)<offsets[policy+1]);
        }
        assert(slab.observation_end==last+1 && slab.observation_slots<=banks);
        assert(last<first+slab.observation_slots*slab.macro_level_stride);
      }
    }
    if (partition && depth==4) {
      CpmlWidths widths;
      widths.fill({12,12});
      const auto tensor=BeamzBuildNativeMovingPlan(f,depth,obs,groups,starts,widths,true,{},false,true);
      assert(tensor.owned==kTensorOwned && tensor.skew==2);
      assert(tensor.surface_points==size_t(kTensorTrailPoints));
      assert(tensor.trail_floats==size_t(4)*tensor.max_frontier*(9*depth-3)*kTensorTrailPoints);
      for(int a=0;a<3;++a)assert(tensor.counts[a]==(n+1+2*depth+tensor.owned[a]-1)/tensor.owned[a]);
    }
    // Contiguous X runs span multiple tiles, change row, skip gaps, and cross
    // component boundaries. Compare against the uncompressed point oracle.
    CpmlWidths widths; widths.fill({12,12});
    int64_t base=0; std::vector<int64_t> keys; std::vector<std::array<int,3>> points;
    for(int c=0;c<6;++c){
      const int ny=f[c].dims[1],nx=f[c].dims[2];
      for(int z:{0,n/2})for(int y:{1,2})for(int x=0;x<nx;++x){
        if(y==2&&x%5==2)continue;
        points.push_back({z,y,x});keys.push_back(base+(int64_t(z)*ny+y)*nx+x);
      }
      base+=f[c].dims[0]*ny*nx;
    }
    for(bool tensor:{false,true}){
      const auto full=BeamzBuildNativeMovingPlan(f,depth,points,groups,starts,widths,partition,{},false,tensor);
      const auto compact=BeamzBuildNativeMovingPlan(f,depth,{},groups,starts,widths,partition,{},false,tensor,keys);
      assert(full.tile_flags==compact.tile_flags && full.tile_ids==compact.tile_ids);
    }
    bool rejected=false;try{BeamzBuildNativeMovingPlan(f,3,{},nullptr,nullptr,12);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
  }
  // Two separated probes in a large volume need two banks at maximum
  // overlap. Moving the second wave to the last reader releases one bank.
  BeamzBuffer large[6]{};
  for(auto& field:large){field.rank=3;field.element_type=kBeamzF32;for(int a=0;a<3;++a)field.dims[a]=600;}
  auto sparse=BeamzBuildNativeMovingPlan(large,16,{{0,0,0},{224,224,224}},nullptr,nullptr,12,true);
  BeamzCompactNativeMovingUniformTiles(sparse,false);
  const auto fixed=BeamzBuildNativeMovingMacroSlab(sparse,kSlabMacros,4,false);
  const auto spaced=BeamzBuildNativeMovingMacroSlab(sparse,kSlabMacros,4,false,82);
  assert(fixed.observation_slots==2 && spaced.observation_slots==1);
  assert(spaced.macro_level_stride>fixed.macro_level_stride && spaced.max_frontier<fixed.max_frontier);

}
""".replace(
            "#define SURFACE_IMPLEMENTATION",
            geometry + "\n" + surface,
        )
    )
    binary = tmp_path / "proof"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            *CUDA_INCLUDE_FLAGS,
            str(source),
            str(ROOT / "cuda/src/temporal/host/native_moving.cc"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
