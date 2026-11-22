#include "native_moving.h"
#include <algorithm>
#include <limits>
#include <cstring>
#include <cmath>
#include <stdexcept>

BeamzNativeMovingPlan BeamzBuildNativeMovingPlan(
    const BeamzBuffer*fields,int depth,const std::vector<std::array<int,3>>&observed_points,
    const BeamzSourceGroupLaunch*groups,const int32_t*const*host_starts,
    const beamz::cuda::moving::CpmlWidths& widths,bool partition_policies,
    const std::vector<beamz::cuda::moving::SourceCell>& sparse_sources,
    bool record_volume, bool tensor, const std::vector<int64_t>& observation_keys, unsigned transparent_faces){
  using namespace beamz::cuda::moving;
  if(!ValidDepth(depth))throw std::invalid_argument("temporal depth must be a multiple of four from 4 to 32");
  BeamzNativeMovingPlan p;p.depth=depth;
  if(tensor){p.owned=kTensorOwned;p.skew=2;}
  const int reach=p.skew*depth;
  int minimum[3]{INT32_MAX,INT32_MAX,INT32_MAX};
  for(int c=0;c<6;++c){
    if(fields[c].rank!=3||fields[c].element_type!=kBeamzF32)throw std::invalid_argument("moving requires six FP32 fields");
    int64_t count=1;
    for(int a=0;a<3;++a){const int64_t n=fields[c].dims[a];
      if(n<1||n>INT32_MAX||count>INT32_MAX/n)throw std::invalid_argument("moving field offsets exceed int32");
      count*=n;p.extent[a]=std::max(p.extent[a],int(n));minimum[a]=std::min(minimum[a],int(n));
    }
  }
  int cpml_low[3]{}, cpml_high_start[3]{INT32_MAX,INT32_MAX,INT32_MAX};
  for(int slot=0;slot<12;++slot){
    const int axis=kCpmlAxes[slot%6];
    const int extent=fields[slot/2].dims[axis];
    const auto [low,high]=widths[slot];
    if(low<0||high<0||int64_t(low)+high>extent)
      throw std::invalid_argument("moving CPML widths exceed the Yee support");
    cpml_low[axis]=std::max(cpml_low[axis],transparent_faces&(1u<<(2*axis))?0:low);
    if(high && !(transparent_faces&(2u<<(2*axis))))cpml_high_start[axis]=std::min(cpml_high_start[axis],extent-high);
  }
  int64_t boxes=1;
  for(int a=0;a<3;++a){
    if(p.extent[a]>INT32_MAX-reach-p.owned[a])throw std::invalid_argument("moving padded coordinates exceed int32");
    p.counts[a]=(int64_t(p.extent[a])+reach+p.owned[a]-1)/p.owned[a];boxes*=p.counts[a];if(boxes>INT32_MAX)throw std::invalid_argument("moving tile IDs exceed int32");
  }
  const int nz=p.counts[0],ny=p.counts[1],nx=p.counts[2],ranks=nz+ny+nx-2;
  p.slots.resize(boxes);p.tile_flags.assign(boxes,0);p.tile_ids.reserve(boxes);p.rank_offsets.push_back(0);
  for(int rank=0;rank<ranks;++rank){const int begin=p.tile_ids.size();
    for(int z=std::max(0,rank-(ny-1)-(nx-1));z<=std::min(rank,nz-1);++z)
    for(int y=std::max(0,rank-z-(nx-1));y<=std::min(ny-1,rank-z);++y){
      const int x=rank-z-y;
      const int tile=(z*ny+y)*nx+x;p.slots[tile]=p.tile_ids.size()-begin;p.tile_ids.push_back(tile);
      const int origin[]={z*p.owned[0],y*p.owned[1],x*p.owned[2]};int low[3],high[3];uint8_t flags=0,transparent_walls=0;
      for(int a=0;a<3;++a){low[a]=origin[a]-reach;high[a]=origin[a]+p.owned[a]-2;
        if((cpml_low[a]&&low[a]<cpml_low[a])||high[a]>=cpml_high_start[a])flags|=uint8_t(2<<a);
        // Boundary specialization covers startup high=origin+B-1 and the
        // last imported low=origin-depth-1, beyond owned update points.
        if(low[a]<=0||high[a]+1>=minimum[a])flags|=kNativeBoundary;
        if((low[a]<=0 && (transparent_faces&(1u<<(2*a)))) ||
           (high[a]+1>=minimum[a] && (transparent_faces&(2u<<(2*a)))))transparent_walls|=uint8_t(2<<a);
      }
      // Single-axis kernels omit bounds checks along the other axes. Preserve
      // every wall axis at a junction between a true PML and a transparent cut.
      if(flags&14)flags|=transparent_walls;
      if(groups&&host_starts)for(int g=0;g<9;++g){
        const auto&coeff=groups[g].coefficients;const int32_t*starts=host_starts[g];
        if(coeff.dims[0]&&!starts)throw std::invalid_argument("moving source starts missing");
        for(int s=0;s<coeff.dims[0];++s){bool hit=true;
          for(int a=0;a<3;++a)hit&=low[a]<int64_t(starts[3*s+a])+coeff.dims[a+1]&&high[a]+(g<3)>=starts[3*s+a];
          if(hit)flags|=kMayHaveSources;
        }
      }
      p.tile_flags[tile]=flags|(record_volume?kMayObserve:0);
    }
    p.rank_offsets.push_back(p.tile_ids.size());p.max_frontier=std::max(p.max_frontier,int(p.tile_ids.size())-begin);
  }
  const auto mark_owners=[&](const auto& point,uint8_t flag){
    for(int a=0;a<3;++a)if(point[a]<0||point[a]>=p.extent[a])
      throw std::invalid_argument("moving effect outside field domain");
    for(int version=1;version<=reach;){
      int owner[3],next=reach+1;
      for(int a=0;a<3;++a){
        owner[a]=(point[a]+version)/p.owned[a];
        next=std::min(next,(owner[a]+1)*p.owned[a]-point[a]);
      }
      const int tile=(owner[0]*ny+owner[1])*nx+owner[2];p.tile_flags[tile]|=flag;
      version=next;
    }
  };
  for(const auto&point:observed_points)mark_owners(point,kMayObserve);
  // A contiguous X run visits every owner between its endpoints at each
  // version. Sampling at tile-width intervals marks that same set exactly.
  size_t cursor=0;
  int64_t base=0;
  for(int field=0;field<6&&cursor<observation_keys.size();++field){
    const auto& f=fields[field];
    const int64_t row=f.dims[2],plane=f.dims[1]*row,end=base+f.dims[0]*plane;
    while(cursor<observation_keys.size()&&observation_keys[cursor]<end){
      const int64_t key=observation_keys[cursor++],local=key-base;
      if(local<0)throw std::invalid_argument("invalid moving observation key order");
      std::array<int,3> point{int(local/plane),int(local/row%f.dims[1]),int(local%row)};
      const int64_t row_end=key+row-point[2];
      int64_t last=key;
      while(cursor<observation_keys.size()&&last+1<row_end&&observation_keys[cursor]==last+1){++cursor;++last;}
      const int last_x=point[2]+int(last-key);
      for(;last_x-point[2]>=p.owned[2];point[2]+=p.owned[2])mark_owners(point,kMayObserve);
      mark_owners(point,kMayObserve);
      if(point[2]!=last_x){point[2]=last_x;mark_owners(point,kMayObserve);}
    }
    base=end;
  }
  if(cursor!=observation_keys.size())throw std::invalid_argument("moving observation key outside field domain");
  for(const auto&source:sparse_sources){
    mark_owners(source.point,kMayHaveSources);
    if(source.pre_e)for(int a=0;a<3;++a)if(source.point[a]){
      auto neighbor=source.point;--neighbor[a];mark_owners(neighbor,kMayHaveSources);
    }
  }
  // Preserve rank antichains. Both sub-launches finish before the next rank;
  // local slots are recomputed after this operational permutation.
  for(int rank=0;rank<ranks;++rank){
    auto begin=p.tile_ids.begin()+p.rank_offsets[rank],end=p.tile_ids.begin()+p.rank_offsets[rank+1];
    if(partition_policies){
      std::stable_sort(begin,end,[&](int a,int b){return Policy(p.tile_flags[a])<Policy(p.tile_flags[b]);});
      auto cursor=begin;
      for(int policy=0;policy<6;++policy){p.policy_offsets.push_back(cursor-p.tile_ids.begin());while(cursor!=end&&Policy(p.tile_flags[*cursor])==policy)++cursor;}
      p.policy_offsets.push_back(end-p.tile_ids.begin());
      if(cursor!=end)throw std::runtime_error("moving policy partition lost a tile");
    }
    if(!partition_policies)std::stable_partition(begin,end,[&](int tile){return p.tile_flags[tile]==0;});
    // Re-find after the legacy stable partition; policy0 is first in both modes.
    p.bulk_counts.push_back(std::find_if(begin,end,[&](int tile){return p.tile_flags[tile]!=0;})-begin);
    for(auto it=begin;it!=end;++it)p.slots[*it]=it-begin;
  }
  p.surface_points=tensor?kTensorTrailPoints:kTrailSurfacePoints;
  const size_t owners=TrailOwners(p.max_frontier,tensor);
  const int rows=tensor?9*depth-3:6*depth;
  if(owners>std::numeric_limits<size_t>::max()/rows/p.surface_points)throw std::invalid_argument("moving trail allocation overflows");
  p.trail_floats=owners*rows*p.surface_points;
  p.metadata_bytes=2*size_t(boxes)*sizeof(int32_t)+size_t(boxes)*sizeof(uint8_t)+size_t(p.epochs)*p.max_frontier*sizeof(uint64_t)+sizeof(uint32_t)+sizeof(int32_t);
  if(p.tile_ids.size()!=size_t(boxes)||p.rank_offsets.back()!=boxes)throw std::runtime_error("moving rank coverage mismatch");
  return p;
}

#include "native_block_material_facts.inc"

// Compact the certified bulk and observer prefixes within each rank.
void BeamzCompactNativeMovingUniformTiles(BeamzNativeMovingPlan& plan, bool enabled) {
  using namespace beamz::cuda::moving;
  const size_t ranks = plan.bulk_counts.size();
  if (plan.rank_offsets.size() != ranks + 1)
    throw std::invalid_argument("compact uniform rank offsets differ");
  std::vector<int32_t>* counts[] = {
      &plan.uniform_noio_counts, &plan.uniform_observer_counts};
  for (auto* count : counts) count->assign(ranks, 0);
  if (plan.policy_offsets.empty()) return;
  if (plan.policy_offsets.size() != ranks * 7)
    throw std::invalid_argument("compact uniform policy offsets differ");
  for (size_t rank = 0; rank < ranks; ++rank) {
    const int begin = plan.rank_offsets[rank], end = plan.rank_offsets[rank + 1];
    for (int policy : {1, 2}) {
      const size_t offset = rank * 7 + policy;
      const int first = plan.policy_offsets[offset], last = plan.policy_offsets[offset + 1];
      if (begin < 0 || begin > first || first > last || last > end ||
          end > int(plan.tile_ids.size()))
        throw std::invalid_argument("compact uniform range outside rank");
      const auto start = plan.tile_ids.begin() + first, finish = plan.tile_ids.begin() + last;
      for (auto it = start; it != finish; ++it) {
        const int tile = *it;
        if (tile < 0 || size_t(tile) >= plan.tile_flags.size() ||
            Policy(plan.tile_flags[tile]) != policy)
          throw std::invalid_argument("compact uniform range contains another policy");
      }
      if (enabled) {
        const auto split = std::stable_partition(start, finish, [&](int tile) {
          return bool(plan.tile_flags[tile] & 64);
        });
        (*counts[policy - 1])[rank] = split - start;
      }
    }
    // All producers and consumers use the same slots after reordering.
    for (int i = begin; i < end; ++i) plan.slots[plan.tile_ids[i]] = i - begin;
  }
}
