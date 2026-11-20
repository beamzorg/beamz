// Host-only affine compiler for overlapping complete temporal macros.
// Tile kernels retain their own physical timestep dependencies.
#pragma once
#include "native_moving.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <queue>
#include <tuple>
#include <vector>

struct BeamzNativeMovingMacroSlab {
 std::vector<int32_t> tile_ids,slots,policy_offsets,uniform_noio_counts,uniform_observer_counts;
 int macro_count=0,level_count=0,macro_level_stride=0,max_frontier=0;
 int observation_end=0,observation_slots=0,overlap_slots=0,scheduling_sms=0;
 size_t active_tiles=0;
 size_t device_bytes()const{return active_tiles*sizeof(int32_t);}
};

inline BeamzNativeMovingMacroSlab BeamzBuildNativeMovingMacroSlab(const BeamzNativeMovingPlan&p,int macros,int raw_slots=3,bool materialize=true,int scheduling_sms=0,std::array<int,3> cut={-1,-1,-1}){
 using namespace beamz::cuda::moving;
 const size_t tiles=p.tile_ids.size(),ranks=p.bulk_counts.size();
 if(scheduling_sms<0||raw_slots<2||raw_slots>16||macros<2||macros>kSlabMacros||!tiles||!ranks||p.tile_flags.size()!=tiles||p.rank_offsets.size()!=ranks+1||p.policy_offsets.size()!=ranks*7||p.uniform_noio_counts.size()!=ranks||p.uniform_observer_counts.size()!=ranks||
    tiles>size_t(INT32_MAX)/macros||p.depth<1)
  throw std::invalid_argument("macro slab requires a prepared partition and bounded encoded IDs");
 BeamzNativeMovingMacroSlab result;result.macro_count=macros;int stride=1;
 for(int a=0;a<3;++a){if(p.owned[a]<1||p.counts[a]<1)throw std::invalid_argument("invalid macro ownership");stride+=(p.skew*p.depth+p.owned[a]-1)/p.owned[a];}
 stride=std::max(stride,int((ranks+raw_slots-1)/raw_slots));
 if(ranks+size_t(macros-1)*stride>size_t(INT32_MAX)/7)throw std::invalid_argument("macro level overflow");
 result.macro_level_stride=stride;result.level_count=int(ranks)+(macros-1)*stride;
 int first_observed=int(ranks);
 result.overlap_slots=raw_slots;result.scheduling_sms=scheduling_sms;
 // Validate physical tiles once. Each macro reuses this same partition.
 std::vector<bool> seen(tiles);
 for(size_t rank=0;rank<ranks;++rank){
  const auto* offsets=p.policy_offsets.data()+rank*7;
  for(int policy=0;policy<6;++policy){
   const int begin=offsets[policy],end=offsets[policy+1];
   if(begin<0||end<begin||size_t(end)>tiles)throw std::invalid_argument("macro partition range invalid");
   const int certified=policy==1?p.uniform_noio_counts[rank]:policy==2?p.uniform_observer_counts[rank]:0;
   if(certified<0||certified>end-begin)throw std::invalid_argument("macro certificate prefix invalid");
   if((policy==0&&begin!=p.rank_offsets[rank])||(policy==5&&end!=p.rank_offsets[rank+1]))
    throw std::invalid_argument("macro partition differs from rank boundaries");
   for(int i=begin;i<end;++i){
    const int tile=p.tile_ids[i];
    if(tile<0||size_t(tile)>=tiles||seen[tile]||Policy(p.tile_flags[tile])!=policy)
     throw std::invalid_argument("macro physical tile partition invalid");
    const int z=tile/(p.counts[1]*p.counts[2]),y=(tile/p.counts[2])%p.counts[1],x=tile%p.counts[2];
    if(size_t(z+y+x)!=rank)throw std::invalid_argument("macro affine rank differs");
    seen[tile]=true;
    if(p.tile_flags[tile]&kMayObserve){
     first_observed=std::min(first_observed,int(rank));
     result.observation_end=int(rank)+1;
    }
   }
  }
 }
 if(std::find(seen.begin(),seen.end(),false)!=seen.end())throw std::invalid_argument("macro physical tile omitted");

 // A longer interval between wavefronts can release a whole history bank.
 // Limit the extra barriers to a 1.5% increase in the launch-cost model
 // used by the automatic controller. Explicit schedules keep their stride.
 const int observed_span=result.observation_end-first_observed;
 const int initial_banks=std::max(1,(observed_span+stride-1)/stride);
 const double tile_work=double(tiles)*32;
 const double cost_limit=(tile_work+double(scheduling_sms)*result.level_count)*1.015;
 for(int banks=initial_banks-1;scheduling_sms>0&&banks>0;--banks){
  const int next_stride=(observed_span+banks-1)/banks;
  const int64_t levels=int64_t(ranks)+int64_t(macros-1)*next_stride;
  if(levels>INT32_MAX/7||tile_work+double(scheduling_sms)*levels>cost_limit)break;
  stride=next_stride;
  result.macro_level_stride=stride;
  result.level_count=int(levels);
 }
 // The final observed rank drains before the next writer to this ring slot.
 result.observation_slots=std::max(1,(result.observation_end-first_observed+stride-1)/stride);
 if(materialize){
  result.tile_ids.reserve(tiles*macros);
  result.slots.assign(tiles*macros,-1);
 }
 // Tile coordinates at version v are [q*B-v, q*B+B-1-v].
 // Outside this backwards Yee cone a producer cannot reach an owned output.
 const auto active=[&](int id,int macro){
  if(cut[0]<0)return true;
  const int axis=cut[0];
  const int q=axis==0?id/(p.counts[1]*p.counts[2]):axis==1?(id/p.counts[2])%p.counts[1]:id%p.counts[2];
  const int first=q*p.owned[axis], remaining=(macros-macro)*p.skew*p.depth;
  return (cut[1]<0 || first+p.owned[axis]>=cut[1]-remaining-1)
      && (cut[2]<0 || first<=cut[2]+remaining+1);
 };
 for(int level=0;level<result.level_count;++level){
  const int newest=std::min(macros-1,level/stride);
  const int oldest=std::max(0,(level-int(ranks)+stride)/stride);
  int frontier=0;
  for(int m=newest;m>=oldest;--m){
   const int rank=level-m*stride;
   if(cut[0]<0)frontier+=p.rank_offsets[rank+1]-p.rank_offsets[rank];
   else for(int i=p.rank_offsets[rank];i<p.rank_offsets[rank+1];++i)frontier+=active(p.tile_ids[i],m);
  }
  result.max_frontier=std::max(result.max_frontier,frontier);
  result.active_tiles+=frontier;
  if(!materialize)continue;
  const size_t level_begin=result.tile_ids.size();
  for(int policy=0;policy<6;++policy){
   result.policy_offsets.push_back(result.tile_ids.size());
   const bool split=policy==1||policy==2;
   int certified_total=0;
   // Preserve physical rank order within each certified and ordinary range.
   for(int pass=split?0:1;pass<2;++pass)for(int m=newest;m>=oldest;--m){
    const int rank=level-m*stride;
    const auto* offsets=p.policy_offsets.data()+rank*7;
    const int begin=offsets[policy],end=offsets[policy+1];
    const int certified=policy==1?p.uniform_noio_counts[rank]:policy==2?p.uniform_observer_counts[rank]:0;

    for(int i=pass?begin+certified:begin;i<(pass?end:begin+certified);++i){
     if(!active(p.tile_ids[i],m))continue;
     if(pass==0)++certified_total;
     const int encoded=int(size_t(m)*tiles)+p.tile_ids[i];
     result.slots[encoded]=result.tile_ids.size()-level_begin;
     result.tile_ids.push_back(encoded);
    }
   }
   if(policy==1)result.uniform_noio_counts.push_back(certified_total);
   if(policy==2)result.uniform_observer_counts.push_back(certified_total);
  }
  result.policy_offsets.push_back(result.tile_ids.size());
 }
 // Preparation also admits a single native macro for the residual steps.
 result.max_frontier=std::max(result.max_frontier,p.max_frontier);
 if(!materialize)return result;
 if(result.tile_ids.size()!=result.active_tiles || (cut[0]<0&&result.active_tiles!=tiles*macros))
  throw std::logic_error("macro encoded task count differs");
 return result;
}

// A face interior has exactly one reader, the next tile along its normal.
// Its CTA imports each version before exporting the same version in place.
// Color each line of tiles once, reusing the slot after its final tile.
inline void BeamzColorMovingFaces(std::vector<int32_t>& slots, const std::array<int,3>& n,
                                  int stride, int cut_axis, int frontier) {
  const size_t tiles=size_t(n[0])*n[1]*n[2],encoded=slots.size();
  const int macros=int(encoded/tiles), step[3]{n[1]*n[2],n[2],1};
  slots.resize(encoded+macros*beamz::cuda::moving::FaceLines(n),-1);
  for(int axis=0;axis<3;++axis){
    const int u=(axis+1)%3,v=(axis+2)%3;
    std::vector<std::tuple<int,int,size_t>> intervals;
    intervals.reserve(size_t(macros)*n[u]*n[v]);
    for(int m=0;m<macros;++m){
      int lo[3]{},hi[3]{n[0],n[1],n[2]};
      if(cut_axis>=0){
        auto present=[&](int q){return slots[m*tiles+size_t(q)*step[cut_axis]]>=0;};
        while(lo[cut_axis]<hi[cut_axis]&&!present(lo[cut_axis]))++lo[cut_axis];
        while(hi[cut_axis]>lo[cut_axis]&&!present(hi[cut_axis]-1))--hi[cut_axis];
      }
      if(lo[axis]>=hi[axis])continue;
      int p[3]{};p[axis]=lo[axis];
      for(p[u]=lo[u];p[u]<hi[u];++p[u])for(p[v]=lo[v];p[v]<hi[v];++p[v]){
        const int begin=m*stride+p[0]+p[1]+p[2];
        intervals.emplace_back(begin,begin+hi[axis]-lo[axis],
            beamz::cuda::moving::FaceLine(n[0],n[1],n[2],p[0],p[1],p[2],m,axis));
      }
    }
    std::sort(intervals.begin(),intervals.end());
    using End=std::pair<int,int>;
    std::priority_queue<End,std::vector<End>,std::greater<End>> end;
    for(const auto& [first,last,line]:intervals){
      int color=end.size();
      if(!end.empty()&&end.top().first<=first){color=end.top().second;end.pop();}
      if(color>=frontier)throw std::logic_error("temporal face frontier exceeded");
      slots[encoded+line]=color;
      end.emplace(last,color);
    }
  }
}
