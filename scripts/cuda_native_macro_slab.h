// Host-only affine compiler for overlapping complete temporal macros.
// Numerical kernels keep their native16/4-step local dependency schedules.
#pragma once
#include "native_moving.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

struct BeamzNativeMovingMacroSlab {
 std::vector<int32_t> tile_ids,slots,policy_offsets,uniform_noio_counts,uniform_observer_counts;
 int macro_count=0,level_count=0,macro_level_stride=0,max_frontier=0;
 size_t device_bytes()const{return tile_ids.size()*sizeof(int32_t);}
};

inline BeamzNativeMovingMacroSlab BeamzBuildNativeMovingMacroSlab(const BeamzNativeMovingPlan&p,int macros){
 using namespace beamz::cuda::moving;
 const size_t tiles=p.tile_ids.size(),ranks=p.bulk_counts.size();
 if(macros<2||macros>8||!tiles||!ranks||p.policy_offsets.size()!=ranks*7||p.uniform_noio_counts.size()!=ranks||p.uniform_observer_counts.size()!=ranks||
    tiles>size_t(INT32_MAX)/macros||p.depth<1)
  throw std::invalid_argument("macro slab requires a prepared partition and bounded encoded IDs");
 BeamzNativeMovingMacroSlab result;result.macro_count=macros;int stride=1;
 for(int a=0;a<3;++a){if(p.owned[a]<1||p.counts[a]<1)throw std::invalid_argument("invalid macro ownership");stride+=(p.depth+p.owned[a]-1)/p.owned[a];}
 if(ranks+size_t(macros-1)*stride>size_t(INT32_MAX)/7)throw std::invalid_argument("macro level overflow");
 result.macro_level_stride=stride;result.level_count=int(ranks)+(macros-1)*stride;
 using Item=std::pair<int32_t,bool>;
 std::vector<std::vector<Item>>buckets(size_t(result.level_count)*6);
 std::vector<bool>seen(tiles,false);
 for(size_t rank=0;rank<ranks;++rank){const auto*offsets=p.policy_offsets.data()+rank*7;
  for(int policy=0;policy<6;++policy){const int begin=offsets[policy],end=offsets[policy+1];
   if(begin<0||end<begin||size_t(end)>tiles)throw std::invalid_argument("macro partition range invalid");
   const int certified=policy==1?p.uniform_noio_counts[rank]:policy==2?p.uniform_observer_counts[rank]:0;
   if(certified<0||certified>end-begin)throw std::invalid_argument("macro certificate prefix invalid");
   for(int i=begin;i<end;++i){const int tile=p.tile_ids[i];
    if(tile<0||size_t(tile)>=tiles||seen[tile]||Policy(p.tile_flags[tile])!=policy)throw std::invalid_argument("macro physical tile partition invalid");
    const int z=tile/(p.counts[1]*p.counts[2]),y=(tile/p.counts[2])%p.counts[1],x=tile%p.counts[2];
    if(size_t(z+y+x)!=rank)throw std::invalid_argument("macro affine rank differs");seen[tile]=true;
    for(int m=0;m<macros;++m)buckets[(rank+m*stride)*6+policy].push_back({int32_t(size_t(m)*tiles+tile),i<begin+certified});
   }
  }
 }
 if(std::find(seen.begin(),seen.end(),false)!=seen.end())throw std::invalid_argument("macro physical tile omitted");
 result.slots.assign(tiles*macros,-1);
 for(int level=0;level<result.level_count;++level){int noio=0,observer=0;const size_t begin_level=result.tile_ids.size();
  for(int policy=0;policy<6;++policy){result.policy_offsets.push_back(result.tile_ids.size());auto&items=buckets[level*6+policy];
   if(policy==1||policy==2){std::stable_partition(items.begin(),items.end(),[](const Item&i){return i.second;});
    const int certified=std::count_if(items.begin(),items.end(),[](const Item&i){return i.second;});if(policy==1)noio=certified;else observer=certified;
   }
   for(const auto&item:items){result.slots[item.first]=result.tile_ids.size()-begin_level;result.tile_ids.push_back(item.first);}
  }
  result.policy_offsets.push_back(result.tile_ids.size());result.uniform_noio_counts.push_back(noio);result.uniform_observer_counts.push_back(observer);
  result.max_frontier=std::max(result.max_frontier,int(result.tile_ids.size()-begin_level));
 }
 if(result.tile_ids.size()!=tiles*macros)throw std::logic_error("macro encoded task count differs");
 return result;
}
