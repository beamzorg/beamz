// Experimental native persistent-tile ABI. No public FFI/default changes.
#pragma once
#include "native_moving.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <utility>

struct BeamzNativePersistentWorkspace {
  const int32_t* job_ids_device=nullptr;       // one prepared topological tile permutation
  const int32_t* previous_holder_device=nullptr; // [tile] previous 4-epoch owner or-1
  int32_t* next_job=nullptr;                  // reset each macro
  int32_t* abort=nullptr;                     // shared global abort, reset only when bad==0
  uint32_t* claimed_epoch=nullptr;            // [tile], initiallyzero, epoch-stamped
  uint32_t* completed_epoch=nullptr;          // [tile], initiallyzero, epoch-stamped
  uint32_t* ready_step=nullptr;               // [4*frontier], reset0 BEFORE releaseclaimtag
  int job_count=0;
  int worker_count=0;                         // session caps to device SMcount
  uint64_t timeout_ns=60000000000ULL;         // per stalled wait, progress restarts timer
};

struct BeamzNativePersistentPlan {
  std::vector<int32_t> job_ids,previous_holder,job_index;
};

inline int BeamzPersistentTileRank(int tile,const BeamzNativeMovingPlan&p){
 return tile/(p.counts[1]*p.counts[2])+(tile/p.counts[2])%p.counts[1]+tile%p.counts[2];
}
inline std::vector<int32_t> BeamzPersistentConsumers(int tile,const BeamzNativeMovingPlan&p){
 const int z=tile/(p.counts[1]*p.counts[2]),y=(tile/p.counts[2])%p.counts[1],x=tile%p.counts[2];
 std::vector<int32_t>out;
 for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
  if(z+dz<p.counts[0]&&y+dy<p.counts[1]&&x+dx<p.counts[2])out.push_back(tile+dz*p.counts[1]*p.counts[2]+dy*p.counts[2]+dx);
 }
 return out;
}
inline uint64_t BeamzPersistentMorton(int tile,const BeamzNativeMovingPlan&p){
 const unsigned z=tile/(p.counts[1]*p.counts[2]),y=(tile/p.counts[2])%p.counts[1],x=tile%p.counts[2];
 uint64_t code=0;for(int bit=0;bit<21;++bit){code|=uint64_t((x>>bit)&1)<<(3*bit);code|=uint64_t((y>>bit)&1)<<(3*bit+1);code|=uint64_t((z>>bit)&1)<<(3*bit+2);}return code;
}
inline BeamzNativePersistentPlan BeamzBuildNativePersistentPlan(const BeamzNativeMovingPlan&p,bool locality=false){
 const size_t count=p.tile_ids.size();
 if(count<1||count>INT32_MAX||p.epochs!=4||p.max_frontier<1||p.slots.size()!=count||p.rank_offsets.size()!=p.bulk_counts.size()+1||
    p.rank_offsets.front()!=0||p.rank_offsets.back()!=int(count)||p.counts[0]<1||p.counts[1]<1||p.counts[2]<1||
    int64_t(p.counts[0])*p.counts[1]*p.counts[2]!=int64_t(count))throw std::invalid_argument("persistent plan geometry differs");
 BeamzNativePersistentPlan out;out.job_ids=p.tile_ids;out.previous_holder.assign(count,-1);out.job_index.assign(count,-1);
 std::vector<int32_t>owners(4ULL*p.max_frontier,-1),seen(count,0);
 for(size_t rank=0;rank<p.bulk_counts.size();++rank){std::vector<uint8_t>used(p.max_frontier,0);
  if(p.rank_offsets[rank]>p.rank_offsets[rank+1])throw std::invalid_argument("persistent ranks are not ordered");
  for(int i=p.rank_offsets[rank];i<p.rank_offsets[rank+1];++i){const int tile=p.tile_ids[i];
   if(tile<0||size_t(tile)>=count||seen[tile]++||BeamzPersistentTileRank(tile,p)!=int(rank)||p.slots[tile]<0||p.slots[tile]>=p.max_frontier||used[p.slots[tile]]++)throw std::invalid_argument("persistent tile/slot permutation differs");
   const int owner=(rank&3)*p.max_frontier+p.slots[tile];out.previous_holder[tile]=owners[owner];owners[owner]=tile;
  }
 }
 if(locality){
  std::vector<std::vector<int32_t>>pred(count),succ(count);std::vector<int>remaining(count);
  for(size_t tile=0;tile<count;++tile){const int z=tile/(p.counts[1]*p.counts[2]),y=(tile/p.counts[2])%p.counts[1],x=tile%p.counts[2];
   for(int bits=1;bits<8;++bits){int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(z>=dz&&y>=dy&&x>=dx)pred[tile].push_back(tile-dz*p.counts[1]*p.counts[2]-dy*p.counts[2]-dx);}
   const int old=out.previous_holder[tile];if(old>=0){pred[tile].push_back(old);auto readers=BeamzPersistentConsumers(old,p);pred[tile].insert(pred[tile].end(),readers.begin(),readers.end());}
   auto&deps=pred[tile];std::sort(deps.begin(),deps.end());deps.erase(std::unique(deps.begin(),deps.end()),deps.end());remaining[tile]=deps.size();
   for(int dep:deps){if(BeamzPersistentTileRank(dep,p)>=BeamzPersistentTileRank(tile,p))throw std::invalid_argument("persistent dependency is not forward in rank");succ[dep].push_back(tile);}
  }
  using Ready=std::pair<uint64_t,int32_t>;std::priority_queue<Ready,std::vector<Ready>,std::greater<Ready>>ready;
  for(size_t t=0;t<count;++t)if(!remaining[t])ready.push({BeamzPersistentMorton(t,p),int32_t(t)});
  out.job_ids.clear();while(!ready.empty()){const int t=ready.top().second;ready.pop();out.job_ids.push_back(t);for(int child:succ[t])if(--remaining[child]==0)ready.push({BeamzPersistentMorton(child,p),child});}
  if(out.job_ids.size()!=count)throw std::invalid_argument("persistent topological queue is cyclic");
 }
 for(size_t i=0;i<count;++i)out.job_index[out.job_ids[i]]=i;
 for(size_t tile=0;tile<count;++tile){const int old=out.previous_holder[tile];
  if(old>=0){if(out.job_index[old]>=out.job_index[tile])throw std::invalid_argument("persistent old holder is not assigned first");for(int reader:BeamzPersistentConsumers(old,p))if(out.job_index[reader]>=out.job_index[tile])throw std::invalid_argument("persistent old reader is not assigned first");}
  const int z=tile/(p.counts[1]*p.counts[2]),y=(tile/p.counts[2])%p.counts[1],x=tile%p.counts[2];
  for(int bits=1;bits<8;++bits){int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(z>=dz&&y>=dy&&x>=dx&&out.job_index[tile-dz*p.counts[1]*p.counts[2]-dy*p.counts[2]-dx]>=out.job_index[tile])throw std::invalid_argument("persistent RAW parent is not assigned first");}
 }
 return out;
}

int BeamzEnqueueNativePersistent(void* stream,const BeamzProgramLaunch&program,
 const BeamzBuffer*final_fields,int macro_step,const BeamzNativeMovingWorkspace&moving,
 const BeamzNativePersistentWorkspace&persistent);
