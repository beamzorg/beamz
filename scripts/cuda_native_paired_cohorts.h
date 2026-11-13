// Adjacent-Z fusion with a five-rank packet ring and typed supernode hazards.
#pragma once
#include "native_closed_cohorts.h"
#include <limits>
#include <numeric>

struct BeamzNativePairedCohorts:BeamzNativeClosedCohorts{
 int original_count=0,plane=0;
 size_t Bytes()const{
  return sizeof(int32_t)*(size_t(7)*tile_ids.size()+size_t(2)*original_count+finish_children.size()+9);
 }
 size_t CapacityBytes()const{
  if(finish_children.size()>size_t(16)*tile_ids.size())throw std::logic_error("Paired completion CSR exceeds16hazards per supernode");
  return Bytes()+(size_t(16)*tile_ids.size()-finish_children.size())*sizeof(int32_t);
 }
};

inline BeamzNativePairedCohorts BeamzBuildNativePairedCohorts(const BeamzNativeMovingPlan&p){
 const int nz=p.counts[0],ny=p.counts[1],nx=p.counts[2],n=p.tile_ids.size();
 if(nz<1||ny<1||nx<1||int64_t(nz)*ny*nx!=n||p.max_frontier<1||p.max_frontier>std::numeric_limits<int>::max()/5||p.slots.size()!=size_t(n))
  throw std::invalid_argument("Paired supernodes require a complete native plan");
 const int plane=ny*nx,nodes=((nz+1)/2)*plane;
 BeamzNativePairedCohorts q;q.epochs=5;q.original_count=n;q.plane=plane;q.owner_count=q.internal_ring_owners=5*p.max_frontier;
 q.owner_slots.assign(n,-1);q.previous_holder.assign(n,-1);q.tile_cohort.assign(nodes,0);
 q.tile_ids.resize(nodes);std::iota(q.tile_ids.begin(),q.tile_ids.end(),0);
 auto original_rank=[&](int t){return t/plane+(t/nx)%ny+t%nx;};
 auto supernode=[&](int t){return (t/plane/2)*plane+t%plane;};
 auto rank=[&](int s){return 2*(s/plane)+(s/nx)%ny+s%nx;};
 std::stable_sort(q.tile_ids.begin(),q.tile_ids.end(),[&](int a,int b){return rank(a)<rank(b);});
 q.cohort_offsets={0,nodes};q.cohort_policies={6};
 std::vector<std::set<int>>start(nodes),finish(nodes);std::vector<int>last(q.owner_count,-1),seen(n,0);
 int prior_rank=-1;
 for(int t:p.tile_ids){
  if(t<0||t>=n||seen[t]++||p.slots[t]<0||p.slots[t]>=p.max_frontier||original_rank(t)<prior_rank)throw std::invalid_argument("Invalid native rank/slot order");
  prior_rank=original_rank(t);const int z=t/plane,y=(t/nx)%ny,x=t%nx,node=supernode(t);
  for(int parent:{z?t-plane:-1,y?t-nx:-1,x?t-1:-1})if(parent>=0&&supernode(parent)!=node)start[node].insert(supernode(parent));
  const int owner=(original_rank(t)%5)*p.max_frontier+p.slots[t],old=last[owner];q.owner_slots[t]=owner;q.previous_holder[t]=old;
  if(old>=0){
   const int oz=old/plane,oy=(old/nx)%ny,ox=old%nx;std::set<int>readers{old};
   for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
    if(oz+dz<nz&&oy+dy<ny&&ox+dx<nx)readers.insert(old+dz*plane+dy*nx+dx);
   }
   for(int reader:readers){const int parent=supernode(reader);
    if(parent==node||rank(parent)>=rank(node))throw std::logic_error("Fused packet retirement self-edge or future supernode");
    finish[node].insert(parent);
   }
  }
  last[owner]=t;
 }
 q.initial_pending.assign(nodes,0);q.initial_queue.assign(nodes,-1);q.initial_entry.assign(nodes,0);q.root_counts={0};
 std::vector<std::vector<int>>finish_children(nodes);
 for(int s=0;s<nodes;++s){
  for(int parent:start[s]){if(rank(parent)>=rank(s))throw std::logic_error("Fused START rank");++q.initial_pending[s];++q.start_edges;}
  for(int parent:finish[s]){++q.initial_pending[s];++q.finish_edges;finish_children[parent].push_back(s);}
 }
 for(int s:q.tile_ids)if(!q.initial_pending[s]){q.initial_queue[q.root_counts[0]++]=s;q.initial_entry[s]=1;}
 if(q.root_counts[0]<1)throw std::logic_error("Fused queue has no initial root");
 q.finish_offsets={0};
 for(auto&row:finish_children){q.finish_children.insert(q.finish_children.end(),row.begin(),row.end());q.finish_offsets.push_back(q.finish_children.size());}
 q.InitializeStack();(void)q.CapacityBytes();return q;
}
