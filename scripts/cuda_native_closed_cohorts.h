// Policy-homogeneous closures with typed START and COMPLETE dependencies.
// Compile only after immutable certificates and physical rank slots are final.
#pragma once
#include "native_moving.h"
#include <algorithm>
#include <functional>
#include <queue>
#include <set>
#include <stdexcept>
#include <vector>

struct BeamzNativeClosedCohorts {
 int epochs=8;
 int owner_count=0,internal_ring_owners=0,export_owners=0,exported_packets=0;
 std::vector<int32_t>owner_slots;
 std::vector<int32_t>tile_ids,cohort_offsets,cohort_policies,tile_cohort;
 std::vector<int32_t>initial_pending,initial_queue,initial_entry,root_counts;
 std::vector<int32_t>initial_next,initial_heads;
 std::vector<int32_t>finish_offsets,finish_children,previous_holder;
 size_t start_edges=0,finish_edges=0;
 size_t Bytes()const{
  return sizeof(int32_t)*(owner_slots.size()+tile_cohort.size()+initial_pending.size()+initial_next.size()+initial_heads.size()+
      finish_offsets.size()+finish_children.size()+root_counts.size()+5*cohort_policies.size()+
      4*tile_ids.size()+1);
 }
 size_t CapacityBytes()const{
  if(finish_children.size()>size_t(8)*tile_ids.size())throw std::logic_error("Completion CSR exceeds8hazards per tile");
  return Bytes()+(size_t(8)*tile_ids.size()-finish_children.size())*sizeof(int32_t);
 }
 void InitializeStack(){
  initial_next.assign(tile_ids.size(),-1);initial_heads.assign(cohort_policies.size(),-1);
  for(size_t c=0;c<cohort_policies.size();++c)for(int i=cohort_offsets[c]+root_counts[c]-1;i>=cohort_offsets[c];--i){
   const int t=initial_queue[i];initial_next[t]=initial_heads[c];initial_heads[c]=t;
  }
 }
};

inline BeamzNativeClosedCohorts BeamzBuildNativeRankClosedCohorts(const BeamzNativeMovingPlan&p,int epochs=8,bool omit_reuse=false,bool single_class=false){
 using namespace beamz::cuda::moving;
 const int n=p.tile_ids.size(),nz=p.counts[0],ny=p.counts[1],nx=p.counts[2];
 if((epochs!=4&&epochs!=8&&epochs!=16&&epochs!=32)||n<1||size_t(nz)*ny*nx!=size_t(n)||p.max_frontier<1||p.slots.size()!=size_t(n)||p.tile_flags.size()!=size_t(n))
  throw std::invalid_argument("Closed cohorts require a complete prepared native plan");
 BeamzNativeClosedCohorts q;q.epochs=epochs;q.owner_count=epochs*p.max_frontier;q.owner_slots.assign(n,-1);q.tile_cohort.assign(n,-1);q.previous_holder.assign(n,-1);
 std::vector<int>position(n,-1),rank(n),classes(n),remaining(n);
 std::vector<std::set<int>>start(n),finish(n),dependencies(n);std::vector<std::vector<int>>children(n);
 for(int i=0;i<n;++i){const int t=p.tile_ids[i];if(t<0||t>=n||position[t]>=0)throw std::invalid_argument("Duplicate closed-cohort tile");position[t]=i;}
 std::vector<int>last(size_t(epochs)*p.max_frontier,-1);
 for(int t:p.tile_ids){
  const int z=t/(ny*nx),y=(t/nx)%ny,x=t%nx;rank[t]=z+y+x;classes[t]=single_class?0:Policy(p.tile_flags[t]);
  if(p.slots[t]<0||p.slots[t]>=p.max_frontier)throw std::invalid_argument("Invalid native physical slot");
  if(z)start[t].insert(t-ny*nx);if(y)start[t].insert(t-nx);if(x)start[t].insert(t-1);
  const int slot=(rank[t]%epochs)*p.max_frontier+p.slots[t],old=last[slot];q.previous_holder[t]=old;
  q.owner_slots[t]=slot;
  if(old>=0&&!omit_reuse){finish[t].insert(old);const int oz=old/(ny*nx),oy=(old/nx)%ny,ox=old%nx;
   for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
    if(oz+dz<nz&&oy+dy<ny&&ox+dx<nx)finish[t].insert(old+dz*ny*nx+dy*nx+dx);}}
  last[slot]=t;dependencies[t]=start[t];dependencies[t].insert(finish[t].begin(),finish[t].end());
  for(int parent:dependencies[t]){
   if(position[parent]>=position[t]||rank[parent]>=rank[t])throw std::invalid_argument("Closed cohort hazard must strictly advance rank");
   children[parent].push_back(t);
  }
  remaining[t]=dependencies[t].size();
 }
 // Earliest available physical task wins the next policy; then consume the
 // entire policy closure. This avoids fragmenting the native shell into many
 // tiny cohorts merely because another policy has more immediately ready IDs.
 std::set<std::pair<int,int>>ready[6];
 for(int t=0;t<n;++t)if(!remaining[t])ready[classes[t]].insert({position[t],t});
 q.cohort_offsets.push_back(0);
 while(q.tile_ids.size()<size_t(n)){
  int policy=-1;
  for(int k=0;k<6;++k)if(!ready[k].empty()&&(policy<0||ready[k].begin()->first<ready[policy].begin()->first))policy=k;
  if(policy<0)throw std::runtime_error("Closed cohort dependency deadlock");
  const int cohort=q.cohort_policies.size();q.cohort_policies.push_back(policy);
  while(!ready[policy].empty()){
   const int t=ready[policy].begin()->second;ready[policy].erase(ready[policy].begin());q.tile_cohort[t]=cohort;q.tile_ids.push_back(t);
   for(int parent:dependencies[t])if(q.tile_cohort[parent]<0)throw std::runtime_error("Future external cohort dependency");
   for(int child:children[t])if(!--remaining[child])ready[classes[child]].insert({position[child],child});
  }
  q.cohort_offsets.push_back(q.tile_ids.size());
 }
 q.initial_pending.assign(n,0);q.initial_queue.assign(n,-1);q.initial_entry.assign(n,0);q.root_counts.assign(q.cohort_policies.size(),0);
 std::vector<std::vector<int>>finish_children(n);
 for(int t=0;t<n;++t){
  const int cohort=q.tile_cohort[t];
  for(int parent:start[t]){
   if(q.tile_cohort[parent]>cohort)throw std::runtime_error("Future START kernel");
   if(q.tile_cohort[parent]==cohort){++q.initial_pending[t];++q.start_edges;}
  }
  for(int parent:finish[t]){
   if(q.tile_cohort[parent]>cohort)throw std::runtime_error("Future COMPLETE kernel");
   if(q.tile_cohort[parent]==cohort){++q.initial_pending[t];++q.finish_edges;finish_children[parent].push_back(t);}
  }
 }
 for(int t:q.tile_ids)if(!q.initial_pending[t]){
  const int cohort=q.tile_cohort[t],slot=q.cohort_offsets[cohort]+q.root_counts[cohort]++;
  q.initial_queue[slot]=t;q.initial_entry[slot]=1;
 }
 q.finish_offsets.push_back(0);
 for(const auto&v:finish_children){q.finish_children.insert(q.finish_children.end(),v.begin(),v.end());q.finish_offsets.push_back(q.finish_children.size());}
 for(size_t c=0;c<q.cohort_policies.size();++c)if(q.root_counts[c]<1)throw std::runtime_error("Closed kernel has no initial root");
 q.InitializeStack();
 return q;
}

// Schedule first, allocate second: internal packets use a small rank ring;
// cross-kernel packets retain an interval color through their last reader.
inline BeamzNativeClosedCohorts BeamzBuildNativeClosedCohorts(const BeamzNativeMovingPlan&p,int epochs=4,bool single_class=false){
 auto q=BeamzBuildNativeRankClosedCohorts(p,epochs,true,single_class);
 const int n=p.tile_ids.size(),nz=p.counts[0],ny=p.counts[1],nx=p.counts[2],ranks=nz+ny+nx-2;
 const int cohorts=q.cohort_policies.size();std::vector<int>last_reader(n),rank(n),position(n);
 std::vector<int>rank_counts(size_t(cohorts)*ranks,0);int internal_frontier=0;
 for(int i=0;i<n;++i)position[q.tile_ids[i]]=i;
 for(int t=0;t<n;++t){
  const int z=t/(ny*nx),y=(t/nx)%ny,x=t%nx,cohort=q.tile_cohort[t];rank[t]=z+y+x;last_reader[t]=cohort;
  for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
   if(z+dz<nz&&y+dy<ny&&x+dx<nx){const int reader=t+dz*ny*nx+dy*nx+dx;
    if(q.tile_cohort[reader]<cohort)throw std::runtime_error("Physics cohort precedes its producer");
    last_reader[t]=std::max(last_reader[t],q.tile_cohort[reader]);}}
  if(last_reader[t]==cohort)internal_frontier=std::max(internal_frontier,++rank_counts[size_t(cohort)*ranks+rank[t]]);
 }
 q.internal_ring_owners=epochs*internal_frontier;int next_owner=q.internal_ring_owners;
 std::fill(rank_counts.begin(),rank_counts.end(),0);
 std::priority_queue<std::pair<int,int>,std::vector<std::pair<int,int>>,std::greater<std::pair<int,int>>>busy;
 std::priority_queue<int,std::vector<int>,std::greater<int>>available;
 for(int cohort=0;cohort<cohorts;++cohort){
  while(!busy.empty()&&busy.top().first<cohort){available.push(busy.top().second);busy.pop();}
  for(int i=q.cohort_offsets[cohort];i<q.cohort_offsets[cohort+1];++i){const int t=q.tile_ids[i];
   if(last_reader[t]>cohort){
    const int slot=available.empty()?next_owner++:available.top();if(!available.empty())available.pop();
    q.owner_slots[t]=slot;busy.push({last_reader[t],slot});++q.exported_packets;
   }else q.owner_slots[t]=(rank[t]%epochs)*internal_frontier+rank_counts[size_t(cohort)*ranks+rank[t]]++;
  }
 }
 q.owner_count=next_owner;q.export_owners=next_owner-q.internal_ring_owners;q.finish_edges=0;
 std::vector<int>last(next_owner,-1);std::vector<std::vector<int>>finish_children(n);
 // Preserve the chosen physics partition; every new storage edge must stay
 // within its writer kernel or point back to a fully completed older kernel.
 for(int t:q.tile_ids){
  const int slot=q.owner_slots[t],old=last[slot],cohort=q.tile_cohort[t];q.previous_holder[t]=old;
  if(old>=0){
   std::set<int>retire{old};const int z=old/(ny*nx),y=(old/nx)%ny,x=old%nx;
   for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
    if(z+dz<nz&&y+dy<ny&&x+dx<nx)retire.insert(old+dz*ny*nx+dy*nx+dx);}
   for(int parent:retire){
    if(q.tile_cohort[parent]>cohort||position[parent]>=position[t])throw std::runtime_error("Export color reused before final reader kernel");
    if(q.tile_cohort[parent]==cohort){++q.initial_pending[t];++q.finish_edges;finish_children[parent].push_back(t);}
   }
  }
  last[slot]=t;
 }
 q.initial_queue.assign(n,-1);q.initial_entry.assign(n,0);q.root_counts.assign(cohorts,0);
 for(int t:q.tile_ids)if(!q.initial_pending[t]){const int cohort=q.tile_cohort[t],slot=q.cohort_offsets[cohort]+q.root_counts[cohort]++;q.initial_queue[slot]=t;q.initial_entry[slot]=1;}
 q.finish_offsets.assign(1,0);q.finish_children.clear();
 for(const auto&v:finish_children){q.finish_children.insert(q.finish_children.end(),v.begin(),v.end());q.finish_offsets.push_back(q.finish_children.size());}
 for(int c=0;c<cohorts;++c)if(q.root_counts[c]<1)throw std::runtime_error("Allocated physics kernel has no initial root");
 q.InitializeStack();
 if(single_class){
  if(q.cohort_policies.size()!=1)throw std::runtime_error("Global worker must own one complete task DAG");
  q.cohort_policies[0]=6;
 }
 return q;
}
