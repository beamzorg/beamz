// Pure CPU dependency compiler for a prepared native moving plan.
// Consumers must use the exact tile order/slots/flags captured with the GPU.
#pragma once
#include "native_moving.h"
#include <algorithm>
#include <set>
#include <stdexcept>
#include <vector>

struct BeamzNativeMovingCohorts {
 std::vector<int32_t> offsets,dependencies;
 std::vector<int32_t> tile_cohort,last_policy_cohort;
 size_t raw_links=0,reuse_links=0,active_cohorts=0;
 size_t unreduced_dependency_count=0;
 int maximum_dependency_distance=0;
};
inline BeamzNativeMovingCohorts BeamzBuildNativeMovingCohorts(const BeamzNativeMovingPlan&p){
 using namespace beamz::cuda::moving;
 const int ranks=int(p.rank_offsets.size())-1;
 if(ranks<1||p.policy_offsets.size()!=size_t(ranks)*7||p.max_frontier<1||p.epochs!=4)
  throw std::invalid_argument("cohort graph requires a prepared six-policy rank partition");
 BeamzNativeMovingCohorts result;result.tile_cohort.assign(p.tile_ids.size(),-1);result.last_policy_cohort.assign(6,-1);
 std::vector<std::set<int32_t>> predecessors(size_t(ranks)*6);
 const int nz=p.counts[0],ny=p.counts[1],nx=p.counts[2];
 if(nz<1||ny<1||nx<1||p.rank_offsets.front()!=0||p.rank_offsets.back()!=int(p.tile_ids.size())||size_t(nz)*ny*nx!=p.tile_ids.size()||p.slots.size()!=p.tile_ids.size()||p.tile_flags.size()!=p.tile_ids.size())
  throw std::invalid_argument("cohort graph tile metadata shape mismatch");
 for(int r=0;r<ranks;++r){
  if(p.rank_offsets[r]<0||p.rank_offsets[r+1]<p.rank_offsets[r]||p.rank_offsets[r+1]>int(p.tile_ids.size())||p.rank_offsets[r+1]-p.rank_offsets[r]>p.max_frontier)
   throw std::invalid_argument("cohort graph rank range invalid");
  if(p.policy_offsets[r*7]!=p.rank_offsets[r]||p.policy_offsets[r*7+6]!=p.rank_offsets[r+1])
   throw std::invalid_argument("cohort graph policy/rank ranges disagree");
  for(int policy=0;policy<6;++policy){const int begin=p.policy_offsets[r*7+policy],end=p.policy_offsets[r*7+policy+1];
   if(begin<0||end<begin||end>int(p.tile_ids.size()))throw std::invalid_argument("cohort graph policy range invalid");
   if(begin!=end){++result.active_cohorts;result.last_policy_cohort[policy]=r*6+policy;}
   for(int i=begin;i<end;++i){const int tile=p.tile_ids[i];
    if(tile<0||tile>=int(p.tile_ids.size())||result.tile_cohort[tile]!=-1||Policy(p.tile_flags[tile])!=policy||p.slots[tile]!=i-p.rank_offsets[r])
     throw std::invalid_argument("cohort graph tile ownership invalid");
    const int z=tile/(ny*nx),y=(tile/nx)%ny,x=tile%nx;
    if(z+y+x!=r)throw std::invalid_argument("cohort graph affine rank mismatch");
    result.tile_cohort[tile]=r*6+policy;
   }
  }
 }
 auto edge=[&](int consumer,int producer){
  if(consumer<0||producer<0||producer/6>=consumer/6)throw std::invalid_argument("cohort graph must strictly advance affine rank");
  result.maximum_dependency_distance=std::max(result.maximum_dependency_distance,consumer/6-producer/6);
  // Same-policy ordering is implicit in the one persistent stream perpolicy.
  if(consumer%6!=producer%6)predecessors[consumer].insert(producer);
 };
 for(size_t tile=0;tile<p.tile_ids.size();++tile){const int z=tile/(ny*nx),y=(tile/nx)%ny,x=tile%nx;
  for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
   if(z<dz||y<dy||x<dx)continue;
   const int producer=int(tile)-dz*ny*nx-dy*nx-dx;
   edge(result.tile_cohort[tile],result.tile_cohort[producer]);++result.raw_links;
  }
 }
 // Follow actual physical owner reuse. A changing frontier need not populate
 // every (epoch,slot) at every rank; never infer an unwritten previous owner.
 std::vector<int32_t> last_owner(size_t(4)*p.max_frontier,-1);
 for(int r=0;r<ranks;++r)for(int i=p.rank_offsets[r];i<p.rank_offsets[r+1];++i){
  const int tile=p.tile_ids[i],owner=(r&3)*p.max_frontier+p.slots[tile],old=last_owner[owner];
  if(old>=0){
   const int old_cohort=result.tile_cohort[old];edge(result.tile_cohort[tile],old_cohort);++result.reuse_links;
   const int z=old/(ny*nx),y=(old/nx)%ny,x=old%nx;
   for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
    if(z+dz>=nz||y+dy>=ny||x+dx>=nx)continue;
    const int reader=old+dz*ny*nx+dy*nx+dx;
    edge(result.tile_cohort[tile],result.tile_cohort[reader]);++result.reuse_links;
   }
  }
  last_owner[owner]=tile;
 }
 for(const auto&deps:predecessors)result.unreduced_dependency_count+=deps.size();
 // Remove waits already implied by another wait or same-policy FIFO. This
 // changes no happens-before relation, and keeps capture/kernel-DAG overhead
 // from growing with every conservative tile-level RAW/WAW/WAR witness.
 // Bound the dense proof workspace; larger plans retain all safe edges.
 const size_t nodes=predecessors.size();
 if(nodes<=4096){
  const size_t words=(nodes+63)/64;
  std::vector<std::vector<uint64_t>>anc(nodes,std::vector<uint64_t>(words,0));
  int previous[6]={-1,-1,-1,-1,-1,-1};
  for(size_t c=0;c<nodes;++c){const int rank=c/6,policy=c%6;
   if(p.policy_offsets[rank*7+policy]==p.policy_offsets[rank*7+policy+1])continue;
   const int fifo=previous[policy];
   auto include=[&](int d){if(d<0)return;for(size_t w=0;w<words;++w)anc[c][w]|=anc[d][w];anc[c][d/64]|=uint64_t(1)<<(d%64);};
   include(fifo);for(int d:predecessors[c])include(d);
   std::vector<int>drop;
   for(int d:predecessors[c]){
    bool implied=fifo>=0&&(d==fifo||(anc[fifo][d/64]&(uint64_t(1)<<(d%64))));
    if(!implied)for(int other:predecessors[c])if(other!=d&&(anc[other][d/64]&(uint64_t(1)<<(d%64)))){implied=true;break;}
    if(implied)drop.push_back(d);
   }
   for(int d:drop)predecessors[c].erase(d);
   previous[policy]=c;
  }
 }
 result.offsets.push_back(0);
 for(const auto&deps:predecessors){result.dependencies.insert(result.dependencies.end(),deps.begin(),deps.end());result.offsets.push_back(result.dependencies.size());}
 return result;
}
