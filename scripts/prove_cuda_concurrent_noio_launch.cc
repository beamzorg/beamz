// Actual host NoIO schedule + outer rank launcher against CUDA event snapshots.
// No CUDA runtime, GPU or floating-point emulation.
#include "native_moving.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

struct StreamState{std::set<int>frontier;int role=-1;};
struct EventState{std::set<int>frontier;bool recorded=false;};
using cudaStream_t=StreamState*;using cudaEvent_t=EventState*;using cudaError_t=int;
constexpr int cudaSuccess=0,cudaErrorInvalidValue=1;
static size_t waits=0,records=0,launches=0,raw_checks=0,reuse_checks=0,concurrency_checks=0;
static const BeamzNativeMovingPlan*prepared=nullptr;
static std::map<int,std::set<int>>graph;
static std::vector<int>tile_node;
static int next_node=1;
static std::map<std::pair<int,int>,int>rank_class_nodes;
static int cudaEventRecord(cudaEvent_t event,cudaStream_t stream){
 if(!event||!stream||stream->frontier.empty())throw std::logic_error("record lacks a prepared stream frontier");
 event->frontier=stream->frontier;event->recorded=true;++records;return cudaSuccess;
}
static int cudaStreamWaitEvent(cudaStream_t stream,cudaEvent_t event,unsigned){
 if(!event||!stream||!event->recorded)throw std::logic_error("wait references an unrecorded event");
 stream->frontier.insert(event->frontier.begin(),event->frontier.end());++waits;return cudaSuccess;
}
struct Parameters{};
static int rank_of(int t){return t/(prepared->counts[1]*prepared->counts[2])+(t/prepared->counts[2])%prepared->counts[1]+t%prepared->counts[2];}
static int launch(cudaStream_t stream,const int32_t*ids,int count,int policy,int axis=-1,bool uniform=false){
 if(!count)return cudaSuccess;
 if(count<0||!stream||stream->frontier.empty())throw std::logic_error("launch lacks a captured input predecessor");
 int node=next_node++,rank=rank_of(ids[0]);graph[node]=stream->frontier;stream->frontier={node};++launches;
 for(int i=0;i<count;++i){int t=ids[i];
  if(t<0||size_t(t)>=prepared->tile_flags.size()||rank_of(t)!=rank||beamz::cuda::moving::Policy(prepared->tile_flags[t])!=policy)throw std::logic_error("launch crosses rank/policy");
  if(tile_node[t]>=0)throw std::logic_error("tile launched twice");
  tile_node[t]=node;
  const int bits=prepared->tile_flags[t]&14;
  if(uniform&&!(prepared->tile_flags[t]&64))throw std::logic_error("uniform launch lacks immutable certificate");
  if(axis>=0&&bits!=(2<<axis))throw std::logic_error("axis launch does not match exact one-hot flag");
  if(uniform&&axis<0&&(bits==2||bits==4||bits==8))throw std::logic_error("generic uniform remainder contains a one-hot tile");
 }
 if(policy==1)rank_class_nodes[{rank,axis>=0?axis:uniform?3:4}]=node;
 return cudaSuccess;
}
template<int D,int L,bool Fast,bool Iso,int Policy>static int Launch(cudaStream_t s,const Parameters&,const int32_t*i,int n){return launch(s,i,n,Policy);}
template<int D,int Axes>static int LaunchNoIOAxis(cudaStream_t s,const Parameters&,const int32_t*i,int n){return launch(s,i,n,1,Axes==1?0:Axes==2?1:2,true);}
template<int D>static int LaunchNoIOUniformRange(cudaStream_t s,const Parameters&,const int32_t*i,int n){return launch(s,i,n,1,-1,true);}
template<int D,int L,bool Iso>static int LaunchObservers(cudaStream_t s,const Parameters&,const int32_t*i,int n,int u){if(u<0||u>n)return cudaErrorInvalidValue;return launch(s,i,n,2);}
template<int D,int L,bool Iso,int Policy>static int LaunchSourcePending(cudaStream_t s,const Parameters&,const int32_t*i,int n,uint32_t,bool){return launch(s,i,n,Policy);}
#include "concurrent_noio_axis_schedule.inc"
#include "actual_outer_noio_rank.inc"

struct Reachability{
 std::vector<std::vector<uint64_t>>bits;
 explicit Reachability(int n):bits(n,std::vector<uint64_t>((n+63)/64)){
  for(int node=1;node<n;++node)for(int d:graph.at(node)){
   if(d>=node)throw std::logic_error("event snapshot introduced a future/cyclic dependency");
   for(size_t w=0;w<bits[node].size();++w)bits[node][w]|=bits[d][w];
   bits[node][d/64]|=uint64_t(1)<<(d%64);
  }
 }
 bool has(int predecessor,int node)const{return bits[node][predecessor/64]&(uint64_t(1)<<(predecessor%64));}
};
static void repartition(BeamzNativeMovingPlan&p){
 using beamz::cuda::moving::Policy;
 p.policy_offsets.clear();
 for(size_t r=0;r<p.bulk_counts.size();++r){auto a=p.tile_ids.begin()+p.rank_offsets[r],b=p.tile_ids.begin()+p.rank_offsets[r+1];
  std::stable_sort(a,b,[&](int x,int y){return Policy(p.tile_flags[x])<Policy(p.tile_flags[y]);});auto it=a;
  for(int k=0;k<6;++k){p.policy_offsets.push_back(it-p.tile_ids.begin());while(it!=b&&Policy(p.tile_flags[*it])==k)++it;}
  p.policy_offsets.push_back(b-p.tile_ids.begin());p.bulk_counts[r]=p.policy_offsets[r*7+1]-p.policy_offsets[r*7];
  for(auto q=a;q!=b;++q)p.slots[*q]=q-a;
 }
}
static void run_case(const std::array<int,3>&dims,int depth,bool stress,bool parallel_policy,bool parallel_axes){
 using namespace beamz::cuda::moving;
 const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 BeamzBuffer f[6]{};for(int c=0;c<6;++c){f[c].rank=3;f[c].element_type=kBeamzF32;for(int a=0;a<3;++a)f[c].dims[a]=dims[a]+add[c][a];}
 auto p=BeamzBuildNativeMovingPlan(f,depth,{{0,0,0},{dims[0]/2,dims[1]/2,dims[2]/2}},nullptr,nullptr,12,true);
 if(stress){const uint8_t choices[]={0,2,4,8,6,10,12,14,32,3,1,16,17,18};
  for(size_t t=0;t<p.tile_flags.size();++t)p.tile_flags[t]=choices[(t*1103515245u+12345u)%14];repartition(p);
 }
 for(size_t t=0;t<p.tile_flags.size();++t)if(Policy(p.tile_flags[t])==1&&t%3)p.tile_flags[t]|=64;
 BeamzCompactNativeMovingUniformNoIO(p,true);BeamzCompactNativeMovingUniformObservers(p,true);
 const auto original=p;BeamzCompactNativeMovingNoIOAxes(p,true);
 // Host permutation preserves every policy/rank cardinality and a one-to-one
 // rank-local slot mapping. Counts bind the exact three one-hot prefixes.
 assert(original.rank_offsets==p.rank_offsets&&original.policy_offsets==p.policy_offsets);
 for(size_t r=0;r<p.bulk_counts.size();++r){
  std::set<int>old(original.tile_ids.begin()+p.rank_offsets[r],original.tile_ids.begin()+p.rank_offsets[r+1]);
  std::set<int>now(p.tile_ids.begin()+p.rank_offsets[r],p.tile_ids.begin()+p.rank_offsets[r+1]);assert(old==now);
  for(int i=p.rank_offsets[r];i<p.rank_offsets[r+1];++i)assert(p.slots[p.tile_ids[i]]==i-p.rank_offsets[r]);
 }
 prepared=&p;StreamState origin,branches[6],axis_streams[3];EventState fork,done[6],axis_fork,axis_done[3];
 BeamzNativeMovingWorkspace w;w.rank_count=p.bulk_counts.size();w.policy_offsets_host=p.policy_offsets.data();w.tile_ids_device=p.tile_ids.data();
 w.uniform_noio_counts_host=p.uniform_noio_counts.data();w.uniform_observer_counts_host=p.uniform_observer_counts.data();w.uniform_noio_axis_counts_host=p.uniform_noio_axis_counts.data();
 w.parallel_policies=parallel_policy;w.parallel_noio_axes=parallel_axes;w.policy_fork_event=&fork;w.noio_axis_fork_event=&axis_fork;
 for(int k=0;k<6;++k){w.policy_streams[k]=branches+k;w.policy_done_events[k]=done+k;}
 for(int k=0;k<3;++k){w.noio_axis_streams[k]=axis_streams+k;w.noio_axis_done_events[k]=axis_done+k;}
 graph.clear();next_node=1;
 for(int macro=0;macro<3;++macro){
  rank_class_nodes.clear();tile_node.assign(p.tile_ids.size(),-1);
  int begin=next_node++;graph[begin]=origin.frontier;origin.frontier={begin};
  int result=depth==4?LaunchRanks<4,1,true>(&origin,Parameters{},w):LaunchRanks<16,1,true>(&origin,Parameters{},w);assert(result==cudaSuccess);
  int end=next_node++;graph[end]=origin.frontier;origin.frontier={end};Reachability reach(next_node);
  for(int node:tile_node)if(node<0||!reach.has(node,end))throw std::logic_error("NoIO child did not reach outer completion");
  auto before=[&](int a,int b){if(a==b||!reach.has(a,b))throw std::logic_error("required producer/consumer join missing");};
  std::vector<int>last(4*p.max_frontier,-1);
  for(size_t r=0;r<p.bulk_counts.size();++r)for(int i=p.rank_offsets[r];i<p.rank_offsets[r+1];++i){int t=p.tile_ids[i];
   int z=t/(p.counts[1]*p.counts[2]),y=(t/p.counts[2])%p.counts[1],x=t%p.counts[2];
   if(!reach.has(begin,tile_node[t]))throw std::logic_error("launch lost initial/previous-macro dependency");
   for(int bits=1;bits<8;++bits){int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(z>=dz&&y>=dy&&x>=dx){before(tile_node[t-dz*p.counts[1]*p.counts[2]-dy*p.counts[2]-dx],tile_node[t]);++raw_checks;}}
   int slot=(r&3)*p.max_frontier+p.slots[t],old=last[slot];
   if(old>=0){before(tile_node[old],tile_node[t]);++reuse_checks;int oz=old/(p.counts[1]*p.counts[2]),oy=(old/p.counts[2])%p.counts[1],ox=old%p.counts[2];
    for(int bits=1;bits<8;++bits){int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(oz+dz<p.counts[0]&&oy+dy<p.counts[1]&&ox+dx<p.counts[2]){before(tile_node[old+dz*p.counts[1]*p.counts[2]+dy*p.counts[2]+dx],tile_node[t]);++reuse_checks;}}
   }last[slot]=t;
  }
  if(parallel_axes)for(size_t r=0;r<p.bulk_counts.size();++r)for(int a=0;a<3;++a){auto found=rank_class_nodes.find({r,a});if(found==rank_class_nodes.end())continue;int axis_node=found->second;
   for(int b=a+1;b<5;++b){auto other=rank_class_nodes.find({r,b});if(other==rank_class_nodes.end())continue;
    if(reach.has(other->second,axis_node)||reach.has(axis_node,other->second))throw std::logic_error("axis fork/join serialized independent same-rank work");++concurrency_checks;
   }
  }
 }
}
int main(){
 try{
  size_t cases=0;
  for(auto dims:std::vector<std::array<int,3>>{{25,25,25},{33,47,61},{64,96,128},{136,136,136},{512,512,512}})
  for(int depth:{4,16})for(bool stress:{false,true})for(bool outer:{false,true})for(bool axes:{false,true}){run_case(dims,depth,stress,outer,axes);++cases;}
  if(!concurrency_checks)throw std::logic_error("concurrent axes were not exercised");
  std::cout<<"{\"cases\":"<<cases<<",\"macros_per_case\":3,\"actual_API_waits\":"<<waits<<",\"actual_API_records\":"<<records<<",\"launches\":"<<launches<<",\"RAW_predecessor_checks\":"<<raw_checks<<",\"ring_reuse_checks\":"<<reuse_checks<<",\"unordered_axis_checks\":"<<concurrency_checks<<",\"pass\":true}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}
}
