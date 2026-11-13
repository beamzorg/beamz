// Execute the actual host launcher against a capture-frontier CUDA mock.
// Event record snapshots a stream frontier; wait snapshots that recorded state.
// No GPU, CUDA runtime or floating-point emulation is involved.
#define BeamzNativeMovingWorkspace BaseMovingWorkspace
#include "native_moving.h"
#undef BeamzNativeMovingWorkspace
#include "cuda_native_cohort_dependencies.h"
#include <cassert>
#include <iostream>
#include <map>
#include <set>

struct BeamzNativeMovingWorkspace:BaseMovingWorkspace {
 const int32_t*uniform_noio_counts_host=nullptr,*uniform_observer_counts_host=nullptr;
 const int32_t*cohort_offsets_host=nullptr,*cohort_dependencies_host=nullptr,*cohort_last_policy_host=nullptr;
 void*const*cohort_events_host=nullptr;int cohort_dependency_count=0;
};
struct StreamState{std::set<int>frontier;int policy=-1;};
struct EventState{std::set<int>frontier;};
using cudaStream_t=StreamState*;using cudaEvent_t=EventState*;using cudaError_t=int;
constexpr int cudaSuccess=0,cudaErrorInvalidValue=1;
static size_t waits=0,records=0,launches=0;
static int base=0;
static const BeamzNativeMovingPlan*prepared=nullptr;
static const BeamzNativeMovingCohorts*cohorts=nullptr;
static std::map<int,std::set<int>>graph;
static std::set<int>launched;
static int cudaEventRecord(cudaEvent_t event,cudaStream_t stream){assert(event&&stream&&!stream->frontier.empty());event->frontier=stream->frontier;++records;return cudaSuccess;}
static int cudaStreamWaitEvent(cudaStream_t stream,cudaEvent_t event,unsigned){assert(event&&stream&&!event->frontier.empty());stream->frontier.insert(event->frontier.begin(),event->frontier.end());++waits;return cudaSuccess;}
struct Parameters{};
static int launch(cudaStream_t stream,const int32_t*ids,int count,int policy){
 assert(count>0&&stream->policy==policy);const int c=cohorts->tile_cohort[*ids],r=c/6;
 assert(c%6==policy&&count==prepared->policy_offsets[r*7+policy+1]-prepared->policy_offsets[r*7+policy]);
 for(int i=0;i<count;++i)assert(cohorts->tile_cohort[ids[i]]==c);
 assert(launched.insert(base+c).second);graph[base+c]=stream->frontier;stream->frontier={base+c};++launches;return cudaSuccess;
}
template<int D,int L,bool Fast,bool Iso,int Policy>static int Launch(cudaStream_t s,const Parameters&,const int32_t*i,int n){return launch(s,i,n,Policy);}
template<int D,int L,bool Iso>static int LaunchNoIO(cudaStream_t s,const Parameters&,const int32_t*i,int n,int u){assert(u>=0&&u<=n);return launch(s,i,n,1);}
template<int D,int L,bool Iso>static int LaunchObservers(cudaStream_t s,const Parameters&,const int32_t*i,int n,int u){assert(u>=0&&u<=n);return launch(s,i,n,2);}
#include "cuda_native_cohort_schedule.inc"

static std::set<int> ancestors(int node){std::set<int>result;std::vector<int>todo{node};while(!todo.empty()){int c=todo.back();todo.pop_back();for(int d:graph.at(c))if(result.insert(d).second)todo.push_back(d);}return result;}
int main(){
 using namespace beamz::cuda::moving;size_t cases=0,checked=0;
 const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 for(auto dims:std::vector<std::array<int,3>>{{64,96,128},{136,136,136},{512,512,512}})for(int depth:{4,16}){
  BeamzBuffer f[6]{};for(int c=0;c<6;++c){f[c].rank=3;f[c].element_type=kBeamzF32;for(int a=0;a<3;++a)f[c].dims[a]=dims[a]+add[c][a];}
  auto p=BeamzBuildNativeMovingPlan(f,depth,{{0,0,0},{dims[0]/2,dims[1]/2,dims[2]/2}},nullptr,nullptr,12,true);auto g=BeamzBuildNativeMovingCohorts(p);prepared=&p;cohorts=&g;
  const int count=g.offsets.size()-1;std::vector<EventState>events(count);std::vector<void*>handles;for(auto&e:events)handles.push_back(&e);
  StreamState origin;StreamState branches[6];EventState fork;
  std::vector<int32_t>zeros(p.bulk_counts.size(),0);BeamzNativeMovingWorkspace w;
  w.rank_count=p.bulk_counts.size();w.policy_offsets_host=p.policy_offsets.data();w.tile_ids_device=p.tile_ids.data();w.policy_fork_event=&fork;
  w.uniform_noio_counts_host=zeros.data();w.uniform_observer_counts_host=zeros.data();
  w.cohort_offsets_host=g.offsets.data();w.cohort_dependencies_host=g.dependencies.data();w.cohort_dependency_count=g.dependencies.size();w.cohort_last_policy_host=g.last_policy_cohort.data();w.cohort_events_host=handles.data();
  for(int policy=0;policy<6;++policy){branches[policy].policy=policy;w.policy_streams[policy]=branches+policy;}
  graph.clear();launched.clear();
  for(int macro=0;macro<3;++macro){base=macro*(count+2);const int begin=base+count,end=base+count+1;graph[begin]=origin.frontier;origin.frontier={begin};
   int result=depth==4?LaunchCohortRanks<4,1,true>(&origin,Parameters{},w):LaunchCohortRanks<16,1,true>(&origin,Parameters{},w);assert(result==cudaSuccess);graph[end]=origin.frontier;origin.frontier={end};
   auto final_ancestors=ancestors(end);
   for(int c=0;c<count;++c){int r=c/6,k=c%6;if(p.policy_offsets[r*7+k]==p.policy_offsets[r*7+k+1])continue;
    auto a=ancestors(base+c);assert(a.count(begin)&&final_ancestors.count(base+c));
    for(int i=g.offsets[c];i<g.offsets[c+1];++i){assert(a.count(base+g.dependencies[i]));++checked;}
   }
  }
  assert(launched.size()==3*g.active_cohorts);++cases;
 }
 std::cout<<"{\"cases\":"<<cases<<",\"macros_per_case\":3,\"actual_API_waits\":"<<waits<<",\"actual_API_records\":"<<records<<",\"cohort_launches\":"<<launches<<",\"capture_dependency_checks\":"<<checked<<",\"pass\":true}\n";
}
