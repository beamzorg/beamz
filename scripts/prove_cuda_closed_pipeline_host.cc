// Included by the host validator with the actual staged scheduler helpers.
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <set>
#include "native_closed_pipeline.h"
#include "native_closed_cohorts.h"
#define __device__
#define __forceinline__ inline
struct ThreadIndex{int x=0;};thread_local ThreadIndex threadIdx;
void __nanosleep(int){std::this_thread::yield();}
void __threadfence(){std::atomic_thread_fence(std::memory_order_seq_cst);}
void __syncthreads(){std::atomic_thread_fence(std::memory_order_seq_cst);}
void atomicOr(int*p,int v){std::atomic_ref<int>(*p).fetch_or(v);}
template<class T>T __ldg(const T*p){return *p;}
template<class T>T __ldcg(const T*p){return *p;}
namespace cuda{enum scope{thread_scope_device,thread_scope_block};
using std::memory_order_acquire;using std::memory_order_release;using std::memory_order_acq_rel;using std::memory_order_relaxed;
template<class T,scope S>using atomic_ref=std::atomic_ref<T>;}
using std::min;
struct Parameters{int nz,ny,nx;};
#include "control.h"

static size_t hazard_checks=0,queue_tasks=0,step_waits=0,queue_cases=0,planner_cases=0;
static void require(bool value,const char*why){if(!value)throw std::runtime_error(why);}

static void audit(const BeamzNativeMovingPlan&p,const BeamzNativeClosedCohorts&q){
 const int n=p.tile_ids.size(),nz=p.counts[0],ny=p.counts[1],nx=p.counts[2],cohorts=q.cohort_policies.size();
 require(q.owner_slots.size()==size_t(n)&&q.tile_ids.size()==size_t(n),"Owner/task shape");
 std::vector<int>position(n,-1),last(q.owner_count,-1),pending(n,0);std::vector<std::vector<int>>finish(n);
 for(int i=0;i<n;++i){const int t=q.tile_ids[i];require(t>=0&&t<n&&position[t]<0,"Task coverage");position[t]=i;}
 for(int t:q.tile_ids){
  const int z=t/(ny*nx),y=(t/nx)%ny,x=t%nx,c=q.tile_cohort[t],owner=q.owner_slots[t];
  require(c>=0&&c<cohorts&&(q.cohort_policies[c]==6||q.cohort_policies[c]==beamz::cuda::moving::Policy(p.tile_flags[t])),"Physics signature");
  require(owner>=0&&owner<q.owner_count,"Owner bounds");
  if(q.cohort_policies[c]==6)require(cohorts==1&&c==0&&owner==((z+y+x)%q.epochs)*p.max_frontier+p.slots[t],"Global worker native owner identity");
  const int parents[3]={z?t-ny*nx:-1,y?t-nx:-1,x?t-1:-1};
  for(int parent:parents)if(parent>=0){require(q.tile_cohort[parent]<=c&&position[parent]<position[t],"START closure");pending[t]+=q.tile_cohort[parent]==c;++hazard_checks;}
  bool exported=false;
  for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
   if(z+dz<nz&&y+dy<ny&&x+dx<nx)exported|=q.tile_cohort[t+dz*ny*nx+dy*nx+dx]!=c;}
  require(exported?(owner>=q.internal_ring_owners):(owner<q.internal_ring_owners),"Export classified as internal");
  const int old=last[owner];require(old==q.previous_holder[t],"Actual owner predecessor");
  if(old>=0){std::set<int>readers{old};const int oz=old/(ny*nx),oy=(old/nx)%ny,ox=old%nx;
   for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
    if(oz+dz<nz&&oy+dy<ny&&ox+dx<nx)readers.insert(old+dz*ny*nx+dy*nx+dx);}
   for(int parent:readers){
    require(q.tile_cohort[parent]<=c&&position[parent]<position[t],"Physical packet reused before final reader");
    if(q.tile_cohort[parent]==c){++pending[t];finish[parent].push_back(t);}++hazard_checks;
   }
  }
  last[owner]=t;
 }
 require(pending==q.initial_pending,"Typed pending count");
 for(int t=0;t<n;++t){
  std::vector<int>actual(q.finish_children.begin()+q.finish_offsets[t],q.finish_children.begin()+q.finish_offsets[t+1]);
  std::sort(actual.begin(),actual.end());std::sort(finish[t].begin(),finish[t].end());require(actual==finish[t],"All7reader completion CSR");
 }
 for(int c=0;c<cohorts;++c){std::set<int>expected,actual;
  for(int i=q.cohort_offsets[c];i<q.cohort_offsets[c+1];++i)if(!pending[q.tile_ids[i]])expected.insert(q.tile_ids[i]);
#ifdef BEAMZ_READY_FIFO
  require(q.initial_heads[c]==0,"Initial FIFO ticket");
  for(int i=q.cohort_offsets[c];i<q.cohort_offsets[c]+q.root_counts[c];++i)require(actual.insert(q.initial_next[i]).second,"Unique FIFO roots");
  for(int i=q.cohort_offsets[c]+q.root_counts[c];i<q.cohort_offsets[c+1];++i)require(q.initial_next[i]==-1,"Unpublished FIFO slot must be empty");
#else
  int next=q.initial_heads[c];
  while(next>=0){require(next<n&&q.tile_cohort[next]==c&&actual.insert(next).second,"Unique root stack");next=q.initial_next[next];}
#endif
  require(actual==expected&&actual.size()==size_t(q.root_counts[c]),"Initial root coverage");
 }
 require(q.Bytes()==sizeof(int32_t)*(size_t(9)*n+q.finish_children.size()+size_t(7)*cohorts+2),"Exact GPU metadata bytes");
 require(q.CapacityBytes()==sizeof(int32_t)*(size_t(17)*n+size_t(7)*cohorts+2),"Exact reserved GPU metadata capacity");
 ++planner_cases;
}

static void stress(const BeamzNativeMovingPlan&p,const BeamzNativeClosedCohorts&plan,int workers,int depth,int seed){
 using namespace closed_pipeline;const int n=p.tile_ids.size(),cohorts=plan.cohort_policies.size();
 BeamzNativeClosedPipelineWorkspace q;
 std::vector<int>next=plan.initial_next,pending=plan.initial_pending,claimed(n),ready(n),heads=plan.initial_heads,enqueued=plan.root_counts,started(cohorts),completed(cohorts),exited(cohorts),proof_started(n),proof_done(n),tags(plan.owner_count,-1);
 for(int t=0;t<n;++t)claimed[t]=pending[t]==0;
 int abort=0,bad=0;q.next=next.data();q.pending=pending.data();q.claimed=claimed.data();q.ready=ready.data();
 q.initial_next=plan.initial_next.data();q.initial_pending=plan.initial_pending.data();q.initial_heads=plan.initial_heads.data();q.root_counts=plan.root_counts.data();
 q.tile_cohort=plan.tile_cohort.data();q.owner_slots=plan.owner_slots.data();q.finish_offsets=plan.finish_offsets.data();q.finish_children=plan.finish_children.data();q.finish_count=plan.finish_children.size();
 q.heads=heads.data();q.enqueued=enqueued.data();q.started=started.data();q.completed=completed.data();q.exited=exited.data();q.abort=&abort;
 q.tile_count=n;q.cohort_count=cohorts;q.max_workers=workers;q.timeout_ns=30000000000ULL;Parameters physics{p.counts[0],p.counts[1],p.counts[2]};
 for(int c=0;c<cohorts;++c){q.cohort=c;q.cohort_tasks=plan.cohort_offsets[c+1]-plan.cohort_offsets[c];q.worker_count=std::min(workers,q.cohort_tasks);
#ifdef BEAMZ_READY_FIFO
  q.cohort_offset=plan.cohort_offsets[c];
#endif
  std::vector<std::thread>threads;
  for(int worker=0;worker<q.worker_count;++worker)threads.emplace_back([&,worker]{
   auto&cold=const_cast<Cold&>(C());cold.q=q;cold.bad=&bad;cold.depth=depth;
   for(;;){
    Pop();if(Tile()<0)break;const int t=Tile(),owner=Owner(physics,t),z=t/(physics.ny*physics.nx),y=(t/physics.nx)%physics.ny,x=t%physics.nx;
    if(std::atomic_ref<int>(proof_started[t]).fetch_add(1))Abort(8192);
    const int old=Acquire(tags.data()+owner);
    if(old!=plan.previous_holder[t])Abort(16384);
    if(old>=0){if(!Acquire(proof_done.data()+old))Abort(16384);
     const int oz=old/(physics.ny*physics.nx),oy=(old/physics.nx)%physics.ny,ox=old%physics.nx;
     for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
      if(oz+dz<physics.nz&&oy+dy<physics.ny&&ox+dx<physics.nx&&!Acquire(proof_done.data()+old+dz*physics.ny*physics.nx+dy*physics.nx+dx))Abort(16384);}}
    Release(tags.data()+owner,t);Started(physics);
    for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(z<dz||y<dy||x<dx)continue;
     const int parent=t-dz*physics.ny*physics.nx-dy*physics.nx-dx;if(Acquire(tags.data()+Owner(physics,parent))!=parent)Abort(32768);}
    for(int step=1;step<=depth&&!Aborted();++step){WaitParents(physics,step);if(Aborted())break;
     if((t+step+worker+seed)%7==0)std::this_thread::yield();Publish(step);}
    if(Aborted())break;Release(proof_done.data()+t,1);Complete();
   }
  });
  for(auto&thread:threads)thread.join();
  require(!bad&&!abort,"Actual closed queue/control aborted");require(completed[c]==q.cohort_tasks&&started[c]==q.cohort_tasks&&enqueued[c]==q.cohort_tasks,"Lost/duplicated cohort tasks");
 }
 for(int t=0;t<n;++t)require(proof_started[t]==1&&proof_done[t]==1&&ready[t]==depth&&pending[t]==0&&claimed[t]==1,"Terminal task state");
 queue_tasks+=n;step_waits+=size_t(n)*depth;++queue_cases;
}

int main(){
 const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 for(auto dims:std::vector<std::array<int,3>>{{25,29,33},{64,72,80},{136,136,136},{512,512,512},{564,564,564},{6000,25,25}})
 for(int depth:{4,16})for(bool single_class:{false,true}){
  BeamzBuffer fields[6]{};for(int c=0;c<6;++c){fields[c].rank=3;fields[c].element_type=kBeamzF32;for(int a=0;a<3;++a)fields[c].dims[a]=dims[a]+add[c][a];}
  BeamzSourceGroupLaunch groups[9]{};const int32_t*starts[9]{};int32_t start[6]={12,12,dims[2]/3,12,12,dims[2]/3};
  for(int g:{5,7}){auto&f=groups[g].coefficients;f.rank=4;f.dims[0]=2;f.dims[1]=dims[0]-24;f.dims[2]=dims[1]-24;f.dims[3]=1;starts[g]=start;}
  auto p=BeamzBuildNativeMovingPlan(fields,depth,{{0,0,0},{dims[0]/2,dims[1]/2,dims[2]/2}},groups,starts,12,true);
  auto q=BeamzBuildNativeClosedCohorts(p,4,single_class);audit(p,q);
  for(size_t t=0;t<p.tile_flags.size();++t)if(t%3)p.tile_flags[t]|=64;
  BeamzCompactNativeMovingUniformNoIO(p,true);BeamzCompactNativeMovingUniformObservers(p,true);
  auto compact=BeamzBuildNativeClosedCohorts(p,4,single_class);audit(p,compact);
  require(q.tile_cohort==compact.tile_cohort&&q.owner_count==compact.owner_count&&q.internal_ring_owners==compact.internal_ring_owners&&q.cohort_policies==compact.cohort_policies,"Material compaction changed physics closures/allocation size");
  if(dims[0]<=136)for(int workers:{1,4,82})for(int seed:{13,37})stress(p,compact,workers,depth,seed);
 }
 std::cout<<"{\"passed\":true,\"planner_cases\":"<<planner_cases<<",\"hazard_checks\":"<<hazard_checks<<",\"queue_cases\":"<<queue_cases<<",\"queue_tasks\":"<<queue_tasks<<",\"step_waits\":"<<step_waits<<"}\n";
}
