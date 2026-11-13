// Actual queue/control methods under matching C++ atomic primitive emulation.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#include "native_paired_cohorts.h"
#include "native_closed_pipeline.h"
#define __device__
#define __forceinline__ inline
struct ThreadIndex{int x=0;};thread_local ThreadIndex threadIdx;
void __nanosleep(int){std::this_thread::yield();}
void __threadfence(){std::atomic_thread_fence(std::memory_order_seq_cst);}
void atomicOr(int*p,int v){std::atomic_ref<int>(*p).fetch_or(v);}
template<class T>T __ldg(const T*p){return *p;}
template<class T>T __ldcg(const T*p){return *p;}
using std::min;
struct Parameters{int nz,ny,nx;};
#include "control.h"
#ifndef BEAMZ_GROUP_THREADS
#define BEAMZ_GROUP_THREADS 512
#endif
static void require(bool v,const char*why){if(!v)throw std::runtime_error(why);}
static size_t cases=0,nodes_checked=0,original_checked=0,waits=0;
static void run(const std::array<int,3>&shape,int depth,int workers,int seed){
 using namespace closed_pipeline;
 BeamzBuffer fields[6]{};const int add[6][3]={{0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1},{0,1,1}};
 for(int c=0;c<6;++c){fields[c].rank=3;fields[c].element_type=kBeamzF32;for(int a=0;a<3;++a)fields[c].dims[a]=shape[a]+add[c][a];}
 BeamzSourceGroupLaunch groups[9]{};const int32_t*starts[9]{};int32_t start[6]={12,12,shape[2]/3,12,12,shape[2]/3};
 for(int g:{5,7}){auto&f=groups[g].coefficients;f.rank=4;f.dims[0]=2;f.dims[1]=shape[0]-24;f.dims[2]=shape[1]-24;f.dims[3]=1;starts[g]=start;}
 const auto p=BeamzBuildNativeMovingPlan(fields,depth,{{0,0,0}},groups,starts,12,true);const auto plan=BeamzBuildNativePairedCohorts(p);
 const int n=plan.tile_ids.size(),original=p.tile_ids.size(),plane=plan.plane;
 std::vector<int>next=plan.initial_next,pending=plan.initial_pending,claimed(n),ready(original),heads=plan.initial_heads,enqueued=plan.root_counts;
 std::vector<int>started(1),completed(1),exited(1),tags(plan.owner_count,-1),proof_started(original),proof_done(original);
 for(int s=0;s<n;++s)claimed[s]=pending[s]==0;
 int bad=0,abort=0;BeamzNativeClosedPipelineWorkspace q;
 q.next=next.data();q.pending=pending.data();q.claimed=claimed.data();q.ready=ready.data();q.tile_cohort=plan.tile_cohort.data();q.owner_slots=plan.owner_slots.data();
 q.finish_offsets=plan.finish_offsets.data();q.finish_children=plan.finish_children.data();q.heads=heads.data();q.enqueued=enqueued.data();q.started=started.data();q.completed=completed.data();q.exited=exited.data();q.abort=&abort;
 q.tile_count=n;q.original_count=original;q.node_plane=plane;q.cohort_count=1;q.cohort_tasks=n;q.worker_count=std::min(workers,n);q.timeout_ns=30000000000ULL;
 Parameters physics{p.counts[0],p.counts[1],p.counts[2]};std::vector<std::thread>threads;
 for(int worker=0;worker<q.worker_count;++worker)threads.emplace_back([&,worker]{
  for(int g=0;g<2;++g){threadIdx.x=g*BEAMZ_GROUP_THREADS;auto&cold=const_cast<Cold&>(C());cold.q=q;cold.bad=&bad;cold.depth=depth;}
  for(;;){
   threadIdx.x=0;Pop();const int s=C().tile;if(s<0)break;
   threadIdx.x=BEAMZ_GROUP_THREADS;C().tile=s;C().parent_ready=0;C().abort_local=0;
   for(int g=0;g<2;++g){threadIdx.x=g*BEAMZ_GROUP_THREADS;const int t=Tile();if(t>=original)continue;
    if(std::atomic_ref<int>(proof_started[t]).fetch_add(1))Abort(8192);
    const int slot=Owner(physics,t),old=Acquire(tags.data()+slot);if(old!=plan.previous_holder[t])Abort(16384);
    if(old>=0){const int oz=old/plane,oy=(old/physics.nx)%physics.ny,ox=old%physics.nx;
     for(int dz=0;dz<=int(oz+1<physics.nz);++dz)for(int dy=0;dy<=int(oy+1<physics.ny);++dy)for(int dx=0;dx<=int(ox+1<physics.nx);++dx)
      if(!Acquire(proof_done.data()+old+dz*plane+dy*physics.nx+dx))Abort(16384);
    }
    Release(tags.data()+slot,t);
   }
   threadIdx.x=0;Started(physics);
   for(int g=0;g<2;++g){threadIdx.x=g*BEAMZ_GROUP_THREADS;const int t=Tile();if(t>=original)continue;
    const int z=t/plane,y=(t/physics.nx)%physics.ny,x=t%physics.nx;
    for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(z<dz||y<dy||x<dx)continue;
     const int parent=t-dz*plane-dy*physics.nx-dx;if(Acquire(tags.data()+Owner(physics,parent))!=parent)Abort(32768);
    }
   }
   for(int step=1;step<=depth;++step)for(int g=0;g<2;++g){threadIdx.x=g*BEAMZ_GROUP_THREADS;if(Tile()>=original)continue;
    WaitParents(physics,step);if(Aborted())break;if((s+step+worker+seed)%7==0)std::this_thread::yield();Publish(step);
   }
   for(int g=0;g<2;++g){threadIdx.x=g*BEAMZ_GROUP_THREADS;if(Tile()<original)Release(proof_done.data()+Tile(),1);}
   threadIdx.x=0;if(Acquire(&abort))break;Complete();
  }
 });
 for(auto&thread:threads)thread.join();
 require(!bad&&!abort,"Actual paired controller aborted");require(started[0]==n&&completed[0]==n&&enqueued[0]==n,"Lost/duplicate supernode");
 for(int s=0;s<n;++s)require(pending[s]==0&&claimed[s]==1,"Terminal supernode queue state");
 for(int t=0;t<original;++t)require(proof_started[t]==1&&proof_done[t]==1&&ready[t]==depth,"Original tile tag/ready coverage");
 ++cases;nodes_checked+=n;original_checked+=original;waits+=size_t(original)*depth;
}
int main(){
 for(auto shape:std::vector<std::array<int,3>>{{25,29,33},{64,72,80},{136,136,136}})for(int depth:{4,16})for(int workers:{1,4,82})for(int seed:{13,37})run(shape,depth,workers,seed);
 std::cout<<"{\"passed\":true,\"cases\":"<<cases<<",\"queue_supernodes\":"<<nodes_checked<<",\"original_tiles\":"<<original_checked<<",\"step_waits\":"<<waits<<",\"scope\":\"Actual staged supernode LIFO and typed START/COMPLETE, original5rank tags/all7reader retirement, lower3/upper2globalparent ready waits, both original ready arrays; CPUleader emulation with C++atomic scopes. Shared H/Evisibility and GPU behavior separate\"}\n";
}
