// Actual generated device control functions, compiled against CPU atomics.
// This validates protocol ordering, NOT CUDA performance or FDTD arithmetic.
#include "native_persistent.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace cuda {
inline constexpr int thread_scope_device=0;
template<class T,int>using atomic_ref=std::atomic_ref<T>;
inline constexpr auto memory_order_acquire=std::memory_order_acquire;
inline constexpr auto memory_order_release=std::memory_order_release;
inline constexpr auto memory_order_relaxed=std::memory_order_relaxed;
}
#define __device__
#define __forceinline__ inline
static int atomicOr(int*word,int value){return std::atomic_ref<int>(*word).fetch_or(value);}
static void __nanosleep(unsigned){std::this_thread::yield();}
#include "actual_persistent_control.inc"

namespace beamz::cuda::moving_persistent {
thread_local Cold local_cold{};
volatile Cold&C(){return local_cold;}
}

static std::atomic<size_t>checked_steps{0},overlaps{0};
static size_t completed_cases=0;
static std::mutex message_mutex;
static std::string failure;

static void run_case(const std::array<int,3>&counts,int depth,int workers,bool locality,int epoch_runs){
 using namespace beamz::cuda::moving_persistent;
 BeamzNativeMovingPlan moving;moving.counts=counts;moving.depth=depth;moving.epochs=4;
 const int count=counts[0]*counts[1]*counts[2],ranks=counts[0]+counts[1]+counts[2]-2;
 moving.bulk_counts.assign(ranks,0);moving.slots.assign(count,0);moving.rank_offsets={0};
 for(int rank=0;rank<ranks;++rank){int slot=0;for(int t=0;t<count;++t)if(BeamzPersistentTileRank(t,moving)==rank){moving.tile_ids.push_back(t);moving.slots[t]=slot++;}
  // Deliberately permute actual rank slots; never assume position==slot.
  if(slot>1)for(int i=moving.rank_offsets.back();i<int(moving.tile_ids.size());++i){int t=moving.tile_ids[i];moving.slots[t]=(moving.slots[t]* (slot-1)+1)%slot;}
  moving.max_frontier=std::max(moving.max_frontier,slot);moving.rank_offsets.push_back(moving.tile_ids.size());
 }
 auto plan=BeamzBuildNativePersistentPlan(moving,locality);
 const int owner_count=4*moving.max_frontier;
 std::vector<uint64_t>tags(owner_count,0),trail(size_t(owner_count)*depth,0);
 std::vector<uint32_t>ready(owner_count,0),claimed(count,0),completed(count,0);
 int32_t next=0,abort=0,bad=0;
 auto owner=[&](int t){return (BeamzPersistentTileRank(t,moving)&3)*moving.max_frontier+moving.slots[t];};
 for(int epoch=1;epoch<=epoch_runs;++epoch){
  next=abort=bad=0;std::vector<std::thread>threads;
  for(int wi=0;wi<workers;++wi)threads.emplace_back([&,wi]{
   auto&c=local_cold;c.jobs=plan.job_ids.data();c.previous=plan.previous_holder.data();c.slots=moving.slots.data();
   c.next=&next;c.abort_global=&abort;c.bad=&bad;c.claimed=claimed.data();c.completed=completed.data();c.ready=ready.data();c.tags=tags.data();
   c.timeout_ns=1000000000ULL;c.count=count;c.nz=counts[0];c.ny=counts[1];c.nx=counts[2];c.frontier=moving.max_frontier;c.epoch=epoch;c.abort_local=0;
   // A delayed, unstarted worker never takes a queue item before executing.
   std::this_thread::sleep_for(std::chrono::microseconds(wi*7));
   try{for(;;){
    if(PollAbort())break;const int job=std::atomic_ref<int32_t>(next).fetch_add(1);if(job>=count)break;c.tile=c.jobs[job];
    Claim();if(c.abort_local)break;
    const int old=plan.previous_holder[c.tile];
    if(old>=0){if(Acquire(completed.data()+old)!=uint32_t(epoch))throw std::logic_error("old holder completion missing");
     for(int reader:BeamzPersistentConsumers(old,moving))if(Acquire(completed.data()+reader)!=uint32_t(epoch))throw std::logic_error("old consumer not retired before claim");}
    WaitUpstream(1);if(c.abort_local)break;
    for(int step=1;step<=depth;++step){
     const int t=c.tile,z=t/(counts[1]*counts[2]),y=(t/counts[2])%counts[1],x=t%counts[2];
     for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(z<dz||y<dy||x<dx)continue;
      const int parent=t-dz*counts[1]*counts[2]-dy*counts[2]-dx;
      if(trail[size_t(owner(parent))*depth+step-1]!=((uint64_t(epoch)<<32)|uint32_t(parent)))throw std::logic_error("STEP ready exposed stale/overwritten trail");
      if(Acquire(completed.data()+parent)!=uint32_t(epoch))++overlaps;
     }
     // One CPU thread represents the barrier-completed/fenced set of CTA
     // writers. The test does not replace the CUDA all-thread fence proof.
     trail[size_t(c.owner)*depth+step-1]=(uint64_t(epoch)<<32)|uint32_t(t);
     std::atomic_thread_fence(std::memory_order_release);
     std::this_thread::sleep_for(std::chrono::microseconds((t+step*7)%5));
     PublishAndWait(step,depth);if(c.abort_local)break;++checked_steps;
    }
    if(c.abort_local)break;
    Release(ready.data()+c.owner,uint32_t(depth));Release(completed.data()+c.tile,uint32_t(epoch));
   }}catch(const std::exception&e){
    {std::lock_guard<std::mutex>lock(message_mutex);if(failure.empty())failure=e.what();}Abort(1024);
   }
  });
  for(auto&t:threads)t.join();
  if(bad||abort||!failure.empty())throw std::logic_error(failure.empty()?"protocol watchdog/abort fired":failure);
  for(int t=0;t<count;++t)if(completed[t]!=uint32_t(epoch))throw std::logic_error("persistent coverage incomplete");
  ++completed_cases;
 }
}
int main(){
 try{
  for(auto counts:std::vector<std::array<int,3>>{{1,1,1},{2,3,4},{8,4,3},{12,9,6}})
  for(int depth:{4,16})for(int workers:{1,3,16})for(bool locality:{false,true})run_case(counts,depth,workers,locality,3);
  if(TestedPublicationSteps<16&&overlaps==0)throw std::logic_error("all RAW dependencies incorrectly waited full completion");
  std::cout<<"{\"cases_with_epochs\":"<<completed_cases<<",\"actual_step_reads\":"<<checked_steps.load()<<",\"reads_before_parent_completion\":"<<overlaps.load()<<",\"pass\":true}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}
}
