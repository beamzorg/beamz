// Experimental resident-worker STEP pipeline. Included after asymmetric_native.cuh.
#pragma once
#include "native_persistent.h"
#include <cuda/atomic>

namespace beamz::cuda::moving_persistent {
using namespace beamz::cuda::moving_native;
struct Cold {
 const int32_t*jobs;const int32_t*previous;const int32_t*slots;
 int32_t*next;int32_t*abort_global;int32_t*bad;
 uint32_t*claimed;uint32_t*completed;uint32_t*ready;
 uint64_t*tags;
 uint64_t timeout_ns;
 int count,nz,ny,nx,frontier;
 uint32_t epoch;
 int tile,owner,abort_local;
};
constexpr size_t ColdOffset=(SourcePendingSharedBytes+15)&~size_t(15);
constexpr size_t PersistentSharedBytes=ColdOffset+sizeof(Cold);
static_assert(PersistentSharedBytes<99*1024,"SM86 persistent shared-memory budget");
__device__ __forceinline__ volatile Cold& C(){
 extern __shared__ float frame[];
 return *reinterpret_cast<volatile Cold*>(reinterpret_cast<char*>(frame)+ColdOffset);
}
template<class T>__device__ __forceinline__ T Acquire(const T*word){
 ::cuda::atomic_ref<T,::cuda::thread_scope_device>a(*const_cast<T*>(word));return a.load(::cuda::memory_order_acquire);
}
template<class T>__device__ __forceinline__ void Release(T*word,T value){
 ::cuda::atomic_ref<T,::cuda::thread_scope_device>a(*word);a.store(value,::cuda::memory_order_release);
}
__device__ __forceinline__ uint64_t Timer(){uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__device__ __forceinline__ void Abort(int bit){atomicOr(C().bad,bit);Release(C().abort_global,int32_t(1));C().abort_local=1;}
__device__ __forceinline__ bool PollAbort(){
 if(Acquire(C().abort_global)){C().abort_local=1;return true;}return false;
}
__device__ __forceinline__ bool WaitCompleted(int tile){
 uint64_t last=Timer();uint32_t previous=0;
 for(;;){if(PollAbort())return false;const uint32_t seen=Acquire(C().completed+tile);
  if(seen==C().epoch)return true;
  if(seen>C().epoch){Abort(512);return false;}
  if(seen>previous){previous=seen;last=Timer();}
  if(Timer()-last>C().timeout_ns){Abort(256);return false;}__nanosleep(64);
 }
}
__device__ __forceinline__ void Claim(){
 if(PollAbort())return;
 const int tile=C().tile,old=C().previous[tile];
 if(old>=0){
  if(!WaitCompleted(old))return;
  const int z=old/(C().ny*C().nx),y=(old/C().nx)%C().ny,x=old%C().nx;
  // Explicit WAR set, even when one of these edges is also a RAW edge.
  for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
   if(z+dz<C().nz&&y+dy<C().ny&&x+dx<C().nx)
    if(!WaitCompleted(old+dz*C().ny*C().nx+dy*C().nx+dx))return;
  }
 }
 const int rank=tile/(C().ny*C().nx)+(tile/C().nx)%C().ny+tile%C().nx;
 const int owner=(rank&3)*C().frontier+C().slots[tile];C().owner=owner;
 // Reader retirement above makes the reset exclusive. Release tag publishes
 // the reset before any consumer may interpret the reused ready counter.
 ::cuda::atomic_ref<uint32_t,::cuda::thread_scope_device>ready(C().ready[owner]);ready.store(0,::cuda::memory_order_relaxed);
 Release(C().tags+owner,(uint64_t(C().epoch)<<32)|uint32_t(tile));
 Release(C().claimed+tile,uint32_t(C().epoch));
}
__device__ __forceinline__ bool WaitProducer(int producer,int owner,int step){
 uint64_t last=Timer(),previous_tag=0;uint32_t previous_ready=0;
 const uint64_t expected=(uint64_t(C().epoch)<<32)|uint32_t(producer);
 for(;;){
  if(PollAbort())return false;
  const uint64_t tag=Acquire(C().tags+owner);
  if(tag==expected){
   const uint32_t ready=Acquire(C().ready+owner);
   if(ready>=uint32_t(step))return true;
   if(ready>previous_ready){previous_ready=ready;last=Timer();}
  }else if(tag!=previous_tag){previous_tag=tag;last=Timer();}
  if(Timer()-last>C().timeout_ns){Abort(256);return false;}__nanosleep(64);
 }
}
__device__ __forceinline__ void WaitUpstream(int step){
 const int tile=C().tile,z=tile/(C().ny*C().nx),y=(tile/C().nx)%C().ny,x=tile%C().nx,rank=z+y+x;
 for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
  if(z>=dz&&y>=dy&&x>=dx){const int producer=tile-dz*C().ny*C().nx-dy*C().nx-dx;
   const int owner=((rank-dz-dy-dx)&3)*C().frontier+C().slots[producer];
   if(!WaitProducer(producer,owner,step))return;
  }
 }
}
__device__ __forceinline__ void PublishAndWait(int step,int last){
 if(PollAbort())return;
 Release(C().ready+C().owner,uint32_t(step));
 if(step<last)WaitUpstream(step+1);
}
struct PersistentEnvironment:Environment<1,false,true,5,false,20,false>{
 using Base=Environment<1,false,true,5,false,20,false>;
 __device__ __forceinline__ PersistentEnvironment(const Parameters&physics,uint8_t facts):Base{physics,facts}{}
 __device__ __forceinline__ float LoadLocal(int slot,int z,int y,int x)const{
  // Ownership handoffs now occur within one kernel across SMs. Bypass L1 for
  // mutable psi; this is coherent .cg, never readonly/noncoherent __ldg.
  return __ldcg(this->p.phase[slot/6].psi[slot%6].input+this->PsiOffset(slot,z,y,x));
 }
 __device__ __forceinline__ float Coefficient(int kind,int index,int z,int y,int x)const{
  if(kind==1&&index>=3&&index<6&&(this->flags&64))return this->p.common_e[index-3];
  return Base::Coefficient(kind,index,z,y,x);
 }
};

// BEAMZ_PERSISTENT_PAIRED_BODY_INSERT

__global__ void ResetPersistent(BeamzNativePersistentWorkspace q,const int32_t*bad){
 if(!threadIdx.x){*q.next_job=0;*q.abort=(*bad!=0);}
}
template<int Depth>
__global__ __launch_bounds__(1024,1) void PersistentTileStep(
 const __grid_constant__ Parameters p,const __grid_constant__ BeamzNativePersistentWorkspace q){
 if(!threadIdx.x){
  C().jobs=q.job_ids_device;C().previous=q.previous_holder_device;C().slots=p.slots;
  C().next=q.next_job;C().abort_global=q.abort;C().bad=p.bad;
  C().claimed=q.claimed_epoch;C().completed=q.completed_epoch;C().ready=q.ready_step;
  C().tags=p.owner_tags;C().timeout_ns=q.timeout_ns;
  C().count=q.job_count;C().nz=p.nz;C().ny=p.ny;C().nx=p.nx;C().frontier=p.frontier;
  C().epoch=Acquire(p.epoch);C().tile=-1;C().owner=-1;C().abort_local=0;
 }
 __syncthreads();
 for(;;){
  if(!threadIdx.x){
   if(PollAbort())C().tile=-1;
   else{const int job=atomicAdd(C().next,1);C().tile=job<C().count?C().jobs[job]:-1;}
  }
  __syncthreads();
  if(C().tile<0)break;
  PersistentPairedBody<Depth>(p);
  // All physical writers are fenced at every RLpair; this barrier also
  // handles empty/early-window exits before the one completion publisher.
  __syncthreads();
  if(!threadIdx.x){
   if(!PollAbort()){
    Release(C().ready+C().owner,uint32_t(Depth));
    Release(C().completed+C().tile,uint32_t(C().epoch));
   }
  }
  __syncthreads();
  if(C().abort_local)break;
 }
}
template<int Depth>
inline cudaError_t LaunchPersistent(cudaStream_t stream,const Parameters&p,const BeamzNativePersistentWorkspace&q){
 const auto kernel=PersistentTileStep<Depth>;
 auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,PersistentSharedBytes);if(status!=cudaSuccess)return status;
 PersistentTileStep<Depth><<<q.worker_count,1024,PersistentSharedBytes,stream>>>(p,q);return cudaGetLastError();
}
} // namespace beamz::cuda::moving_persistent

int BeamzEnqueueNativePersistent(void*opaque,const BeamzProgramLaunch&program,const BeamzBuffer*final_fields,
 int macro_step,const BeamzNativeMovingWorkspace&w,const BeamzNativePersistentWorkspace&q){
 using namespace beamz::cuda::moving_native;
 const auto validation=BeamzValidateNativeProgram(program);if(validation)return validation;
 if(!(w.data_flags&beamz::cuda::moving::kBytePackedE)||!(w.data_flags&beamz::cuda::moving::kLosslessIsotropicH)||
    (w.source_presence_mask!=0&&w.source_presence_mask!=20)||q.job_count!=w.tile_count||q.worker_count<1||q.worker_count>q.job_count||
    q.job_count>INT32_MAX-q.worker_count||q.timeout_ns<1000000ULL||!q.job_ids_device||!q.previous_holder_device||!q.next_job||!q.abort||
    !q.claimed_epoch||!q.completed_epoch||!q.ready_step)return cudaErrorInvalidValue;
 Parameters p;int layout=0;if(!Prepare(program,final_fields,w,macro_step,&p,&layout)||layout!=1)return cudaErrorInvalidValue;
 const auto stream=static_cast<cudaStream_t>(opaque);
 BeginEpoch<<<1,1,0,stream>>>(w.epoch,w.bad);auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
 beamz::cuda::moving_persistent::ResetPersistent<<<1,1,0,stream>>>(q,w.bad);status=cudaGetLastError();if(status!=cudaSuccess)return status;
 if(w.depth==4)return beamz::cuda::moving_persistent::LaunchPersistent<4>(stream,p,q);
 if(w.depth==16)return beamz::cuda::moving_persistent::LaunchPersistent<16>(stream,p,q);
 return cudaErrorInvalidValue;
}
