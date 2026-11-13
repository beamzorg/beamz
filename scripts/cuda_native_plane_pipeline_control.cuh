// Included inside beamz::cuda::moving_native after Parameters is defined.
namespace plane_pipeline {
constexpr size_t ColdOffset=62144;
struct Cold {
 BeamzNativePlanePipelineWorkspace q;
 int32_t*bad;
 int tile,depth,abort_local,parent_ready;
};
constexpr size_t SharedBytes=ColdOffset+sizeof(Cold);
static_assert(SharedBytes<64512,"Native paired profiles plus queue cold state must retain64KB carveout");
__device__ __forceinline__ volatile Cold& C(){
 extern __shared__ float frame[];
 return *reinterpret_cast<volatile Cold*>(reinterpret_cast<char*>(frame)+ColdOffset);
}
template<class T>__device__ __forceinline__ T Acquire(const T*pointer){
 ::cuda::atomic_ref<T,::cuda::thread_scope_device>value(*const_cast<T*>(pointer));return value.load(::cuda::memory_order_acquire);
}
template<class T>__device__ __forceinline__ void Release(T*pointer,T value){
 ::cuda::atomic_ref<T,::cuda::thread_scope_device>target(*pointer);target.store(value,::cuda::memory_order_release);
}
__device__ __forceinline__ uint64_t Timer(){uint64_t result;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(result));return result;}
__device__ __forceinline__ int Tile(){return C().tile;}
__device__ __forceinline__ int Owner(const Parameters&p,int tile){
 const int z=tile/(p.ny*p.nx);return (z%3)*p.ny*p.nx+tile%(p.ny*p.nx);
}
__device__ __forceinline__ void Abort(int bit){atomicOr(C().bad,bit);Release(C().q.abort,int32_t(1));C().abort_local=1;}
__device__ __forceinline__ bool PlaneSafe(int z){
 if(z<3)return true;
 const int count=C().q.ny*C().q.nx;
 return Acquire(C().q.plane_done+z-3)==count&&Acquire(C().q.plane_done+z-2)==count;
}
__device__ __forceinline__ void TryEnqueue(int tile){
 if(tile<0||tile>=C().q.tile_count||Acquire(C().q.pending+tile)||!PlaneSafe(tile/(C().q.ny*C().q.nx)))return;
 ::cuda::atomic_ref<int32_t,::cuda::thread_scope_device>claim(C().q.claimed[tile]);int32_t expected=0;
 if(!claim.compare_exchange_strong(expected,1,::cuda::memory_order_acq_rel,::cuda::memory_order_relaxed))return;
 ::cuda::atomic_ref<int32_t,::cuda::thread_scope_device>tail(*C().q.tail);
 const int ticket=tail.fetch_add(1,::cuda::memory_order_relaxed);
 if(ticket<0||ticket>=C().q.tile_count){Abort(2048);return;}
 C().q.queue[ticket]=tile;Release(C().q.entry_ready+ticket,int32_t(1));
}
// Called only by thread0 after the existing native owner-tag CTA barrier.
__device__ __forceinline__ void Started(){
 const int tile=C().tile,ny=C().q.ny,nx=C().q.nx,z=tile/(ny*nx),y=(tile/nx)%ny,x=tile%nx;
 const int next[3]={z+1<C().q.nz?tile+ny*nx:-1,y+1<ny?tile+nx:-1,x+1<nx?tile+1:-1};
 for(int axis=0;axis<3;++axis){const int child=next[axis];if(child<0)continue;
  ::cuda::atomic_ref<int32_t,::cuda::thread_scope_device>pending(C().q.pending[child]);
  const int previous=pending.fetch_sub(1,::cuda::memory_order_acq_rel);
  if(previous<1){Abort(2048);return;}if(previous==1)TryEnqueue(child);
 }
 ::cuda::atomic_ref<int32_t,::cuda::thread_scope_device>count(*C().q.started);count.fetch_add(1,::cuda::memory_order_relaxed);
}
__device__ __forceinline__ bool ProducerReady(int tile,int step){return tile<0||Acquire(C().q.ready+tile)>=step;}
__device__ __forceinline__ void Skipped(int step){if(step>0&&!threadIdx.x)Release(C().q.ready+C().tile,int32_t(step));}
__device__ __forceinline__ void WaitParents(int step){
 if(step<1)return;
 if(!threadIdx.x){
  const int tile=C().tile,ny=C().q.ny,nx=C().q.nx,z=tile/(ny*nx),y=(tile/nx)%ny,x=tile%nx;
  const int parent[3]={z?tile-ny*nx:-1,y?tile-nx:-1,x?tile-1:-1};const uint64_t start=Timer();
  while(!(ProducerReady(parent[0],step)&&ProducerReady(parent[1],step)&&ProducerReady(parent[2],step))){
   if(Acquire(C().q.abort)){C().abort_local=1;break;}
   if(Timer()-start>C().q.timeout_ns){Abort(256);break;}__nanosleep(32);
  }
  ::cuda::atomic_ref<int,::cuda::thread_scope_block>relay(*const_cast<int*>(&C().parent_ready));relay.store(step,::cuda::memory_order_release);
 }
 ::cuda::atomic_ref<int,::cuda::thread_scope_block>relay(*const_cast<int*>(&C().parent_ready));
 while(relay.load(::cuda::memory_order_acquire)<step)__nanosleep(32);
}
__device__ __forceinline__ bool Aborted(){return C().abort_local!=0;}
__device__ __forceinline__ void Publish(int step){
 // Conservative first prototype: every payload/psi/raw writer fences before
 // the CTA join. The leader then publishes device-scoped readiness.
 __threadfence();__syncthreads();
 if(!threadIdx.x)Release(C().q.ready+C().tile,int32_t(step));
}
__device__ __forceinline__ void Complete(){
 Release(C().q.ready+C().tile,int32_t(C().depth));
 ::cuda::atomic_ref<int32_t,::cuda::thread_scope_device>total(*C().q.completed);total.fetch_add(1,::cuda::memory_order_acq_rel);
 const int z=C().tile/(C().q.ny*C().q.nx),plane_size=C().q.ny*C().q.nx;
 ::cuda::atomic_ref<int32_t,::cuda::thread_scope_device>done(C().q.plane_done[z]);
 const int before=done.fetch_add(1,::cuda::memory_order_acq_rel);
 if(before<0||before>=plane_size){Abort(2048);return;}
 if(before+1==plane_size){
  for(int offset=2;offset<=3;++offset){const int future=z+offset;if(future>=C().q.nz)continue;
   for(int t=future*plane_size;t<(future+1)*plane_size;++t)TryEnqueue(t);
  }
 }
}
__device__ __forceinline__ void Pop(){
 C().tile=-1;C().abort_local=0;C().parent_ready=0;
 if(Acquire(C().q.abort))return;
 ::cuda::atomic_ref<int32_t,::cuda::thread_scope_device>head(*C().q.head);
 const int ticket=head.fetch_add(1,::cuda::memory_order_relaxed);if(ticket>=C().q.tile_count)return;
 const uint64_t start=Timer();
 while(!Acquire(C().q.entry_ready+ticket)){
  if(Acquire(C().q.abort))return;
  if(Timer()-start>C().q.timeout_ns){Abort(256);return;}__nanosleep(32);
 }
 C().tile=C().q.queue[ticket];if(C().tile<0||C().tile>=C().q.tile_count){Abort(2048);C().tile=-1;}
}
} // namespace plane_pipeline
