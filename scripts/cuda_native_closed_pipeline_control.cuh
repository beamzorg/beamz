// Included inside moving_native after Parameters. All ready dependencies
// refer to tasks whose parents have started, or to fully completed old readers.
namespace closed_pipeline {
constexpr size_t ColdOffset=62144;
struct Cold {
 BeamzNativeClosedPipelineWorkspace q;
 int32_t*bad;
 int tile,depth,abort_local,parent_ready;
};
constexpr size_t SharedBytes=ColdOffset+sizeof(Cold);
static_assert(SharedBytes<64512,"Queue state must preserve native64KB carveout");
__device__ __forceinline__ volatile Cold& C(){extern __shared__ float frame[];return *reinterpret_cast<volatile Cold*>(reinterpret_cast<char*>(frame)+ColdOffset);}
__device__ __forceinline__ int Acquire(const int32_t*pointer){
 int value;asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(value):"l"(__cvta_generic_to_global(pointer)):"memory");return value;
}
__device__ __forceinline__ void Release(int32_t*pointer,int value){
 asm volatile("st.release.gpu.global.u32 [%0], %1;"::"l"(__cvta_generic_to_global(pointer)),"r"(value):"memory");
}
__device__ __forceinline__ int CasAcqRel(int32_t*pointer,int expected,int desired){
 int value;asm volatile("atom.acq_rel.gpu.global.cas.b32 %0, [%1], %2, %3;":"=r"(value):"l"(__cvta_generic_to_global(pointer)),"r"(expected),"r"(desired):"memory");return value;
}
__device__ __forceinline__ int AddAcqRel(int32_t*pointer,int increment){
 int value;asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;":"=r"(value):"l"(__cvta_generic_to_global(pointer)),"r"(increment):"memory");return value;
}
__device__ __forceinline__ int AddRelaxed(int32_t*pointer,int increment){
 int value;asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;":"=r"(value):"l"(__cvta_generic_to_global(pointer)),"r"(increment):"memory");return value;
}
__device__ __forceinline__ int AcquireBlock(const volatile int*pointer){
 int value;asm volatile("ld.acquire.cta.shared.u32 %0, [%1];":"=r"(value):"r"(unsigned(__cvta_generic_to_shared(const_cast<const int*>(pointer)))):"memory");return value;
}
__device__ __forceinline__ void ReleaseBlock(volatile int*pointer,int value){
 asm volatile("st.release.cta.shared.u32 [%0], %1;"::"r"(unsigned(__cvta_generic_to_shared(const_cast<int*>(pointer)))),"r"(value):"memory");
}
__device__ __forceinline__ uint64_t Timer(){uint64_t result;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(result));return result;}
__device__ __forceinline__ int Tile(){return C().tile;}
__device__ __forceinline__ int Owner(const Parameters&,int tile){return __ldg(C().q.owner_slots+tile);}
__device__ __forceinline__ void Abort(int bit){atomicOr(C().bad,bit);Release(C().q.abort,int32_t(1));C().abort_local=1;}
__device__ __forceinline__ bool Aborted(){return C().abort_local!=0;}

// A logical tile is inserted once and removed once. Nodes never re-enter the
// stack until the next whole-macro reset, so there is no recycled-node ABA.
__device__ __forceinline__ void Push(int tile){
 if(CasAcqRel(C().q.claimed+tile,0,1)!=0){Abort(2048);return;}
 int32_t*head=C().q.heads+C().q.cohort;int old=Acquire(head);
 for(;;){C().q.next[tile]=old;const int actual=CasAcqRel(head,old,tile);if(actual==old)break;old=actual;}
 AddRelaxed(C().q.enqueued+C().q.cohort,1);
}
__device__ __forceinline__ void Arrive(int child){
 if(__ldg(C().q.tile_cohort+child)!=C().q.cohort)return;
 const int previous=AddAcqRel(C().q.pending+child,-1);
 if(previous<1){Abort(2048);return;}if(previous==1)Push(child);
}
// Thread0 only, after owner-tag store and the original native CTA barrier.
__device__ __forceinline__ void Started(const Parameters&p){
 const int tile=C().tile,z=tile/(p.ny*p.nx),y=(tile/p.nx)%p.ny,x=tile%p.nx;
 if(z+1<p.nz)Arrive(tile+p.ny*p.nx);if(y+1<p.ny)Arrive(tile+p.nx);if(x+1<p.nx)Arrive(tile+1);
 AddRelaxed(C().q.started+C().q.cohort,1);
}
__device__ __forceinline__ void Skipped(int step){if(step>0&&!threadIdx.x)Release(C().q.ready+C().tile,int32_t(step));}
__device__ __forceinline__ void WaitParents(const Parameters&p,int step,bool startup=false){
 if(step<1)return;
 if(AcquireBlock(&C().parent_ready)>=step)return;
 if(!threadIdx.x){
  const int tile=C().tile,z=tile/(p.ny*p.nx),y=(tile/p.nx)%p.ny,x=tile%p.nx;const uint64_t start=Timer();int floor=0;
  for(;;){
   if(startup){
    // Skipped readiness is not transitive at late clipped initialization.
    // Equal high-side start times can leave a valid diagonal E corner behind
    // three axial parents that have only published skipped versions.
    floor=C().depth;
    for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
     if(z>=dz&&y>=dy&&x>=dx)floor=min(floor,Acquire(C().q.ready+tile-dz*p.ny*p.nx-dy*p.nx-dx));}
   }else{
    const int a=z?Acquire(C().q.ready+tile-p.ny*p.nx):C().depth;
    const int b=y?Acquire(C().q.ready+tile-p.nx):C().depth;
    const int c=x?Acquire(C().q.ready+tile-1):C().depth;
    floor=min(a,min(b,c));
   }
   if(floor>=step)break;
   if(Acquire(C().q.abort)){C().abort_local=1;floor=C().depth;break;}
   if(Timer()-start>C().q.timeout_ns){Abort(256);floor=C().depth;break;}__nanosleep(32);
  }
  ReleaseBlock(&C().parent_ready,floor);
 }
 while(AcquireBlock(&C().parent_ready)<step)__nanosleep(32);
}
__device__ __forceinline__ void Publish(int step){
 __threadfence();__syncthreads();if(!threadIdx.x)Release(C().q.ready+C().tile,int32_t(step));
}
__device__ __forceinline__ void Complete(){
 Release(C().q.ready+C().tile,int32_t(C().depth));
 const int begin=__ldg(C().q.finish_offsets+C().tile),end=__ldg(C().q.finish_offsets+C().tile+1);
 for(int i=begin;i<end;++i)Arrive(__ldg(C().q.finish_children+i));
 AddAcqRel(C().q.completed+C().q.cohort,1);
}
__device__ __forceinline__ void Pop(){
 C().tile=-1;C().abort_local=0;C().parent_ready=0;if(Acquire(C().q.abort))return;
 int32_t*head=C().q.heads+C().q.cohort;const uint64_t start=Timer();
 for(;;){
  const int old=Acquire(head);
  if(old>=0){
   if(old>=C().q.tile_count||__ldg(C().q.tile_cohort+old)!=C().q.cohort){Abort(2048);return;}
   const int32_t next=__ldcg(C().q.next+old);
   if(CasAcqRel(head,old,next)==old){C().tile=old;return;}
  }else{
   if(Acquire(C().q.completed+C().q.cohort)==C().q.cohort_tasks)return;
   if(Acquire(C().q.abort))return;__nanosleep(32);
  }
  if(Timer()-start>C().q.timeout_ns){Abort(256);return;}
 }
}
} // namespace closed_pipeline
