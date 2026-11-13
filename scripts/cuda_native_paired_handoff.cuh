// Included after phase geometry and native support/import helpers.
namespace closed_pipeline{
__device__ __forceinline__ volatile Cold& Peer(){
 extern __shared__ float frame[];return *reinterpret_cast<volatile Cold*>(reinterpret_cast<char*>(frame)+(1-Group())*GroupStride+ColdOffset);
}
__device__ __forceinline__ bool PeerActive(int n){return n>=Peer().first&&n<=Peer().last;}
__device__ __forceinline__ bool ProducerOverlap(int n){return Group()==0&&n>=C().first&&n<=C().last&&PeerActive(n);}
__device__ __forceinline__ bool KeepLastE(int n){return ProducerOverlap(n)&&n<Peer().last&&n<C().depth;}
// Every consuming thread acquires the CTA flag before loading shared payload.
// Thread0 alone owns the timer/abort path; no extra group join is required.
__device__ __forceinline__ bool WaitLocal(const volatile int*flag,int step){
 if(!LocalThread()){
  const uint64_t start=Timer();
  while(AcquireBlock(flag)<step){
   if(Acquire(C().q.abort)){ReleaseBlock(&C().abort_local,1);break;}
   if(Timer()-start>C().q.timeout_ns){Abort(256);break;}__nanosleep(32);
  }
 }
 while(AcquireBlock(flag)<step){if(AcquireBlock(&C().abort_local))return false;__nanosleep(32);}
 return !Aborted();
}
__device__ __forceinline__ bool BeforeHOverwrite(int n){return !ProducerOverlap(n-1)||WaitLocal(&C().e_ack,n-1);}
__device__ __forceinline__ bool BeforeEOverwrite(int n){return !ProducerOverlap(n)||WaitLocal(&C().h_ack,n);}
template<bool Electric>__device__ __forceinline__ bool BeforeGhostImport(int n){
 if(Group()==0||!PeerActive(n))return true;
 return WaitLocal(Electric?&Peer().e_ready:&Peer().h_ready,n);
}
__device__ __forceinline__ void HVisible(int n){
 if(!LocalThread()){ReleaseBlock(&C().h_ready,n);if(Group()==1&&PeerActive(n))ReleaseBlock(&Peer().h_ack,n);}
}
__device__ __forceinline__ void GhostERead(int n){if(!LocalThread()&&Group()==1&&PeerActive(n))ReleaseBlock(&Peer().e_ack,n);}
__device__ __forceinline__ void EVisible(int n){if(!LocalThread())ReleaseBlock(&C().e_ready,n);}
__device__ __forceinline__ bool ElideInternal(int n,int dz,int dy,int dx){
 return dz==BZ&&dy<BY&&dx<BX&&ProducerOverlap(n);
}
} // namespace closed_pipeline

template<int Depth,int T,bool Electric>
__device__ __forceinline__ void PairedScatterGhost(const Parameters&p,float*frame,const size_t*bases,int n,int oz,int oy,int ox){
 if(!closed_pipeline::BeforeGhostImport<Electric>(n))return;
 for(int i=closed_pipeline::LocalThread();i<PhaseGhostPoints;i+=T){
  const auto d=PhaseGhostLogical(i),s=PhaseGhostPhysicalSlot(i,n);const int slot=PhaseSlot(s.z,s.y,s.x);FieldPacket v{};
  if(closed_pipeline::Group()==1&&d.z==0&&d.y>0&&d.x>0){
   const int z=oz-n-1,y=oy-n-1+d.y,x=ox-n-1+d.x;const unsigned mask=NativeSupportMask(p,z,y,x);
   const float*peer=reinterpret_cast<const float*>(reinterpret_cast<const char*>(frame)-closed_pipeline::GroupStride);
   const int parent_z=(-n-1+BZ)&(CZ-1),source=PhaseSlot(parent_z,s.y,s.x);
   if constexpr(Electric){if(mask&8u)v.ex=peer[source];if(mask&16u)v.ey=peer[PAD+source];if(mask&32u)v.ez=peer[2*PAD+source];}
   else{if(mask&1u)v.hx=peer[source];if(mask&2u)v.hy=peer[PAD+source];if(mask&4u)v.hz=peer[2*PAD+source];}
  }else v=ImportPacket<Depth,false,true>(p,bases,n,oz,oy,ox,d.z,d.y,d.x);
  if constexpr(Electric){frame[slot]=v.ex;frame[PAD+slot]=v.ey;frame[2*PAD+slot]=v.ez;}
  else{frame[slot]=v.hx;frame[PAD+slot]=v.hy;frame[2*PAD+slot]=v.hz;}
 }
}
