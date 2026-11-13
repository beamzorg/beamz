// Research-only destructive cache retirement. Every nominal packet reader ACKs
// exactly once, including clipped/empty windows. Counter publication and slot
// reuse are ordered through the closed scheduler's STARTED/COMPLETE protocol.
namespace closed_pipeline {
constexpr int PacketReady=16;
__device__ __forceinline__ int OrAcqRel(int32_t*pointer,int bits){
 int value;asm volatile("atom.acq_rel.gpu.global.or.b32 %0, [%1], %2;":"=r"(value):"l"(__cvta_generic_to_global(pointer)),"r"(bits):"memory");return value;
}
__device__ __forceinline__ void DiscardLine(uint64_t address){
 asm volatile("discard.global.L2 [%0], 128;"::"l"(address):"memory");
}
// Only lines wholly inside a private version row are eligible. Edge lines may
// contain another version's data and therefore must retain their contents.
__device__ __forceinline__ void DiscardPacketWarp(const Parameters&p,int owner,int step){
 const uint64_t begin=__cvta_generic_to_global(p.trails+((size_t(owner)*C().depth+step-1)*6*SurfacePoints));
 const uint64_t end=begin+6*SurfacePoints*sizeof(float),first=(begin+127)&~uint64_t(127),last=end&~uint64_t(127);
 for(uint64_t line=first+(threadIdx.x&31)*128;line<last;line+=32*128)DiscardLine(line);
}
__device__ __forceinline__ void InitializePackets(const Parameters&p){
 if(threadIdx.x<C().depth-1){
  const int tile=Tile(),z=tile/(p.ny*p.nx),y=(tile/p.nx)%p.ny,x=tile%p.nx;
  const int readers=(1+int(z+1<p.nz))*(1+int(y+1<p.ny))*(1+int(x+1<p.nx))-1;
  C().q.packet_refs[size_t(Owner(p,tile))*C().depth+threadIdx.x]=readers;
 }
}
// The tracer assigns one warp to each of the seven incoming packet edges.
// Independent per-edge ACK floors avoid a CTA barrier or a serial seven-parent
// loop while field registers are live. Each edge's RMW acquire chain joins
// previous readers before a warp barrier transfers ordering to discard lanes.
__device__ __forceinline__ void AckPacketsUpTo(const Parameters&p,int step){
 const int edge=threadIdx.x/32;if(edge>=7)return;
 const int bits=edge+1,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
 const int last=min(step,C().depth-1),tile=Tile(),z=tile/(p.ny*p.nx),y=(tile/p.nx)%p.ny,x=tile%p.nx;
 if(z>=dz&&y>=dy&&x>=dx){
  const int owner=Owner(p,tile-dz*p.ny*p.nx-dy*p.nx-dx);
  for(int n=C().acked_step[edge]+1;n<=last;++n){
   int old=0;
   if(!(threadIdx.x&31)){
    old=AddAcqRel(C().q.packet_refs+size_t(owner)*C().depth+n-1,-1);
    if(!(old&15)||(old&~31))Abort(4096);
   }
   old=__shfl_sync(0xffffffffu,old,0);__syncwarp();
   if(old==(PacketReady|1))DiscardPacketWarp(p,owner,n);
   __syncwarp();
  }
 }
 if(!(threadIdx.x&31))C().acked_step[edge]=max(C().acked_step[edge],last);
 __syncwarp();
}
__device__ __forceinline__ void PublishPacket(const Parameters&p,int step){
 __threadfence();__syncthreads();
 if(threadIdx.x<32){
  const int owner=Owner(p,Tile());int old=-1;
  if(step<C().depth&&!(threadIdx.x&31)){
   old=OrAcqRel(C().q.packet_refs+size_t(owner)*C().depth+step-1,PacketReady);
   if((old&~15)||(old&15)>7)Abort(4096);
  }
  old=__shfl_sync(0xffffffffu,old,0);__syncwarp();
  if(old==0)DiscardPacketWarp(p,owner,step);
  __syncwarp();
  if(!threadIdx.x)Release(C().q.ready+C().tile,int32_t(step));
 }
}
} // namespace closed_pipeline
