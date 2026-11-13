// A causal last-use proof replaces seven reader counters. For an interior
// producer q, its distal reader q+(1,1,1) waits for three axial parents at n.
// Their actual n publications transitively retire the other six readers'
// imports. At the distal reader's import join, every global read of row n is
// therefore complete. The active-window guard excludes skipped publication.
namespace closed_pipeline {
__device__ __forceinline__ void RetireDistalPacket(const Parameters&p,int step){
 if(threadIdx.x>=32||step>=C().depth)return;
 const int tile=Tile(),z=tile/(p.ny*p.nx),y=(tile/p.nx)%p.ny,x=tile%p.nx;
 if(!z||!y||!x||step>min(z*BZ-1,min(y*BY-1,x*BX-1)))return;
 const int producer=tile-p.ny*p.nx-p.nx-1,owner=Owner(p,producer);
 const uint64_t begin=__cvta_generic_to_global(p.trails+((size_t(owner)*C().depth+step-1)*6*SurfacePoints));
 const uint64_t end=begin+6*SurfacePoints*sizeof(float),first=(begin+127)&~uint64_t(127),last=end&~uint64_t(127);
 __syncwarp();
 for(uint64_t line=first+(threadIdx.x&31)*128;line<last;line+=32*128)
  asm volatile("discard.global.L2 [%0], 128;"::"l"(line):"memory");
 __syncwarp();
}
} // namespace closed_pipeline
