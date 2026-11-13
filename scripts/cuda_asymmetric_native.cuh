// Private complete moving-cache consumer. Included at update.cu end only in
// immutable research builds; no production dispatch/default changes.
#pragma once
#include "native_moving.h"
#include "compact_full_edge.cuh"
#include "asymmetric_native_cell_generated.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#ifndef BEAMZ_NATIVE_ASYMMETRIC_SWIZZLE
#define BEAMZ_NATIVE_ASYMMETRIC_SWIZZLE 0
#endif
namespace beamz::cuda::moving_native {
constexpr int BZ=5,BY=15,BX=31,CZ=6,CY=16,CX=32,Threads=512;
constexpr int PAD=CZ*CY*CX,SurfacePoints=BZ*BY*BX-(BZ-1)*(BY-1)*(BX-1);
constexpr int LowXPoints=BZ*BY;
constexpr size_t SharedBytes=4ULL*PAD*sizeof(float)+8*sizeof(size_t)+sizeof(int)+2ULL*LowXPoints*sizeof(float);
struct Source {
 const float* coefficients=nullptr;const float* waveforms=nullptr;
 const int32_t* starts=nullptr;const int32_t* current_step=nullptr;
 int count=0,nz=0,ny=0,nx=0,ntime=0;
};
struct Parameters {
 MstepCompactEdgeView phase[2];Source sources[6];
 beamz::cuda::deep::Observation observation;
 const int32_t* slots;const uint8_t* flags;float* trails;
 uint64_t* owner_tags;const uint32_t* epoch;int32_t* bad;
 int nz,ny,nx,frontier,macro_step;float h_source;
};
__device__ __forceinline__ int WrapZ(int z){const int r=z%CZ;return r<0?r+CZ:r;}
__device__ __forceinline__ int Slot(int z,int y,int x){
 z=WrapZ(z);y&=15;x&=31;
#if BEAMZ_NATIVE_ASYMMETRIC_SWIZZLE
 x^=y|((z&1)<<4);
#endif
 return (z*CY+y)*CX+x;
}
__device__ __forceinline__ float& At(float*shared,int field,int z,int y,int x){return shared[(2*(field/3)+(field%3)/2)*PAD+Slot(z,y,x)];}
__host__ __device__ __forceinline__ int Surface(int z,int y,int x){
 if(z==BZ-1)return y*BX+x;
 if(y==BY-1)return BY*BX+z*BX+x;
 if(x==BX-1)return BY*BX+(BZ-1)*BX+z*(BY-1)+y;
 return -1;
}
__device__ __forceinline__ bool Contains(const MstepCompactField&f,int z,int y,int x){return z>=0&&y>=0&&x>=0&&z<f.z&&y<f.y&&x<f.x;}
__host__ __device__ __forceinline__ const MstepCompactField& InitialField(const Parameters&p,int field){return p.phase[field/3].own[field%3];}
__host__ __device__ __forceinline__ const MstepCompactField& FinalField(const Parameters&p,int field){return p.phase[field/3].output[field%3];}
__device__ __forceinline__ float Initial(const Parameters&p,int field,int z,int y,int x){
 const auto&f=InitialField(p,field);return Contains(f,z,y,x)?f.data[z*f.plane+y*f.row+x]:0.f;
}
__device__ __forceinline__ void Observe(const Parameters&p,int field,int version,int z,int y,int x,float value){
 const auto&o=p.observation;if(!o.values)return;
 if(z<0||y<0||x<0||z>=o.mask_nz||y>=o.mask_ny||x>=o.mask_nx)return;
 const int cell=(z*o.mask_ny+y)*o.mask_nx+x;
 const uint32_t word=__ldg(o.publication_words+(cell>>2));
 if(!((word>>(8*(cell&3)))&(1u<<field)))return;
 const auto&f=FinalField(p,field);
 const int64_t key=o.field_base[field]+(static_cast<int64_t>(z)*f.y+y)*f.x+x;
 int lo=0,hi=o.key_count;while(lo<hi){const int middle=(lo+hi)/2;if(o.keys[middle]<key)lo=middle+1;else hi=middle;}
 if(lo<o.key_count&&o.keys[lo]==key)o.values[static_cast<size_t>(version-1)*o.key_count+lo]=value;
}
// Fast is admitted only for source/CPML/observation/native-wall-free OWNED
// trajectories. Startup field reads always retain real support checks.
template<int Layout,bool Fast,bool IsotropicH> struct Environment {
 const Parameters&p;uint8_t flags;
 __device__ __forceinline__ bool Valid(int field,int z,int y,int x)const{
  if constexpr(Fast)return true;return Contains(FinalField(p,field),z,y,x);
 }
 __device__ __forceinline__ int Packed(int slot,int z,int y,int x)const{
  const int family=slot/6,term=slot%6,axis=beamz::cuda::yee::CpmlAxis(term);
  const auto&f=p.phase[family].output[term/2];const int coord=axis==0?z:axis==1?y:x;
  const int size=axis==0?f.z:axis==1?f.y:f.x;
  return coord<12?coord:24+coord-size;
 }
 __device__ __forceinline__ bool LocalActive(int slot,int z,int y,int x)const{
  if constexpr(Fast)return false;
  const int family=slot/6,term=slot%6,axis=beamz::cuda::yee::CpmlAxis(term);
  if(!(flags&(2u<<axis)))return false;
  const auto&f=p.phase[family].output[term/2];const int coord=axis==0?z:axis==1?y:x;
  const int size=axis==0?f.z:axis==1?f.y:f.x;
  return coord<12||coord>=size-12;
 }
 __device__ __forceinline__ int PsiOffset(int slot,int z,int y,int x)const{
  const int family=slot/6,term=slot%6,axis=beamz::cuda::yee::CpmlAxis(term);
  const auto&psi=p.phase[family].psi[term];const int packed=Packed(slot,z,y,x);
  return (axis==0?packed:z)*psi.plane+(axis==1?packed:y)*psi.row+(axis==2?packed:x);
 }
 __device__ __forceinline__ float LoadLocal(int slot,int z,int y,int x)const{
  // Mutable within the same CTA/window: ordinary coherent loads, never __ldg.
  return p.phase[slot/6].psi[slot%6].input[PsiOffset(slot,z,y,x)];
 }
 __device__ __forceinline__ void StoreLocal(int slot,int z,int y,int x,float value)const{
  p.phase[slot/6].psi[slot%6].output[PsiOffset(slot,z,y,x)]=value;
 }
 __device__ __forceinline__ float Coefficient(int kind,int index,int z,int y,int x)const{
  if(kind>=2){const auto&psi=p.phase[index/6].psi[index%6];const int packed=Packed(index,z,y,x);return kind==2?__ldg(psi.a+packed):kind==3?__ldg(psi.b+packed):__ldg(psi.inv+packed);}
  if(index<3){if constexpr(IsotropicH)return kind==0?1.f:p.h_source;
   return kind==0?__ldg(p.phase[0].decay[index]):__ldg(p.phase[0].source[index]);}
  if(kind==0)return 1.f;
  const int c=index-3;const auto&v=p.phase[1];const auto&f=v.output[c];const int linear=(z*f.y+y)*f.x+x;
  if constexpr(Layout==2){const int h=__ldg(reinterpret_cast<const int32_t*>(v.codes[c])+(linear>>5));return __ldg(v.tables[c]+(h<0?~h:h+(linear&31)));}
  else return beamz::cuda::yee::PackedMaterialSource(v.tables[c],v.table_sizes[c],v.codes[c],linear);
 }
 __device__ __forceinline__ float Difference(int phase,int source,int axis,int z,int y,int x,float center,float adjacent)const{
  if constexpr(!Fast){const auto&f=InitialField(p,source);const int coordinate=axis==0?z:axis==1?y:x;const int size=axis==0?f.z:axis==1?f.y:f.x;
   if(!phase){if(coordinate+1>=size)return 0.f;}
   else if(coordinate==0||coordinate==size)return 0.f;
  }
  return beamz::cuda::yee::ScaleYeeDifference(phase?__fsub_rn(center,adjacent):__fsub_rn(adjacent,center),p.phase[phase].inv);
 }
 __device__ __forceinline__ float Finish(int field,int version,int z,int y,int x,float value)const{
  if constexpr(!Fast){
   if(flags&beamz::cuda::moving::kMayHaveSources){const auto&s=p.sources[field];
    for(int source=0;source<s.count;++source){const int sz=z-s.starts[3*source],sy=y-s.starts[3*source+1],sx=x-s.starts[3*source+2];
     if(sz<0||sy<0||sx<0||sz>=s.nz||sy>=s.ny||sx>=s.nx)continue;
     int time=*s.current_step+p.macro_step+version-1;time=max(0,min(time,s.ntime-1));
     const int offset=((source*s.nz+sz)*s.ny+sy)*s.nx+sx;
     // Initial admission is coincident multi-source: match io.cu split rounds.
     value=__fadd_rn(value,__fmul_rn(__ldg(s.coefficients+offset),__ldg(s.waveforms+source*s.ntime+time)));
    }
   }
   if(flags&beamz::cuda::moving::kMayObserve)Observe(p,field,version,z,y,x,value);
  }
  return value;
 }
};
template<int Depth>
__device__ __forceinline__ float Import(const Parameters&p,const size_t*bases,int field,int version,
 int oz,int oy,int ox,int dz,int dy,int dx){
 const int z=oz-1+dz-version,y=oy-1+dy-version,x=ox-1+dx-version;
 if(!Contains(InitialField(p,field),z,y,x))return 0.f;
 const int neighbor=(dz==0)|((dy==0)<<1)|((dx==0)<<2);
 const int surface=Surface(dz?dz-1:BZ-1,dy?dy-1:BY-1,dx?dx-1:BX-1);
 const size_t base=bases[neighbor];
 if(!neighbor||surface<0||base==static_cast<size_t>(-1)){atomicOr(p.bad,2);return __int_as_float(0x7fffffff);}
 return p.trails[base+((version-1)*6+field)*SurfacePoints+surface];
}
template<int Depth>
__device__ __forceinline__ void StageLowX(const Parameters&p,float*shared,float*stage,const size_t*bases,int version,int oz,int oy,int ox){
 const int head=WrapZ(-version-1),yh=(-version-1)&15,xh=(-version-1)&31;
 for(int i=threadIdx.x;i<LowXPoints;i+=Threads){const int dz=1+i/BY,dy=1+i%BY;
  const int z=oz-1+dz-version,y=oy-1+dy-version,x=ox-1-version;
  const int surface=Surface(dz-1,dy-1,BX-1);const size_t base=bases[4];int j=head+dz;if(j>=CZ)j-=CZ;const int sy=(yh+dy)&15;
  #pragma unroll
  for(int field=0;field<6;++field){float value=0.f;
   if(Contains(InitialField(p,field),z,y,x)){if(base==static_cast<size_t>(-1)){atomicOr(p.bad,2);value=__int_as_float(0x7fffffff);}else value=p.trails[base+((version-1)*6+field)*SurfacePoints+surface];}
   if(field==1)stage[i]=value;else if(field==4)stage[LowXPoints+i]=value;else At(shared,field,j,sy,xh)=value;
  }
 }
}
__device__ __forceinline__ void RetireLowX(const float*stage,float(&hy)[CZ],float(&ey)[CZ],int version,int sy,int sx){
 const int head=WrapZ(-version-1),yh=(-version-1)&15,xh=(-version-1)&31;
 if(sx==xh&&sy!=yh){const int dy=(sy-yh)&15;
  #pragma unroll
  for(int j=0;j<CZ;++j)if(j!=head){int dz=j-head;if(dz<0)dz+=CZ;const int index=(dz-1)*BY+dy-1;hy[j]=stage[index];ey[j]=stage[LowXPoints+index];}
 }
}
__global__ void BeginEpoch(uint32_t*epoch,int32_t*bad){if(!threadIdx.x){if(*epoch==0xffffffffu){atomicOr(bad,4);return;}++*epoch;}}

template<int Depth,int Layout,bool Fast,bool IsotropicH>
__global__ __launch_bounds__(Threads,(Fast?2:1)) void Asymmetric(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 extern __shared__ float shared[];size_t*bases=reinterpret_cast<size_t*>(shared+4*PAD);
 volatile int*phase_step=reinterpret_cast<volatile int*>(bases+8);
 float*low_stage=shared+4*PAD+8*sizeof(size_t)/sizeof(float)+1;
 const int tile=tiles[blockIdx.x],bz=tile/(p.nx*p.ny),by=(tile/p.nx)%p.ny,bx=tile%p.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX,rank=bz+by+bx;
 const uint64_t epoch=static_cast<uint64_t>(*p.epoch)<<32;
 if(!threadIdx.x){const int slot=(rank&3)*p.frontier+p.slots[tile];p.owner_tags[slot]=epoch|static_cast<uint32_t>(tile);}
 __syncthreads();
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;size_t base=static_cast<size_t>(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*p.nx*p.ny-dy*p.nx-dx;const int slot=((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer];
   if(bits&&p.owner_tags[slot]!=(epoch|static_cast<uint32_t>(producer)))atomicOr(p.bad,1);else base=static_cast<size_t>(slot)*Depth*6*SurfacePoints;
  }bases[bits]=base;
 }
 const int sx=threadIdx.x&31,sy=threadIdx.x/32;float hy[CZ],ey[CZ];
 #pragma unroll
 for(int j=0;j<CZ;++j){const int z=oz+(j==CZ-1?-1:j),y=oy+(sy==15?-1:sy),x=ox+(sx==31?-1:sx);
  // Mandatory real support clipping, even for Fast owned trajectories.
  hy[j]=Initial(p,1,z,y,x);ey[j]=Initial(p,4,z,y,x);
  #pragma unroll
  for(int family=0;family<2;++family){At(shared,family*3,j,sy,sx)=Initial(p,family*3,z,y,x);At(shared,family*3+2,j,sy,sx)=Initial(p,family*3+2,z,y,x);}
 }
 if(!threadIdx.x)*phase_step=1;__syncthreads();
 const uint8_t flags=Fast?0:p.flags[tile];Environment<Layout,Fast,IsotropicH>env{p,flags};
 #pragma unroll 1
 for(int n=1;n<=Depth;++n){
  {const int step=*phase_step,head=WrapZ(-step-1),yh=(-step-1)&15,xh=(-step-1)&31,low=-step-1;
   const int y=oy+low+((sy-yh)&15),x=ox+low+((sx-xh)&31);
   #pragma unroll
   for(int j=0;j<CZ;++j){const float next_ey=__shfl_sync(0xffffffffu,ey[j],(sx+1)&31);
    if(j!=head&&sy!=yh&&sx!=xh){const int z=oz+low+WrapZ(j-head);
     const auto read=[&] __device__(int field,int,int dz,int dy,int dx){if(field<3)return field==1?hy[j]:At(shared,field,j,sy,sx);if(field==4){if(dz)return ey[(j+1)%CZ];if(dx)return next_ey;return ey[j];}return At(shared,field,j+dz,sy+dy,sx+dx);};
     #pragma unroll
     for(int c=0;c<3;++c){float value=0.f;if(env.Valid(c,z,y,x)){
      if(c==0)value=beamz::cuda::trace_native::AdvanceNativeCell<0,0,false>(env,read,n,z,y,x);
      else if(c==1)value=beamz::cuda::trace_native::AdvanceNativeCell<0,1,false>(env,read,n,z,y,x);
      else value=beamz::cuda::trace_native::AdvanceNativeCell<0,2,false>(env,read,n,z,y,x);
     }if(c==1)hy[j]=value;else At(shared,c,j,sy,sx)=value;}
    }
   }
  }
  __syncthreads(); // B1: oldE reads / prior publications retire.
  {const int head=WrapZ(-n-1),yh=(-n-1)&15,xh=(-n-1)&31;
   const int dy=(sy-yh)&15,dx=(sx-xh)&31;
   #pragma unroll
   for(int j=0;j<CZ;++j)if(j==head||sy==yh){const int dz=WrapZ(j-head);
    #pragma unroll
    for(int c=0;c<3;++c){const float h=Import<Depth>(p,bases,c,n,oz,oy,ox,dz,dy,dx),e=Import<Depth>(p,bases,c+3,n,oz,oy,ox,dz,dy,dx);if(c==1){hy[j]=h;ey[j]=e;}else{At(shared,c,j,sy,sx)=h;At(shared,c+3,j,sy,sx)=e;}}
   }
  }
  StageLowX<Depth>(p,shared,low_stage,bases,n,oz,oy,ox);
  if(!threadIdx.x)*phase_step=n+1;__syncthreads(); // B2 publishes allcurrentH.
  RetireLowX(low_stage,hy,ey,n,sy,sx);
  {const int step=*phase_step-1,head=WrapZ(-step-1),yh=(-step-1)&15,xh=(-step-1)&31,low=-step-1;
   const int y=oy+low+((sy-yh)&15),x=ox+low+((sx-xh)&31);
   #pragma unroll
   for(int j=0;j<CZ;++j){const float previous_hy=__shfl_sync(0xffffffffu,hy[j],(sx-1)&31);
    if(j!=head&&sy!=yh&&sx!=xh){const int z=oz+low+WrapZ(j-head);
     const auto read=[&] __device__(int field,int,int dz,int dy,int dx){if(field>=3)return field==4?ey[j]:At(shared,field,j,sy,sx);if(field==1){if(dz)return hy[(j+CZ-1)%CZ];if(dx)return previous_hy;return hy[j];}return At(shared,field,j+dz,sy+dy,sx+dx);};
     #pragma unroll
     for(int c=0;c<3;++c){float value=0.f;if(env.Valid(c+3,z,y,x)){
      if(c==0)value=beamz::cuda::trace_native::AdvanceNativeCell<1,0,false>(env,read,n,z,y,x);
      else if(c==1)value=beamz::cuda::trace_native::AdvanceNativeCell<1,1,false>(env,read,n,z,y,x);
      else value=beamz::cuda::trace_native::AdvanceNativeCell<1,2,false>(env,read,n,z,y,x);
     }if(c==1)ey[j]=value;else At(shared,c+3,j,sy,sx)=value;}
    }
   }
  }
  __syncthreads(); // B3: all current E values published before output readers.
  {const int head=WrapZ(-n-1),yh=(-n-1)&15,xh=(-n-1)&31,low=-n-1;
   const int dy=(sy-yh)&15,dx=(sx-xh)&31,y=oy+low+dy,x=ox+low+dx;
   #pragma unroll
   for(int j=0;j<CZ;++j)if(j!=head&&sy!=yh&&sx!=xh){const int dz=WrapZ(j-head),z=oz+low+dz,surface=Surface(dz-1,dy-1,dx-1);
    // Actual shared loads stay restricted to the old high-face union or the
    // terminal version. Next H cannot clobber a pending high-face reader.
    if(surface>=0||n==Depth){
     #pragma unroll
     for(int field=0;field<6;++field){const float value=field==1?hy[j]:field==4?ey[j]:At(shared,field,j,sy,sx);
      if(surface>=0)p.trails[bases[0]+((n-1)*6+field)*SurfacePoints+surface]=value;
      if(n==Depth){const auto&f=FinalField(p,field);if(Contains(f,z,y,x))f.data[z*f.plane+y*f.row+x]=value;}
     }
    }
   }
  }
 }
}

inline bool MakeSource(const BeamzSourceGroupLaunch&g,Source*out){
 *out={};if(g.coefficients.dims[0]==0)return true;
 if(g.coefficients.rank!=4||g.coefficients.element_type!=kBeamzF32||
    g.waveforms.rank!=2||g.waveforms.element_type!=kBeamzF32||
    g.starts.element_type!=kBeamzS32||g.current_step.element_type!=kBeamzS32||
    g.coefficients.dims[0]<=1||!g.coincident||g.waveforms.dims[0]!=g.coefficients.dims[0])return false;
 for(int a=0;a<4;++a)if(g.coefficients.dims[a]<1||g.coefficients.dims[a]>INT32_MAX)return false;
 if(g.waveforms.dims[1]<1||g.waveforms.dims[1]>INT32_MAX)return false;
 out->coefficients=static_cast<const float*>(g.coefficients.data);out->waveforms=static_cast<const float*>(g.waveforms.data);
 out->starts=static_cast<const int32_t*>(g.starts.data);out->current_step=static_cast<const int32_t*>(g.current_step.data);
 out->count=g.coefficients.dims[0];out->nz=g.coefficients.dims[1];out->ny=g.coefficients.dims[2];out->nx=g.coefficients.dims[3];out->ntime=g.waveforms.dims[1];return true;
}
inline bool Prepare(const BeamzProgramLaunch&program,const BeamzBuffer*final,
 const BeamzNativeMovingWorkspace&w,int macro,Parameters*out,int*layout){
 using namespace beamz::cuda::moving;
 if(!final||!out||!layout||macro<0||!program.source_groups||program.source_group_count!=9||
    program.h_ab.tensor||program.e_ab.tensor||w.epochs!=4||w.owned[0]!=BZ||w.owned[1]!=BY||w.owned[2]!=BX||
    (w.depth!=4&&w.depth!=16)||w.max_frontier<1||w.rank_count<1||w.tile_count<1||
    !w.rank_offsets_host||!w.bulk_counts_host||!w.tile_ids_device||!w.slots_device||!w.tile_flags_device||!w.trails||!w.owner_tags||!w.epoch||!w.bad)return false;
 for(int c=0;c<3;++c)if(program.source_groups[c].coefficients.dims[0])return false;
 auto h=program.h_ab,e=program.e_ab;
 for(int c=0;c<3;++c){h.outputs[c]=final[c];e.outputs[c]=final[3+c];}
 *out={};int h_layout=0;
 if(!MakeMstepCompactEdge(h,&out->phase[0],&h_layout)||!MakeMstepCompactEdge(e,&out->phase[1],layout)||(*layout!=1&&*layout!=2))return false;
 for(int term=0;term<6;++term){if(h.inputs[31+term].data!=h.outputs[3+term].data||e.inputs[31+term].data!=e.outputs[3+term].data)return false;}
 int minimum[3]={INT32_MAX,INT32_MAX,INT32_MAX},maximum[3]={0,0,0};
 for(int field=0;field<6;++field){const auto&f=FinalField(*out,field);const int dim[3]={f.z,f.y,f.x};for(int a=0;a<3;++a){minimum[a]=std::min(minimum[a],dim[a]);maximum[a]=std::max(maximum[a],dim[a]);}
  for(int input=0;input<6;++input)if(f.data==InitialField(*out,input).data)return false; // alias mode is not admitted yet.
 }
 for(int field=0;field<6;++field){const auto&f=FinalField(*out,field);const int dz=field==2||field==3||field==4,dy=field==1||field==3||field==5,dx=field==0||field==4||field==5;
  if(f.z!=minimum[0]+dz||f.y!=minimum[1]+dy||f.x!=minimum[2]+dx)return false;
 }
 const int sizes[3]={BZ,BY,BX};int64_t tiles=1;
 for(int a=0;a<3;++a){if(w.counts[a]!=(int64_t(maximum[a])+w.depth+sizes[a]-1)/sizes[a])return false;tiles*=w.counts[a];}
 if(tiles!=w.tile_count||w.rank_count!=w.counts[0]+w.counts[1]+w.counts[2]-2||w.rank_offsets_host[0]!=0||w.rank_offsets_host[w.rank_count]!=w.tile_count)return false;
 for(int rank=0;rank<w.rank_count;++rank){const int count=w.rank_offsets_host[rank+1]-w.rank_offsets_host[rank];if(count<0||count>w.max_frontier||w.bulk_counts_host[rank]<0||w.bulk_counts_host[rank]>count)return false;}
 const size_t owners=4ULL*w.max_frontier;
 if(owners>std::numeric_limits<size_t>::max()/w.depth/6/SurfacePoints||w.trail_floats<owners*w.depth*6*SurfacePoints)return false;
 for(int field=0;field<6;++field)if(!MakeSource(program.source_groups[3+field],&out->sources[field]))return false;
 if(w.observation.key_count<0||(w.observation.key_count&&(!w.observation.keys||!w.observation.values||!w.observation.publication_words)))return false;
 out->observation=w.observation;out->slots=w.slots_device;out->flags=w.tile_flags_device;out->trails=w.trails;out->owner_tags=w.owner_tags;out->epoch=w.epoch;out->bad=w.bad;
 out->nz=w.counts[0];out->ny=w.counts[1];out->nx=w.counts[2];out->frontier=w.max_frontier;out->macro_step=macro;out->h_source=w.h_source[0];return true;
}
template<int Depth,int Layout,bool Fast,bool IsotropicH>
inline cudaError_t Launch(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count){
 if(!count)return cudaSuccess;
 const auto kernel=Asymmetric<Depth,Layout,Fast,IsotropicH>;
 auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
 Asymmetric<Depth,Layout,Fast,IsotropicH><<<count,Threads,SharedBytes,stream>>>(p,ids);return cudaGetLastError();
}
template<int Depth,int Layout,bool IsotropicH>
inline cudaError_t LaunchRanks(cudaStream_t stream,const Parameters&p,const BeamzNativeMovingWorkspace&w){
 for(int rank=0;rank<w.rank_count;++rank){const int start=w.rank_offsets_host[rank],count=w.rank_offsets_host[rank+1]-start,bulk=w.bulk_counts_host[rank];
  auto status=Launch<Depth,Layout,true,IsotropicH>(stream,p,w.tile_ids_device+start,bulk);if(status!=cudaSuccess)return status;
  status=Launch<Depth,Layout,false,IsotropicH>(stream,p,w.tile_ids_device+start+bulk,count-bulk);if(status!=cudaSuccess)return status;
 }
 return cudaSuccess;
}
template<int Depth,int Layout>
inline cudaError_t DispatchClass(cudaStream_t stream,const Parameters&p,const BeamzNativeMovingWorkspace&w){
 if(w.data_flags&beamz::cuda::moving::kLosslessIsotropicH)return LaunchRanks<Depth,Layout,true>(stream,p,w);
 return LaunchRanks<Depth,Layout,false>(stream,p,w);
}
} // namespace beamz::cuda::moving_native

int BeamzEnqueueNativeAsymmetric(void*opaque_stream,const BeamzProgramLaunch&program,
 const BeamzBuffer*final_fields,int macro_step,const BeamzNativeMovingWorkspace&w){
 using namespace beamz::cuda::moving_native;
 const auto validation=BeamzValidateNativeProgram(program);if(validation)return validation;
 Parameters p;int layout=0;if(!Prepare(program,final_fields,w,macro_step,&p,&layout))return cudaErrorInvalidValue;
 const auto stream=static_cast<cudaStream_t>(opaque_stream);
 BeginEpoch<<<1,1,0,stream>>>(w.epoch,w.bad);auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
 if(w.depth==4){if(layout==2)return DispatchClass<4,2>(stream,p,w);return DispatchClass<4,1>(stream,p,w);}
 if(w.depth==16){if(layout==2)return DispatchClass<16,2>(stream,p,w);return DispatchClass<16,1>(stream,p,w);}
 return cudaErrorInvalidValue;
}
