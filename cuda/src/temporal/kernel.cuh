// Shared bounded temporal CUDA engine. Shapes and source topology are captured
// descriptor facts; no benchmark dimensions, source masks or term counts select code.
#pragma once
#ifndef BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
#define BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES 0
#endif
#ifndef BEAMZ_CUDA_NATIVE_MOVING_POLICIES
#define BEAMZ_CUDA_NATIVE_MOVING_POLICIES 0
#endif
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
constexpr int BZ=7,BY=15,BX=31,CZ=8,CY=16,CX=32,Threads=512;
constexpr int PAD=CZ*CY*CX,SurfacePoints=BZ*BY*BX-(BZ-1)*(BY-1)*(BX-1);
constexpr int LowXPoints=BZ*BY;
constexpr size_t SharedBytes=3ULL*PAD*sizeof(float)+8*sizeof(size_t)+3*sizeof(int)+6ULL*LowXPoints*sizeof(float);
constexpr size_t ProfileCachedSharedBytes=SharedBytes+864ULL*sizeof(float);
static_assert(SharedBytes==51748&&ProfileCachedSharedBytes==55204,"Exact appended profile layout");
struct Source {
 const float* coefficients=nullptr;const float* waveforms=nullptr;
 const int32_t* starts=nullptr;const int32_t* current_step=nullptr;
 int count=0,nz=0,ny=0,nx=0,ntime=0;
};
struct Parameters {
 MstepCompactEdgeView phase[2];Source sources[6];
 beamz::cuda::deep::Observation observation;
 const int32_t* observation_tiles;const int32_t* observation_leaf;
 int observation_ny,observation_nx;
 const int32_t* slots;const uint8_t* flags;float* trails;
 uint64_t* owner_tags;const uint32_t* epoch;int32_t* bad;
 int nz,ny,nx,frontier,macro_step,macro_level_stride;int physical_n[3];float h_source;float common_e[3];
};
__device__ __forceinline__ int PhysicalTileID(const Parameters&p,int encoded){return encoded%(p.nz*p.ny*p.nx);}
template<int Depth>__device__ __forceinline__ int MacroStepOffset(const Parameters&p,int encoded){return (encoded/(p.nz*p.ny*p.nx))*Depth;}

__device__ __forceinline__ int WrapZ(int z){return z&(CZ-1);}
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
__device__ __forceinline__ const MstepCompactField& MacroFinalField(const Parameters&p,int field,int bank){return (bank&1)?InitialField(p,field):FinalField(p,field);}
__device__ __forceinline__ float Initial(const Parameters&p,int field,int z,int y,int x,int bank=0){
 const auto&f=(bank&1)?FinalField(p,field):InitialField(p,field);return Contains(f,z,y,x)?f.data[z*f.plane+y*f.row+x]:0.f;
}
__device__ __forceinline__ void Observe(const Parameters&p,int field,int version,int z,int y,int x,float value){
 const auto&o=p.observation;if(!o.values)return;
 if(z<0||y<0||x<0||z>=o.mask_nz||y>=o.mask_ny||x>=o.mask_nx)return;
#if BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION
 const int tile=(z*p.observation_ny+(y>>3))*p.observation_nx+(x>>4);
 const int encoded=__ldg(p.observation_tiles+tile);
 if(encoded<0||!(encoded&(1u<<field)))return;
 const int slot=encoded>>6;
 const size_t leaf=size_t(slot)*128*6+(size_t(y&7)*16+(x&15))*6+field;
 const int raw=__ldg(p.observation_leaf+leaf);
 if(raw>=0)o.values[size_t(version-1)*o.key_count+raw]=value;
#else
 const int cell=(z*o.mask_ny+y)*o.mask_nx+x;
 const uint32_t word=__ldg(o.publication_words+(cell>>2));
 if(!((word>>(8*(cell&3)))&(1u<<field)))return;
 const auto&f=FinalField(p,field);
 const int64_t key=o.field_base[field]+(static_cast<int64_t>(z)*f.y+y)*f.x+x;
 int lo=0,hi=o.key_count;while(lo<hi){const int middle=(lo+hi)/2;if(o.keys[middle]<key)lo=middle+1;else hi=middle;}
 if(lo<o.key_count&&o.keys[lo]==key)o.values[static_cast<size_t>(version-1)*o.key_count+lo]=value;
#endif
}

// Each slot is a distinct family/term; everykind retains its own24FP32 bits.
__host__ __device__ __forceinline__ int IndependentProfileCacheIndex(int slot,int kind,int packed){
 return (slot*3+(kind-2))*24+packed;
}
__device__ __forceinline__ float ReadIndependentProfileCache(const float*profiles,int slot,int kind,int packed){
 return profiles[IndependentProfileCacheIndex(slot,kind,packed)];
}
__device__ __forceinline__ void StageIndependentCpmlProfiles(const Parameters&p,float*profiles){
 if(threadIdx.x<512)for(int i=threadIdx.x;i<864;i+=512){
  const int profile=i/24,packed=i%24,slot=profile/3,kind=profile%3;
  const auto&psi=p.phase[slot/6].psi[slot%6];
  const float*input=kind==0?psi.a:kind==1?psi.b:psi.inv;
  profiles[i]=__ldg(input+packed);
 }
}

// Fast is admitted only for source/CPML/observation/native-wall-free OWNED
// trajectories. Startup field reads always retain real support checks.
template<int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5),bool UniformE=false,int ActiveAxes=-1,bool CachedProfiles=false> struct Environment {
 static constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);
 static constexpr bool HasCpml=Policy==1||Policy==5;
 static constexpr bool HasSources=Policy==3||Policy==4||Policy==5;
 static constexpr bool HasObserve=Policy==2||Policy==4||Policy==5;
 const Parameters&p;uint8_t flags;const float*profiles=nullptr;int step_offset=0,raw_step_offset=0;
 __device__ __forceinline__ bool Valid(int field,int z,int y,int x)const{
  if constexpr(SupportSafe)return true;
  const int nz=p.physical_n[0],ny=p.physical_n[1],nx=p.physical_n[2];
  if(z<0||y<0||x<0||z>nz||y>ny||x>nx)return false;
  const bool iz=z<nz,iy=y<ny,ix=x<nx;
  return field==0?(iz&&iy):field==1?(iz&&ix):field==2?(iy&&ix):field==3?ix:field==4?iy:iz;
 }
 __device__ __forceinline__ int Packed(int slot,int z,int y,int x)const{
  const int family=slot/6,term=slot%6,axis=beamz::cuda::yee::CpmlAxis(term);
  const int coord=axis==0?z:axis==1?y:x;
  const int size=p.physical_n[axis]+family;
  return coord<12?coord:24+coord-size;
 }
 __device__ __forceinline__ bool LocalActive(int slot,int z,int y,int x)const{
  if constexpr(!HasCpml)return false;
  const int family=slot/6,term=slot%6,axis=beamz::cuda::yee::CpmlAxis(term);
  if constexpr(ActiveAxes>=0){if(!(ActiveAxes&(1u<<axis)))return false;}
  else if(!(flags&(2u<<axis)))return false;
  const int coord=axis==0?z:axis==1?y:x;
  const int size=p.physical_n[axis]+family;
  return coord<12||coord>=size-12;
 }
 __device__ __forceinline__ int PsiOffset(int slot,int z,int y,int x)const{
  const int family=slot/6,term=slot%6,axis=beamz::cuda::yee::CpmlAxis(term),c=term/2;
  const int packed=Packed(slot,z,y,x);
  // Prepare proves dense canonical CPML slabs, so strides follow geometry for every shape.
  const int ny=axis==1?24:p.physical_n[1]+(family?(c!=1):(c==1));
  const int nx=axis==2?24:p.physical_n[2]+(family?(c!=0):(c==0));
  return ((axis==0?packed:z)*ny+(axis==1?packed:y))*nx+(axis==2?packed:x);
 }
 __device__ __forceinline__ float LoadLocal(int slot,int z,int y,int x)const{
  // Mutable within the same CTA/window: ordinary coherent loads, never __ldg.
  return p.phase[slot/6].psi[slot%6].input[PsiOffset(slot,z,y,x)];
 }
 __device__ __forceinline__ void StoreLocal(int slot,int z,int y,int x,float value)const{
  p.phase[slot/6].psi[slot%6].output[PsiOffset(slot,z,y,x)]=value;
 }
 __device__ __forceinline__ float Coefficient(int kind,int index,int z,int y,int x)const{
  if(kind>=2){const int packed=Packed(index,z,y,x);
   if constexpr(CachedProfiles)return ReadIndependentProfileCache(profiles,index,kind,packed);
   else{const auto&psi=p.phase[index/6].psi[index%6];return kind==2?__ldg(psi.a+packed):kind==3?__ldg(psi.b+packed):__ldg(psi.inv+packed);}
  }
  if(index<3){if constexpr(IsotropicH)return kind==0?1.f:p.h_source;
   return kind==0?__ldg(p.phase[0].decay[index]):__ldg(p.phase[0].source[index]);}
  if(kind==0)return 1.f;
  if constexpr(UniformE)return p.common_e[index-3];
  const int c=index-3;const auto&v=p.phase[1];const int linear=(z*(p.physical_n[1]+(c!=1))+y)*(p.physical_n[2]+(c!=0))+x;
  if constexpr(Layout==2){const int h=__ldg(reinterpret_cast<const int32_t*>(v.codes[c])+(linear>>5));return __ldg(v.tables[c]+(h<0?~h:h+(linear&31)));}
  else return beamz::cuda::yee::PackedMaterialSource(v.tables[c],v.table_sizes[c],v.codes[c],linear);
 }
 __device__ __forceinline__ float Difference(int phase,int source,int axis,int z,int y,int x,float center,float adjacent)const{
  // H's forward wall guard is unreachable after the output support check.
  if constexpr(!SupportSafe){if(phase){const int coordinate=axis==0?z:axis==1?y:x;
   if(coordinate==0||coordinate==p.physical_n[axis])return 0.f;
  }}
  return beamz::cuda::yee::ScaleYeeDifference(phase?__fsub_rn(center,adjacent):__fsub_rn(adjacent,center),p.phase[phase].inv);
 }
 __device__ __forceinline__ float Finish(int field,int version,int z,int y,int x,float value)const{
  if constexpr(HasSources||HasObserve){
   if constexpr(HasSources)if(flags&beamz::cuda::moving::kMayHaveSources){
    const auto&s=p.sources[field];
    if(s.count){
     // Captured coincident starts are equal; one contributor is coincident by definition.
     const int sz=z-s.starts[0],sy=y-s.starts[1],sx=x-s.starts[2];
     if(sz>=0&&sy>=0&&sx>=0&&sz<s.nz&&sy<s.ny&&sx<s.nx){
      int time=*s.current_step+p.macro_step+step_offset+version-1;time=max(0,min(time,s.ntime-1));
      const int spatial=(sz*s.ny+sy)*s.nx+sx,volume=s.nz*s.ny*s.nx;
      // Ordinary single-source injection contracts; coincident scatter products are rounded first.
      if(s.count==1)value=__fmaf_rn(__ldg(s.coefficients+spatial),__ldg(s.waveforms+time),value);
      else{
       #pragma unroll 4
       for(int source=0;source<s.count;++source)
        value=__fadd_rn(value,__fmul_rn(__ldg(s.coefficients+source*volume+spatial),__ldg(s.waveforms+source*s.ntime+time)));
      }
     }
    }
   }
   if constexpr(HasObserve)if(flags&beamz::cuda::moving::kMayObserve)Observe(p,field,version+raw_step_offset,z,y,x,value);
  }
  return value;
 }
};
// Canonical staggering was proved by Prepare before this kernel can run.
// Bounds are rectangular; imports outside every support never touch trails.
__device__ __forceinline__ unsigned NativeSupportMask(const Parameters&p,int z,int y,int x){
 const int nz=p.physical_n[0],ny=p.physical_n[1],nx=p.physical_n[2];
 if(z<0||y<0||x<0||z>nz||y>ny||x>nx)return 0;
 const unsigned iz=z<nz,iy=y<ny,ix=x<nx;
 return (iz&&iy)|((iz&&ix)<<1)|((iy&&ix)<<2)|(ix<<3)|(iy<<4)|(iz<<5);
}
struct FieldPacket {float hx,hy,hz,ex,ey,ez;};
template<int Depth,bool Fast,bool ReuseTerminal=true>
__device__ __forceinline__ FieldPacket ImportPacket(const Parameters&p,const size_t*bases,int version,
 int oz,int oy,int ox,int dz,int dy,int dx){
 unsigned mask=63;
 if constexpr(!Fast)mask=NativeSupportMask(p,oz-1+dz-version,oy-1+dy-version,ox-1+dx-version);
 if(!mask)return {};
 const int neighbor=(dz==0)|((dy==0)<<1)|((dx==0)<<2);
 const int surface=Surface(dz?dz-1:BZ-1,dy?dy-1:BY-1,dx?dx-1:BX-1);
 const size_t base=bases[neighbor];
 if(!neighbor||surface<0||base==static_cast<size_t>(-1)){
  atomicOr(p.bad,2);const float bad=__int_as_float(0x7fffffff);
  return {mask&1?bad:0.f,mask&2?bad:0.f,mask&4?bad:0.f,mask&8?bad:0.f,mask&16?bad:0.f,mask&32?bad:0.f};
 }
 if constexpr(ReuseTerminal){if(version==Depth)mask&=7u;}
 const float*point=p.trails+base+static_cast<size_t>(version-1)*6*SurfacePoints+surface;
 return {mask&1?__ldcg(point+0*SurfacePoints):0.f,mask&2?__ldcg(point+1*SurfacePoints):0.f,
         mask&4?__ldcg(point+2*SurfacePoints):0.f,mask&8?__ldcg(point+3*SurfacePoints):0.f,
         mask&16?__ldcg(point+4*SurfacePoints):0.f,mask&32?__ldcg(point+5*SurfacePoints):0.f};
}
__global__ void BeginEpoch(uint32_t*epoch,int32_t*bad){if(!threadIdx.x){if(*epoch==0xffffffffu){atomicOr(bad,4);return;}++*epoch;}}

template<int Depth,bool Safe,int T,bool ReuseTerminal=true>
__device__ __forceinline__ void StageRegisterLow(const Parameters&p,float*stage,const size_t*bases,int n,int oz,int oy,int ox){
 for(int i=threadIdx.x;i<LowXPoints;i+=T){const int dz=1+i/BY,dy=1+i%BY;
  const auto v=ImportPacket<Depth,Safe,ReuseTerminal>(p,bases,n,oz,oy,ox,dz,dy,0);
  stage[i]=v.hx;stage[LowXPoints+i]=v.hy;stage[2*LowXPoints+i]=v.hz;
  stage[3*LowXPoints+i]=v.ex;stage[4*LowXPoints+i]=v.ey;stage[5*LowXPoints+i]=v.ez;
 }
}
template<int T,int Slots>
__device__ __forceinline__ void RetireRegisterLow(const float*stage,float(&h)[3][Slots],float(&e)[3][Slots],int n,int sy,int sx){
 const int hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31,dy=(sy-hy)&15;
 if(sx==hx&&sy!=hy){
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zslot=threadIdx.x/512+j*(T/512);if(zslot!=hz){const int dz=(zslot-hz)&7,key=(dz-1)*BY+dy-1;
   #pragma unroll
   for(int c=0;c<3;++c){h[c][j]=stage[c*LowXPoints+key];e[c][j]=stage[(c+3)*LowXPoints+key];}
  }}
 }
}

template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool CachedProfiles=false>
__device__ __forceinline__ void UniformBody(const Parameters&p,const int32_t*tiles){
 constexpr int T=(Fast||((Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512,Slots=PAD/T,ZStep=T/512;
 constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);
 static_assert(!CachedProfiles||(!Fast&&IsotropicH&&Policy==1&&UniformE),"Profile staging only certified NoIO1024");
 extern __shared__ float frame[];size_t*bases=reinterpret_cast<size_t*>(frame+3*PAD);
 volatile int*phase_step=reinterpret_cast<volatile int*>(bases+8);
 volatile int*window=phase_step+1;
 float*low_stage=frame+3*PAD+8*sizeof(size_t)/sizeof(float)+3;
 float*profiles=CachedProfiles?frame+SharedBytes/sizeof(float):nullptr;
 if constexpr(CachedProfiles)StageIndependentCpmlProfiles(p,profiles);
 const int encoded=tiles[blockIdx.x],macro_index=MacroStepOffset<1>(p,encoded),tile=PhysicalTileID(p,encoded),bz=tile/(p.nx*p.ny),by=(tile/p.nx)%p.ny,bx=tile%p.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX,rank=bz+by+bx+macro_index*p.macro_level_stride;
 const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[encoded];p.owner_tags[owner]=epoch|uint32_t(encoded);}
 __syncthreads();
 if constexpr(!Fast){
  if(!threadIdx.x){
   const int mz=p.phase[0].own[0].z+1,my=p.phase[0].own[0].y+1,mx=p.phase[0].own[1].x+1;
   window[0]=max(1,max(oz-mz+1,max(oy-my+1,ox-mx+1)));
   window[1]=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));
  }
  __syncthreads();
  // The owner tag remains published even when no component can consume trails.
  if(window[0]>window[1])return;
 }
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;size_t base=size_t(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*p.nx*p.ny-dy*p.nx-dx;const int owner=((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer+macro_index*p.nz*p.ny*p.nx];
   if(bits&&p.owner_tags[owner]!=(epoch|uint32_t(producer+macro_index*p.nz*p.ny*p.nx)))atomicOr(p.bad,1);else base=size_t(owner)*Depth*6*SurfacePoints;
  }bases[bits]=base;
 }
 const int sx=threadIdx.x&31,sy=(threadIdx.x/32)&15,zbase=threadIdx.x/512;
 float h[3][Slots],e[3][Slots];
 // A late start must see the neighbor base table before reading trails.
 if constexpr(!Fast)__syncthreads();
 if constexpr(Fast){
 #pragma unroll
 for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+(zs==7?-1:zs),y=oy+(sy==15?-1:sy),x=ox+(sx==31?-1:sx);
  #pragma unroll
  for(int c=0;c<3;++c){h[c][j]=Initial(p,c,z,y,x,macro_index);e[c][j]=Initial(p,c+3,z,y,x,macro_index);frame[c*PAD+Slot(zs,sy,sx)]=e[c][j];}
 }
 __syncthreads();
 }else{
  const int first=window[0],prev=first-1;
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;
   FieldPacket v{};
   if(first==1){const int z=oz+(zs==7?-1:zs),y=oy+(sy==15?-1:sy),x=ox+(sx==31?-1:sx);
    v={Initial(p,0,z,y,x,macro_index),Initial(p,1,z,y,x,macro_index),Initial(p,2,z,y,x,macro_index),Initial(p,3,z,y,x,macro_index),Initial(p,4,z,y,x,macro_index),Initial(p,5,z,y,x,macro_index)};
   }else{
    const int hz=(-prev-1)&7,hy=(-prev-1)&15,hx=(-prev-1)&31;
    // Previous owned volume is wholly invalid; only the low ghost union may live.
    if(zs==hz||sy==hy||sx==hx)v=ImportPacket<Depth,false,(Layout==1)>(p,bases,prev,oz,oy,ox,(zs-hz)&7,(sy-hy)&15,(sx-hx)&31);
   }
   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;
   e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;
   frame[Slot(zs,sy,sx)]=v.ex;frame[PAD+Slot(zs,sy,sx)]=v.ey;frame[2*PAD+Slot(zs,sy,sx)]=v.ez;
  }

  __syncthreads();
 }
 const uint8_t flags=Policy==0?0:Policy==2?1:Policy==3?16:Policy==4?17:p.flags[tile];
 Environment<Layout,Fast,IsotropicH,Policy,UniformE,ActiveAxes,CachedProfiles>env{p,flags,profiles};env.step_offset=MacroStepOffset<Depth>(p,tiles[blockIdx.x]);env.raw_step_offset=(MacroStepOffset<1>(p,tiles[blockIdx.x])%3)*Depth;
 #pragma unroll 1
 for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n){
  {int step;asm volatile("mov.b32 %0,%1;":"=r"(step):"r"(n));const int hz=(-step-1)&7,hy=(-step-1)&15,hx=(-step-1)&31,low=-step-1;
   const int dy=(sy-hy)&15,dx=(sx-hx)&31,y=oy+low+dy,x=ox+low+dx;
   #pragma unroll
   for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+low+((zs-hz)&7);
    if(zs!=hz&&sy!=hy&&sx!=hx){
     const auto read=[&] __device__(int field,int,int dz,int ddy,int ddx){
      if(field<3)return h[field][j];const int c=field-3;
      if(!dz&&!ddy&&!ddx)return e[c][j];return frame[c*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];
     };
     #pragma unroll
     for(int c=0;c<3;++c){float value=0.f;if(env.Valid(c,z,y,x)){
      if(c==0)value=beamz::cuda::trace_native::AdvanceNativeCell<0,0,false>(env,read,n,z,y,x);
      else if(c==1)value=beamz::cuda::trace_native::AdvanceNativeCell<0,1,false>(env,read,n,z,y,x);
      else value=beamz::cuda::trace_native::AdvanceNativeCell<0,2,false>(env,read,n,z,y,x);
     }h[c][j]=value;}
    }
   }
  }
  const int hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31,low=-n-1;
  const int dy=(sy-hy)&15,dx=(sx-hx)&31;
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;if(zs==hz||sy==hy){const int dz=(zs-hz)&7;
   const auto v=ImportPacket<Depth,SupportSafe,(Layout==1)>(p,bases,n,oz,oy,ox,dz,dy,dx);
   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;
  }}
  StageRegisterLow<Depth,SupportSafe,T,(Layout==1)>(p,low_stage,bases,n,oz,oy,ox);
  __syncthreads(); // B1 retires all old-E frame readers and publishes low imports.
  RetireRegisterLow<T,Slots>(low_stage,h,e,n,sy,sx);
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;
   #pragma unroll
   for(int c=0;c<3;++c)frame[c*PAD+Slot(zs,sy,sx)]=h[c][j];
  }
  __syncthreads(); // B2 publishes all current H values.
  {int step;asm volatile("mov.b32 %0,%1;":"=r"(step):"r"(n));const int ehz=(-step-1)&7,ehy=(-step-1)&15,ehx=(-step-1)&31,elow=-step-1;
   const int y=oy+elow+((sy-ehy)&15),x=ox+elow+((sx-ehx)&31);
   #pragma unroll
   for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+elow+((zs-ehz)&7);
    if(zs!=ehz&&sy!=ehy&&sx!=ehx){
     const auto read=[&] __device__(int field,int,int dz,int ddy,int ddx){
      if(field>=3)return e[field-3][j];if(!dz&&!ddy&&!ddx)return h[field][j];return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];
     };
     #pragma unroll
     for(int c=0;c<3;++c){float value=0.f;if(env.Valid(c+3,z,y,x)){
      if(c==0)value=beamz::cuda::trace_native::AdvanceNativeCell<1,0,false>(env,read,n,z,y,x);
      else if(c==1)value=beamz::cuda::trace_native::AdvanceNativeCell<1,1,false>(env,read,n,z,y,x);
      else value=beamz::cuda::trace_native::AdvanceNativeCell<1,2,false>(env,read,n,z,y,x);
     }e[c][j]=value;}
    }
   }
  }
  if(n<(Fast?Depth:window[1])){
  __syncthreads(); // B3 retires current-H frame readers before E overwrite.
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;
   #pragma unroll
   for(int c=0;c<3;++c)frame[c*PAD+Slot(zs,sy,sx)]=e[c][j];
  }

  __syncthreads(); // B4 publishes E and the next phase counter.
  }
  const int y=oy+low+dy,x=ox+low+dx;
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;if(zs!=hz&&sy!=hy&&sx!=hx){const int dz=(zs-hz)&7,z=oz+low+dz,surface=Surface(dz-1,dy-1,dx-1);
   if(surface>=0){
    #pragma unroll
    for(int c=0;c<3;++c){
    p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=h[c][j];if(Layout!=1||n<Depth)p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=e[c][j];}
   }
   if(n==Depth){
    #pragma unroll
    for(int c=0;c<3;++c){
    const auto&hf=MacroFinalField(p,c,macro_index);const auto&ef=MacroFinalField(p,c+3,macro_index);
     if(Contains(hf,z,y,x))hf.data[z*hf.plane+y*hf.row+x]=h[c][j];
     if(Contains(ef,z,y,x))ef.data[z*ef.plane+y*ef.row+x]=e[c][j];
    }
   }
  }}
 }
}


template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>
__global__ __launch_bounds__((Fast?1024:512),1) void Asymmetric(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 if constexpr(Policy==1&&IsotropicH){
  // Host compact range contains only noncertified NoIO IDs.
  UniformBody<Depth,Layout,false,true,1,false>(p,tiles);
 }else if constexpr(Policy==2&&IsotropicH){
  // Noncertified observer subrange; canonical spatial E lookup remains.
  UniformBody<Depth,Layout,false,true,2,false>(p,tiles);
 }else if constexpr(Fast||(Policy==1&&IsotropicH)){
  // CTA-uniform immutable flag; barriers remain converged in either complete body.
  if(p.flags[PhysicalTileID(p,tiles[blockIdx.x])]&64)UniformBody<Depth,Layout,Fast,IsotropicH,Policy,true>(p,tiles);
  else UniformBody<Depth,Layout,Fast,IsotropicH,Policy,false>(p,tiles);
 }else UniformBody<Depth,Layout,Fast,IsotropicH,Policy,false>(p,tiles);
}

template<int Depth,int Layout=1>
__global__ __launch_bounds__(1024,1) void AsymmetricNoIOUniform1024(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 // Original full certified NoIO grid; selector uniform before any barrier.
 switch(p.flags[PhysicalTileID(p,tiles[blockIdx.x])]&14){
  case 2:UniformBody<Depth,Layout,false,true,1,true,1,true>(p,tiles);return;
  case 4:UniformBody<Depth,Layout,false,true,1,true,2,true>(p,tiles);return;
  case 8:UniformBody<Depth,Layout,false,true,1,true,4,true>(p,tiles);return;
  default:UniformBody<Depth,Layout,false,true,1,true,-1,true>(p,tiles);return;
 }
}

template<int Depth,int Layout=1>
__global__ __launch_bounds__(1024,1) void AsymmetricObserverUniform1024(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 UniformBody<Depth,Layout,false,true,2,true>(p,tiles);
}

// Private source-only storage schedule; source-bound to the proven shared-Ey body.
template<int Depth,bool Safe,int T>
__device__ __forceinline__ void StageSourcePendingH(const Parameters&p,float*stage,const size_t*bases,int n,int oz,int oy,int ox){
 for(int i=threadIdx.x;i<LowXPoints;i+=T){const int dz=1+i/BY,dy=1+i%BY;
  const auto v=ImportPacket<Depth,Safe>(p,bases,n,oz,oy,ox,dz,dy,0);
  stage[i]=v.hx;stage[LowXPoints+i]=v.hy;stage[2*LowXPoints+i]=v.hz;
 }
}
// Two-plane publication groups; use after the reviewed base SourcePendingBody helpers.
template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool CachedProfiles=false>
__device__ __forceinline__ void SourcePendingPairedBody(const Parameters&p,const int32_t*tiles){
 constexpr int T=1024;
 constexpr int XY=CY*CX,ZGroups=T>XY?T/XY:1,YGroups=XY>T?XY/T:1;
 constexpr int LocalZ=CZ/ZGroups,Slots=LocalZ*YGroups;
 static_assert(CZ==8&&CY==16&&CX==32&&T==1024&&LocalZ==4&&YGroups==1,"Paired source mapping only");
 constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);
 extern __shared__ float frame[];
 float*ey_shared=frame+2*PAD;
 float*snapshots=frame+3*PAD;
 size_t*bases=reinterpret_cast<size_t*>(snapshots+2048);
 volatile int*phase_step=reinterpret_cast<volatile int*>(bases+8);
 volatile int*window=phase_step+1;
 float*low_stage=reinterpret_cast<float*>(bases+8)+3;
 float*profiles=CachedProfiles?low_stage+3*LowXPoints:nullptr;
 if constexpr(CachedProfiles)StageIndependentCpmlProfiles(p,profiles);
 const int encoded=tiles[blockIdx.x],macro_index=MacroStepOffset<1>(p,encoded),tile=PhysicalTileID(p,encoded),bz=tile/(p.nx*p.ny),by=(tile/p.nx)%p.ny,bx=tile%p.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX,rank=bz+by+bx+macro_index*p.macro_level_stride;
 const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[encoded];p.owner_tags[owner]=epoch|uint32_t(encoded);}
 __syncthreads();
 if constexpr(!Fast){
  if(!threadIdx.x){const int mz=p.phase[0].own[0].z+1,my=p.phase[0].own[0].y+1,mx=p.phase[0].own[1].x+1;
   window[0]=max(1,max(oz-mz+1,max(oy-my+1,ox-mx+1)));
   window[1]=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));
  }
  __syncthreads();if(window[0]>window[1])return;
 }
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;size_t base=size_t(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*p.nx*p.ny-dy*p.nx-dx;const int owner=((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer+macro_index*p.nz*p.ny*p.nx];
   if(bits&&p.owner_tags[owner]!=(epoch|uint32_t(producer+macro_index*p.nz*p.ny*p.nx)))atomicOr(p.bad,1);else base=size_t(owner)*Depth*6*SurfacePoints;
  }bases[bits]=base;
 }
 const int sx=threadIdx.x&31,sy=(threadIdx.x/32)&(CY-1),zbase=(threadIdx.x/XY)*LocalZ;
 float h[3][Slots],pending_ex0[YGroups],pending_ez0[YGroups],pending_ex1[YGroups],pending_ez1[YGroups];
 const auto get_ey=[&] __device__(int zs,int y,int x,int){return ey_shared[Slot(zs,y,x)];};
 const auto set_ey=[&] __device__(int zs,int y,int,float value){ey_shared[Slot(zs,y,sx)]=value;};
 __syncthreads();
 const int first=Fast?1:window[0],prev=first-1;
 #pragma unroll
 for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX);
  #pragma unroll
  for(int j=0;j<LocalZ;++j){const int zs=zbase+j,si=gy*LocalZ+j;FieldPacket v{};
   if(first==1){const int z=oz+(zs==CZ-1?-1:zs),y=oy+(ys==CY-1?-1:ys),x=ox+(sx==31?-1:sx);
    v={Initial(p,0,z,y,x,macro_index),Initial(p,1,z,y,x,macro_index),Initial(p,2,z,y,x,macro_index),Initial(p,3,z,y,x,macro_index),Initial(p,4,z,y,x,macro_index),Initial(p,5,z,y,x,macro_index)};
   }else{const int hz=(-prev-1)&(CZ-1),hy=(-prev-1)&(CY-1),hx=(-prev-1)&31;
    if(zs==hz||ys==hy||sx==hx)v=ImportPacket<Depth,false>(p,bases,prev,oz,oy,ox,(zs-hz)&(CZ-1),(ys-hy)&(CY-1),(sx-hx)&31);
   }
   h[0][si]=v.hx;h[1][si]=v.hy;h[2][si]=v.hz;set_ey(zs,ys,si,v.ey);
   frame[Slot(zs,ys,sx)]=v.ex;frame[PAD+Slot(zs,ys,sx)]=v.ez;
  }
 }
 if(!threadIdx.x)*phase_step=first;__syncthreads();
 const uint8_t flags=Policy==0?0:Policy==2?1:Policy==3?16:Policy==4?17:p.flags[tile];
 Environment<Layout,Fast,IsotropicH,Policy,UniformE,ActiveAxes,CachedProfiles>env{p,flags,profiles};env.step_offset=MacroStepOffset<Depth>(p,tiles[blockIdx.x]);env.raw_step_offset=(MacroStepOffset<1>(p,tiles[blockIdx.x])%3)*Depth;
 #pragma unroll 1
 for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n){
  {const int step=*phase_step,hz=(-step-1)&(CZ-1),hy=(-step-1)&(CY-1),hx=(-step-1)&31,low=-step-1;
   #pragma unroll
   for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),y=oy+low+((ys-hy)&(CY-1)),x=ox+low+((sx-hx)&31);
    #pragma unroll
    for(int j=0;j<LocalZ;++j){const int zs=zbase+j,si=gy*LocalZ+j,z=oz+low+((zs-hz)&(CZ-1));
     const float ey_x=get_ey(zs,ys,sx+1,si);
     if(zs!=hz&&ys!=hy&&sx!=hx){
      const auto read=[&] __device__(int field,int,int dz,int ddy,int ddx){
       if(field<3)return h[field][si];
       if(field==4){if(ddx)return ey_x;if(dz)return get_ey((zs+1)&(CZ-1),ys,sx,si+1);return get_ey(zs,ys,sx,si);}
       return frame[(field==3?0:1)*PAD+Slot(zs+dz,ys+ddy,sx+ddx)];
      };
      #pragma unroll
      for(int c=0;c<3;++c){float value=0.f;if(env.Valid(c,z,y,x)){
       if(c==0)value=beamz::cuda::trace_native::AdvanceNativeCell<0,0,false>(env,read,n,z,y,x);
       else if(c==1)value=beamz::cuda::trace_native::AdvanceNativeCell<0,1,false>(env,read,n,z,y,x);
       else value=beamz::cuda::trace_native::AdvanceNativeCell<0,2,false>(env,read,n,z,y,x);
      }h[c][si]=value;}
     }
    }
   }
  }
  const int hz=(-n-1)&(CZ-1),hy=(-n-1)&(CY-1),hx=(-n-1)&31,low=-n-1;
  #pragma unroll
  for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),dy=(ys-hy)&(CY-1),dx=(sx-hx)&31;
   #pragma unroll
   for(int j=0;j<LocalZ;++j){const int zs=zbase+j,si=gy*LocalZ+j;if(zs==hz||ys==hy){
    const auto v=ImportPacket<Depth,SupportSafe>(p,bases,n,oz,oy,ox,(zs-hz)&(CZ-1),dy,dx);h[0][si]=v.hx;h[1][si]=v.hy;h[2][si]=v.hz;
   }}
  }
  StageSourcePendingH<Depth,SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);
  __syncthreads(); // R0: retire all old E readers; publish low-X H imports.
  #pragma unroll
  for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),dy=(ys-hy)&(CY-1),dx=(sx-hx)&31;
   #pragma unroll
   for(int j=0;j<LocalZ;++j){const int zs=zbase+j,si=gy*LocalZ+j,dz=(zs-hz)&(CZ-1);
    if(sx==hx&&ys!=hy&&zs!=hz){const int key=(dz-1)*BY+dy-1;h[0][si]=low_stage[key];h[1][si]=low_stage[LowXPoints+key];h[2][si]=low_stage[2*LowXPoints+key];}
    if(zs==hz||ys==hy||sx==hx){const auto v=ImportPacket<Depth,SupportSafe>(p,bases,n,oz,oy,ox,dz,dy,dx);frame[Slot(zs,ys,sx)]=v.ex;frame[PAD+Slot(zs,ys,sx)]=v.ez;set_ey(zs,ys,si,v.ey);}
    if constexpr(ZGroups==2)if(j==LocalZ-1){const int key=(zbase/LocalZ)*XY+ys*CX+sx;snapshots[key]=h[0][si];snapshots[1024+key]=h[1][si];}
   }
  }
  #pragma unroll
  for(int pair=0;pair<LocalZ/2;++pair){const int j0=2*pair,j1=j0+1;
   float ex0[YGroups],ez0[YGroups],ex1[YGroups],ez1[YGroups];
   #pragma unroll
   for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),s0=gy*LocalZ+j0,s1=s0+1;
    ex0[gy]=frame[Slot(zbase+j0,ys,sx)];ez0[gy]=frame[PAD+Slot(zbase+j0,ys,sx)];
    ex1[gy]=frame[Slot(zbase+j1,ys,sx)];ez1[gy]=frame[PAD+Slot(zbase+j1,ys,sx)];
    frame[Slot(zbase+j0,ys,sx)]=h[0][s0];frame[PAD+Slot(zbase+j0,ys,sx)]=h[2][s0];
    frame[Slot(zbase+j1,ys,sx)]=h[0][s1];frame[PAD+Slot(zbase+j1,ys,sx)]=h[2][s1];
   }
   __syncthreads(); // Ppair publishes BOTH planes and retires ALL previous-pair XY readers.
   if(pair){
    #pragma unroll
    for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX);
     frame[Slot(zbase+j0-2,ys,sx)]=pending_ex0[gy];frame[PAD+Slot(zbase+j0-2,ys,sx)]=pending_ez0[gy];
     frame[Slot(zbase+j0-1,ys,sx)]=pending_ex1[gy];frame[PAD+Slot(zbase+j0-1,ys,sx)]=pending_ez1[gy];
    }
   }
   {const int j=j0,zs=zbase+j;
   #pragma unroll
   for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),si=gy*LocalZ+j,snapshot_key=((zbase/LocalZ)^1)*XY+ys*CX+sx;
    const float hy_x=__shfl_sync(0xffffffffu,h[1][si],(sx+31)&31);
    const int dz=(zs-hz)&(CZ-1),dy=(ys-hy)&(CY-1),dx=(sx-hx)&31,z=oz+low+dz,y=oy+low+dy,x=ox+low+dx;
    if(zs!=hz&&ys!=hy&&sx!=hx){
     const auto read=[&] __device__(int field,int,int ddz,int ddy,int ddx){
      if(field>=3)return field==3?ex0[gy]:field==4?get_ey(zs,ys,sx,si):ez0[gy];
      if(!ddz&&!ddy&&!ddx)return h[field][si];
      if(ddz){if(j)return h[field][si-1];if constexpr(ZGroups==2)return snapshots[field*1024+snapshot_key];else return h[field][si+LocalZ-1];}
      if(field==1)return hy_x;
      return frame[(field==0?0:1)*PAD+Slot(zs,ys+ddy,sx+ddx)];
     };
     ex0[gy]=env.Valid(3,z,y,x)?beamz::cuda::trace_native::AdvanceNativeCell<1,0,false>(env,read,n,z,y,x):0.f;
     const float new_ey=env.Valid(4,z,y,x)?beamz::cuda::trace_native::AdvanceNativeCell<1,1,false>(env,read,n,z,y,x):0.f;set_ey(zs,ys,si,new_ey);
     ez0[gy]=env.Valid(5,z,y,x)?beamz::cuda::trace_native::AdvanceNativeCell<1,2,false>(env,read,n,z,y,x):0.f;
    }

    if(zs!=hz&&ys!=hy&&sx!=hx){const int surface=Surface(dz-1,dy-1,dx-1);
     if(surface>=0){
    #pragma unroll
    for(int c=0;c<3;++c){const float ev=c==0?ex0[gy]:c==1?get_ey(zs,ys,sx,si):ez0[gy];
      p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=h[c][si];if(Layout!=1||n<Depth)p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=ev;}
   }
   if(n==Depth){
    #pragma unroll
    for(int c=0;c<3;++c){const float ev=c==0?ex0[gy]:c==1?get_ey(zs,ys,sx,si):ez0[gy];
      const auto&hf=MacroFinalField(p,c,macro_index);const auto&ef=MacroFinalField(p,c+3,macro_index);if(Contains(hf,z,y,x))hf.data[z*hf.plane+y*hf.row+x]=h[c][si];if(Contains(ef,z,y,x))ef.data[z*ef.plane+y*ef.row+x]=ev;}
   }
    }
   }
   }
   {const int j=j1,zs=zbase+j;
   #pragma unroll
   for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),si=gy*LocalZ+j,snapshot_key=((zbase/LocalZ)^1)*XY+ys*CX+sx;
    const float hy_x=__shfl_sync(0xffffffffu,h[1][si],(sx+31)&31);
    const int dz=(zs-hz)&(CZ-1),dy=(ys-hy)&(CY-1),dx=(sx-hx)&31,z=oz+low+dz,y=oy+low+dy,x=ox+low+dx;
    if(zs!=hz&&ys!=hy&&sx!=hx){
     const auto read=[&] __device__(int field,int,int ddz,int ddy,int ddx){
      if(field>=3)return field==3?ex1[gy]:field==4?get_ey(zs,ys,sx,si):ez1[gy];
      if(!ddz&&!ddy&&!ddx)return h[field][si];
      if(ddz){if(j)return h[field][si-1];if constexpr(ZGroups==2)return snapshots[field*1024+snapshot_key];else return h[field][si+LocalZ-1];}
      if(field==1)return hy_x;
      return frame[(field==0?0:1)*PAD+Slot(zs,ys+ddy,sx+ddx)];
     };
     ex1[gy]=env.Valid(3,z,y,x)?beamz::cuda::trace_native::AdvanceNativeCell<1,0,false>(env,read,n,z,y,x):0.f;
     const float new_ey=env.Valid(4,z,y,x)?beamz::cuda::trace_native::AdvanceNativeCell<1,1,false>(env,read,n,z,y,x):0.f;set_ey(zs,ys,si,new_ey);
     ez1[gy]=env.Valid(5,z,y,x)?beamz::cuda::trace_native::AdvanceNativeCell<1,2,false>(env,read,n,z,y,x):0.f;
    }

    if(zs!=hz&&ys!=hy&&sx!=hx){const int surface=Surface(dz-1,dy-1,dx-1);
     if(surface>=0){
    #pragma unroll
    for(int c=0;c<3;++c){const float ev=c==0?ex1[gy]:c==1?get_ey(zs,ys,sx,si):ez1[gy];
      p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=h[c][si];if(Layout!=1||n<Depth)p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=ev;}
   }
   if(n==Depth){
    #pragma unroll
    for(int c=0;c<3;++c){const float ev=c==0?ex1[gy]:c==1?get_ey(zs,ys,sx,si):ez1[gy];
      const auto&hf=MacroFinalField(p,c,macro_index);const auto&ef=MacroFinalField(p,c+3,macro_index);if(Contains(hf,z,y,x))hf.data[z*hf.plane+y*hf.row+x]=h[c][si];if(Contains(ef,z,y,x))ef.data[z*ef.plane+y*ef.row+x]=ev;}
   }
    }
   }
   }
   #pragma unroll
   for(int gy=0;gy<YGroups;++gy){pending_ex0[gy]=ex0[gy];pending_ez0[gy]=ez0[gy];pending_ex1[gy]=ex1[gy];pending_ez1[gy]=ez1[gy];}
  }
  __syncthreads(); // RLpair retires BOTH final planes' H XY readers.
  #pragma unroll
  for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX);
   frame[Slot(zbase+LocalZ-2,ys,sx)]=pending_ex0[gy];frame[PAD+Slot(zbase+LocalZ-2,ys,sx)]=pending_ez0[gy];
   frame[Slot(zbase+LocalZ-1,ys,sx)]=pending_ex1[gy];frame[PAD+Slot(zbase+LocalZ-1,ys,sx)]=pending_ez1[gy];
  }
  if(!threadIdx.x)*phase_step=n+1;
  __syncthreads(); // RE: publish all current E and retire scratch readers for next H.
 }
}


constexpr size_t SourcePendingSharedBytes=(3ULL*PAD+2048)*sizeof(float)+8*sizeof(size_t)+3*sizeof(int)+3ULL*LowXPoints*sizeof(float);
static_assert(SourcePendingSharedBytes==58680,"Shared paired-plane source schedule");
template<int Depth,int Layout,bool IsotropicH,int Policy,int Axes=-1>
__device__ __forceinline__ void TemporalEffectBody(const Parameters&p,const int32_t*ids){
 if(p.flags[PhysicalTileID(p,ids[blockIdx.x])]&64)
  SourcePendingPairedBody<Depth,Layout,false,IsotropicH,Policy,true,Axes>(p,ids);
 else SourcePendingPairedBody<Depth,Layout,false,IsotropicH,Policy,false,Axes>(p,ids);
}
template<int Depth,int Layout,bool IsotropicH,int Policy>
__global__ __launch_bounds__(1024,1) void TemporalEffects(
 const __grid_constant__ Parameters p,const int32_t*ids){
 static_assert(Policy>=1&&Policy<=5,"Shared boundary/observer/source effect classes");
 if constexpr(Policy==5){
  // A trajectory's exact CPML axis effects are already proved by the planner.
  // Specialize the common single-axis case without separating launch cohorts.
  switch(p.flags[PhysicalTileID(p,ids[blockIdx.x])]&14){
   case 2:TemporalEffectBody<Depth,Layout,IsotropicH,Policy,1>(p,ids);return;
   case 4:TemporalEffectBody<Depth,Layout,IsotropicH,Policy,2>(p,ids);return;
   case 8:TemporalEffectBody<Depth,Layout,IsotropicH,Policy,4>(p,ids);return;
   default:TemporalEffectBody<Depth,Layout,IsotropicH,Policy>(p,ids);return;
  }
 }else TemporalEffectBody<Depth,Layout,IsotropicH,Policy>(p,ids);
}

inline bool MakeSource(const BeamzSourceGroupLaunch&g,Source*out){
 *out={};if(g.coefficients.dims[0]==0)return true;
 if(g.coefficients.rank!=4||g.coefficients.element_type!=kBeamzF32||
    g.waveforms.rank!=2||g.waveforms.element_type!=kBeamzF32||
    g.starts.element_type!=kBeamzS32||g.current_step.element_type!=kBeamzS32||
    g.coefficients.dims[0]<1||(g.coefficients.dims[0]>1&&!g.coincident)||g.waveforms.dims[0]!=g.coefficients.dims[0])return false;
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
 for(int axis=0;axis<3;++axis)out->physical_n[axis]=minimum[axis];
 const int sizes[3]={BZ,BY,BX};int64_t tiles=1;
 for(int a=0;a<3;++a){if(w.counts[a]!=(int64_t(maximum[a])+w.depth+sizes[a]-1)/sizes[a])return false;tiles*=w.counts[a];}
 if(tiles!=w.tile_count||w.rank_count!=w.counts[0]+w.counts[1]+w.counts[2]-2||w.rank_offsets_host[0]!=0||w.rank_offsets_host[w.rank_count]!=w.tile_count)return false;
 for(int rank=0;rank<w.rank_count;++rank){const int count=w.rank_offsets_host[rank+1]-w.rank_offsets_host[rank];if(count<0||count>w.max_frontier||w.bulk_counts_host[rank]<0||w.bulk_counts_host[rank]>count)return false;}
#if BEAMZ_CUDA_NATIVE_MOVING_POLICIES
 if(!w.policy_offsets_host)return false;
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
 if(w.parallel_policies){
  if(!w.policy_fork_event)return false;
  for(int policy=0;policy<6;++policy)if(!w.policy_streams[policy]||!w.policy_done_events[policy])return false;
 }
#endif
 for(int rank=0;rank<w.rank_count;++rank){const auto*offsets=w.policy_offsets_host+rank*7;
  if(offsets[0]!=w.rank_offsets_host[rank]||offsets[6]!=w.rank_offsets_host[rank+1])return false;
  for(int policy=0;policy<6;++policy)if(offsets[policy]>offsets[policy+1])return false;
 }
#endif
 const size_t owners=4ULL*w.max_frontier;
 if(owners>std::numeric_limits<size_t>::max()/w.depth/6/SurfacePoints||w.trail_floats<owners*w.depth*6*SurfacePoints)return false;
 for(int field=0;field<6;++field)if(!MakeSource(program.source_groups[3+field],&out->sources[field]))return false;
 uint32_t actual_source_presence=0;for(int field=0;field<6;++field)if(out->sources[field].count>0)actual_source_presence|=1u<<field;
 if(actual_source_presence!=w.source_presence_mask)return false;
 if(w.observation.key_count<0||(w.observation.key_count&&(!w.observation.keys||!w.observation.values||!w.observation.publication_words)))return false;
#if BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION
 if(w.observation.key_count&&(!w.observation_tiles||!w.observation_leaf||
    w.observation_grid[0]!=w.observation.mask_nz||
    w.observation_grid[1]!=(w.observation.mask_ny+7)/8||
    w.observation_grid[2]!=(w.observation.mask_nx+15)/16))return false;
#endif
 out->observation_tiles=w.observation_tiles;out->observation_leaf=w.observation_leaf;
 out->observation_ny=w.observation_grid[1];out->observation_nx=w.observation_grid[2];
 out->observation=w.observation;out->slots=w.slots_device;out->flags=w.tile_flags_device;out->trails=w.trails;out->owner_tags=w.owner_tags;out->epoch=w.epoch;out->bad=w.bad;
 out->nz=w.counts[0];out->ny=w.counts[1];out->nx=w.counts[2];out->frontier=w.max_frontier;out->macro_step=macro;out->macro_level_stride=w.slab_macro_stride;out->h_source=w.h_source[0];std::copy(w.common_e,w.common_e+3,out->common_e);return true;
}
template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>
inline cudaError_t Launch(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count){
 if(!count)return cudaSuccess;
 const auto kernel=Asymmetric<Depth,Layout,Fast,IsotropicH,Policy>;
 auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
 Asymmetric<Depth,Layout,Fast,IsotropicH,Policy><<<count,(Fast?1024:512),SharedBytes,stream>>>(p,ids);
 status=cudaGetLastError();if(status!=cudaSuccess)return status;
 return cudaSuccess;
}
template<int Depth,int Layout,bool IsotropicH,int Policy>
inline cudaError_t LaunchSources(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count);
template<int Depth,int Layout,bool IsotropicH>
inline cudaError_t LaunchNoIO(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count,int uniform_count){
 if(uniform_count<0||uniform_count>count)return cudaErrorInvalidValue;
 if constexpr(IsotropicH){
  // Existing policy branch/join; both subranges are disjoint rank-local IDs.
  auto status=LaunchSources<Depth,Layout,IsotropicH,1>(stream,p,ids+uniform_count,count-uniform_count);
  if(status!=cudaSuccess||!uniform_count)return status;
  const auto kernel=AsymmetricNoIOUniform1024<Depth,Layout>;
  status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,ProfileCachedSharedBytes);if(status!=cudaSuccess)return status;
  AsymmetricNoIOUniform1024<Depth,Layout><<<uniform_count,1024,ProfileCachedSharedBytes,stream>>>(p,ids);
  return cudaGetLastError();
 }else{
  if(uniform_count)return cudaErrorInvalidValue;
  return LaunchSources<Depth,Layout,IsotropicH,1>(stream,p,ids,count);
 }
}
template<int Depth,int Layout,bool IsotropicH>
inline cudaError_t LaunchObservers(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count,int uniform_count){
 if(uniform_count<0||uniform_count>count)return cudaErrorInvalidValue;
 if constexpr(IsotropicH){
  auto status=LaunchSources<Depth,Layout,IsotropicH,2>(stream,p,ids+uniform_count,count-uniform_count);
  if(status!=cudaSuccess||!uniform_count)return status;
  const auto kernel=AsymmetricObserverUniform1024<Depth,Layout>;
  status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
  AsymmetricObserverUniform1024<Depth,Layout><<<uniform_count,1024,SharedBytes,stream>>>(p,ids);
  return cudaGetLastError();
 }else{
  if(uniform_count)return cudaErrorInvalidValue;
  return LaunchSources<Depth,Layout,IsotropicH,2>(stream,p,ids,count);
 }
}
template<int Depth,int Layout,bool IsotropicH,int Policy>
inline cudaError_t LaunchSources(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count){
 if(!count)return cudaSuccess;
 const auto kernel=TemporalEffects<Depth,Layout,IsotropicH,Policy>;
 auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SourcePendingSharedBytes);
 if(status!=cudaSuccess)return status;
 TemporalEffects<Depth,Layout,IsotropicH,Policy><<<count,1024,SourcePendingSharedBytes,stream>>>(p,ids);
 return cudaGetLastError();
}
template<int Depth,int Layout,bool IsotropicH>
inline cudaError_t LaunchRanks(cudaStream_t stream,const Parameters&p,const BeamzNativeMovingWorkspace&w){
#if BEAMZ_CUDA_NATIVE_MOVING_POLICIES
 for(int rank=0;rank<w.rank_count;++rank){const auto*offsets=w.policy_offsets_host+rank*7;
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
  const bool parallel=w.parallel_policies;
  if(parallel){const auto status=cudaEventRecord(static_cast<cudaEvent_t>(w.policy_fork_event),stream);if(status!=cudaSuccess)return status;}
#endif
  for(int policy=0;policy<6;++policy){const int start=offsets[policy],count=offsets[policy+1]-start;
   if(!count)continue;cudaError_t status=cudaSuccess;cudaStream_t branch=stream;
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
   if(parallel){branch=static_cast<cudaStream_t>(w.policy_streams[policy]);status=cudaStreamWaitEvent(branch,static_cast<cudaEvent_t>(w.policy_fork_event),0);if(status!=cudaSuccess)return status;}
#endif
   switch(policy){
    case 0:status=Launch<Depth,Layout,true,IsotropicH,0>(branch,p,w.tile_ids_device+start,count);break;
    case 1:status=LaunchNoIO<Depth,Layout,IsotropicH>(branch,p,w.tile_ids_device+start,count,w.uniform_noio_counts_host[rank]);break;
    case 2:status=LaunchObservers<Depth,Layout,IsotropicH>(branch,p,w.tile_ids_device+start,count,w.uniform_observer_counts_host[rank]);break;
    case 3:status=LaunchSources<Depth,Layout,IsotropicH,3>(branch,p,w.tile_ids_device+start,count);break;
    case 4:status=LaunchSources<Depth,Layout,IsotropicH,4>(branch,p,w.tile_ids_device+start,count);break;
    case 5:status=LaunchSources<Depth,Layout,IsotropicH,5>(branch,p,w.tile_ids_device+start,count);break;
   }
   if(status!=cudaSuccess)return status;
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
   if(parallel){status=cudaEventRecord(static_cast<cudaEvent_t>(w.policy_done_events[policy]),branch);if(status!=cudaSuccess)return status;}
#endif
  }
#if BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES
  if(parallel)for(int policy=0;policy<6;++policy){
   if(offsets[policy]==offsets[policy+1])continue;
   const auto status=cudaStreamWaitEvent(stream,static_cast<cudaEvent_t>(w.policy_done_events[policy]),0);if(status!=cudaSuccess)return status;
  }
#endif
 }
 return cudaSuccess;
#else
 for(int rank=0;rank<w.rank_count;++rank){const int start=w.rank_offsets_host[rank],count=w.rank_offsets_host[rank+1]-start,bulk=w.bulk_counts_host[rank];
  auto status=Launch<Depth,Layout,true,IsotropicH>(stream,p,w.tile_ids_device+start,bulk);if(status!=cudaSuccess)return status;
  status=LaunchSources<Depth,Layout,IsotropicH,5>(stream,p,w.tile_ids_device+start+bulk,count-bulk);if(status!=cudaSuccess)return status;
 }
 return cudaSuccess;
#endif
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
#if BEAMZ_CUDA_NATIVE_MOVING_POLICIES
 if(!w.uniform_noio_counts_host||!w.uniform_observer_counts_host)return cudaErrorInvalidValue;
#endif
 const auto stream=static_cast<cudaStream_t>(opaque_stream);
 BeginEpoch<<<1,1,0,stream>>>(w.epoch,w.bad);auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
 if(w.depth==4){if(layout==2)return DispatchClass<4,2>(stream,p,w);return DispatchClass<4,1>(stream,p,w);}
 if(w.depth==16){if(layout==2)return DispatchClass<16,2>(stream,p,w);return DispatchClass<16,1>(stream,p,w);}
 return cudaErrorInvalidValue;
}

int BeamzEnqueueNativeMacroSlab(void*opaque_stream,const BeamzProgramLaunch&program,
 const BeamzBuffer*final_fields,int macro_step,const BeamzNativeMovingWorkspace&base,int level_begin,int level_end,bool begin_epoch){
 using namespace beamz::cuda::moving_native;
 const auto validation=BeamzValidateNativeProgram(program);if(validation)return validation;
 Parameters p;int layout=0;if(!Prepare(program,final_fields,base,macro_step,&p,&layout))return cudaErrorInvalidValue;
 if(!base.slab_tile_ids_device||!base.slab_policy_offsets_host||!base.slab_noio_counts_host||!base.slab_observer_counts_host||base.slab_level_count<1)return cudaErrorInvalidValue;
 BeamzNativeMovingWorkspace slab=base;
 slab.tile_ids_device=base.slab_tile_ids_device;slab.policy_offsets_host=base.slab_policy_offsets_host;
 slab.uniform_noio_counts_host=base.slab_noio_counts_host;slab.uniform_observer_counts_host=base.slab_observer_counts_host;
 if(level_begin<0||level_end<=level_begin||level_end>base.slab_level_count)return cudaErrorInvalidValue;
 slab.policy_offsets_host+=size_t(level_begin)*7;slab.uniform_noio_counts_host+=level_begin;slab.uniform_observer_counts_host+=level_begin;
 slab.rank_count=level_end-level_begin;const auto&w=slab;
 if(!base.slab_slots_device||base.slab_max_frontier<1)return cudaErrorInvalidValue;
 p.slots=base.slab_slots_device;p.frontier=base.slab_max_frontier;
#if BEAMZ_CUDA_NATIVE_MOVING_POLICIES
 if(!w.uniform_noio_counts_host||!w.uniform_observer_counts_host)return cudaErrorInvalidValue;
#endif
 const auto stream=static_cast<cudaStream_t>(opaque_stream);
 if(begin_epoch){BeginEpoch<<<1,1,0,stream>>>(w.epoch,w.bad);auto status=cudaGetLastError();if(status!=cudaSuccess)return status;}
 if(w.depth==4){if(layout==2)return DispatchClass<4,2>(stream,p,w);return DispatchClass<4,1>(stream,p,w);}
 if(w.depth==16){if(layout==2)return DispatchClass<16,2>(stream,p,w);return DispatchClass<16,1>(stream,p,w);}
 return cudaErrorInvalidValue;
}
