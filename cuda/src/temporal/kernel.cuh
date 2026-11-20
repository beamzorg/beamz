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
#include "temporal/stencils/compact_full_edge.cuh"
#include "cell.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#ifndef BEAMZ_NATIVE_ASYMMETRIC_SWIZZLE
#define BEAMZ_NATIVE_ASYMMETRIC_SWIZZLE 0
#endif
namespace beamz::cuda::moving_native {
template<int Material,bool UniformMetric> struct KernelFacts {
 static constexpr int material=Material;
 static constexpr bool uniform_metric=UniformMetric;
};
constexpr int BZ=7,BY=15,BX=31,CZ=8,CY=16,CX=32,Threads=512;
constexpr int PAD=CZ*CY*CX,SurfacePoints=beamz::cuda::moving::kTrailSurfacePoints;
constexpr int LowXPoints=BZ*BY;
constexpr size_t SharedBytes=3ULL*PAD*sizeof(float)+16*sizeof(size_t)+3*sizeof(int)+6ULL*LowXPoints*sizeof(float);
constexpr size_t ProfileCachedSharedBytes=SharedBytes+864ULL*sizeof(float);
static_assert(SharedBytes==51812&&ProfileCachedSharedBytes==55268,"Exact appended profile layout");
struct Source {
 const float* coefficients=nullptr;const float* waveforms=nullptr;
 const int32_t* starts=nullptr;const int32_t* current_step=nullptr;
 int count=0,nz=0,ny=0,nx=0,ntime=0;
 bool coincident=false,disjoint=false;
};
enum GeneralEffect : unsigned {
 WideBoundary=1, GeneralSource=2, DenseMaterial=4, VolumeOutput=8,
};
struct Parameters {
 MstepCompactEdgeView phase[2];Source sources[6],pre_e[3];
 const int32_t* current_step;
 const beamz::cuda::moving::Scatter* scatters;
 int scatter_offsets[10];
 const beamz::cuda::moving::VolumeRecording* volumes;
 int volume_offsets[7];
 unsigned pre_e_mask;
 unsigned general_effects;
 beamz::cuda::deep::Observation observation;
 const int32_t* observation_tiles;const int32_t* observation_leaf;
 int observation_ny,observation_nx,observation_nz,observation_origin[3];
 const int32_t* slots;const int32_t* face_lines;const uint8_t* flags;float* trails;
 uint64_t* owner_tags;const uint32_t* epoch;int32_t* bad;
 int nz,ny,nx,frontier,macro_step,macro_level_stride;int physical_n[3];float h_source;float common_e[3];int raw_slots=3;int source_wave_count=0;int source_wave_offsets[6]{};
 unsigned bf16_psi;
 int material_layout,depth;
 bool isotropic_h;
};
__device__ __forceinline__ int PhysicalTileID(const Parameters&p,int encoded){return encoded%(p.nz*p.ny*p.nx);}
__device__ __forceinline__ int MacroIndex(const Parameters&p,int encoded){return encoded/(p.nz*p.ny*p.nx);}

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
 if(z==BZ-1)return y*CX+x;
 if(y==BY-1)return BY*CX+z*CX+x;
 if(x==BX-1)return BY*CX+(BZ-1)*CX+z*CY+y;
 return -1;
}
template<int Axes=-1>__device__ __forceinline__ bool Contains(const MstepCompactField&f,int z,int y,int x){
 if constexpr(Axes==1)return unsigned(z)<unsigned(f.z);
 if constexpr(Axes==2)return unsigned(y)<unsigned(f.y);
 if constexpr(Axes==4)return unsigned(x)<unsigned(f.x);
 return z>=0&&y>=0&&x>=0&&z<f.z&&y<f.y&&x<f.x;}
__host__ __device__ __forceinline__ const MstepCompactField& InitialField(const Parameters&p,int field){return p.phase[field/3].own[field%3];}
__host__ __device__ __forceinline__ const MstepCompactField& FinalField(const Parameters&p,int field){return p.phase[field/3].output[field%3];}
__device__ __forceinline__ const MstepCompactField& MacroFinalField(const Parameters&p,int field,int bank){return (bank&1)?InitialField(p,field):FinalField(p,field);}
template<int Axes=-1>__device__ __forceinline__ float Initial(const Parameters&p,int field,int z,int y,int x,int bank=0){
 const auto&f=(bank&1)?FinalField(p,field):InitialField(p,field);return Contains<Axes>(f,z,y,x)?__ldcg(f.data+z*f.plane+y*f.row+x):0.f;
}
__device__ __forceinline__ void Observe(const Parameters&p,int field,int version,int z,int y,int x,float value){
 const auto&o=p.observation;if(!o.values)return;
#if BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION
 const int iz=z-p.observation_origin[0],iy=(y>>3)-p.observation_origin[1],ix=(x>>4)-p.observation_origin[2];
 if(unsigned(iz)>=unsigned(p.observation_nz)||unsigned(iy)>=unsigned(p.observation_ny)||unsigned(ix)>=unsigned(p.observation_nx))return;
 const int tile=(iz*p.observation_ny+iy)*p.observation_nx+ix;
 const int encoded=__ldg(p.observation_tiles+tile);
 if(encoded<0||!(encoded&(1u<<field)))return;
 const int slot=encoded>>6;
 const uint64_t row=__ldg(reinterpret_cast<const uint64_t*>(p.observation_leaf)+size_t(slot)*48+field*8+(y&7));
 const unsigned mask=unsigned(row&65535u),column=x&15,bit=1u<<column;
 if(mask&bit){const int raw=int(row>>16)+__popc(mask&(bit-1));o.values[size_t(version-1)*o.key_count+raw]=value;}
#else
 if(z<0||y<0||x<0||z>=o.mask_nz||y>=o.mask_ny||x>=o.mask_nx)return;
 const int cell=(z*o.mask_ny+y)*o.mask_nx+x;
 const uint32_t word=__ldg(o.publication_words+(cell>>2));
 if(!((word>>(8*(cell&3)))&(1u<<field)))return;
 const auto&f=FinalField(p,field);
 const int64_t key=o.field_base[field]+(static_cast<int64_t>(z)*f.y+y)*f.x+x;
 int lo=0,hi=o.key_count;while(lo<hi){const int middle=(lo+hi)/2;if(o.keys[middle]<key)lo=middle+1;else hi=middle;}
 if(lo<o.key_count&&o.keys[lo]==key)o.values[static_cast<size_t>(version-1)*o.key_count+lo]=value;
#endif
}

__device__ __noinline__ void RecordVolume(const Parameters&p,int field,int offset,int version,
                                          int z,int y,int x,float value){
 const int start=*p.current_step, step=start+p.macro_step+offset+version;
 const auto& f=FinalField(p,field);
 const int point=(z*f.y+y)*f.x+x;
 for(int i=p.volume_offsets[field];i<p.volume_offsets[field+1];++i){
  const auto& record=p.volumes[i];
  const int frame=step/record.interval;
  if(step%record.interval==0&&frame<=record.capacity)
   record.frames[size_t(frame-start/record.interval-1)*record.points+point]=value*record.sign;
 }
}

// Each slot is a distinct family/term; everykind retains its own24FP32 bits.
__host__ __device__ __forceinline__ int IndependentProfileCacheIndex(int slot,int kind,int packed){
 return (slot*3+(kind-2))*24+packed;
}
__device__ __forceinline__ float ReadIndependentProfileCache(const float*profiles,int slot,int kind,int packed){
 return profiles[IndependentProfileCacheIndex(slot,kind,packed)];
}
template<int Axes=-1>__device__ __forceinline__ void StageIndependentCpmlProfiles(const Parameters&p,float*profiles){
 if(threadIdx.x<512)for(int i=threadIdx.x;i<864;i+=512){
  const int profile=i/24,packed=i%24,slot=profile/3,kind=profile%3;
  if constexpr(Axes>0)if(!(Axes&(1<<beamz::cuda::yee::CpmlAxis(slot%6))))continue;
  const auto&psi=p.phase[slot/6].psi[slot%6];
  const float*input=kind==0?psi.a:kind==1?psi.b:psi.inv;
  profiles[i]=__ldg(input+packed);
 }
}

// Source waveform indices depend on time/contributor, never spatial cell.
__device__ __forceinline__ float SourceWave(const Parameters&p,const float*stage,int field,int source,int version){
 return stage[(p.source_wave_offsets[field]+source)*p.depth+version-1];
}
__device__ __forceinline__ void StageSourceWaves(const Parameters&p,int step_offset,float*stage){
 #pragma unroll
 for(int field=0;field<6;++field){const auto&s=p.sources[field];
  for(int i=threadIdx.x;i<s.count*p.depth;i+=1024){const int source=i/p.depth,version=i%p.depth;
   int time=*s.current_step+p.macro_step+step_offset+version;time=max(0,min(time,s.ntime-1));
   stage[(p.source_wave_offsets[field]+source)*p.depth+version]=__ldg(s.waveforms+source*s.ntime+time);
  }
 }
}

__device__ __noinline__ float ApplySparse(const beamz::cuda::moving::Scatter* sources,
    int first,int last,int index,int time,float value) {
 for(int s=first;s<last;++s){
  const auto& source=sources[s];
  int low=0,high=source.count;
  while(low<high){const int mid=low+(high-low)/2;if(source.indices[mid]<index)low=mid+1;else high=mid;}
  if(low==source.count||source.indices[low]!=index)continue;
  const float amp=source.waveform[min(time,source.samples-1)];
  const int begin=source.ends?(low?source.ends[low-1]:0):low;
  const int end=source.ends?source.ends[low]:low+1;
  for(int j=begin;j<end;++j)value=__fadd_rn(value,__fmul_rn(source.coefficients[j],amp));
 }
 return value;
}

__device__ __noinline__ float ApplyUnstagedGroup(const Source& source,int time,
    int z,int y,int x,float value) {
 const int volume=source.nz*source.ny*source.nx;
 time=min(time,source.ntime-1);
 for(int s=0;s<source.count;++s){
  const int start=source.coincident?0:3*s;
  const int64_t sz=int64_t(z)-source.starts[start],sy=int64_t(y)-source.starts[start+1],sx=int64_t(x)-source.starts[start+2];
  if(sz<0||sy<0||sx<0||sz>=source.nz||sy>=source.ny||sx>=source.nx)continue;
  const int spatial=(int(sz)*source.ny+int(sy))*source.nx+int(sx);
  const float a=source.coefficients[s*volume+spatial],b=source.waveforms[s*source.ntime+time];
  value=(source.count==1||source.disjoint)?__fmaf_rn(a,b,value):__fadd_rn(value,__fmul_rn(a,b));
 }
 return value;
}

// Fast is admitted only for source/CPML/observation/native-wall-free OWNED
// trajectories. Startup field reads always retain real support checks.
template<class Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5),bool UniformE=false,int ActiveAxes=-1,bool CachedProfiles=false,bool StagedSources=false,bool General=false> struct Environment {
 // This specialization requires packed lossless E and a unit isotropic H decay.
 static constexpr bool UnitDecay = IsotropicH;
 static constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);
 static constexpr bool HasCpml=Policy==1||Policy==5;
 static constexpr bool HasSources=Policy==3||Policy==4||Policy==5;
 static constexpr bool HasObserve=Policy==2||Policy==4||Policy==5;
 const Parameters&p;uint8_t flags;const float*profiles=nullptr;int step_offset=0,raw_step_offset=0;const float*source_waves=nullptr;
 __device__ __forceinline__ bool Valid(int field,int z,int y,int x)const{
  if constexpr(SupportSafe)return true;
  const int nz=p.physical_n[0],ny=p.physical_n[1],nx=p.physical_n[2];
  // The symmetric CPML12 footprint encloses the whole one-cell import halo.
  // A certified single-axis trajectory is interior on the two remaining axes.
  if constexpr(!General&&HasCpml&&ActiveAxes==1)return unsigned(z)<unsigned(nz+(field==2||field==3||field==4));
  if constexpr(!General&&HasCpml&&ActiveAxes==2)return unsigned(y)<unsigned(ny+(field==1||field==3||field==5));
  if constexpr(!General&&HasCpml&&ActiveAxes==4)return unsigned(x)<unsigned(nx+(field==0||field==4||field==5));
  if(z<0||y<0||x<0||z>nz||y>ny||x>nx)return false;
  const bool iz=z<nz,iy=y<ny,ix=x<nx;
  return field==0?(iz&&iy):field==1?(iz&&ix):field==2?(iy&&ix):field==3?ix:field==4?iy:iz;
 }
 __device__ __forceinline__ int Packed(int slot,int z,int y,int x)const{
  const int family=slot/6,term=slot%6,axis=beamz::cuda::yee::CpmlAxis(term);
  const int coord=axis==0?z:axis==1?y:x;
  const int size=p.physical_n[axis]+family;
  if constexpr(!General)return coord<12?coord:24+coord-size;
  const auto&psi=p.phase[family].psi[term];
  return coord<psi.low?coord:psi.low+coord-(size-psi.high);
 }
 __device__ __forceinline__ bool LocalActive(int slot,int z,int y,int x)const{
  if constexpr(!HasCpml)return false;
  const int family=slot/6,term=slot%6,axis=beamz::cuda::yee::CpmlAxis(term);
  if constexpr(ActiveAxes>=0){if(!(ActiveAxes&(1u<<axis)))return false;}
  else if(!(flags&(2u<<axis)))return false;
  const int coord=axis==0?z:axis==1?y:x;
  const int size=p.physical_n[axis]+family;
  if constexpr(!General)return coord<12||coord>=size-12;
  const auto&psi=p.phase[family].psi[term];
  return coord<psi.low||coord>=size-psi.high;
 }
 __device__ __forceinline__ int PsiOffset(int slot,int z,int y,int x)const{
  const int family=slot/6,term=slot%6,axis=beamz::cuda::yee::CpmlAxis(term);
  const int packed=Packed(slot,z,y,x);
  if constexpr(!General){
   const int c=term/2;
   const int ny=axis==1?24:p.physical_n[1]+(family?(c!=1):(c==1));
   const int nx=axis==2?24:p.physical_n[2]+(family?(c!=0):(c==0));
   return ((axis==0?packed:z)*ny+(axis==1?packed:y))*nx+(axis==2?packed:x);
  }
  const auto&psi=p.phase[family].psi[term];
  return (axis==0?packed:z)*psi.plane+(axis==1?packed:y)*psi.row+(axis==2?packed:x);
 }
 __device__ __forceinline__ float LoadLocal(int slot,int z,int y,int x)const{
  // Mutable within the same CTA/window: ordinary coherent loads, never __ldg.
  const auto* input=p.phase[slot/6].psi[slot%6].input;
  const int offset=PsiOffset(slot,z,y,x);
  if constexpr(General)if(p.bf16_psi&(1u<<slot))
   return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(input)[offset]);
  return input[offset];
 }
 __device__ __forceinline__ void StoreLocal(int slot,int z,int y,int x,float value)const{
  auto* output=p.phase[slot/6].psi[slot%6].output;
  const int offset=PsiOffset(slot,z,y,x);
  if constexpr(General)if(p.bf16_psi&(1u<<slot)){
   reinterpret_cast<__nv_bfloat16*>(output)[offset]=__float2bfloat16_rn(value);
   return;
  }
  output[offset]=value;
 }
 __device__ __forceinline__ float Coefficient(int kind,int index,int z,int y,int x)const{
  if(kind>=2){const int packed=Packed(index,z,y,x);
   if constexpr(CachedProfiles)return ReadIndependentProfileCache(profiles,index,kind,packed);
   else{const auto&psi=p.phase[index/6].psi[index%6];return kind==2?__ldg(psi.a+packed):kind==3?__ldg(psi.b+packed):__ldg(psi.inv+packed);}
  }
  if(index<3){
   if constexpr(IsotropicH)return kind==0?1.f:p.h_source;
   if constexpr(General)if(p.isotropic_h)return kind==0?1.f:p.h_source;
   const auto&v=p.phase[0];
   if constexpr(!General)return kind==0?__ldg(v.decay[index]):__ldg(v.source[index]);
   return kind==0?CompactCoefficient(v.decay[index],v.decay_strides[index],z,y,x):
                  CompactCoefficient(v.source[index],v.source_strides[index],z,y,x);}
  const int c=index-3;const auto&v=p.phase[1];
  if constexpr(General&&!IsotropicH)if(v.dense_e)return kind==0?CompactCoefficient(v.decay[c],v.decay_strides[c],z,y,x):
                                                      CompactCoefficient(v.source[c],v.source_strides[c],z,y,x);
  if(kind==0)return 1.f;
  if constexpr(UniformE)return p.common_e[c];
  const int linear=(z*(p.physical_n[1]+(c!=1))+y)*(p.physical_n[2]+(c!=0))+x;
  if constexpr(General||Layout::material==2)if(!General||p.material_layout==2){
   const int h=__ldg(reinterpret_cast<const int32_t*>(v.codes[c])+(linear>>5));
   return __ldg(v.tables[c]+(h<0?~h:h+(linear&31)));
  }
  return beamz::cuda::yee::PackedMaterialSource(v.tables[c],v.table_sizes[c],v.codes[c],linear);
 }
 __device__ __forceinline__ float Difference(int phase,int source,int axis,int z,int y,int x,float center,float adjacent)const{
  const auto&v=p.phase[phase];
  const int coordinate=axis==0?z:axis==1?y:x;
  if constexpr(!General&&!SupportSafe)
   if(phase&&(ActiveAxes<0||(ActiveAxes&(1<<axis)))&&(coordinate==0||coordinate==p.physical_n[axis]))return 0.f;
  float metric;
  if constexpr(Layout::uniform_metric)metric=v.inv;
  else metric=v.metric_kind==0?v.inv:__ldg(v.metrics[axis]+(v.metric_kind==1?0:coordinate));
  // H's forward wall guard is unreachable after the output support check.
  if constexpr(General&&!SupportSafe){if(phase){
   if(coordinate==0)return v.metallic_edges&(1<<(2*axis))?beamz::cuda::yee::ScaleYeeDifference(center,metric):0.f;
   if(coordinate==p.physical_n[axis])return v.metallic_edges&(2<<(2*axis))?beamz::cuda::yee::ScaleYeeDifference(-adjacent,metric):0.f;
  }}
  return beamz::cuda::yee::ScaleYeeDifference(phase?__fsub_rn(center,adjacent):__fsub_rn(adjacent,center),metric);
 }
 __device__ __forceinline__ float Sparse(int group,int field,int time,int z,int y,int x,float value)const{
  const int first=p.scatter_offsets[group],last=p.scatter_offsets[group+1];
  if(first==last)return value;
  const auto& f=FinalField(p,field);
  return ApplySparse(p.scatters,first,last,(z*f.y+y)*f.x+x,time,value);
 }
 __device__ __forceinline__ float PreElectric(int field,int version,int z,int y,int x,float value)const{
  if constexpr(General&&HasSources){
   if(field>=3&&(p.pre_e_mask&(1u<<(field-3)))&&(flags&beamz::cuda::moving::kMayHaveSources)
      &&Contains(FinalField(p,field),z,y,x)){
    const int time=*p.current_step+p.macro_step+step_offset+version-1;
    value=ApplyUnstagedGroup(p.pre_e[field-3],time,z,y,x,value);
    return Sparse(field-3,field,time,z,y,x,value);
   }
  }
  return value;
 }
 __device__ __forceinline__ float Finish(int field,int version,int z,int y,int x,float value)const{
  bool constrained=false;
  if constexpr(General&&!SupportSafe){
   const int metallic=p.phase[field/3].metallic_edges;
   if(metallic){
    const auto&f=FinalField(p,field);
    const int dims[3]{f.z,f.y,f.x};
    constrained=beamz::cuda::yee::PecConstrained(dims,field/3,field%3,metallic,z,y,x);
    if(constrained)value=0.f;
   }
  }
  if constexpr(HasSources||HasObserve){
   if constexpr(HasSources)if(!constrained&&(flags&beamz::cuda::moving::kMayHaveSources)){
    const auto&s=p.sources[field];
    if(s.count&&(!General||s.count==1||s.coincident)){
     // Captured coincident starts are equal; one contributor is coincident by definition.
     const int sz=z-s.starts[0],sy=y-s.starts[1],sx=x-s.starts[2];
     if(sz>=0&&sy>=0&&sx>=0&&sz<s.nz&&sy<s.ny&&sx<s.nx){
      int time=0;if constexpr(!StagedSources){time=*s.current_step+p.macro_step+step_offset+version-1;time=max(0,min(time,s.ntime-1));}
      const int spatial=(sz*s.ny+sy)*s.nx+sx,volume=s.nz*s.ny*s.nx;
      // Ordinary single-source injection contracts; coincident scatter products are rounded first.
      if(s.count==1)value=__fmaf_rn(__ldg(s.coefficients+spatial),(StagedSources?SourceWave(p,source_waves,field,0,version):__ldg(s.waveforms+time)),value);
      else{
       #pragma unroll 4
       for(int source=0;source<s.count;++source)
        value=__fadd_rn(value,__fmul_rn(__ldg(s.coefficients+source*volume+spatial),(StagedSources?SourceWave(p,source_waves,field,source,version):__ldg(s.waveforms+source*s.ntime+time))));
      }
     }
    }else if constexpr(General){
     if(s.count)value=ApplyUnstagedGroup(s,*p.current_step+p.macro_step+step_offset+version-1,z,y,x,value);
    }
    if constexpr(General)if(p.scatters){
     const int time=*p.current_step+p.macro_step+step_offset+version-1;
     value=Sparse(field+3,field,time,z,y,x,value);
    }
   }
   if constexpr(HasObserve)if(flags&beamz::cuda::moving::kMayObserve){
    Observe(p,field,version+raw_step_offset,z,y,x,value);
    if constexpr(General)if(p.volumes&&p.volume_offsets[field]!=p.volume_offsets[field+1])
      RecordVolume(p,field,step_offset,version,z,y,x,value);
   }
  }
  return value;
 }
};
template<int Phase,int Component,bool CheckSupport,class Env,class Read>
__device__ __forceinline__ float AdvanceCell(Env& env,Read read,int version,int z,int y,int x) {
 const auto with_pre=[&] __device__(int field,int step,int dz,int dy,int dx){
  const float value=read(field,step,dz,dy,dx);
  return env.PreElectric(field,version,z+dz,y+dy,x+dx,value);
 };
 return beamz::cuda::trace_native::AdvanceNativeCell<Phase,Component,CheckSupport,Env::UnitDecay>(env,with_pre,version,z,y,x);
}
// Canonical staggering was proved by Prepare before this kernel can run.
// Bounds are rectangular; imports outside every support never touch trails.
template<int Axes=-1>__device__ __forceinline__ unsigned NativeSupportMask(const Parameters&p,int z,int y,int x){
 if constexpr(Axes==1||Axes==2||Axes==4){
  constexpr int axis=Axes==1?0:Axes==2?1:2;
  constexpr unsigned upper=Axes==1?28u:Axes==2?42u:49u;
  const int coordinate=axis==0?z:axis==1?y:x,n=p.physical_n[axis];
  return unsigned(coordinate)<unsigned(n)?63u:coordinate==n?upper:0u;
 }
 const int nz=p.physical_n[0],ny=p.physical_n[1],nx=p.physical_n[2];
 if(z<0||y<0||x<0||z>nz||y>ny||x>nx)return 0;
 const unsigned iz=z<nz,iy=y<ny,ix=x<nx;
 return (iz&&iy)|((iz&&ix)<<1)|((iy&&ix)<<2)|(ix<<3)|(iy<<4)|(iz<<5);
}
__device__ __forceinline__ int EdgeKey(int z,int y,int x){
 if(z==BZ-1 && y==BY-1)return x;
 if(z==BZ-1 && x==BX-1)return BX+y;
 if(y==BY-1 && x==BX-1)return BX+BY+z;
 return -1;
}
__device__ __forceinline__ size_t PointBase(const size_t*bases,int neighbor,int z,int y,int x,int surface){
 const int edge=EdgeKey(z,y,x);
 return edge>=0?bases[8+neighbor]+edge:bases[z==BZ-1?1:y==BY-1?2:4]+surface;
}
__device__ __forceinline__ void StageTrailBases(const Parameters&p,size_t*bases,int encoded,int bz,int by,int bx,int macro,int rank,uint64_t epoch){
 using namespace beamz::cuda::moving;
 static_assert(BX+BY+BZ<=kEdgePitch && SurfacePoints%kEdgePitch==0);
 const int bits=threadIdx.x;
 if(bits>=8)return;
 const size_t stride=size_t(6*p.depth-3)*SurfacePoints;
 if(bits==1||bits==2||bits==4){
  const int axis=bits==1?0:bits==2?1:2;
  bases[bits]=size_t(p.face_lines[FaceLine(p.nz,p.ny,p.nx,bz,by,bx,macro,axis)])*stride;
 }
 const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
 size_t edge=size_t(-2);
 if(bz>=dz&&by>=dy&&bx>=dx){
  const int producer=encoded-dz*p.nx*p.ny-dy*p.nx-dx,slot=p.slots[producer];
  if(slot>=0){
   const int owner=((rank-dz-dy-dx)&3)*p.frontier+slot;
   if(bits && p.owner_tags[owner]!=(epoch|uint32_t(producer))){atomicOr(p.bad,1);edge=size_t(-1);}
   else edge=(size_t(p.frontier)+owner/kEdgeOwnersPerRow)*stride+(owner%kEdgeOwnersPerRow)*kEdgePitch;
  }
 }
 bases[8+bits]=edge;
}
struct FieldPacket {float hx,hy,hz,ex,ey,ez;};
template<bool Fast,bool ReuseTerminal=true,int Axes=-1>
__device__ __forceinline__ FieldPacket ImportPacket(const Parameters&p,const size_t*bases,int version,
 int oz,int oy,int ox,int dz,int dy,int dx){
 unsigned mask=63;
 if constexpr(!Fast)mask=NativeSupportMask<Axes>(p,oz-1+dz-version,oy-1+dy-version,ox-1+dx-version);
 if(!mask)return {};
 const int neighbor=(dz==0)|((dy==0)<<1)|((dx==0)<<2);
 const int surface=Surface(dz?dz-1:BZ-1,dy?dy-1:BY-1,dx?dx-1:BX-1);
 const size_t base=bases[8+neighbor];
 if(base==size_t(-2))return {};
 if(!neighbor||surface<0||base==static_cast<size_t>(-1)){
  atomicOr(p.bad,2);const float bad=__int_as_float(0x7fffffff);
  return {mask&1?bad:0.f,mask&2?bad:0.f,mask&4?bad:0.f,mask&8?bad:0.f,mask&16?bad:0.f,mask&32?bad:0.f};
 }
 if constexpr(ReuseTerminal){if(version==p.depth)mask&=7u;}
 const float*point=p.trails+PointBase(bases,neighbor,dz?dz-1:BZ-1,dy?dy-1:BY-1,dx?dx-1:BX-1,surface)+static_cast<size_t>(version-1)*6*SurfacePoints;
 return {mask&1?__ldcg(point+0*SurfacePoints):0.f,mask&2?__ldcg(point+1*SurfacePoints):0.f,
         mask&4?__ldcg(point+2*SurfacePoints):0.f,mask&8?__ldcg(point+3*SurfacePoints):0.f,
         mask&16?__ldcg(point+4*SurfacePoints):0.f,mask&32?__ldcg(point+5*SurfacePoints):0.f};
}
__global__ void BeginEpoch(uint32_t*epoch,int32_t*bad){if(!threadIdx.x){if(*epoch==0xffffffffu){atomicOr(bad,4);return;}++*epoch;}}

template<bool Safe,int T,bool ReuseTerminal=true,int Axes=-1>
__device__ __forceinline__ void StageRegisterLow(const Parameters&p,float*stage,const size_t*bases,int n,int oz,int oy,int ox){
 for(int i=threadIdx.x;i<LowXPoints;i+=T){const int dz=1+i/BY,dy=1+i%BY;
  const auto v=ImportPacket<Safe,ReuseTerminal,Axes>(p,bases,n,oz,oy,ox,dz,dy,0);
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

template<class Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool CachedProfiles=false>
__device__ __forceinline__ void UniformBody(const Parameters&p,const int32_t*tiles){
 constexpr int T=(Fast||((Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512,Slots=PAD/T,ZStep=T/512;
 constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);
 static_assert(!CachedProfiles||(!Fast&&IsotropicH&&Policy==1&&UniformE),"Profile staging only certified NoIO1024");
 extern __shared__ float frame[];size_t*bases=reinterpret_cast<size_t*>(frame+3*PAD);
 volatile int*phase_step=reinterpret_cast<volatile int*>(bases+16);
 volatile int*window=phase_step+1;
 float*low_stage=frame+3*PAD+16*sizeof(size_t)/sizeof(float)+3;
 float*profiles=CachedProfiles?frame+SharedBytes/sizeof(float):nullptr;
 if constexpr(CachedProfiles)StageIndependentCpmlProfiles<ActiveAxes>(p,profiles);
 const int encoded=tiles[blockIdx.x],macro_index=MacroIndex(p,encoded),tile=PhysicalTileID(p,encoded),bz=tile/(p.nx*p.ny),by=(tile/p.nx)%p.ny,bx=tile%p.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX,rank=bz+by+bx+macro_index*p.macro_level_stride;
 const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[encoded];p.owner_tags[owner]=epoch|uint32_t(encoded);}
 __syncthreads();
 if constexpr(!Fast){
  if(!threadIdx.x){
   const int mz=p.phase[0].own[0].z+1,my=p.phase[0].own[0].y+1,mx=p.phase[0].own[1].x+1;
   window[0]=max(1,max(oz-mz+1,max(oy-my+1,ox-mx+1)));
   window[1]=min(p.depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));
  }
  __syncthreads();
  // The owner tag remains published even when no component can consume trails.
  if(window[0]>window[1])return;
 }
 StageTrailBases(p,bases,encoded,bz,by,bx,macro_index,rank,epoch);
 const int sx=threadIdx.x&31,sy=(threadIdx.x/32)&15,zbase=threadIdx.x/512;
 float h[3][Slots],e[3][Slots];
 // A late start must see the neighbor base table before reading trails.
 if constexpr(!Fast)__syncthreads();
 if constexpr(Fast){
 #pragma unroll
 for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+(zs==7?-1:zs),y=oy+(sy==15?-1:sy),x=ox+(sx==31?-1:sx);
  #pragma unroll
  for(int c=0;c<3;++c){h[c][j]=Initial<ActiveAxes>(p,c,z,y,x,macro_index);e[c][j]=Initial<ActiveAxes>(p,c+3,z,y,x,macro_index);frame[c*PAD+Slot(zs,sy,sx)]=e[c][j];}
 }
 __syncthreads();
 }else{
  const int first=window[0],prev=first-1;
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;
   FieldPacket v{};
   if(first==1){const int z=oz+(zs==7?-1:zs),y=oy+(sy==15?-1:sy),x=ox+(sx==31?-1:sx);
    v={Initial<ActiveAxes>(p,0,z,y,x,macro_index),Initial<ActiveAxes>(p,1,z,y,x,macro_index),Initial<ActiveAxes>(p,2,z,y,x,macro_index),Initial<ActiveAxes>(p,3,z,y,x,macro_index),Initial<ActiveAxes>(p,4,z,y,x,macro_index),Initial<ActiveAxes>(p,5,z,y,x,macro_index)};
   }else{
    const int hz=(-prev-1)&7,hy=(-prev-1)&15,hx=(-prev-1)&31;
    // Previous owned volume is wholly invalid; only the low ghost union may live.
    if(zs==hz||sy==hy||sx==hx)v=ImportPacket<false,true,ActiveAxes>(p,bases,prev,oz,oy,ox,(zs-hz)&7,(sy-hy)&15,(sx-hx)&31);
   }
   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;
   e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;
   frame[Slot(zs,sy,sx)]=v.ex;frame[PAD+Slot(zs,sy,sx)]=v.ey;frame[2*PAD+Slot(zs,sy,sx)]=v.ez;
  }

  __syncthreads();
 }
 const uint8_t flags=Policy==0?0:Policy==2?1:Policy==3?16:Policy==4?17:p.flags[tile];
 Environment<Layout,Fast,IsotropicH,Policy,UniformE,ActiveAxes,CachedProfiles>env{p,flags,profiles};env.step_offset=macro_index*p.depth;env.raw_step_offset=(macro_index%p.raw_slots)*p.depth;
 #pragma unroll 1
 for(int n=(Fast?1:window[0]);n<=(Fast?p.depth:window[1]);++n){
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
      if(c==0)value=AdvanceCell<0,0,false>(env,read,n,z,y,x);
      else if(c==1)value=AdvanceCell<0,1,false>(env,read,n,z,y,x);
      else value=AdvanceCell<0,2,false>(env,read,n,z,y,x);
     }h[c][j]=value;}
    }
   }
  }
  const int hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31,low=-n-1;
  const int dy=(sy-hy)&15,dx=(sx-hx)&31;
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;if(zs==hz||sy==hy){const int dz=(zs-hz)&7;
   const auto v=ImportPacket<SupportSafe,true,ActiveAxes>(p,bases,n,oz,oy,ox,dz,dy,dx);
   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;
  }}
  StageRegisterLow<SupportSafe,T,true,ActiveAxes>(p,low_stage,bases,n,oz,oy,ox);
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
      if(c==0)value=AdvanceCell<1,0,false>(env,read,n,z,y,x);
      else if(c==1)value=AdvanceCell<1,1,false>(env,read,n,z,y,x);
      else value=AdvanceCell<1,2,false>(env,read,n,z,y,x);
     }e[c][j]=value;}
    }
   }
  }
  if(n<(Fast?p.depth:window[1])){
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
   if(surface>=0){const size_t dest=PointBase(bases,0,dz-1,dy-1,dx-1,surface);
    #pragma unroll
    for(int c=0;c<3;++c){
    p.trails[dest+((n-1)*6+c)*SurfacePoints]=h[c][j];if(n<p.depth)p.trails[dest+((n-1)*6+c+3)*SurfacePoints]=e[c][j];}
   }
   if(n==p.depth){
    #pragma unroll
    for(int c=0;c<3;++c){
    const auto&hf=MacroFinalField(p,c,macro_index);const auto&ef=MacroFinalField(p,c+3,macro_index);
     if(Contains<ActiveAxes>(hf,z,y,x))hf.data[z*hf.plane+y*hf.row+x]=h[c][j];
     if(Contains<ActiveAxes>(ef,z,y,x))ef.data[z*ef.plane+y*ef.row+x]=e[c][j];
    }
   }
  }}
 }

}


template<class Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>
__global__ __launch_bounds__((Fast?1024:512),1) void Asymmetric(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 if constexpr(Policy==1&&IsotropicH){
  // Host compact range contains only noncertified NoIO IDs.
  UniformBody<Layout,false,true,1,false>(p,tiles);
 }else if constexpr(Policy==2&&IsotropicH){
  // Noncertified observer subrange; canonical spatial E lookup remains.
  UniformBody<Layout,false,true,2,false>(p,tiles);
 }else if constexpr(Fast||(Policy==1&&IsotropicH)){
  // CTA-uniform immutable flag; barriers remain converged in either complete body.
  if(p.flags[PhysicalTileID(p,tiles[blockIdx.x])]&64)UniformBody<Layout,Fast,IsotropicH,Policy,true>(p,tiles);
  else UniformBody<Layout,Fast,IsotropicH,Policy,false>(p,tiles);
 }else UniformBody<Layout,Fast,IsotropicH,Policy,false>(p,tiles);
}

template<class Layout=KernelFacts<1,true>>
__global__ __launch_bounds__(1024,1) void AsymmetricNoIOUniform1024(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 // Original full certified NoIO grid; selector uniform before any barrier.
 switch(p.flags[PhysicalTileID(p,tiles[blockIdx.x])]&14){
  case 2:UniformBody<Layout,false,true,1,true,1,true>(p,tiles);return;
  case 4:UniformBody<Layout,false,true,1,true,2,true>(p,tiles);return;
  case 8:UniformBody<Layout,false,true,1,true,4,true>(p,tiles);return;
  default:UniformBody<Layout,false,true,1,true,-1,true>(p,tiles);return;
 }
}

template<class Layout=KernelFacts<1,true>>
__global__ __launch_bounds__(1024,1) void AsymmetricObserverUniform1024(
 const __grid_constant__ Parameters p,const int32_t*tiles){
 UniformBody<Layout,false,true,2,true>(p,tiles);
}

// Private source-only storage schedule; source-bound to the proven shared-Ey body.
template<bool Safe,int T>
__device__ __forceinline__ void StageSourcePendingH(const Parameters&p,float*stage,const size_t*bases,int n,int oz,int oy,int ox){
 for(int i=threadIdx.x;i<LowXPoints;i+=T){const int dz=1+i/BY,dy=1+i%BY;
  const auto v=ImportPacket<Safe>(p,bases,n,oz,oy,ox,dz,dy,0);
  stage[i]=v.hx;stage[LowXPoints+i]=v.hy;stage[2*LowXPoints+i]=v.hz;
 }
}
// Two-plane publication groups; use after the reviewed base SourcePendingBody helpers.
template<class Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool CachedProfiles=false,bool StagedSources=false,bool General=false>
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
 volatile int*phase_step=reinterpret_cast<volatile int*>(bases+16);
 volatile int*window=phase_step+1;
 float*low_stage=reinterpret_cast<float*>(bases+16)+3;
 float*profiles=CachedProfiles?low_stage+3*LowXPoints:nullptr;
 if constexpr(CachedProfiles)StageIndependentCpmlProfiles(p,profiles);
 const int encoded=tiles[blockIdx.x],macro_index=MacroIndex(p,encoded),tile=PhysicalTileID(p,encoded),bz=tile/(p.nx*p.ny),by=(tile/p.nx)%p.ny,bx=tile%p.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX,rank=bz+by+bx+macro_index*p.macro_level_stride;
 if constexpr(StagedSources)if(p.flags[tile]&beamz::cuda::moving::kMayHaveSources)StageSourceWaves(p,macro_index*p.depth,frame+58744/sizeof(float));
 const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[encoded];p.owner_tags[owner]=epoch|uint32_t(encoded);}
 __syncthreads();
 if constexpr(!Fast){
  if(!threadIdx.x){const int mz=p.phase[0].own[0].z+1,my=p.phase[0].own[0].y+1,mx=p.phase[0].own[1].x+1;
   window[0]=max(1,max(oz-mz+1,max(oy-my+1,ox-mx+1)));
   window[1]=min(p.depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));
  }
  __syncthreads();if(window[0]>window[1])return;
 }
 StageTrailBases(p,bases,encoded,bz,by,bx,macro_index,rank,epoch);
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
    if(zs==hz||ys==hy||sx==hx)v=ImportPacket<false>(p,bases,prev,oz,oy,ox,(zs-hz)&(CZ-1),(ys-hy)&(CY-1),(sx-hx)&31);
   }
   h[0][si]=v.hx;h[1][si]=v.hy;h[2][si]=v.hz;set_ey(zs,ys,si,v.ey);
   frame[Slot(zs,ys,sx)]=v.ex;frame[PAD+Slot(zs,ys,sx)]=v.ez;
  }
 }
 if(!threadIdx.x)*phase_step=first;__syncthreads();
 const uint8_t flags=Policy==0?0:Policy==2?1:Policy==3?16:Policy==4?17:p.flags[tile];
 Environment<Layout,Fast,IsotropicH,Policy,UniformE,ActiveAxes,CachedProfiles,StagedSources,General>env{p,flags,profiles};env.source_waves=frame+58744/sizeof(float);env.step_offset=macro_index*p.depth;env.raw_step_offset=(macro_index%p.raw_slots)*p.depth;
 #pragma unroll 1
 for(int n=(Fast?1:window[0]);n<=(Fast?p.depth:window[1]);++n){
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
       if(c==0)value=AdvanceCell<0,0,false>(env,read,n,z,y,x);
       else if(c==1)value=AdvanceCell<0,1,false>(env,read,n,z,y,x);
       else value=AdvanceCell<0,2,false>(env,read,n,z,y,x);
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
    const auto v=ImportPacket<SupportSafe>(p,bases,n,oz,oy,ox,(zs-hz)&(CZ-1),dy,dx);h[0][si]=v.hx;h[1][si]=v.hy;h[2][si]=v.hz;
   }}
  }
  StageSourcePendingH<SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);
  __syncthreads(); // R0: retire all old E readers; publish low-X H imports.
  #pragma unroll
  for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),dy=(ys-hy)&(CY-1),dx=(sx-hx)&31;
   #pragma unroll
   for(int j=0;j<LocalZ;++j){const int zs=zbase+j,si=gy*LocalZ+j,dz=(zs-hz)&(CZ-1);
    if(sx==hx&&ys!=hy&&zs!=hz){const int key=(dz-1)*BY+dy-1;h[0][si]=low_stage[key];h[1][si]=low_stage[LowXPoints+key];h[2][si]=low_stage[2*LowXPoints+key];}
    if(zs==hz||ys==hy||sx==hx){const auto v=ImportPacket<SupportSafe>(p,bases,n,oz,oy,ox,dz,dy,dx);frame[Slot(zs,ys,sx)]=v.ex;frame[PAD+Slot(zs,ys,sx)]=v.ez;set_ey(zs,ys,si,v.ey);}
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
     ex0[gy]=env.Valid(3,z,y,x)?AdvanceCell<1,0,false>(env,read,n,z,y,x):0.f;
     const float new_ey=env.Valid(4,z,y,x)?AdvanceCell<1,1,false>(env,read,n,z,y,x):0.f;set_ey(zs,ys,si,new_ey);
     ez0[gy]=env.Valid(5,z,y,x)?AdvanceCell<1,2,false>(env,read,n,z,y,x):0.f;
    }

    if(zs!=hz&&ys!=hy&&sx!=hx){const int surface=Surface(dz-1,dy-1,dx-1);
     if(surface>=0){const size_t dest=PointBase(bases,0,dz-1,dy-1,dx-1,surface);
    #pragma unroll
    for(int c=0;c<3;++c){const float ev=c==0?ex0[gy]:c==1?get_ey(zs,ys,sx,si):ez0[gy];
      p.trails[dest+((n-1)*6+c)*SurfacePoints]=h[c][si];if(n<p.depth)p.trails[dest+((n-1)*6+c+3)*SurfacePoints]=ev;}
   }
   if(n==p.depth){
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
     ex1[gy]=env.Valid(3,z,y,x)?AdvanceCell<1,0,false>(env,read,n,z,y,x):0.f;
     const float new_ey=env.Valid(4,z,y,x)?AdvanceCell<1,1,false>(env,read,n,z,y,x):0.f;set_ey(zs,ys,si,new_ey);
     ez1[gy]=env.Valid(5,z,y,x)?AdvanceCell<1,2,false>(env,read,n,z,y,x):0.f;
    }

    if(zs!=hz&&ys!=hy&&sx!=hx){const int surface=Surface(dz-1,dy-1,dx-1);
     if(surface>=0){const size_t dest=PointBase(bases,0,dz-1,dy-1,dx-1,surface);
    #pragma unroll
    for(int c=0;c<3;++c){const float ev=c==0?ex1[gy]:c==1?get_ey(zs,ys,sx,si):ez1[gy];
      p.trails[dest+((n-1)*6+c)*SurfacePoints]=h[c][si];if(n<p.depth)p.trails[dest+((n-1)*6+c+3)*SurfacePoints]=ev;}
   }
   if(n==p.depth){
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


constexpr size_t SourcePendingSharedBytes=(3ULL*PAD+2048)*sizeof(float)+16*sizeof(size_t)+3*sizeof(int)+3ULL*LowXPoints*sizeof(float);
static_assert(SourcePendingSharedBytes==58744,"Shared paired-plane source schedule");
template<class Layout,bool IsotropicH,int Policy,int Axes=-1,bool StagedSources=false,bool General=false>
__device__ __forceinline__ void TemporalEffectBody(const Parameters&p,const int32_t*ids){
 if(p.flags[PhysicalTileID(p,ids[blockIdx.x])]&64)
  SourcePendingPairedBody<Layout,false,IsotropicH,Policy,true,Axes,false,StagedSources,General>(p,ids);
 else SourcePendingPairedBody<Layout,false,IsotropicH,Policy,false,Axes,false,StagedSources,General>(p,ids);
}
template<class Layout,bool IsotropicH,int Policy,bool StagedSources=false,bool General=false>
__global__ __launch_bounds__(1024,1) void TemporalEffects(
 const __grid_constant__ Parameters p,const int32_t*ids){
 static_assert(Policy>=1&&Policy<=5,"Shared boundary/observer/source effect classes");
 if constexpr(Policy==5&&!General){
  // A trajectory's exact CPML axis effects are already proved by the planner.
  // Specialize the common single-axis case without separating launch cohorts.
  switch(p.flags[PhysicalTileID(p,ids[blockIdx.x])]&14){
   case 2:TemporalEffectBody<Layout,IsotropicH,Policy,1,StagedSources>(p,ids);return;
   case 4:TemporalEffectBody<Layout,IsotropicH,Policy,2,StagedSources>(p,ids);return;
   case 8:TemporalEffectBody<Layout,IsotropicH,Policy,4,StagedSources>(p,ids);return;
   default:TemporalEffectBody<Layout,IsotropicH,Policy,-1,StagedSources>(p,ids);return;
  }
 }else TemporalEffectBody<Layout,IsotropicH,Policy,-1,StagedSources,General>(p,ids);
}

inline bool MakeSource(const BeamzSourceGroupLaunch&g,Source*out){
 *out={};if(g.coefficients.dims[0]==0)return true;
 if(g.coefficients.rank!=4||g.coefficients.element_type!=kBeamzF32||
    g.waveforms.rank!=2||g.waveforms.element_type!=kBeamzF32||
    g.starts.element_type!=kBeamzS32||g.current_step.element_type!=kBeamzS32||
    g.coefficients.dims[0]<1||g.waveforms.dims[0]!=g.coefficients.dims[0])return false;
 for(int a=0;a<4;++a)if(g.coefficients.dims[a]<1||g.coefficients.dims[a]>INT32_MAX)return false;
 if(g.waveforms.dims[1]<1||g.waveforms.dims[1]>INT32_MAX)return false;
 out->coefficients=static_cast<const float*>(g.coefficients.data);out->waveforms=static_cast<const float*>(g.waveforms.data);
 out->starts=static_cast<const int32_t*>(g.starts.data);out->current_step=static_cast<const int32_t*>(g.current_step.data);
 out->count=g.coefficients.dims[0];out->nz=g.coefficients.dims[1];out->ny=g.coefficients.dims[2];out->nx=g.coefficients.dims[3];out->ntime=g.waveforms.dims[1];
 out->coincident=g.coincident;out->disjoint=g.disjoint;return true;
}
inline bool Prepare(const BeamzProgramLaunch&program,const BeamzBuffer*final,
 const BeamzNativeMovingWorkspace&w,int macro,Parameters*out,int*layout){
 using namespace beamz::cuda::moving;
 const bool tensor=program.e_ab.tensor!=nullptr;
 const std::array<int,3> sizes=tensor?kTensorOwned:std::array<int,3>{BZ,BY,BX};
 if(!final||!out||!layout||macro<0||!program.source_groups||program.source_group_count!=9||
    program.h_ab.tensor||w.epochs!=4||!std::equal(sizes.begin(),sizes.end(),w.owned)||(tensor&&w.depth!=4)||
    !ValidDepth(w.depth)||w.max_frontier<1||w.rank_count<1||w.tile_count<1||
    !w.rank_offsets_host||!w.bulk_counts_host||!w.tile_flags_device||!w.bad||(!w.rolling&&(!w.tile_ids_device||!w.slots_device||!w.trails||!w.owner_tags||!w.epoch)))return false;
 auto h=program.h_ab,e=program.e_ab;
 for(int c=0;c<3;++c){h.outputs[c]=final[c];e.outputs[c]=final[3+c];}
 if(w.raw_slots<1||w.raw_slots>16)return false;
 *out={};int h_layout=0;out->raw_slots=w.raw_slots;
 const bool dense_e=e.inputs[9].element_type==kBeamzF32;
 if(!MakeMstepCompactEdge(h,&out->phase[0],&h_layout,false,w.cpml_widths.data())||
    !MakeMstepCompactEdge(e,&out->phase[1],layout,dense_e,w.cpml_widths.data()+6))return false;
 out->material_layout=*layout;out->depth=w.depth;
 out->isotropic_h=w.data_flags&kLosslessIsotropicH;
 if(dense_e&&(w.data_flags&kLosslessIsotropicH))return false;
 if(h.uniform_cpml_thickness!=12||e.uniform_cpml_thickness!=12||h.metallic_edges||e.metallic_edges)
   out->general_effects|=WideBoundary;
 if(dense_e||std::any_of(h.inputs+6,h.inputs+12,[](const auto& coefficient){return coefficient.rank!=0;}))
   out->general_effects|=DenseMaterial;
 for(int term=0;term<h.nterms;++term){
  if(h.inputs[31+term].data!=h.outputs[3+term].data||e.inputs[31+term].data!=e.outputs[3+term].data)return false;
  if(h.inputs[31+term].element_type==kBeamzBF16)out->bf16_psi|=1u<<term;
  if(e.inputs[31+term].element_type==kBeamzBF16)out->bf16_psi|=1u<<(6+term);
 }
 if(out->bf16_psi)out->general_effects|=WideBoundary;
 int minimum[3]={INT32_MAX,INT32_MAX,INT32_MAX},maximum[3]={0,0,0};
 for(int field=0;field<6;++field){const auto&f=FinalField(*out,field);const int dim[3]={f.z,f.y,f.x};for(int a=0;a<3;++a){minimum[a]=std::min(minimum[a],dim[a]);maximum[a]=std::max(maximum[a],dim[a]);}
  for(int input=0;input<6;++input)if(field!=input&&f.data==InitialField(*out,input).data)return false; // Only same-component aliasing is proved.
 }
 for(int field=0;field<6;++field){const auto&f=FinalField(*out,field);const int dz=field==2||field==3||field==4,dy=field==1||field==3||field==5,dx=field==0||field==4||field==5;
  if(f.z!=minimum[0]+dz||f.y!=minimum[1]+dy||f.x!=minimum[2]+dx)return false;
 }
 for(int axis=0;axis<3;++axis)out->physical_n[axis]=minimum[axis];
 int64_t tiles=1;
 const int reach=(tensor?2:1)*w.depth;
 for(int a=0;a<3;++a){if(w.counts[a]!=(int64_t(maximum[a])+reach+sizes[a]-1)/sizes[a])return false;tiles*=w.counts[a];}
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
 const size_t owners=beamz::cuda::moving::TrailOwners(w.max_frontier,tensor);
 const size_t rows=(tensor?9:6)*w.depth-3, surface=tensor?kTensorTrailPoints:SurfacePoints;
 if(!w.rolling&&(owners>std::numeric_limits<size_t>::max()/rows/surface||w.trail_floats<owners*rows*surface))return false;
 for(int field=0;field<6;++field)if(!MakeSource(program.source_groups[3+field],&out->sources[field]))return false;
 out->current_step=static_cast<const int32_t*>(program.source_groups[0].current_step.data);
 out->scatters=w.scatters;
 std::copy(w.scatter_offsets.begin(),w.scatter_offsets.end(),out->scatter_offsets);
 out->volumes=w.volumes;
 std::copy(w.volume_offsets.begin(),w.volume_offsets.end(),out->volume_offsets);
 for(int c=0;c<3;++c){
  if(!MakeSource(program.source_groups[c],&out->pre_e[c]))return false;
  if(out->pre_e[c].count||w.scatter_offsets[c]!=w.scatter_offsets[c+1])out->pre_e_mask|=1u<<c;
 }
 int64_t wave_count=0;bool needs_general=out->pre_e_mask||out->scatters;
 for(int field=0;field<6;++field){
  const auto& source=out->sources[field];
  out->source_wave_offsets[field]=wave_count<=(65536-58744)/(w.depth*sizeof(float))?int(wave_count):0;
  wave_count+=source.count;needs_general|=source.count>1&&!source.coincident;
 }
 out->source_wave_count=!needs_general&&wave_count<=(65536-58744)/(w.depth*sizeof(float))?int(wave_count):-1;
 if(needs_general)out->general_effects|=GeneralSource;
 if(out->volumes)out->general_effects|=VolumeOutput;

 uint32_t actual_source_presence=0;for(int field=0;field<6;++field)if(out->sources[field].count>0)actual_source_presence|=1u<<field;
 if(actual_source_presence!=w.source_presence_mask)return false;
 if(w.observation.key_count<0||(w.observation.key_count&&!w.observation.values))return false;
#if !BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION
 if(w.observation.key_count&&(!w.observation.keys||!w.observation.publication_words))return false;
#endif
#if BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION
 if(w.observation.key_count){
  if(!w.observation_tiles||!w.observation_leaf)return false;
  const int full[]={w.observation.mask_nz,(w.observation.mask_ny+7)/8,(w.observation.mask_nx+15)/16};
  for(int a=0;a<3;++a)if(w.observation_grid[a]<1||w.observation_origin[a]<0||w.observation_origin[a]>full[a]-w.observation_grid[a])return false;
 }
#endif
 out->observation_tiles=w.observation_tiles;out->observation_leaf=w.observation_leaf;
 std::copy_n(w.observation_origin,3,out->observation_origin);out->observation_nz=w.observation_grid[0];out->observation_ny=w.observation_grid[1];out->observation_nx=w.observation_grid[2];
 out->observation=w.observation;out->slots=w.slots_device;out->face_lines=w.slots_device?w.slots_device+w.tile_count:nullptr;out->flags=w.tile_flags_device;out->trails=w.trails;out->owner_tags=w.owner_tags;out->epoch=w.epoch;out->bad=w.bad;
 out->nz=w.counts[0];out->ny=w.counts[1];out->nx=w.counts[2];out->frontier=w.max_frontier;out->macro_step=macro;out->macro_level_stride=w.slab_macro_stride;out->h_source=w.h_source[0];std::copy(w.common_e,w.common_e+3,out->common_e);return true;
}
template<class Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>
inline cudaError_t Launch(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count){
 if(!count)return cudaSuccess;
 const auto kernel=Asymmetric<Layout,Fast,IsotropicH,Policy>;
 auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
 Asymmetric<Layout,Fast,IsotropicH,Policy><<<count,(Fast?1024:512),SharedBytes,stream>>>(p,ids);
 status=cudaGetLastError();if(status!=cudaSuccess)return status;
 return cudaSuccess;
}
template<class Layout,bool IsotropicH,int Policy>
inline cudaError_t LaunchSources(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count);
template<class Layout,bool IsotropicH>
inline cudaError_t LaunchNoIO(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count,int uniform_count){
 if(uniform_count<0||uniform_count>count)return cudaErrorInvalidValue;
 if constexpr(IsotropicH){
  // Existing policy branch/join; both subranges are disjoint rank-local IDs.
  auto status=LaunchSources<Layout,IsotropicH,1>(stream,p,ids+uniform_count,count-uniform_count);
  if(status!=cudaSuccess||!uniform_count)return status;
  const auto kernel=AsymmetricNoIOUniform1024<Layout>;
  status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,ProfileCachedSharedBytes);if(status!=cudaSuccess)return status;
  AsymmetricNoIOUniform1024<Layout><<<uniform_count,1024,ProfileCachedSharedBytes,stream>>>(p,ids);
  return cudaGetLastError();
 }else{
  if(uniform_count)return cudaErrorInvalidValue;
  return LaunchSources<Layout,IsotropicH,1>(stream,p,ids,count);
 }
}
template<class Layout,bool IsotropicH>
inline cudaError_t LaunchObservers(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count,int uniform_count){
 if(uniform_count<0||uniform_count>count)return cudaErrorInvalidValue;
 if constexpr(IsotropicH){
  auto status=LaunchSources<Layout,IsotropicH,2>(stream,p,ids+uniform_count,count-uniform_count);
  if(status!=cudaSuccess||!uniform_count)return status;
  const auto kernel=AsymmetricObserverUniform1024<Layout>;
  status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes);if(status!=cudaSuccess)return status;
  AsymmetricObserverUniform1024<Layout><<<uniform_count,1024,SharedBytes,stream>>>(p,ids);
  return cudaGetLastError();
 }else{
  if(uniform_count)return cudaErrorInvalidValue;
  return LaunchSources<Layout,IsotropicH,2>(stream,p,ids,count);
 }
}
template<class Layout,bool IsotropicH,int Policy,bool StagedSources,bool General=false>
inline cudaError_t LaunchSourceWaves(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count){
 if(!count)return cudaSuccess;
 const auto kernel=TemporalEffects<Layout,IsotropicH,Policy,StagedSources,General>;
 auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(SourcePendingSharedBytes+(StagedSources?size_t(p.source_wave_count)*p.depth*sizeof(float):0)));
 if(status!=cudaSuccess)return status;
 TemporalEffects<Layout,IsotropicH,Policy,StagedSources,General><<<count,1024,(SourcePendingSharedBytes+(StagedSources?size_t(p.source_wave_count)*p.depth*sizeof(float):0)),stream>>>(p,ids);
 return cudaGetLastError();
}
template<class Layout,bool IsotropicH,int Policy>
inline cudaError_t LaunchSources(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count){
 if constexpr(Policy>=3){if(p.source_wave_count>0)return LaunchSourceWaves<Layout,IsotropicH,Policy,true>(stream,p,ids,count);}
 return LaunchSourceWaves<Layout,IsotropicH,Policy,false>(stream,p,ids,count);
}
template<int Policy=5>
inline cudaError_t LaunchGeneral(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count){
 // The planner proves whether this rank range can omit sources and observation.
 return LaunchSourceWaves<KernelFacts<0,false>,false,Policy,false,true>(stream,p,ids,count);
}
template<class Layout,bool IsotropicH>
inline cudaError_t LaunchRanks(cudaStream_t stream,const Parameters&p,const BeamzNativeMovingWorkspace&w){
#if BEAMZ_CUDA_NATIVE_MOVING_POLICIES
 constexpr unsigned effects_by_policy[]{DenseMaterial,DenseMaterial|WideBoundary,
   DenseMaterial|VolumeOutput,DenseMaterial|GeneralSource,DenseMaterial|GeneralSource|VolumeOutput,~0u};
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
   if(p.general_effects&effects_by_policy[policy]){
    status=policy==1?LaunchGeneral<1>(branch,p,w.tile_ids_device+start,count):
                     LaunchGeneral<>(branch,p,w.tile_ids_device+start,count);
   }else switch(policy){
    case 0:status=Launch<Layout,true,IsotropicH,0>(branch,p,w.tile_ids_device+start,count);break;
    case 1:status=LaunchNoIO<Layout,IsotropicH>(branch,p,w.tile_ids_device+start,count,w.uniform_noio_counts_host[rank]);break;
    case 2:status=LaunchObservers<Layout,IsotropicH>(branch,p,w.tile_ids_device+start,count,w.uniform_observer_counts_host[rank]);break;
    case 3:status=LaunchSources<Layout,IsotropicH,3>(branch,p,w.tile_ids_device+start,count);break;
    case 4:status=LaunchSources<Layout,IsotropicH,4>(branch,p,w.tile_ids_device+start,count);break;
    case 5:status=LaunchSources<Layout,IsotropicH,5>(branch,p,w.tile_ids_device+start,count);break;
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
  const int classic_bulk=(p.general_effects&DenseMaterial)?0:bulk;
  auto status=Launch<Layout,true,IsotropicH>(stream,p,w.tile_ids_device+start,classic_bulk);if(status!=cudaSuccess)return status;
  if(p.general_effects)status=LaunchGeneral<>(stream,p,w.tile_ids_device+start+classic_bulk,count-classic_bulk);
  else status=LaunchSources<Layout,IsotropicH,5>(stream,p,w.tile_ids_device+start+bulk,count-bulk);
  if(status!=cudaSuccess)return status;
 }
 return cudaSuccess;
#endif
}

template<class Layout>
inline cudaError_t DispatchClass(cudaStream_t stream,const Parameters&p,const BeamzNativeMovingWorkspace&w){
 if(w.data_flags&beamz::cuda::moving::kLosslessIsotropicH)return LaunchRanks<Layout,true>(stream,p,w);
 return LaunchRanks<Layout,false>(stream,p,w);
}
template<int Material>
inline cudaError_t DispatchMetric(cudaStream_t stream,const Parameters&p,const BeamzNativeMovingWorkspace&w){
 if(p.phase[0].metric_kind==0&&p.phase[1].metric_kind==0)return DispatchClass<KernelFacts<Material,true>>(stream,p,w);
 return DispatchClass<KernelFacts<Material,false>>(stream,p,w);
}
#include "temporal/stencils/tensor_wavefront.cuh"
} // namespace beamz::cuda::moving_native

#include "rolling.cuh"

int BeamzPrepareNativeRolling(void*opaque,const BeamzProgramLaunch&program,
 const BeamzBuffer*final,const BeamzNativeMovingWorkspace&w){
 using namespace beamz::cuda;
 moving_native::Parameters p;int layout=0;
 if(!w.rolling||!w.rolling_masks||!moving_native::Prepare(program,final,w,0,&p,&layout)||
    program.e_ab.tensor||p.general_effects&(moving_native::WideBoundary|moving_native::DenseMaterial)||
    layout!=2||!p.isotropic_h||p.bf16_psi||p.phase[0].metric_kind||p.phase[1].metric_kind||
    w.depth%8)return cudaErrorNotSupported;
 for(const auto& band:w.rolling->bands){
  const auto status=rolling::Prepare(static_cast<cudaStream_t>(opaque),p,band,w.rolling_masks);
  if(status!=cudaSuccess)return status;
 }
 return cudaSuccess;
}

int BeamzEnqueueNativeAsymmetric(void*opaque_stream,const BeamzProgramLaunch&program,
 const BeamzBuffer*final_fields,int macro_step,const BeamzNativeMovingWorkspace&w){
 using namespace beamz::cuda::moving_native;
 const auto validation=BeamzValidateNativeProgram(program);if(validation)return validation;
 Parameters p;int layout=0;if(!Prepare(program,final_fields,w,macro_step,&p,&layout))return cudaErrorInvalidValue;
#if BEAMZ_CUDA_NATIVE_MOVING_POLICIES
 if(!w.uniform_noio_counts_host||!w.uniform_observer_counts_host)return cudaErrorInvalidValue;
#endif
 const auto stream=static_cast<cudaStream_t>(opaque_stream);
 if(w.rolling){
  // Odd windows write into retired Z planes. The even sweep restores the
  // canonical addresses before observations, peer copies or the ordinary tail.
  for(int field=0;field<6;++field){
   const auto& f=InitialField(p,field);
   p.phase[field/3].output[field%3].data=const_cast<float*>(f.data)-size_t(w.rolling_band+4)*f.plane;
  }
  for(int i=0;i<12;++i){
   auto& psi=p.phase[i/6].psi[i%6];
   const int planes=beamz::cuda::moving::kCpmlAxes[i%6]==0?psi.low+psi.high:w.rolling_band+4;
   if(psi.input)psi.output=const_cast<float*>(psi.input)-size_t(planes)*psi.plane;
  }
  const auto& bands=w.rolling->bands;
  for(int phase=0;phase<w.depth;phase+=4){
   for(size_t k=0;k<bands.size();++k){
    const size_t index=(phase/4)%2?bands.size()-1-k:k;
    const auto [pieces,count]=w.rolling->View(index,macro_step+phase);
    for(size_t i=0;i<count;++i){
     const auto status=beamz::cuda::rolling::Launch(stream,p,pieces[i],phase,w.rolling_masks);
     if(status!=cudaSuccess)return status;
    }
   }
   for(int field=0;field<6;++field)std::swap(p.phase[field/3].own[field%3].data,p.phase[field/3].output[field%3].data);
   for(auto& phase_view:p.phase)for(auto& psi:phase_view.psi){
    const auto* old=psi.input;psi.input=psi.output;psi.output=const_cast<float*>(old);
   }
  }
  return cudaSuccess;
 }
 BeginEpoch<<<1,1,0,stream>>>(w.epoch,w.bad);auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
 if(program.e_ab.tensor)return LaunchTensorRanks(stream,p,w,program.e_ab);
 return layout==1?DispatchMetric<1>(stream,p,w):DispatchMetric<2>(stream,p,w);
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
 p.face_lines=p.slots+size_t(base.tile_count)*(1+(base.slab_level_count-base.rank_count)/base.slab_macro_stride);
#if BEAMZ_CUDA_NATIVE_MOVING_POLICIES
 if(!w.uniform_noio_counts_host||!w.uniform_observer_counts_host)return cudaErrorInvalidValue;
#endif
 const auto stream=static_cast<cudaStream_t>(opaque_stream);
 if(begin_epoch){BeginEpoch<<<1,1,0,stream>>>(w.epoch,w.bad);auto status=cudaGetLastError();if(status!=cudaSuccess)return status;}
 if(program.e_ab.tensor)return LaunchTensorRanks(stream,p,w,program.e_ab);
 return layout==1?DispatchMetric<1>(stream,p,w):DispatchMetric<2>(stream,p,w);
}
