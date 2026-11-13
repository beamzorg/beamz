
#pragma once
#include "deep_temporal_observation.cuh"
// Requires real PairCpmlPhase/FusedYeePhase/PairFieldOutputs/FusedHSources,
// PairCpmlGrid, Read3D/Read/BufferContains and canonical Yee helpers.
namespace beamz { namespace cuda { namespace deep {
__device__ __forceinline__ float SafeRead(const BeamzBuffer &field,int z,int y,int x) {
  return BufferContains(field,z,y,x)?Read3D(field,z,y,x):0.f;
}
__device__ __forceinline__ float Difference(const PairCpmlPhase &p,int phase,int component,int axis,
    int z,int y,int x,float center,float adjacent) {
  const int coord=axis==0?z:axis==1?y:x;
  const int extent=p.field.inputs[3+component].dims[axis];
  if(phase && ((coord==0 && !(p.metallic_edges&(1<<(2*axis)))) ||
               (coord==extent && !(p.metallic_edges&(1<<(2*axis+1))))))return 0.f;
  return beamz::cuda::yee::ScaleYeeDifference(phase?center-adjacent:adjacent-center,p.field.inv_resolution);
}
__device__ __forceinline__ void Publish(const Observation &o, const BeamzBuffer &field,
    int c,int step,int z,int y,int x,float value) {
  if (!o.values) return;
  if (o.publication_words) {
    if(z<0||y<0||x<0||z>=o.mask_nz||y>=o.mask_ny||x>=o.mask_nx)return;
    const int cell=(z*o.mask_ny+y)*o.mask_nx+x;
    const uint32_t word=__ldg(o.publication_words+(cell>>2));
    if(!((word>>(8*(cell&3)))&(1u<<c)))return;
  }
  const int64_t key=o.field_base[c]+(static_cast<int64_t>(z)*field.dims[1]+y)*field.dims[2]+x;
  int low=0,high=o.key_count;
  while(low<high) { const int mid=(low+high)/2;if(o.keys[mid]<key)low=mid+1;else high=mid; }
  if(low<o.key_count && o.keys[low]==key)o.values[step*o.key_count+low]=value;
}
__device__ __forceinline__ int PsiIndex(const PairCpmlPhase &p,int term,int c,int z,int y,int x,int *packed) {
  if(!BufferContains(p.field.outputs[c],z,y,x))return -1;
  const int axis=beamz::cuda::yee::CpmlAxis(term);
  const int coord=axis==0?z:axis==1?y:x;
  if(!beamz::cuda::yee::CpmlPackedCoordinate(coord,p.field.outputs[c].dims[axis],p.thickness,p.thickness,packed))return -1;
  const int pz=axis==0?*packed:z,py=axis==1?*packed:y,px=axis==2?*packed:x;
  return pz*p.psi_plane[term]+py*p.psi_row[term]+px;
}
template<bool Half> __device__ __forceinline__ float ReadPsi(const PairCpmlPhase &p,int term,int z,int y,int x) {
  int packed;const int index=PsiIndex(p,term,term/2,z,y,x,&packed);
  if(index<0)return 0.f;
  if constexpr(Half)return __bfloat162float(static_cast<const __nv_bfloat16 *>(p.psi[term])[index]);
  else return static_cast<const float *>(p.psi[term])[index];
}
__device__ __forceinline__ float Profile(const PairCpmlPhase &p,int term,int kind,int z,int y,int x) {
  int packed;if(PsiIndex(p,term,term/2,z,y,x,&packed)<0)return kind==0?0.f:1.f;
  return p.profiles[3*term+kind][packed];
}
template<bool Half> __device__ __forceinline__ void WritePsi(const PairCpmlPhase &p,int term,int z,int y,int x,float value) {
  int packed;const int index=PsiIndex(p,term,term/2,z,y,x,&packed);if(index<0)return;
  if constexpr(Half)static_cast<__nv_bfloat16 *>(p.final_psi[term])[index]=__float2bfloat16_rn(value);
  else static_cast<float *>(p.final_psi[term])[index]=value;
}


constexpr int DeepWarpD4M0F32ScalarHBytePackedSource20Threads=480;
constexpr int DeepWarpD4M0F32ScalarHBytePackedSource20SharedBytes=21504;
template<bool Packed> __global__ __launch_bounds__(480,1) void DeepWarpD4M0F32ScalarHBytePackedSource20(
    const __grid_constant__ PairCpmlPhase h,const __grid_constant__ PairCpmlPhase e,
    const __grid_constant__ PairFieldOutputs final,const __grid_constant__ FusedHSources hs,
    const __grid_constant__ FusedHSources es,const __grid_constant__ Observation observation,PairCpmlGrid grid) {
  if(blockDim.x!=480)return;
  extern __shared__ float s[];
  auto coordinate=[&](int axis,int block,int size) {const int low=(grid.high[axis]+size-1)/size;
    return (0&(1<<axis)) ? (block<low?block*size:grid.boundary[axis]+(block-low)*size) : grid.high[axis]+block*size;};
  auto end=[&](int axis,int block,int origin,int size) {const int low=(grid.high[axis]+size-1)/size;
    return min(origin+size,(0&(1<<axis))?(block<low?grid.high[axis]:grid.extent[axis]):grid.boundary[axis]);};
  const int ox=coordinate(2,blockIdx.x,25),oy=coordinate(1,blockIdx.y,8),oz=coordinate(0,blockIdx.z,64);
  const int end_x=end(2,blockIdx.x,ox,25),end_y=end(1,blockIdx.y,oy,8),end_z=end(0,blockIdx.z,oz,64);
  const int qx=threadIdx.x&31,qy=threadIdx.x>>5;
  __shared__ bool active_sources[6];
  if(threadIdx.x<6){const int c=threadIdx.x;const auto &g=c<3?hs.groups[c]:es.groups[c-3];
    const auto *starts=static_cast<const int *>(g.starts.data);bool active=false;
    for(int i=0;i<g.coefficients.dims[0];++i)
      active |= starts[3*i]<oz+68 && starts[3*i]+g.coefficients.dims[1]>oz-4 &&
                starts[3*i+1]<oy+12 && starts[3*i+1]+g.coefficients.dims[2]>oy-4 &&
                starts[3*i+2]<ox+29 && starts[3*i+2]+g.coefficients.dims[3]>ox-4;
    active_sources[c]=active;}
  __syncthreads();
  float previous_y_0=0.f;
float previous_y_1=0.f;
float previous_y_2=0.f;
float previous_y_3=0.f;
float previous_y_4=0.f;
float previous_y_5=0.f;
float previous_y_6=0.f;
  for(int wave=0;wave<end_z-oz+7;++wave){
    float current_H_0=0.f,old_H_0=0.f;
float current_H_1=0.f,old_H_1=0.f;
float current_H_2=0.f,old_H_2=0.f;
float current_E_0=0.f,old_E_0=0.f;
float current_E_1=0.f,old_E_1=0.f;
float current_E_2=0.f,old_E_2=0.f;
    {
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-0,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) {
const float derivative_0=Difference(p,0,2,1,z,y,x,SafeRead(p.field.inputs[5],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[5],z+(0),y+(1),x+(0)));
const float derivative_1=Difference(p,0,1,0,z,y,x,SafeRead(p.field.inputs[4],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[4],z+(1),y+(0),x+(0)));
const float derivative_2=Difference(p,0,0,0,z,y,x,SafeRead(p.field.inputs[3],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[3],z+(1),y+(0),x+(0)));
const float derivative_3=Difference(p,0,2,2,z,y,x,SafeRead(p.field.inputs[5],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[5],z+(0),y+(0),x+(1)));
const float derivative_4=Difference(p,0,1,2,z,y,x,SafeRead(p.field.inputs[4],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[4],z+(0),y+(0),x+(1)));
const float derivative_5=Difference(p,0,0,1,z,y,x,SafeRead(p.field.inputs[3],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[3],z+(0),y+(1),x+(0)));
const float stage0_0 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage0_1 = (-stage0_0);
const float stage0_2 = 1.0f;
const float stage0_3 = derivative_0;
const float stage0_4 = __fmul_rn(stage0_2, stage0_3);
const float stage0_5 = -1.0f;
const float stage0_6 = derivative_1;
const float stage0_7 = __fmul_rn(stage0_5, stage0_6);
const float stage0_8 = __fadd_rn(stage0_4, stage0_7);
const float stage0_9 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage0_10 = SafeRead(p.field.inputs[0],z,y,x);
const float stage0_11 = __fmul_rn(stage0_9, stage0_10);
const float stage0_12 = __fmaf_rn(stage0_1, stage0_8, stage0_11);
const float stage0_13 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage0_14 = (-stage0_13);
const float stage0_15 = derivative_2;
const float stage0_16 = __fmul_rn(stage0_2, stage0_15);
const float stage0_17 = derivative_3;
const float stage0_18 = __fmul_rn(stage0_5, stage0_17);
const float stage0_19 = __fadd_rn(stage0_16, stage0_18);
const float stage0_20 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage0_21 = SafeRead(p.field.inputs[1],z,y,x);
const float stage0_22 = __fmul_rn(stage0_20, stage0_21);
const float stage0_23 = __fmaf_rn(stage0_14, stage0_19, stage0_22);
const float stage0_24 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage0_25 = (-stage0_24);
const float stage0_26 = derivative_4;
const float stage0_27 = __fmul_rn(stage0_2, stage0_26);
const float stage0_28 = derivative_5;
const float stage0_29 = __fmul_rn(stage0_5, stage0_28);
const float stage0_30 = __fadd_rn(stage0_27, stage0_29);
const float stage0_31 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage0_32 = SafeRead(p.field.inputs[2],z,y,x);
const float stage0_33 = __fmul_rn(stage0_31, stage0_32);
const float stage0_34 = __fmaf_rn(stage0_25, stage0_30, stage0_33);
next_0 = stage0_12;
next_1 = stage0_23;
next_2 = stage0_34;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,0,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,0,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,0);
if(owned) {
Publish(observation,p.field.outputs[2],2,0,z,y,x,next_2);
}
}
}
if(wave>0 && qx>=0 && qx<32 && qy>=0 && qy<15) old_H_0=s[0+(0*15+qy+(0)-0)*32+qx-0];
if(wave>0 && qx>=0 && qx<32 && qy>=0 && qy<15) old_H_2=s[0+(1*15+qy+(0)-0)*32+qx-0];
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
}
__syncthreads();
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) s[0+(0*15+qy+(0)-0)*32+qx-0]=current_H_0;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) s[0+(1*15+qy+(0)-0)*32+qx-0]=current_H_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-0,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_0;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[0+(1*15+qy+(-1)-0)*32+qx-0]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[0+(0*15+qy+(-1)-0)*32+qx-0]);
const float stage1_0 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage1_1 = 1.0f;
const float stage1_2 = derivative_0;
const float stage1_3 = __fmul_rn(stage1_1, stage1_2);
const float stage1_4 = -1.0f;
const float stage1_5 = derivative_1;
const float stage1_6 = __fmul_rn(stage1_4, stage1_5);
const float stage1_7 = __fadd_rn(stage1_3, stage1_6);
const float stage1_8 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage1_9 = SafeRead(p.field.inputs[0],z,y,x);
const float stage1_10 = __fmul_rn(stage1_8, stage1_9);
const float stage1_11 = __fmaf_rn(stage1_0, stage1_7, stage1_10);
const float stage1_12 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage1_13 = derivative_2;
const float stage1_14 = __fmul_rn(stage1_1, stage1_13);
const float stage1_15 = derivative_3;
const float stage1_16 = __fmul_rn(stage1_4, stage1_15);
const float stage1_17 = __fadd_rn(stage1_14, stage1_16);
const float stage1_18 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage1_19 = SafeRead(p.field.inputs[1],z,y,x);
const float stage1_20 = __fmul_rn(stage1_18, stage1_19);
const float stage1_21 = __fmaf_rn(stage1_12, stage1_17, stage1_20);
const float stage1_22 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage1_23 = derivative_4;
const float stage1_24 = __fmul_rn(stage1_1, stage1_23);
const float stage1_25 = derivative_5;
const float stage1_26 = __fmul_rn(stage1_4, stage1_25);
const float stage1_27 = __fadd_rn(stage1_24, stage1_26);
const float stage1_28 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage1_29 = SafeRead(p.field.inputs[2],z,y,x);
const float stage1_30 = __fmul_rn(stage1_28, stage1_29);
const float stage1_31 = __fmaf_rn(stage1_22, stage1_27, stage1_30);
next_0 = stage1_11;
next_1 = stage1_21;
next_2 = stage1_31;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,0,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,0);
if(owned) {
Publish(observation,p.field.outputs[1],4,0,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,0,z,y,x,next_2);
}
}
}
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-1,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<15) old_E_0=s[960+(0*14+qy+(0)-1)*32+qx-1];
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<15) old_E_1=previous_y_1;
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<15) old_E_2=s[960+(1*14+qy+(0)-1)*32+qx-1];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[960+(1*14+qy+(1)-1)*32+qx-1]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[960+(0*14+qy+(1)-1)*32+qx-1]);
const float stage2_0 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage2_1 = (-stage2_0);
const float stage2_2 = 1.0f;
const float stage2_3 = derivative_0;
const float stage2_4 = __fmul_rn(stage2_2, stage2_3);
const float stage2_5 = -1.0f;
const float stage2_6 = derivative_1;
const float stage2_7 = __fmul_rn(stage2_5, stage2_6);
const float stage2_8 = __fadd_rn(stage2_4, stage2_7);
const float stage2_9 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage2_10 = old_H_0;
const float stage2_11 = __fmul_rn(stage2_9, stage2_10);
const float stage2_12 = __fmaf_rn(stage2_1, stage2_8, stage2_11);
const float stage2_13 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage2_14 = (-stage2_13);
const float stage2_15 = derivative_2;
const float stage2_16 = __fmul_rn(stage2_2, stage2_15);
const float stage2_17 = derivative_3;
const float stage2_18 = __fmul_rn(stage2_5, stage2_17);
const float stage2_19 = __fadd_rn(stage2_16, stage2_18);
const float stage2_20 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage2_21 = old_H_1;
const float stage2_22 = __fmul_rn(stage2_20, stage2_21);
const float stage2_23 = __fmaf_rn(stage2_14, stage2_19, stage2_22);
const float stage2_24 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage2_25 = (-stage2_24);
const float stage2_26 = derivative_4;
const float stage2_27 = __fmul_rn(stage2_2, stage2_26);
const float stage2_28 = derivative_5;
const float stage2_29 = __fmul_rn(stage2_5, stage2_28);
const float stage2_30 = __fadd_rn(stage2_27, stage2_29);
const float stage2_31 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage2_32 = old_H_2;
const float stage2_33 = __fmul_rn(stage2_31, stage2_32);
const float stage2_34 = __fmaf_rn(stage2_25, stage2_30, stage2_33);
next_0 = stage2_12;
next_1 = stage2_23;
next_2 = stage2_34;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,1,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,1,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,1);
if(owned) {
Publish(observation,p.field.outputs[2],2,1,z,y,x,next_2);
}
}
}
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) previous_y_0=current_H_1;
if(wave>2 && qx>=1 && qx<31 && qy>=1 && qy<14) old_H_0=s[1856+(0*13+qy+(0)-1)*32+qx-1];
if(wave>2 && qx>=1 && qx<31 && qy>=1 && qy<14) old_H_2=s[1856+(1*13+qy+(0)-1)*32+qx-1];
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
}
__syncthreads();
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) s[1856+(0*13+qy+(0)-1)*32+qx-1]=current_H_0;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) s[1856+(1*13+qy+(0)-1)*32+qx-1]=current_H_2;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) s[960+(0*14+qy+(0)-1)*32+qx-1]=current_E_0;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) s[960+(1*14+qy+(0)-1)*32+qx-1]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-1,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_2;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[1856+(1*13+qy+(-1)-1)*32+qx-1]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[1856+(0*13+qy+(-1)-1)*32+qx-1]);
const float stage3_0 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage3_1 = 1.0f;
const float stage3_2 = derivative_0;
const float stage3_3 = __fmul_rn(stage3_1, stage3_2);
const float stage3_4 = -1.0f;
const float stage3_5 = derivative_1;
const float stage3_6 = __fmul_rn(stage3_4, stage3_5);
const float stage3_7 = __fadd_rn(stage3_3, stage3_6);
const float stage3_8 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage3_9 = old_E_0;
const float stage3_10 = __fmul_rn(stage3_8, stage3_9);
const float stage3_11 = __fmaf_rn(stage3_0, stage3_7, stage3_10);
const float stage3_12 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage3_13 = derivative_2;
const float stage3_14 = __fmul_rn(stage3_1, stage3_13);
const float stage3_15 = derivative_3;
const float stage3_16 = __fmul_rn(stage3_4, stage3_15);
const float stage3_17 = __fadd_rn(stage3_14, stage3_16);
const float stage3_18 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage3_19 = old_E_1;
const float stage3_20 = __fmul_rn(stage3_18, stage3_19);
const float stage3_21 = __fmaf_rn(stage3_12, stage3_17, stage3_20);
const float stage3_22 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage3_23 = derivative_4;
const float stage3_24 = __fmul_rn(stage3_1, stage3_23);
const float stage3_25 = derivative_5;
const float stage3_26 = __fmul_rn(stage3_4, stage3_25);
const float stage3_27 = __fadd_rn(stage3_24, stage3_26);
const float stage3_28 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage3_29 = old_E_2;
const float stage3_30 = __fmul_rn(stage3_28, stage3_29);
const float stage3_31 = __fmaf_rn(stage3_22, stage3_27, stage3_30);
next_0 = stage3_11;
next_1 = stage3_21;
next_2 = stage3_31;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,1,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,1);
if(owned) {
Publish(observation,p.field.outputs[1],4,1,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,1,z,y,x,next_2);
}
}
}
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) previous_y_1=current_E_1;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-2,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<14) old_E_0=s[2688+(0*12+qy+(0)-2)*32+qx-2];
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<14) old_E_1=previous_y_3;
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<14) old_E_2=s[2688+(1*12+qy+(0)-2)*32+qx-2];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[2688+(1*12+qy+(1)-2)*32+qx-2]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[2688+(0*12+qy+(1)-2)*32+qx-2]);
const float stage4_0 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage4_1 = (-stage4_0);
const float stage4_2 = 1.0f;
const float stage4_3 = derivative_0;
const float stage4_4 = __fmul_rn(stage4_2, stage4_3);
const float stage4_5 = -1.0f;
const float stage4_6 = derivative_1;
const float stage4_7 = __fmul_rn(stage4_5, stage4_6);
const float stage4_8 = __fadd_rn(stage4_4, stage4_7);
const float stage4_9 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage4_10 = old_H_0;
const float stage4_11 = __fmul_rn(stage4_9, stage4_10);
const float stage4_12 = __fmaf_rn(stage4_1, stage4_8, stage4_11);
const float stage4_13 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage4_14 = (-stage4_13);
const float stage4_15 = derivative_2;
const float stage4_16 = __fmul_rn(stage4_2, stage4_15);
const float stage4_17 = derivative_3;
const float stage4_18 = __fmul_rn(stage4_5, stage4_17);
const float stage4_19 = __fadd_rn(stage4_16, stage4_18);
const float stage4_20 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage4_21 = old_H_1;
const float stage4_22 = __fmul_rn(stage4_20, stage4_21);
const float stage4_23 = __fmaf_rn(stage4_14, stage4_19, stage4_22);
const float stage4_24 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage4_25 = (-stage4_24);
const float stage4_26 = derivative_4;
const float stage4_27 = __fmul_rn(stage4_2, stage4_26);
const float stage4_28 = derivative_5;
const float stage4_29 = __fmul_rn(stage4_5, stage4_28);
const float stage4_30 = __fadd_rn(stage4_27, stage4_29);
const float stage4_31 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage4_32 = old_H_2;
const float stage4_33 = __fmul_rn(stage4_31, stage4_32);
const float stage4_34 = __fmaf_rn(stage4_25, stage4_30, stage4_33);
next_0 = stage4_12;
next_1 = stage4_23;
next_2 = stage4_34;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,2,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,2,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,2);
if(owned) {
Publish(observation,p.field.outputs[2],2,2,z,y,x,next_2);
}
}
}
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) previous_y_2=current_H_1;
if(wave>4 && qx>=2 && qx<30 && qy>=2 && qy<13) old_H_0=s[3456+(0*11+qy+(0)-2)*32+qx-2];
if(wave>4 && qx>=2 && qx<30 && qy>=2 && qy<13) old_H_2=s[3456+(1*11+qy+(0)-2)*32+qx-2];
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
}
__syncthreads();
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) s[3456+(0*11+qy+(0)-2)*32+qx-2]=current_H_0;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) s[3456+(1*11+qy+(0)-2)*32+qx-2]=current_H_2;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) s[2688+(0*12+qy+(0)-2)*32+qx-2]=current_E_0;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) s[2688+(1*12+qy+(0)-2)*32+qx-2]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-2,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_4;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[3456+(1*11+qy+(-1)-2)*32+qx-2]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[3456+(0*11+qy+(-1)-2)*32+qx-2]);
const float stage5_0 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage5_1 = 1.0f;
const float stage5_2 = derivative_0;
const float stage5_3 = __fmul_rn(stage5_1, stage5_2);
const float stage5_4 = -1.0f;
const float stage5_5 = derivative_1;
const float stage5_6 = __fmul_rn(stage5_4, stage5_5);
const float stage5_7 = __fadd_rn(stage5_3, stage5_6);
const float stage5_8 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage5_9 = old_E_0;
const float stage5_10 = __fmul_rn(stage5_8, stage5_9);
const float stage5_11 = __fmaf_rn(stage5_0, stage5_7, stage5_10);
const float stage5_12 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage5_13 = derivative_2;
const float stage5_14 = __fmul_rn(stage5_1, stage5_13);
const float stage5_15 = derivative_3;
const float stage5_16 = __fmul_rn(stage5_4, stage5_15);
const float stage5_17 = __fadd_rn(stage5_14, stage5_16);
const float stage5_18 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage5_19 = old_E_1;
const float stage5_20 = __fmul_rn(stage5_18, stage5_19);
const float stage5_21 = __fmaf_rn(stage5_12, stage5_17, stage5_20);
const float stage5_22 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage5_23 = derivative_4;
const float stage5_24 = __fmul_rn(stage5_1, stage5_23);
const float stage5_25 = derivative_5;
const float stage5_26 = __fmul_rn(stage5_4, stage5_25);
const float stage5_27 = __fadd_rn(stage5_24, stage5_26);
const float stage5_28 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage5_29 = old_E_2;
const float stage5_30 = __fmul_rn(stage5_28, stage5_29);
const float stage5_31 = __fmaf_rn(stage5_22, stage5_27, stage5_30);
next_0 = stage5_11;
next_1 = stage5_21;
next_2 = stage5_31;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,2,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,2);
if(owned) {
Publish(observation,p.field.outputs[1],4,2,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,2,z,y,x,next_2);
}
}
}
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) previous_y_3=current_E_1;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-3,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<13) old_E_0=s[4160+(0*10+qy+(0)-3)*32+qx-3];
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<13) old_E_1=previous_y_5;
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<13) old_E_2=s[4160+(1*10+qy+(0)-3)*32+qx-3];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[4160+(1*10+qy+(1)-3)*32+qx-3]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[4160+(0*10+qy+(1)-3)*32+qx-3]);
const float stage6_0 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage6_1 = (-stage6_0);
const float stage6_2 = 1.0f;
const float stage6_3 = derivative_0;
const float stage6_4 = __fmul_rn(stage6_2, stage6_3);
const float stage6_5 = -1.0f;
const float stage6_6 = derivative_1;
const float stage6_7 = __fmul_rn(stage6_5, stage6_6);
const float stage6_8 = __fadd_rn(stage6_4, stage6_7);
const float stage6_9 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage6_10 = old_H_0;
const float stage6_11 = __fmul_rn(stage6_9, stage6_10);
const float stage6_12 = __fmaf_rn(stage6_1, stage6_8, stage6_11);
const float stage6_13 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage6_14 = (-stage6_13);
const float stage6_15 = derivative_2;
const float stage6_16 = __fmul_rn(stage6_2, stage6_15);
const float stage6_17 = derivative_3;
const float stage6_18 = __fmul_rn(stage6_5, stage6_17);
const float stage6_19 = __fadd_rn(stage6_16, stage6_18);
const float stage6_20 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage6_21 = old_H_1;
const float stage6_22 = __fmul_rn(stage6_20, stage6_21);
const float stage6_23 = __fmaf_rn(stage6_14, stage6_19, stage6_22);
const float stage6_24 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage6_25 = (-stage6_24);
const float stage6_26 = derivative_4;
const float stage6_27 = __fmul_rn(stage6_2, stage6_26);
const float stage6_28 = derivative_5;
const float stage6_29 = __fmul_rn(stage6_5, stage6_28);
const float stage6_30 = __fadd_rn(stage6_27, stage6_29);
const float stage6_31 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage6_32 = old_H_2;
const float stage6_33 = __fmul_rn(stage6_31, stage6_32);
const float stage6_34 = __fmaf_rn(stage6_25, stage6_30, stage6_33);
next_0 = stage6_12;
next_1 = stage6_23;
next_2 = stage6_34;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,3,z,y,x,next_0);
static_cast<float *>(final.fields[0].data)[BeamzOffset3D<false>(final.fields[0],z,y,x)]=next_0;
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,3,z,y,x,next_1);
static_cast<float *>(final.fields[1].data)[BeamzOffset3D<false>(final.fields[1],z,y,x)]=next_1;
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,3);
if(owned) {
Publish(observation,p.field.outputs[2],2,3,z,y,x,next_2);
static_cast<float *>(final.fields[2].data)[BeamzOffset3D<false>(final.fields[2],z,y,x)]=next_2;
}
}
}
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) previous_y_4=current_H_1;
if(wave>6 && qx>=3 && qx<29 && qy>=3 && qy<12) old_H_0=s[4800+(0*9+qy+(0)-3)*32+qx-3];
if(wave>6 && qx>=3 && qx<29 && qy>=3 && qy<12) old_H_2=s[4800+(1*9+qy+(0)-3)*32+qx-3];
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
}
__syncthreads();
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) s[4800+(0*9+qy+(0)-3)*32+qx-3]=current_H_0;
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) s[4800+(1*9+qy+(0)-3)*32+qx-3]=current_H_2;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) s[4160+(0*10+qy+(0)-3)*32+qx-3]=current_E_0;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) s[4160+(1*10+qy+(0)-3)*32+qx-3]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-3,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_6;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
if(wave>=7 && qx>=4 && qx<29 && qy>=4 && qy<12) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[4800+(1*9+qy+(-1)-3)*32+qx-3]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[4800+(0*9+qy+(-1)-3)*32+qx-3]);
const float stage7_0 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage7_1 = 1.0f;
const float stage7_2 = derivative_0;
const float stage7_3 = __fmul_rn(stage7_1, stage7_2);
const float stage7_4 = -1.0f;
const float stage7_5 = derivative_1;
const float stage7_6 = __fmul_rn(stage7_4, stage7_5);
const float stage7_7 = __fadd_rn(stage7_3, stage7_6);
const float stage7_8 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage7_9 = old_E_0;
const float stage7_10 = __fmul_rn(stage7_8, stage7_9);
const float stage7_11 = __fmaf_rn(stage7_0, stage7_7, stage7_10);
const float stage7_12 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage7_13 = derivative_2;
const float stage7_14 = __fmul_rn(stage7_1, stage7_13);
const float stage7_15 = derivative_3;
const float stage7_16 = __fmul_rn(stage7_4, stage7_15);
const float stage7_17 = __fadd_rn(stage7_14, stage7_16);
const float stage7_18 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage7_19 = old_E_1;
const float stage7_20 = __fmul_rn(stage7_18, stage7_19);
const float stage7_21 = __fmaf_rn(stage7_12, stage7_17, stage7_20);
const float stage7_22 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage7_23 = derivative_4;
const float stage7_24 = __fmul_rn(stage7_1, stage7_23);
const float stage7_25 = derivative_5;
const float stage7_26 = __fmul_rn(stage7_4, stage7_25);
const float stage7_27 = __fadd_rn(stage7_24, stage7_26);
const float stage7_28 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage7_29 = old_E_2;
const float stage7_30 = __fmul_rn(stage7_28, stage7_29);
const float stage7_31 = __fmaf_rn(stage7_22, stage7_27, stage7_30);
next_0 = stage7_11;
next_1 = stage7_21;
next_2 = stage7_31;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,3,z,y,x,next_0);
static_cast<float *>(final.fields[3].data)[BeamzOffset3D<false>(final.fields[3],z,y,x)]=next_0;
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,3);
if(owned) {
Publish(observation,p.field.outputs[1],4,3,z,y,x,next_1);
static_cast<float *>(final.fields[4].data)[BeamzOffset3D<false>(final.fields[4],z,y,x)]=next_1;
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,3,z,y,x,next_2);
static_cast<float *>(final.fields[5].data)[BeamzOffset3D<false>(final.fields[5],z,y,x)]=next_2;
}
}
}
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) previous_y_5=current_E_1;
if(wave>=7 && qx>=4 && qx<29 && qy>=4 && qy<12) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
}
}
    if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) previous_y_6=current_H_1;
  }
}


constexpr int DeepWarpD4M1F32ScalarHBytePackedSource20Threads=480;
constexpr int DeepWarpD4M1F32ScalarHBytePackedSource20SharedBytes=21504;
template<bool Packed> __global__ __launch_bounds__(480,1) void DeepWarpD4M1F32ScalarHBytePackedSource20(
    const __grid_constant__ PairCpmlPhase h,const __grid_constant__ PairCpmlPhase e,
    const __grid_constant__ PairFieldOutputs final,const __grid_constant__ FusedHSources hs,
    const __grid_constant__ FusedHSources es,const __grid_constant__ Observation observation,PairCpmlGrid grid) {
  if(blockDim.x!=480)return;
  extern __shared__ float s[];
  auto coordinate=[&](int axis,int block,int size) {const int low=(grid.high[axis]+size-1)/size;
    return (1&(1<<axis)) ? (block<low?block*size:grid.boundary[axis]+(block-low)*size) : grid.high[axis]+block*size;};
  auto end=[&](int axis,int block,int origin,int size) {const int low=(grid.high[axis]+size-1)/size;
    return min(origin+size,(1&(1<<axis))?(block<low?grid.high[axis]:grid.extent[axis]):grid.boundary[axis]);};
  const int ox=coordinate(2,blockIdx.x,25),oy=coordinate(1,blockIdx.y,8),oz=coordinate(0,blockIdx.z,64);
  const int end_x=end(2,blockIdx.x,ox,25),end_y=end(1,blockIdx.y,oy,8),end_z=end(0,blockIdx.z,oz,64);
  const int qx=threadIdx.x&31,qy=threadIdx.x>>5;
  __shared__ bool active_sources[6];
  if(threadIdx.x<6){const int c=threadIdx.x;const auto &g=c<3?hs.groups[c]:es.groups[c-3];
    const auto *starts=static_cast<const int *>(g.starts.data);bool active=false;
    for(int i=0;i<g.coefficients.dims[0];++i)
      active |= starts[3*i]<oz+68 && starts[3*i]+g.coefficients.dims[1]>oz-4 &&
                starts[3*i+1]<oy+12 && starts[3*i+1]+g.coefficients.dims[2]>oy-4 &&
                starts[3*i+2]<ox+29 && starts[3*i+2]+g.coefficients.dims[3]>ox-4;
    active_sources[c]=active;}
  __syncthreads();
  float previous_y_0=0.f;
float previous_y_1=0.f;
float previous_y_2=0.f;
float previous_y_3=0.f;
float previous_y_4=0.f;
float previous_y_5=0.f;
float previous_y_6=0.f;
float previous_psi_0_1=0.f;
float previous_psi_0_2=0.f;
float previous_psi_1_1=0.f;
float previous_psi_1_2=0.f;
float previous_psi_2_1=0.f;
float previous_psi_2_2=0.f;
float previous_psi_3_1=0.f;
float previous_psi_3_2=0.f;
float previous_psi_4_1=0.f;
float previous_psi_4_2=0.f;
float previous_psi_5_1=0.f;
float previous_psi_5_2=0.f;
  for(int wave=0;wave<end_z-oz+7;++wave){
    float current_H_0=0.f,old_H_0=0.f;
float current_H_1=0.f,old_H_1=0.f;
float current_H_2=0.f,old_H_2=0.f;
float current_E_0=0.f,old_E_0=0.f;
float current_E_1=0.f,old_E_1=0.f;
float current_E_2=0.f,old_E_2=0.f;
float current_psi_H_1=0.f;
float current_psi_H_2=0.f;
float current_psi_E_1=0.f;
float current_psi_E_2=0.f;
    {
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-0,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) {
const float derivative_0=Difference(p,0,2,1,z,y,x,SafeRead(p.field.inputs[5],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[5],z+(0),y+(1),x+(0)));
const float derivative_1=Difference(p,0,1,0,z,y,x,SafeRead(p.field.inputs[4],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[4],z+(1),y+(0),x+(0)));
const float derivative_2=Difference(p,0,0,0,z,y,x,SafeRead(p.field.inputs[3],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[3],z+(1),y+(0),x+(0)));
const float derivative_3=Difference(p,0,2,2,z,y,x,SafeRead(p.field.inputs[5],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[5],z+(0),y+(0),x+(1)));
const float derivative_4=Difference(p,0,1,2,z,y,x,SafeRead(p.field.inputs[4],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[4],z+(0),y+(0),x+(1)));
const float derivative_5=Difference(p,0,0,1,z,y,x,SafeRead(p.field.inputs[3],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[3],z+(0),y+(1),x+(0)));
const float stage0_0 = Profile(p,1,1,z,y,x);
const float stage0_1 = ReadPsi<false>(p,1,z,y,x);
const float stage0_2 = Profile(p,1,0,z,y,x);
const float stage0_3 = derivative_1;
const float stage0_4 = __fmul_rn(stage0_2, stage0_3);
const float stage0_5 = __fmaf_rn(stage0_0, stage0_1, stage0_4);
const float stage0_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage0_7 = (-stage0_6);
const float stage0_8 = 1.0f;
const float stage0_9 = derivative_0;
const float stage0_10 = __fmul_rn(stage0_8, stage0_9);
const float stage0_11 = -1.0f;
const float stage0_12 = Profile(p,1,2,z,y,x);
const float stage0_13 = __fmaf_rn(stage0_3, stage0_12, stage0_5);
const float stage0_14 = __fmul_rn(stage0_11, stage0_13);
const float stage0_15 = __fadd_rn(stage0_10, stage0_14);
const float stage0_16 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage0_17 = SafeRead(p.field.inputs[0],z,y,x);
const float stage0_18 = __fmul_rn(stage0_16, stage0_17);
const float stage0_19 = __fmaf_rn(stage0_7, stage0_15, stage0_18);
const float stage0_20 = Profile(p,2,1,z,y,x);
const float stage0_21 = ReadPsi<false>(p,2,z,y,x);
const float stage0_22 = Profile(p,2,0,z,y,x);
const float stage0_23 = derivative_2;
const float stage0_24 = __fmul_rn(stage0_22, stage0_23);
const float stage0_25 = __fmaf_rn(stage0_20, stage0_21, stage0_24);
const float stage0_26 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage0_27 = (-stage0_26);
const float stage0_28 = Profile(p,2,2,z,y,x);
const float stage0_29 = __fmaf_rn(stage0_23, stage0_28, stage0_25);
const float stage0_30 = __fmul_rn(stage0_8, stage0_29);
const float stage0_31 = derivative_3;
const float stage0_32 = __fmul_rn(stage0_11, stage0_31);
const float stage0_33 = __fadd_rn(stage0_30, stage0_32);
const float stage0_34 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage0_35 = SafeRead(p.field.inputs[1],z,y,x);
const float stage0_36 = __fmul_rn(stage0_34, stage0_35);
const float stage0_37 = __fmaf_rn(stage0_27, stage0_33, stage0_36);
const float stage0_38 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage0_39 = (-stage0_38);
const float stage0_40 = derivative_4;
const float stage0_41 = __fmul_rn(stage0_8, stage0_40);
const float stage0_42 = derivative_5;
const float stage0_43 = __fmul_rn(stage0_11, stage0_42);
const float stage0_44 = __fadd_rn(stage0_41, stage0_43);
const float stage0_45 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage0_46 = SafeRead(p.field.inputs[2],z,y,x);
const float stage0_47 = __fmul_rn(stage0_45, stage0_46);
const float stage0_48 = __fmaf_rn(stage0_39, stage0_44, stage0_47);
next_psi_1 = stage0_5;
next_0 = stage0_19;
next_psi_2 = stage0_25;
next_1 = stage0_37;
next_2 = stage0_48;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,0,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,0,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,0);
if(owned) {
Publish(observation,p.field.outputs[2],2,0,z,y,x,next_2);
}
}
}
if(wave>0 && qx>=0 && qx<32 && qy>=0 && qy<15) old_H_0=s[0+(0*15+qy+(0)-0)*32+qx-0];
if(wave>0 && qx>=0 && qx<32 && qy>=0 && qy<15) old_H_2=s[0+(1*15+qy+(0)-0)*32+qx-0];
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_1=next_psi_1;
current_psi_H_2=next_psi_2;
}
__syncthreads();
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) s[0+(0*15+qy+(0)-0)*32+qx-0]=current_H_0;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) s[0+(1*15+qy+(0)-0)*32+qx-0]=current_H_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-0,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_0;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[0+(1*15+qy+(-1)-0)*32+qx-0]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[0+(0*15+qy+(-1)-0)*32+qx-0]);
const float stage1_0 = Profile(p,1,1,z,y,x);
const float stage1_1 = ReadPsi<false>(p,1,z,y,x);
const float stage1_2 = Profile(p,1,0,z,y,x);
const float stage1_3 = derivative_1;
const float stage1_4 = __fmul_rn(stage1_2, stage1_3);
const float stage1_5 = __fmaf_rn(stage1_0, stage1_1, stage1_4);
const float stage1_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage1_7 = 1.0f;
const float stage1_8 = derivative_0;
const float stage1_9 = __fmul_rn(stage1_7, stage1_8);
const float stage1_10 = -1.0f;
const float stage1_11 = Profile(p,1,2,z,y,x);
const float stage1_12 = __fmaf_rn(stage1_3, stage1_11, stage1_5);
const float stage1_13 = __fmul_rn(stage1_10, stage1_12);
const float stage1_14 = __fadd_rn(stage1_9, stage1_13);
const float stage1_15 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage1_16 = SafeRead(p.field.inputs[0],z,y,x);
const float stage1_17 = __fmul_rn(stage1_15, stage1_16);
const float stage1_18 = __fmaf_rn(stage1_6, stage1_14, stage1_17);
const float stage1_19 = Profile(p,2,1,z,y,x);
const float stage1_20 = ReadPsi<false>(p,2,z,y,x);
const float stage1_21 = Profile(p,2,0,z,y,x);
const float stage1_22 = derivative_2;
const float stage1_23 = __fmul_rn(stage1_21, stage1_22);
const float stage1_24 = __fmaf_rn(stage1_19, stage1_20, stage1_23);
const float stage1_25 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage1_26 = Profile(p,2,2,z,y,x);
const float stage1_27 = __fmaf_rn(stage1_22, stage1_26, stage1_24);
const float stage1_28 = __fmul_rn(stage1_7, stage1_27);
const float stage1_29 = derivative_3;
const float stage1_30 = __fmul_rn(stage1_10, stage1_29);
const float stage1_31 = __fadd_rn(stage1_28, stage1_30);
const float stage1_32 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage1_33 = SafeRead(p.field.inputs[1],z,y,x);
const float stage1_34 = __fmul_rn(stage1_32, stage1_33);
const float stage1_35 = __fmaf_rn(stage1_25, stage1_31, stage1_34);
const float stage1_36 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage1_37 = derivative_4;
const float stage1_38 = __fmul_rn(stage1_7, stage1_37);
const float stage1_39 = derivative_5;
const float stage1_40 = __fmul_rn(stage1_10, stage1_39);
const float stage1_41 = __fadd_rn(stage1_38, stage1_40);
const float stage1_42 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage1_43 = SafeRead(p.field.inputs[2],z,y,x);
const float stage1_44 = __fmul_rn(stage1_42, stage1_43);
const float stage1_45 = __fmaf_rn(stage1_36, stage1_41, stage1_44);
next_psi_1 = stage1_5;
next_0 = stage1_18;
next_psi_2 = stage1_24;
next_1 = stage1_35;
next_2 = stage1_45;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,0,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,0);
if(owned) {
Publish(observation,p.field.outputs[1],4,0,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,0,z,y,x,next_2);
}
}
}
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_1=next_psi_1;
current_psi_E_2=next_psi_2;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-1,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<15) old_E_0=s[960+(0*14+qy+(0)-1)*32+qx-1];
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<15) old_E_1=previous_y_1;
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<15) old_E_2=s[960+(1*14+qy+(0)-1)*32+qx-1];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[960+(1*14+qy+(1)-1)*32+qx-1]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[960+(0*14+qy+(1)-1)*32+qx-1]);
const float stage2_0 = Profile(p,1,1,z,y,x);
const float stage2_1 = previous_psi_0_1;
const float stage2_2 = Profile(p,1,0,z,y,x);
const float stage2_3 = derivative_1;
const float stage2_4 = __fmul_rn(stage2_2, stage2_3);
const float stage2_5 = __fmaf_rn(stage2_0, stage2_1, stage2_4);
const float stage2_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage2_7 = (-stage2_6);
const float stage2_8 = 1.0f;
const float stage2_9 = derivative_0;
const float stage2_10 = __fmul_rn(stage2_8, stage2_9);
const float stage2_11 = -1.0f;
const float stage2_12 = Profile(p,1,2,z,y,x);
const float stage2_13 = __fmaf_rn(stage2_3, stage2_12, stage2_5);
const float stage2_14 = __fmul_rn(stage2_11, stage2_13);
const float stage2_15 = __fadd_rn(stage2_10, stage2_14);
const float stage2_16 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage2_17 = old_H_0;
const float stage2_18 = __fmul_rn(stage2_16, stage2_17);
const float stage2_19 = __fmaf_rn(stage2_7, stage2_15, stage2_18);
const float stage2_20 = Profile(p,2,1,z,y,x);
const float stage2_21 = previous_psi_0_2;
const float stage2_22 = Profile(p,2,0,z,y,x);
const float stage2_23 = derivative_2;
const float stage2_24 = __fmul_rn(stage2_22, stage2_23);
const float stage2_25 = __fmaf_rn(stage2_20, stage2_21, stage2_24);
const float stage2_26 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage2_27 = (-stage2_26);
const float stage2_28 = Profile(p,2,2,z,y,x);
const float stage2_29 = __fmaf_rn(stage2_23, stage2_28, stage2_25);
const float stage2_30 = __fmul_rn(stage2_8, stage2_29);
const float stage2_31 = derivative_3;
const float stage2_32 = __fmul_rn(stage2_11, stage2_31);
const float stage2_33 = __fadd_rn(stage2_30, stage2_32);
const float stage2_34 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage2_35 = old_H_1;
const float stage2_36 = __fmul_rn(stage2_34, stage2_35);
const float stage2_37 = __fmaf_rn(stage2_27, stage2_33, stage2_36);
const float stage2_38 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage2_39 = (-stage2_38);
const float stage2_40 = derivative_4;
const float stage2_41 = __fmul_rn(stage2_8, stage2_40);
const float stage2_42 = derivative_5;
const float stage2_43 = __fmul_rn(stage2_11, stage2_42);
const float stage2_44 = __fadd_rn(stage2_41, stage2_43);
const float stage2_45 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage2_46 = old_H_2;
const float stage2_47 = __fmul_rn(stage2_45, stage2_46);
const float stage2_48 = __fmaf_rn(stage2_39, stage2_44, stage2_47);
next_psi_1 = stage2_5;
next_0 = stage2_19;
next_psi_2 = stage2_25;
next_1 = stage2_37;
next_2 = stage2_48;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,1,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,1,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,1);
if(owned) {
Publish(observation,p.field.outputs[2],2,1,z,y,x,next_2);
}
}
}
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) previous_y_0=current_H_1;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) previous_psi_0_1=current_psi_H_1;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) previous_psi_0_2=current_psi_H_2;
if(wave>2 && qx>=1 && qx<31 && qy>=1 && qy<14) old_H_0=s[1856+(0*13+qy+(0)-1)*32+qx-1];
if(wave>2 && qx>=1 && qx<31 && qy>=1 && qy<14) old_H_2=s[1856+(1*13+qy+(0)-1)*32+qx-1];
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_1=next_psi_1;
current_psi_H_2=next_psi_2;
}
__syncthreads();
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) s[1856+(0*13+qy+(0)-1)*32+qx-1]=current_H_0;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) s[1856+(1*13+qy+(0)-1)*32+qx-1]=current_H_2;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) s[960+(0*14+qy+(0)-1)*32+qx-1]=current_E_0;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) s[960+(1*14+qy+(0)-1)*32+qx-1]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-1,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_2;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[1856+(1*13+qy+(-1)-1)*32+qx-1]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[1856+(0*13+qy+(-1)-1)*32+qx-1]);
const float stage3_0 = Profile(p,1,1,z,y,x);
const float stage3_1 = previous_psi_1_1;
const float stage3_2 = Profile(p,1,0,z,y,x);
const float stage3_3 = derivative_1;
const float stage3_4 = __fmul_rn(stage3_2, stage3_3);
const float stage3_5 = __fmaf_rn(stage3_0, stage3_1, stage3_4);
const float stage3_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage3_7 = 1.0f;
const float stage3_8 = derivative_0;
const float stage3_9 = __fmul_rn(stage3_7, stage3_8);
const float stage3_10 = -1.0f;
const float stage3_11 = Profile(p,1,2,z,y,x);
const float stage3_12 = __fmaf_rn(stage3_3, stage3_11, stage3_5);
const float stage3_13 = __fmul_rn(stage3_10, stage3_12);
const float stage3_14 = __fadd_rn(stage3_9, stage3_13);
const float stage3_15 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage3_16 = old_E_0;
const float stage3_17 = __fmul_rn(stage3_15, stage3_16);
const float stage3_18 = __fmaf_rn(stage3_6, stage3_14, stage3_17);
const float stage3_19 = Profile(p,2,1,z,y,x);
const float stage3_20 = previous_psi_1_2;
const float stage3_21 = Profile(p,2,0,z,y,x);
const float stage3_22 = derivative_2;
const float stage3_23 = __fmul_rn(stage3_21, stage3_22);
const float stage3_24 = __fmaf_rn(stage3_19, stage3_20, stage3_23);
const float stage3_25 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage3_26 = Profile(p,2,2,z,y,x);
const float stage3_27 = __fmaf_rn(stage3_22, stage3_26, stage3_24);
const float stage3_28 = __fmul_rn(stage3_7, stage3_27);
const float stage3_29 = derivative_3;
const float stage3_30 = __fmul_rn(stage3_10, stage3_29);
const float stage3_31 = __fadd_rn(stage3_28, stage3_30);
const float stage3_32 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage3_33 = old_E_1;
const float stage3_34 = __fmul_rn(stage3_32, stage3_33);
const float stage3_35 = __fmaf_rn(stage3_25, stage3_31, stage3_34);
const float stage3_36 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage3_37 = derivative_4;
const float stage3_38 = __fmul_rn(stage3_7, stage3_37);
const float stage3_39 = derivative_5;
const float stage3_40 = __fmul_rn(stage3_10, stage3_39);
const float stage3_41 = __fadd_rn(stage3_38, stage3_40);
const float stage3_42 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage3_43 = old_E_2;
const float stage3_44 = __fmul_rn(stage3_42, stage3_43);
const float stage3_45 = __fmaf_rn(stage3_36, stage3_41, stage3_44);
next_psi_1 = stage3_5;
next_0 = stage3_18;
next_psi_2 = stage3_24;
next_1 = stage3_35;
next_2 = stage3_45;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,1,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,1);
if(owned) {
Publish(observation,p.field.outputs[1],4,1,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,1,z,y,x,next_2);
}
}
}
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) previous_y_1=current_E_1;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) previous_psi_1_1=current_psi_E_1;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) previous_psi_1_2=current_psi_E_2;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_1=next_psi_1;
current_psi_E_2=next_psi_2;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-2,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<14) old_E_0=s[2688+(0*12+qy+(0)-2)*32+qx-2];
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<14) old_E_1=previous_y_3;
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<14) old_E_2=s[2688+(1*12+qy+(0)-2)*32+qx-2];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[2688+(1*12+qy+(1)-2)*32+qx-2]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[2688+(0*12+qy+(1)-2)*32+qx-2]);
const float stage4_0 = Profile(p,1,1,z,y,x);
const float stage4_1 = previous_psi_2_1;
const float stage4_2 = Profile(p,1,0,z,y,x);
const float stage4_3 = derivative_1;
const float stage4_4 = __fmul_rn(stage4_2, stage4_3);
const float stage4_5 = __fmaf_rn(stage4_0, stage4_1, stage4_4);
const float stage4_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage4_7 = (-stage4_6);
const float stage4_8 = 1.0f;
const float stage4_9 = derivative_0;
const float stage4_10 = __fmul_rn(stage4_8, stage4_9);
const float stage4_11 = -1.0f;
const float stage4_12 = Profile(p,1,2,z,y,x);
const float stage4_13 = __fmaf_rn(stage4_3, stage4_12, stage4_5);
const float stage4_14 = __fmul_rn(stage4_11, stage4_13);
const float stage4_15 = __fadd_rn(stage4_10, stage4_14);
const float stage4_16 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage4_17 = old_H_0;
const float stage4_18 = __fmul_rn(stage4_16, stage4_17);
const float stage4_19 = __fmaf_rn(stage4_7, stage4_15, stage4_18);
const float stage4_20 = Profile(p,2,1,z,y,x);
const float stage4_21 = previous_psi_2_2;
const float stage4_22 = Profile(p,2,0,z,y,x);
const float stage4_23 = derivative_2;
const float stage4_24 = __fmul_rn(stage4_22, stage4_23);
const float stage4_25 = __fmaf_rn(stage4_20, stage4_21, stage4_24);
const float stage4_26 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage4_27 = (-stage4_26);
const float stage4_28 = Profile(p,2,2,z,y,x);
const float stage4_29 = __fmaf_rn(stage4_23, stage4_28, stage4_25);
const float stage4_30 = __fmul_rn(stage4_8, stage4_29);
const float stage4_31 = derivative_3;
const float stage4_32 = __fmul_rn(stage4_11, stage4_31);
const float stage4_33 = __fadd_rn(stage4_30, stage4_32);
const float stage4_34 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage4_35 = old_H_1;
const float stage4_36 = __fmul_rn(stage4_34, stage4_35);
const float stage4_37 = __fmaf_rn(stage4_27, stage4_33, stage4_36);
const float stage4_38 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage4_39 = (-stage4_38);
const float stage4_40 = derivative_4;
const float stage4_41 = __fmul_rn(stage4_8, stage4_40);
const float stage4_42 = derivative_5;
const float stage4_43 = __fmul_rn(stage4_11, stage4_42);
const float stage4_44 = __fadd_rn(stage4_41, stage4_43);
const float stage4_45 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage4_46 = old_H_2;
const float stage4_47 = __fmul_rn(stage4_45, stage4_46);
const float stage4_48 = __fmaf_rn(stage4_39, stage4_44, stage4_47);
next_psi_1 = stage4_5;
next_0 = stage4_19;
next_psi_2 = stage4_25;
next_1 = stage4_37;
next_2 = stage4_48;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,2,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,2,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,2);
if(owned) {
Publish(observation,p.field.outputs[2],2,2,z,y,x,next_2);
}
}
}
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) previous_y_2=current_H_1;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) previous_psi_2_1=current_psi_H_1;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) previous_psi_2_2=current_psi_H_2;
if(wave>4 && qx>=2 && qx<30 && qy>=2 && qy<13) old_H_0=s[3456+(0*11+qy+(0)-2)*32+qx-2];
if(wave>4 && qx>=2 && qx<30 && qy>=2 && qy<13) old_H_2=s[3456+(1*11+qy+(0)-2)*32+qx-2];
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_1=next_psi_1;
current_psi_H_2=next_psi_2;
}
__syncthreads();
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) s[3456+(0*11+qy+(0)-2)*32+qx-2]=current_H_0;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) s[3456+(1*11+qy+(0)-2)*32+qx-2]=current_H_2;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) s[2688+(0*12+qy+(0)-2)*32+qx-2]=current_E_0;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) s[2688+(1*12+qy+(0)-2)*32+qx-2]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-2,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_4;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[3456+(1*11+qy+(-1)-2)*32+qx-2]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[3456+(0*11+qy+(-1)-2)*32+qx-2]);
const float stage5_0 = Profile(p,1,1,z,y,x);
const float stage5_1 = previous_psi_3_1;
const float stage5_2 = Profile(p,1,0,z,y,x);
const float stage5_3 = derivative_1;
const float stage5_4 = __fmul_rn(stage5_2, stage5_3);
const float stage5_5 = __fmaf_rn(stage5_0, stage5_1, stage5_4);
const float stage5_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage5_7 = 1.0f;
const float stage5_8 = derivative_0;
const float stage5_9 = __fmul_rn(stage5_7, stage5_8);
const float stage5_10 = -1.0f;
const float stage5_11 = Profile(p,1,2,z,y,x);
const float stage5_12 = __fmaf_rn(stage5_3, stage5_11, stage5_5);
const float stage5_13 = __fmul_rn(stage5_10, stage5_12);
const float stage5_14 = __fadd_rn(stage5_9, stage5_13);
const float stage5_15 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage5_16 = old_E_0;
const float stage5_17 = __fmul_rn(stage5_15, stage5_16);
const float stage5_18 = __fmaf_rn(stage5_6, stage5_14, stage5_17);
const float stage5_19 = Profile(p,2,1,z,y,x);
const float stage5_20 = previous_psi_3_2;
const float stage5_21 = Profile(p,2,0,z,y,x);
const float stage5_22 = derivative_2;
const float stage5_23 = __fmul_rn(stage5_21, stage5_22);
const float stage5_24 = __fmaf_rn(stage5_19, stage5_20, stage5_23);
const float stage5_25 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage5_26 = Profile(p,2,2,z,y,x);
const float stage5_27 = __fmaf_rn(stage5_22, stage5_26, stage5_24);
const float stage5_28 = __fmul_rn(stage5_7, stage5_27);
const float stage5_29 = derivative_3;
const float stage5_30 = __fmul_rn(stage5_10, stage5_29);
const float stage5_31 = __fadd_rn(stage5_28, stage5_30);
const float stage5_32 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage5_33 = old_E_1;
const float stage5_34 = __fmul_rn(stage5_32, stage5_33);
const float stage5_35 = __fmaf_rn(stage5_25, stage5_31, stage5_34);
const float stage5_36 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage5_37 = derivative_4;
const float stage5_38 = __fmul_rn(stage5_7, stage5_37);
const float stage5_39 = derivative_5;
const float stage5_40 = __fmul_rn(stage5_10, stage5_39);
const float stage5_41 = __fadd_rn(stage5_38, stage5_40);
const float stage5_42 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage5_43 = old_E_2;
const float stage5_44 = __fmul_rn(stage5_42, stage5_43);
const float stage5_45 = __fmaf_rn(stage5_36, stage5_41, stage5_44);
next_psi_1 = stage5_5;
next_0 = stage5_18;
next_psi_2 = stage5_24;
next_1 = stage5_35;
next_2 = stage5_45;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,2,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,2);
if(owned) {
Publish(observation,p.field.outputs[1],4,2,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,2,z,y,x,next_2);
}
}
}
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) previous_y_3=current_E_1;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) previous_psi_3_1=current_psi_E_1;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) previous_psi_3_2=current_psi_E_2;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_1=next_psi_1;
current_psi_E_2=next_psi_2;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-3,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<13) old_E_0=s[4160+(0*10+qy+(0)-3)*32+qx-3];
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<13) old_E_1=previous_y_5;
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<13) old_E_2=s[4160+(1*10+qy+(0)-3)*32+qx-3];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[4160+(1*10+qy+(1)-3)*32+qx-3]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[4160+(0*10+qy+(1)-3)*32+qx-3]);
const float stage6_0 = Profile(p,1,1,z,y,x);
const float stage6_1 = previous_psi_4_1;
const float stage6_2 = Profile(p,1,0,z,y,x);
const float stage6_3 = derivative_1;
const float stage6_4 = __fmul_rn(stage6_2, stage6_3);
const float stage6_5 = __fmaf_rn(stage6_0, stage6_1, stage6_4);
const float stage6_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage6_7 = (-stage6_6);
const float stage6_8 = 1.0f;
const float stage6_9 = derivative_0;
const float stage6_10 = __fmul_rn(stage6_8, stage6_9);
const float stage6_11 = -1.0f;
const float stage6_12 = Profile(p,1,2,z,y,x);
const float stage6_13 = __fmaf_rn(stage6_3, stage6_12, stage6_5);
const float stage6_14 = __fmul_rn(stage6_11, stage6_13);
const float stage6_15 = __fadd_rn(stage6_10, stage6_14);
const float stage6_16 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage6_17 = old_H_0;
const float stage6_18 = __fmul_rn(stage6_16, stage6_17);
const float stage6_19 = __fmaf_rn(stage6_7, stage6_15, stage6_18);
const float stage6_20 = Profile(p,2,1,z,y,x);
const float stage6_21 = previous_psi_4_2;
const float stage6_22 = Profile(p,2,0,z,y,x);
const float stage6_23 = derivative_2;
const float stage6_24 = __fmul_rn(stage6_22, stage6_23);
const float stage6_25 = __fmaf_rn(stage6_20, stage6_21, stage6_24);
const float stage6_26 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage6_27 = (-stage6_26);
const float stage6_28 = Profile(p,2,2,z,y,x);
const float stage6_29 = __fmaf_rn(stage6_23, stage6_28, stage6_25);
const float stage6_30 = __fmul_rn(stage6_8, stage6_29);
const float stage6_31 = derivative_3;
const float stage6_32 = __fmul_rn(stage6_11, stage6_31);
const float stage6_33 = __fadd_rn(stage6_30, stage6_32);
const float stage6_34 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage6_35 = old_H_1;
const float stage6_36 = __fmul_rn(stage6_34, stage6_35);
const float stage6_37 = __fmaf_rn(stage6_27, stage6_33, stage6_36);
const float stage6_38 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage6_39 = (-stage6_38);
const float stage6_40 = derivative_4;
const float stage6_41 = __fmul_rn(stage6_8, stage6_40);
const float stage6_42 = derivative_5;
const float stage6_43 = __fmul_rn(stage6_11, stage6_42);
const float stage6_44 = __fadd_rn(stage6_41, stage6_43);
const float stage6_45 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage6_46 = old_H_2;
const float stage6_47 = __fmul_rn(stage6_45, stage6_46);
const float stage6_48 = __fmaf_rn(stage6_39, stage6_44, stage6_47);
next_psi_1 = stage6_5;
next_0 = stage6_19;
next_psi_2 = stage6_25;
next_1 = stage6_37;
next_2 = stage6_48;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,3,z,y,x,next_0);
static_cast<float *>(final.fields[0].data)[BeamzOffset3D<false>(final.fields[0],z,y,x)]=next_0;
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,3,z,y,x,next_1);
static_cast<float *>(final.fields[1].data)[BeamzOffset3D<false>(final.fields[1],z,y,x)]=next_1;
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,3);
if(owned) {
Publish(observation,p.field.outputs[2],2,3,z,y,x,next_2);
static_cast<float *>(final.fields[2].data)[BeamzOffset3D<false>(final.fields[2],z,y,x)]=next_2;
}
}
if(owned) WritePsi<false>(p,1,z,y,x,next_psi_1);
if(owned) WritePsi<false>(p,2,z,y,x,next_psi_2);
}
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) previous_y_4=current_H_1;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) previous_psi_4_1=current_psi_H_1;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) previous_psi_4_2=current_psi_H_2;
if(wave>6 && qx>=3 && qx<29 && qy>=3 && qy<12) old_H_0=s[4800+(0*9+qy+(0)-3)*32+qx-3];
if(wave>6 && qx>=3 && qx<29 && qy>=3 && qy<12) old_H_2=s[4800+(1*9+qy+(0)-3)*32+qx-3];
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_1=next_psi_1;
current_psi_H_2=next_psi_2;
}
__syncthreads();
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) s[4800+(0*9+qy+(0)-3)*32+qx-3]=current_H_0;
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) s[4800+(1*9+qy+(0)-3)*32+qx-3]=current_H_2;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) s[4160+(0*10+qy+(0)-3)*32+qx-3]=current_E_0;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) s[4160+(1*10+qy+(0)-3)*32+qx-3]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-3,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_6;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
if(wave>=7 && qx>=4 && qx<29 && qy>=4 && qy<12) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[4800+(1*9+qy+(-1)-3)*32+qx-3]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[4800+(0*9+qy+(-1)-3)*32+qx-3]);
const float stage7_0 = Profile(p,1,1,z,y,x);
const float stage7_1 = previous_psi_5_1;
const float stage7_2 = Profile(p,1,0,z,y,x);
const float stage7_3 = derivative_1;
const float stage7_4 = __fmul_rn(stage7_2, stage7_3);
const float stage7_5 = __fmaf_rn(stage7_0, stage7_1, stage7_4);
const float stage7_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage7_7 = 1.0f;
const float stage7_8 = derivative_0;
const float stage7_9 = __fmul_rn(stage7_7, stage7_8);
const float stage7_10 = -1.0f;
const float stage7_11 = Profile(p,1,2,z,y,x);
const float stage7_12 = __fmaf_rn(stage7_3, stage7_11, stage7_5);
const float stage7_13 = __fmul_rn(stage7_10, stage7_12);
const float stage7_14 = __fadd_rn(stage7_9, stage7_13);
const float stage7_15 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage7_16 = old_E_0;
const float stage7_17 = __fmul_rn(stage7_15, stage7_16);
const float stage7_18 = __fmaf_rn(stage7_6, stage7_14, stage7_17);
const float stage7_19 = Profile(p,2,1,z,y,x);
const float stage7_20 = previous_psi_5_2;
const float stage7_21 = Profile(p,2,0,z,y,x);
const float stage7_22 = derivative_2;
const float stage7_23 = __fmul_rn(stage7_21, stage7_22);
const float stage7_24 = __fmaf_rn(stage7_19, stage7_20, stage7_23);
const float stage7_25 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage7_26 = Profile(p,2,2,z,y,x);
const float stage7_27 = __fmaf_rn(stage7_22, stage7_26, stage7_24);
const float stage7_28 = __fmul_rn(stage7_7, stage7_27);
const float stage7_29 = derivative_3;
const float stage7_30 = __fmul_rn(stage7_10, stage7_29);
const float stage7_31 = __fadd_rn(stage7_28, stage7_30);
const float stage7_32 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage7_33 = old_E_1;
const float stage7_34 = __fmul_rn(stage7_32, stage7_33);
const float stage7_35 = __fmaf_rn(stage7_25, stage7_31, stage7_34);
const float stage7_36 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage7_37 = derivative_4;
const float stage7_38 = __fmul_rn(stage7_7, stage7_37);
const float stage7_39 = derivative_5;
const float stage7_40 = __fmul_rn(stage7_10, stage7_39);
const float stage7_41 = __fadd_rn(stage7_38, stage7_40);
const float stage7_42 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage7_43 = old_E_2;
const float stage7_44 = __fmul_rn(stage7_42, stage7_43);
const float stage7_45 = __fmaf_rn(stage7_36, stage7_41, stage7_44);
next_psi_1 = stage7_5;
next_0 = stage7_18;
next_psi_2 = stage7_24;
next_1 = stage7_35;
next_2 = stage7_45;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,3,z,y,x,next_0);
static_cast<float *>(final.fields[3].data)[BeamzOffset3D<false>(final.fields[3],z,y,x)]=next_0;
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,3);
if(owned) {
Publish(observation,p.field.outputs[1],4,3,z,y,x,next_1);
static_cast<float *>(final.fields[4].data)[BeamzOffset3D<false>(final.fields[4],z,y,x)]=next_1;
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,3,z,y,x,next_2);
static_cast<float *>(final.fields[5].data)[BeamzOffset3D<false>(final.fields[5],z,y,x)]=next_2;
}
}
if(owned) WritePsi<false>(p,1,z,y,x,next_psi_1);
if(owned) WritePsi<false>(p,2,z,y,x,next_psi_2);
}
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) previous_y_5=current_E_1;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) previous_psi_5_1=current_psi_E_1;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) previous_psi_5_2=current_psi_E_2;
if(wave>=7 && qx>=4 && qx<29 && qy>=4 && qy<12) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_1=next_psi_1;
current_psi_E_2=next_psi_2;
}
}
    if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) previous_y_6=current_H_1;
  }
}


constexpr int DeepWarpD4M2F32ScalarHBytePackedSource20Threads=480;
constexpr int DeepWarpD4M2F32ScalarHBytePackedSource20SharedBytes=21504;
template<bool Packed> __global__ __launch_bounds__(480,1) void DeepWarpD4M2F32ScalarHBytePackedSource20(
    const __grid_constant__ PairCpmlPhase h,const __grid_constant__ PairCpmlPhase e,
    const __grid_constant__ PairFieldOutputs final,const __grid_constant__ FusedHSources hs,
    const __grid_constant__ FusedHSources es,const __grid_constant__ Observation observation,PairCpmlGrid grid) {
  if(blockDim.x!=480)return;
  extern __shared__ float s[];
  auto coordinate=[&](int axis,int block,int size) {const int low=(grid.high[axis]+size-1)/size;
    return (2&(1<<axis)) ? (block<low?block*size:grid.boundary[axis]+(block-low)*size) : grid.high[axis]+block*size;};
  auto end=[&](int axis,int block,int origin,int size) {const int low=(grid.high[axis]+size-1)/size;
    return min(origin+size,(2&(1<<axis))?(block<low?grid.high[axis]:grid.extent[axis]):grid.boundary[axis]);};
  const int ox=coordinate(2,blockIdx.x,25),oy=coordinate(1,blockIdx.y,8),oz=coordinate(0,blockIdx.z,64);
  const int end_x=end(2,blockIdx.x,ox,25),end_y=end(1,blockIdx.y,oy,8),end_z=end(0,blockIdx.z,oz,64);
  const int qx=threadIdx.x&31,qy=threadIdx.x>>5;
  __shared__ bool active_sources[6];
  if(threadIdx.x<6){const int c=threadIdx.x;const auto &g=c<3?hs.groups[c]:es.groups[c-3];
    const auto *starts=static_cast<const int *>(g.starts.data);bool active=false;
    for(int i=0;i<g.coefficients.dims[0];++i)
      active |= starts[3*i]<oz+68 && starts[3*i]+g.coefficients.dims[1]>oz-4 &&
                starts[3*i+1]<oy+12 && starts[3*i+1]+g.coefficients.dims[2]>oy-4 &&
                starts[3*i+2]<ox+29 && starts[3*i+2]+g.coefficients.dims[3]>ox-4;
    active_sources[c]=active;}
  __syncthreads();
  float previous_y_0=0.f;
float previous_y_1=0.f;
float previous_y_2=0.f;
float previous_y_3=0.f;
float previous_y_4=0.f;
float previous_y_5=0.f;
float previous_y_6=0.f;
float previous_psi_0_0=0.f;
float previous_psi_0_5=0.f;
float previous_psi_1_0=0.f;
float previous_psi_1_5=0.f;
float previous_psi_2_0=0.f;
float previous_psi_2_5=0.f;
float previous_psi_3_0=0.f;
float previous_psi_3_5=0.f;
float previous_psi_4_0=0.f;
float previous_psi_4_5=0.f;
float previous_psi_5_0=0.f;
float previous_psi_5_5=0.f;
  for(int wave=0;wave<end_z-oz+7;++wave){
    float current_H_0=0.f,old_H_0=0.f;
float current_H_1=0.f,old_H_1=0.f;
float current_H_2=0.f,old_H_2=0.f;
float current_E_0=0.f,old_E_0=0.f;
float current_E_1=0.f,old_E_1=0.f;
float current_E_2=0.f,old_E_2=0.f;
float current_psi_H_0=0.f;
float current_psi_H_5=0.f;
float current_psi_E_0=0.f;
float current_psi_E_5=0.f;
    {
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-0,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_5=0.f;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) {
const float derivative_0=Difference(p,0,2,1,z,y,x,SafeRead(p.field.inputs[5],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[5],z+(0),y+(1),x+(0)));
const float derivative_1=Difference(p,0,1,0,z,y,x,SafeRead(p.field.inputs[4],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[4],z+(1),y+(0),x+(0)));
const float derivative_2=Difference(p,0,0,0,z,y,x,SafeRead(p.field.inputs[3],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[3],z+(1),y+(0),x+(0)));
const float derivative_3=Difference(p,0,2,2,z,y,x,SafeRead(p.field.inputs[5],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[5],z+(0),y+(0),x+(1)));
const float derivative_4=Difference(p,0,1,2,z,y,x,SafeRead(p.field.inputs[4],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[4],z+(0),y+(0),x+(1)));
const float derivative_5=Difference(p,0,0,1,z,y,x,SafeRead(p.field.inputs[3],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[3],z+(0),y+(1),x+(0)));
const float stage0_0 = Profile(p,0,1,z,y,x);
const float stage0_1 = ReadPsi<false>(p,0,z,y,x);
const float stage0_2 = Profile(p,0,0,z,y,x);
const float stage0_3 = derivative_0;
const float stage0_4 = __fmul_rn(stage0_2, stage0_3);
const float stage0_5 = __fmaf_rn(stage0_0, stage0_1, stage0_4);
const float stage0_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage0_7 = (-stage0_6);
const float stage0_8 = 1.0f;
const float stage0_9 = Profile(p,0,2,z,y,x);
const float stage0_10 = __fmaf_rn(stage0_3, stage0_9, stage0_5);
const float stage0_11 = __fmul_rn(stage0_8, stage0_10);
const float stage0_12 = -1.0f;
const float stage0_13 = derivative_1;
const float stage0_14 = __fmul_rn(stage0_12, stage0_13);
const float stage0_15 = __fadd_rn(stage0_11, stage0_14);
const float stage0_16 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage0_17 = SafeRead(p.field.inputs[0],z,y,x);
const float stage0_18 = __fmul_rn(stage0_16, stage0_17);
const float stage0_19 = __fmaf_rn(stage0_7, stage0_15, stage0_18);
const float stage0_20 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage0_21 = (-stage0_20);
const float stage0_22 = derivative_2;
const float stage0_23 = __fmul_rn(stage0_8, stage0_22);
const float stage0_24 = derivative_3;
const float stage0_25 = __fmul_rn(stage0_12, stage0_24);
const float stage0_26 = __fadd_rn(stage0_23, stage0_25);
const float stage0_27 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage0_28 = SafeRead(p.field.inputs[1],z,y,x);
const float stage0_29 = __fmul_rn(stage0_27, stage0_28);
const float stage0_30 = __fmaf_rn(stage0_21, stage0_26, stage0_29);
const float stage0_31 = Profile(p,5,1,z,y,x);
const float stage0_32 = ReadPsi<false>(p,5,z,y,x);
const float stage0_33 = Profile(p,5,0,z,y,x);
const float stage0_34 = derivative_5;
const float stage0_35 = __fmul_rn(stage0_33, stage0_34);
const float stage0_36 = __fmaf_rn(stage0_31, stage0_32, stage0_35);
const float stage0_37 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage0_38 = (-stage0_37);
const float stage0_39 = derivative_4;
const float stage0_40 = __fmul_rn(stage0_8, stage0_39);
const float stage0_41 = Profile(p,5,2,z,y,x);
const float stage0_42 = __fmaf_rn(stage0_34, stage0_41, stage0_36);
const float stage0_43 = __fmul_rn(stage0_12, stage0_42);
const float stage0_44 = __fadd_rn(stage0_40, stage0_43);
const float stage0_45 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage0_46 = SafeRead(p.field.inputs[2],z,y,x);
const float stage0_47 = __fmul_rn(stage0_45, stage0_46);
const float stage0_48 = __fmaf_rn(stage0_38, stage0_44, stage0_47);
next_psi_0 = stage0_5;
next_0 = stage0_19;
next_1 = stage0_30;
next_psi_5 = stage0_36;
next_2 = stage0_48;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,0,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,0,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,0);
if(owned) {
Publish(observation,p.field.outputs[2],2,0,z,y,x,next_2);
}
}
}
if(wave>0 && qx>=0 && qx<32 && qy>=0 && qy<15) old_H_0=s[0+(0*15+qy+(0)-0)*32+qx-0];
if(wave>0 && qx>=0 && qx<32 && qy>=0 && qy<15) old_H_2=s[0+(1*15+qy+(0)-0)*32+qx-0];
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_0=next_psi_0;
current_psi_H_5=next_psi_5;
}
__syncthreads();
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) s[0+(0*15+qy+(0)-0)*32+qx-0]=current_H_0;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) s[0+(1*15+qy+(0)-0)*32+qx-0]=current_H_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-0,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_0;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_5=0.f;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[0+(1*15+qy+(-1)-0)*32+qx-0]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[0+(0*15+qy+(-1)-0)*32+qx-0]);
const float stage1_0 = Profile(p,0,1,z,y,x);
const float stage1_1 = ReadPsi<false>(p,0,z,y,x);
const float stage1_2 = Profile(p,0,0,z,y,x);
const float stage1_3 = derivative_0;
const float stage1_4 = __fmul_rn(stage1_2, stage1_3);
const float stage1_5 = __fmaf_rn(stage1_0, stage1_1, stage1_4);
const float stage1_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage1_7 = 1.0f;
const float stage1_8 = Profile(p,0,2,z,y,x);
const float stage1_9 = __fmaf_rn(stage1_3, stage1_8, stage1_5);
const float stage1_10 = __fmul_rn(stage1_7, stage1_9);
const float stage1_11 = -1.0f;
const float stage1_12 = derivative_1;
const float stage1_13 = __fmul_rn(stage1_11, stage1_12);
const float stage1_14 = __fadd_rn(stage1_10, stage1_13);
const float stage1_15 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage1_16 = SafeRead(p.field.inputs[0],z,y,x);
const float stage1_17 = __fmul_rn(stage1_15, stage1_16);
const float stage1_18 = __fmaf_rn(stage1_6, stage1_14, stage1_17);
const float stage1_19 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage1_20 = derivative_2;
const float stage1_21 = __fmul_rn(stage1_7, stage1_20);
const float stage1_22 = derivative_3;
const float stage1_23 = __fmul_rn(stage1_11, stage1_22);
const float stage1_24 = __fadd_rn(stage1_21, stage1_23);
const float stage1_25 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage1_26 = SafeRead(p.field.inputs[1],z,y,x);
const float stage1_27 = __fmul_rn(stage1_25, stage1_26);
const float stage1_28 = __fmaf_rn(stage1_19, stage1_24, stage1_27);
const float stage1_29 = Profile(p,5,1,z,y,x);
const float stage1_30 = ReadPsi<false>(p,5,z,y,x);
const float stage1_31 = Profile(p,5,0,z,y,x);
const float stage1_32 = derivative_5;
const float stage1_33 = __fmul_rn(stage1_31, stage1_32);
const float stage1_34 = __fmaf_rn(stage1_29, stage1_30, stage1_33);
const float stage1_35 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage1_36 = derivative_4;
const float stage1_37 = __fmul_rn(stage1_7, stage1_36);
const float stage1_38 = Profile(p,5,2,z,y,x);
const float stage1_39 = __fmaf_rn(stage1_32, stage1_38, stage1_34);
const float stage1_40 = __fmul_rn(stage1_11, stage1_39);
const float stage1_41 = __fadd_rn(stage1_37, stage1_40);
const float stage1_42 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage1_43 = SafeRead(p.field.inputs[2],z,y,x);
const float stage1_44 = __fmul_rn(stage1_42, stage1_43);
const float stage1_45 = __fmaf_rn(stage1_35, stage1_41, stage1_44);
next_psi_0 = stage1_5;
next_0 = stage1_18;
next_1 = stage1_28;
next_psi_5 = stage1_34;
next_2 = stage1_45;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,0,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,0);
if(owned) {
Publish(observation,p.field.outputs[1],4,0,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,0,z,y,x,next_2);
}
}
}
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_0=next_psi_0;
current_psi_E_5=next_psi_5;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-1,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<15) old_E_0=s[960+(0*14+qy+(0)-1)*32+qx-1];
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<15) old_E_1=previous_y_1;
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<15) old_E_2=s[960+(1*14+qy+(0)-1)*32+qx-1];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_5=0.f;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[960+(1*14+qy+(1)-1)*32+qx-1]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[960+(0*14+qy+(1)-1)*32+qx-1]);
const float stage2_0 = Profile(p,0,1,z,y,x);
const float stage2_1 = previous_psi_0_0;
const float stage2_2 = Profile(p,0,0,z,y,x);
const float stage2_3 = derivative_0;
const float stage2_4 = __fmul_rn(stage2_2, stage2_3);
const float stage2_5 = __fmaf_rn(stage2_0, stage2_1, stage2_4);
const float stage2_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage2_7 = (-stage2_6);
const float stage2_8 = 1.0f;
const float stage2_9 = Profile(p,0,2,z,y,x);
const float stage2_10 = __fmaf_rn(stage2_3, stage2_9, stage2_5);
const float stage2_11 = __fmul_rn(stage2_8, stage2_10);
const float stage2_12 = -1.0f;
const float stage2_13 = derivative_1;
const float stage2_14 = __fmul_rn(stage2_12, stage2_13);
const float stage2_15 = __fadd_rn(stage2_11, stage2_14);
const float stage2_16 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage2_17 = old_H_0;
const float stage2_18 = __fmul_rn(stage2_16, stage2_17);
const float stage2_19 = __fmaf_rn(stage2_7, stage2_15, stage2_18);
const float stage2_20 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage2_21 = (-stage2_20);
const float stage2_22 = derivative_2;
const float stage2_23 = __fmul_rn(stage2_8, stage2_22);
const float stage2_24 = derivative_3;
const float stage2_25 = __fmul_rn(stage2_12, stage2_24);
const float stage2_26 = __fadd_rn(stage2_23, stage2_25);
const float stage2_27 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage2_28 = old_H_1;
const float stage2_29 = __fmul_rn(stage2_27, stage2_28);
const float stage2_30 = __fmaf_rn(stage2_21, stage2_26, stage2_29);
const float stage2_31 = Profile(p,5,1,z,y,x);
const float stage2_32 = previous_psi_0_5;
const float stage2_33 = Profile(p,5,0,z,y,x);
const float stage2_34 = derivative_5;
const float stage2_35 = __fmul_rn(stage2_33, stage2_34);
const float stage2_36 = __fmaf_rn(stage2_31, stage2_32, stage2_35);
const float stage2_37 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage2_38 = (-stage2_37);
const float stage2_39 = derivative_4;
const float stage2_40 = __fmul_rn(stage2_8, stage2_39);
const float stage2_41 = Profile(p,5,2,z,y,x);
const float stage2_42 = __fmaf_rn(stage2_34, stage2_41, stage2_36);
const float stage2_43 = __fmul_rn(stage2_12, stage2_42);
const float stage2_44 = __fadd_rn(stage2_40, stage2_43);
const float stage2_45 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage2_46 = old_H_2;
const float stage2_47 = __fmul_rn(stage2_45, stage2_46);
const float stage2_48 = __fmaf_rn(stage2_38, stage2_44, stage2_47);
next_psi_0 = stage2_5;
next_0 = stage2_19;
next_1 = stage2_30;
next_psi_5 = stage2_36;
next_2 = stage2_48;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,1,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,1,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,1);
if(owned) {
Publish(observation,p.field.outputs[2],2,1,z,y,x,next_2);
}
}
}
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) previous_y_0=current_H_1;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) previous_psi_0_0=current_psi_H_0;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) previous_psi_0_5=current_psi_H_5;
if(wave>2 && qx>=1 && qx<31 && qy>=1 && qy<14) old_H_0=s[1856+(0*13+qy+(0)-1)*32+qx-1];
if(wave>2 && qx>=1 && qx<31 && qy>=1 && qy<14) old_H_2=s[1856+(1*13+qy+(0)-1)*32+qx-1];
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_0=next_psi_0;
current_psi_H_5=next_psi_5;
}
__syncthreads();
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) s[1856+(0*13+qy+(0)-1)*32+qx-1]=current_H_0;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) s[1856+(1*13+qy+(0)-1)*32+qx-1]=current_H_2;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) s[960+(0*14+qy+(0)-1)*32+qx-1]=current_E_0;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) s[960+(1*14+qy+(0)-1)*32+qx-1]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-1,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_2;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_5=0.f;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[1856+(1*13+qy+(-1)-1)*32+qx-1]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[1856+(0*13+qy+(-1)-1)*32+qx-1]);
const float stage3_0 = Profile(p,0,1,z,y,x);
const float stage3_1 = previous_psi_1_0;
const float stage3_2 = Profile(p,0,0,z,y,x);
const float stage3_3 = derivative_0;
const float stage3_4 = __fmul_rn(stage3_2, stage3_3);
const float stage3_5 = __fmaf_rn(stage3_0, stage3_1, stage3_4);
const float stage3_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage3_7 = 1.0f;
const float stage3_8 = Profile(p,0,2,z,y,x);
const float stage3_9 = __fmaf_rn(stage3_3, stage3_8, stage3_5);
const float stage3_10 = __fmul_rn(stage3_7, stage3_9);
const float stage3_11 = -1.0f;
const float stage3_12 = derivative_1;
const float stage3_13 = __fmul_rn(stage3_11, stage3_12);
const float stage3_14 = __fadd_rn(stage3_10, stage3_13);
const float stage3_15 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage3_16 = old_E_0;
const float stage3_17 = __fmul_rn(stage3_15, stage3_16);
const float stage3_18 = __fmaf_rn(stage3_6, stage3_14, stage3_17);
const float stage3_19 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage3_20 = derivative_2;
const float stage3_21 = __fmul_rn(stage3_7, stage3_20);
const float stage3_22 = derivative_3;
const float stage3_23 = __fmul_rn(stage3_11, stage3_22);
const float stage3_24 = __fadd_rn(stage3_21, stage3_23);
const float stage3_25 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage3_26 = old_E_1;
const float stage3_27 = __fmul_rn(stage3_25, stage3_26);
const float stage3_28 = __fmaf_rn(stage3_19, stage3_24, stage3_27);
const float stage3_29 = Profile(p,5,1,z,y,x);
const float stage3_30 = previous_psi_1_5;
const float stage3_31 = Profile(p,5,0,z,y,x);
const float stage3_32 = derivative_5;
const float stage3_33 = __fmul_rn(stage3_31, stage3_32);
const float stage3_34 = __fmaf_rn(stage3_29, stage3_30, stage3_33);
const float stage3_35 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage3_36 = derivative_4;
const float stage3_37 = __fmul_rn(stage3_7, stage3_36);
const float stage3_38 = Profile(p,5,2,z,y,x);
const float stage3_39 = __fmaf_rn(stage3_32, stage3_38, stage3_34);
const float stage3_40 = __fmul_rn(stage3_11, stage3_39);
const float stage3_41 = __fadd_rn(stage3_37, stage3_40);
const float stage3_42 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage3_43 = old_E_2;
const float stage3_44 = __fmul_rn(stage3_42, stage3_43);
const float stage3_45 = __fmaf_rn(stage3_35, stage3_41, stage3_44);
next_psi_0 = stage3_5;
next_0 = stage3_18;
next_1 = stage3_28;
next_psi_5 = stage3_34;
next_2 = stage3_45;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,1,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,1);
if(owned) {
Publish(observation,p.field.outputs[1],4,1,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,1,z,y,x,next_2);
}
}
}
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) previous_y_1=current_E_1;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) previous_psi_1_0=current_psi_E_0;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) previous_psi_1_5=current_psi_E_5;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_0=next_psi_0;
current_psi_E_5=next_psi_5;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-2,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<14) old_E_0=s[2688+(0*12+qy+(0)-2)*32+qx-2];
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<14) old_E_1=previous_y_3;
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<14) old_E_2=s[2688+(1*12+qy+(0)-2)*32+qx-2];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_5=0.f;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[2688+(1*12+qy+(1)-2)*32+qx-2]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[2688+(0*12+qy+(1)-2)*32+qx-2]);
const float stage4_0 = Profile(p,0,1,z,y,x);
const float stage4_1 = previous_psi_2_0;
const float stage4_2 = Profile(p,0,0,z,y,x);
const float stage4_3 = derivative_0;
const float stage4_4 = __fmul_rn(stage4_2, stage4_3);
const float stage4_5 = __fmaf_rn(stage4_0, stage4_1, stage4_4);
const float stage4_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage4_7 = (-stage4_6);
const float stage4_8 = 1.0f;
const float stage4_9 = Profile(p,0,2,z,y,x);
const float stage4_10 = __fmaf_rn(stage4_3, stage4_9, stage4_5);
const float stage4_11 = __fmul_rn(stage4_8, stage4_10);
const float stage4_12 = -1.0f;
const float stage4_13 = derivative_1;
const float stage4_14 = __fmul_rn(stage4_12, stage4_13);
const float stage4_15 = __fadd_rn(stage4_11, stage4_14);
const float stage4_16 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage4_17 = old_H_0;
const float stage4_18 = __fmul_rn(stage4_16, stage4_17);
const float stage4_19 = __fmaf_rn(stage4_7, stage4_15, stage4_18);
const float stage4_20 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage4_21 = (-stage4_20);
const float stage4_22 = derivative_2;
const float stage4_23 = __fmul_rn(stage4_8, stage4_22);
const float stage4_24 = derivative_3;
const float stage4_25 = __fmul_rn(stage4_12, stage4_24);
const float stage4_26 = __fadd_rn(stage4_23, stage4_25);
const float stage4_27 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage4_28 = old_H_1;
const float stage4_29 = __fmul_rn(stage4_27, stage4_28);
const float stage4_30 = __fmaf_rn(stage4_21, stage4_26, stage4_29);
const float stage4_31 = Profile(p,5,1,z,y,x);
const float stage4_32 = previous_psi_2_5;
const float stage4_33 = Profile(p,5,0,z,y,x);
const float stage4_34 = derivative_5;
const float stage4_35 = __fmul_rn(stage4_33, stage4_34);
const float stage4_36 = __fmaf_rn(stage4_31, stage4_32, stage4_35);
const float stage4_37 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage4_38 = (-stage4_37);
const float stage4_39 = derivative_4;
const float stage4_40 = __fmul_rn(stage4_8, stage4_39);
const float stage4_41 = Profile(p,5,2,z,y,x);
const float stage4_42 = __fmaf_rn(stage4_34, stage4_41, stage4_36);
const float stage4_43 = __fmul_rn(stage4_12, stage4_42);
const float stage4_44 = __fadd_rn(stage4_40, stage4_43);
const float stage4_45 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage4_46 = old_H_2;
const float stage4_47 = __fmul_rn(stage4_45, stage4_46);
const float stage4_48 = __fmaf_rn(stage4_38, stage4_44, stage4_47);
next_psi_0 = stage4_5;
next_0 = stage4_19;
next_1 = stage4_30;
next_psi_5 = stage4_36;
next_2 = stage4_48;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,2,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,2,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,2);
if(owned) {
Publish(observation,p.field.outputs[2],2,2,z,y,x,next_2);
}
}
}
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) previous_y_2=current_H_1;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) previous_psi_2_0=current_psi_H_0;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) previous_psi_2_5=current_psi_H_5;
if(wave>4 && qx>=2 && qx<30 && qy>=2 && qy<13) old_H_0=s[3456+(0*11+qy+(0)-2)*32+qx-2];
if(wave>4 && qx>=2 && qx<30 && qy>=2 && qy<13) old_H_2=s[3456+(1*11+qy+(0)-2)*32+qx-2];
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_0=next_psi_0;
current_psi_H_5=next_psi_5;
}
__syncthreads();
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) s[3456+(0*11+qy+(0)-2)*32+qx-2]=current_H_0;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) s[3456+(1*11+qy+(0)-2)*32+qx-2]=current_H_2;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) s[2688+(0*12+qy+(0)-2)*32+qx-2]=current_E_0;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) s[2688+(1*12+qy+(0)-2)*32+qx-2]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-2,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_4;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_5=0.f;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[3456+(1*11+qy+(-1)-2)*32+qx-2]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[3456+(0*11+qy+(-1)-2)*32+qx-2]);
const float stage5_0 = Profile(p,0,1,z,y,x);
const float stage5_1 = previous_psi_3_0;
const float stage5_2 = Profile(p,0,0,z,y,x);
const float stage5_3 = derivative_0;
const float stage5_4 = __fmul_rn(stage5_2, stage5_3);
const float stage5_5 = __fmaf_rn(stage5_0, stage5_1, stage5_4);
const float stage5_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage5_7 = 1.0f;
const float stage5_8 = Profile(p,0,2,z,y,x);
const float stage5_9 = __fmaf_rn(stage5_3, stage5_8, stage5_5);
const float stage5_10 = __fmul_rn(stage5_7, stage5_9);
const float stage5_11 = -1.0f;
const float stage5_12 = derivative_1;
const float stage5_13 = __fmul_rn(stage5_11, stage5_12);
const float stage5_14 = __fadd_rn(stage5_10, stage5_13);
const float stage5_15 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage5_16 = old_E_0;
const float stage5_17 = __fmul_rn(stage5_15, stage5_16);
const float stage5_18 = __fmaf_rn(stage5_6, stage5_14, stage5_17);
const float stage5_19 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage5_20 = derivative_2;
const float stage5_21 = __fmul_rn(stage5_7, stage5_20);
const float stage5_22 = derivative_3;
const float stage5_23 = __fmul_rn(stage5_11, stage5_22);
const float stage5_24 = __fadd_rn(stage5_21, stage5_23);
const float stage5_25 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage5_26 = old_E_1;
const float stage5_27 = __fmul_rn(stage5_25, stage5_26);
const float stage5_28 = __fmaf_rn(stage5_19, stage5_24, stage5_27);
const float stage5_29 = Profile(p,5,1,z,y,x);
const float stage5_30 = previous_psi_3_5;
const float stage5_31 = Profile(p,5,0,z,y,x);
const float stage5_32 = derivative_5;
const float stage5_33 = __fmul_rn(stage5_31, stage5_32);
const float stage5_34 = __fmaf_rn(stage5_29, stage5_30, stage5_33);
const float stage5_35 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage5_36 = derivative_4;
const float stage5_37 = __fmul_rn(stage5_7, stage5_36);
const float stage5_38 = Profile(p,5,2,z,y,x);
const float stage5_39 = __fmaf_rn(stage5_32, stage5_38, stage5_34);
const float stage5_40 = __fmul_rn(stage5_11, stage5_39);
const float stage5_41 = __fadd_rn(stage5_37, stage5_40);
const float stage5_42 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage5_43 = old_E_2;
const float stage5_44 = __fmul_rn(stage5_42, stage5_43);
const float stage5_45 = __fmaf_rn(stage5_35, stage5_41, stage5_44);
next_psi_0 = stage5_5;
next_0 = stage5_18;
next_1 = stage5_28;
next_psi_5 = stage5_34;
next_2 = stage5_45;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,2,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,2);
if(owned) {
Publish(observation,p.field.outputs[1],4,2,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,2,z,y,x,next_2);
}
}
}
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) previous_y_3=current_E_1;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) previous_psi_3_0=current_psi_E_0;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) previous_psi_3_5=current_psi_E_5;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_0=next_psi_0;
current_psi_E_5=next_psi_5;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-3,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<13) old_E_0=s[4160+(0*10+qy+(0)-3)*32+qx-3];
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<13) old_E_1=previous_y_5;
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<13) old_E_2=s[4160+(1*10+qy+(0)-3)*32+qx-3];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_5=0.f;
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[4160+(1*10+qy+(1)-3)*32+qx-3]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[4160+(0*10+qy+(1)-3)*32+qx-3]);
const float stage6_0 = Profile(p,0,1,z,y,x);
const float stage6_1 = previous_psi_4_0;
const float stage6_2 = Profile(p,0,0,z,y,x);
const float stage6_3 = derivative_0;
const float stage6_4 = __fmul_rn(stage6_2, stage6_3);
const float stage6_5 = __fmaf_rn(stage6_0, stage6_1, stage6_4);
const float stage6_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage6_7 = (-stage6_6);
const float stage6_8 = 1.0f;
const float stage6_9 = Profile(p,0,2,z,y,x);
const float stage6_10 = __fmaf_rn(stage6_3, stage6_9, stage6_5);
const float stage6_11 = __fmul_rn(stage6_8, stage6_10);
const float stage6_12 = -1.0f;
const float stage6_13 = derivative_1;
const float stage6_14 = __fmul_rn(stage6_12, stage6_13);
const float stage6_15 = __fadd_rn(stage6_11, stage6_14);
const float stage6_16 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage6_17 = old_H_0;
const float stage6_18 = __fmul_rn(stage6_16, stage6_17);
const float stage6_19 = __fmaf_rn(stage6_7, stage6_15, stage6_18);
const float stage6_20 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage6_21 = (-stage6_20);
const float stage6_22 = derivative_2;
const float stage6_23 = __fmul_rn(stage6_8, stage6_22);
const float stage6_24 = derivative_3;
const float stage6_25 = __fmul_rn(stage6_12, stage6_24);
const float stage6_26 = __fadd_rn(stage6_23, stage6_25);
const float stage6_27 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage6_28 = old_H_1;
const float stage6_29 = __fmul_rn(stage6_27, stage6_28);
const float stage6_30 = __fmaf_rn(stage6_21, stage6_26, stage6_29);
const float stage6_31 = Profile(p,5,1,z,y,x);
const float stage6_32 = previous_psi_4_5;
const float stage6_33 = Profile(p,5,0,z,y,x);
const float stage6_34 = derivative_5;
const float stage6_35 = __fmul_rn(stage6_33, stage6_34);
const float stage6_36 = __fmaf_rn(stage6_31, stage6_32, stage6_35);
const float stage6_37 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage6_38 = (-stage6_37);
const float stage6_39 = derivative_4;
const float stage6_40 = __fmul_rn(stage6_8, stage6_39);
const float stage6_41 = Profile(p,5,2,z,y,x);
const float stage6_42 = __fmaf_rn(stage6_34, stage6_41, stage6_36);
const float stage6_43 = __fmul_rn(stage6_12, stage6_42);
const float stage6_44 = __fadd_rn(stage6_40, stage6_43);
const float stage6_45 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage6_46 = old_H_2;
const float stage6_47 = __fmul_rn(stage6_45, stage6_46);
const float stage6_48 = __fmaf_rn(stage6_38, stage6_44, stage6_47);
next_psi_0 = stage6_5;
next_0 = stage6_19;
next_1 = stage6_30;
next_psi_5 = stage6_36;
next_2 = stage6_48;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,3,z,y,x,next_0);
static_cast<float *>(final.fields[0].data)[BeamzOffset3D<false>(final.fields[0],z,y,x)]=next_0;
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,3,z,y,x,next_1);
static_cast<float *>(final.fields[1].data)[BeamzOffset3D<false>(final.fields[1],z,y,x)]=next_1;
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,3);
if(owned) {
Publish(observation,p.field.outputs[2],2,3,z,y,x,next_2);
static_cast<float *>(final.fields[2].data)[BeamzOffset3D<false>(final.fields[2],z,y,x)]=next_2;
}
}
if(owned) WritePsi<false>(p,0,z,y,x,next_psi_0);
if(owned) WritePsi<false>(p,5,z,y,x,next_psi_5);
}
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) previous_y_4=current_H_1;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) previous_psi_4_0=current_psi_H_0;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) previous_psi_4_5=current_psi_H_5;
if(wave>6 && qx>=3 && qx<29 && qy>=3 && qy<12) old_H_0=s[4800+(0*9+qy+(0)-3)*32+qx-3];
if(wave>6 && qx>=3 && qx<29 && qy>=3 && qy<12) old_H_2=s[4800+(1*9+qy+(0)-3)*32+qx-3];
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_0=next_psi_0;
current_psi_H_5=next_psi_5;
}
__syncthreads();
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) s[4800+(0*9+qy+(0)-3)*32+qx-3]=current_H_0;
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) s[4800+(1*9+qy+(0)-3)*32+qx-3]=current_H_2;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) s[4160+(0*10+qy+(0)-3)*32+qx-3]=current_E_0;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) s[4160+(1*10+qy+(0)-3)*32+qx-3]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-3,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_6;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_5=0.f;
if(wave>=7 && qx>=4 && qx<29 && qy>=4 && qy<12) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[4800+(1*9+qy+(-1)-3)*32+qx-3]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[4800+(0*9+qy+(-1)-3)*32+qx-3]);
const float stage7_0 = Profile(p,0,1,z,y,x);
const float stage7_1 = previous_psi_5_0;
const float stage7_2 = Profile(p,0,0,z,y,x);
const float stage7_3 = derivative_0;
const float stage7_4 = __fmul_rn(stage7_2, stage7_3);
const float stage7_5 = __fmaf_rn(stage7_0, stage7_1, stage7_4);
const float stage7_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage7_7 = 1.0f;
const float stage7_8 = Profile(p,0,2,z,y,x);
const float stage7_9 = __fmaf_rn(stage7_3, stage7_8, stage7_5);
const float stage7_10 = __fmul_rn(stage7_7, stage7_9);
const float stage7_11 = -1.0f;
const float stage7_12 = derivative_1;
const float stage7_13 = __fmul_rn(stage7_11, stage7_12);
const float stage7_14 = __fadd_rn(stage7_10, stage7_13);
const float stage7_15 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage7_16 = old_E_0;
const float stage7_17 = __fmul_rn(stage7_15, stage7_16);
const float stage7_18 = __fmaf_rn(stage7_6, stage7_14, stage7_17);
const float stage7_19 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage7_20 = derivative_2;
const float stage7_21 = __fmul_rn(stage7_7, stage7_20);
const float stage7_22 = derivative_3;
const float stage7_23 = __fmul_rn(stage7_11, stage7_22);
const float stage7_24 = __fadd_rn(stage7_21, stage7_23);
const float stage7_25 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage7_26 = old_E_1;
const float stage7_27 = __fmul_rn(stage7_25, stage7_26);
const float stage7_28 = __fmaf_rn(stage7_19, stage7_24, stage7_27);
const float stage7_29 = Profile(p,5,1,z,y,x);
const float stage7_30 = previous_psi_5_5;
const float stage7_31 = Profile(p,5,0,z,y,x);
const float stage7_32 = derivative_5;
const float stage7_33 = __fmul_rn(stage7_31, stage7_32);
const float stage7_34 = __fmaf_rn(stage7_29, stage7_30, stage7_33);
const float stage7_35 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage7_36 = derivative_4;
const float stage7_37 = __fmul_rn(stage7_7, stage7_36);
const float stage7_38 = Profile(p,5,2,z,y,x);
const float stage7_39 = __fmaf_rn(stage7_32, stage7_38, stage7_34);
const float stage7_40 = __fmul_rn(stage7_11, stage7_39);
const float stage7_41 = __fadd_rn(stage7_37, stage7_40);
const float stage7_42 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage7_43 = old_E_2;
const float stage7_44 = __fmul_rn(stage7_42, stage7_43);
const float stage7_45 = __fmaf_rn(stage7_35, stage7_41, stage7_44);
next_psi_0 = stage7_5;
next_0 = stage7_18;
next_1 = stage7_28;
next_psi_5 = stage7_34;
next_2 = stage7_45;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,3,z,y,x,next_0);
static_cast<float *>(final.fields[3].data)[BeamzOffset3D<false>(final.fields[3],z,y,x)]=next_0;
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,3);
if(owned) {
Publish(observation,p.field.outputs[1],4,3,z,y,x,next_1);
static_cast<float *>(final.fields[4].data)[BeamzOffset3D<false>(final.fields[4],z,y,x)]=next_1;
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,3,z,y,x,next_2);
static_cast<float *>(final.fields[5].data)[BeamzOffset3D<false>(final.fields[5],z,y,x)]=next_2;
}
}
if(owned) WritePsi<false>(p,0,z,y,x,next_psi_0);
if(owned) WritePsi<false>(p,5,z,y,x,next_psi_5);
}
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) previous_y_5=current_E_1;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) previous_psi_5_0=current_psi_E_0;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) previous_psi_5_5=current_psi_E_5;
if(wave>=7 && qx>=4 && qx<29 && qy>=4 && qy<12) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_0=next_psi_0;
current_psi_E_5=next_psi_5;
}
}
    if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) previous_y_6=current_H_1;
  }
}


constexpr int DeepWarpD4M3F32ScalarHBytePackedSource20Threads=352;
constexpr int DeepWarpD4M3F32ScalarHBytePackedSource20SharedBytes=40448;
template<bool Packed> __global__ __launch_bounds__(352,1) void DeepWarpD4M3F32ScalarHBytePackedSource20(
    const __grid_constant__ PairCpmlPhase h,const __grid_constant__ PairCpmlPhase e,
    const __grid_constant__ PairFieldOutputs final,const __grid_constant__ FusedHSources hs,
    const __grid_constant__ FusedHSources es,const __grid_constant__ Observation observation,PairCpmlGrid grid) {
  if(blockDim.x!=352)return;
  extern __shared__ float s[];
  auto coordinate=[&](int axis,int block,int size) {const int low=(grid.high[axis]+size-1)/size;
    return (3&(1<<axis)) ? (block<low?block*size:grid.boundary[axis]+(block-low)*size) : grid.high[axis]+block*size;};
  auto end=[&](int axis,int block,int origin,int size) {const int low=(grid.high[axis]+size-1)/size;
    return min(origin+size,(3&(1<<axis))?(block<low?grid.high[axis]:grid.extent[axis]):grid.boundary[axis]);};
  const int ox=coordinate(2,blockIdx.x,25),oy=coordinate(1,blockIdx.y,4),oz=coordinate(0,blockIdx.z,64);
  const int end_x=end(2,blockIdx.x,ox,25),end_y=end(1,blockIdx.y,oy,4),end_z=end(0,blockIdx.z,oz,64);
  const int qx=threadIdx.x&31,qy=threadIdx.x>>5;
  __shared__ bool active_sources[6];
  if(threadIdx.x<6){const int c=threadIdx.x;const auto &g=c<3?hs.groups[c]:es.groups[c-3];
    const auto *starts=static_cast<const int *>(g.starts.data);bool active=false;
    for(int i=0;i<g.coefficients.dims[0];++i)
      active |= starts[3*i]<oz+68 && starts[3*i]+g.coefficients.dims[1]>oz-4 &&
                starts[3*i+1]<oy+8 && starts[3*i+1]+g.coefficients.dims[2]>oy-4 &&
                starts[3*i+2]<ox+29 && starts[3*i+2]+g.coefficients.dims[3]>ox-4;
    active_sources[c]=active;}
  __syncthreads();
  float previous_y_0=0.f;
float previous_y_1=0.f;
float previous_y_2=0.f;
float previous_y_3=0.f;
float previous_y_4=0.f;
float previous_y_5=0.f;
float previous_y_6=0.f;
  for(int wave=0;wave<end_z-oz+7;++wave){
    float current_H_0=0.f,old_H_0=0.f;
float current_H_1=0.f,old_H_1=0.f;
float current_H_2=0.f,old_H_2=0.f;
float current_E_0=0.f,old_E_0=0.f;
float current_E_1=0.f,old_E_1=0.f;
float current_E_2=0.f,old_E_2=0.f;
float current_psi_H_0=0.f;
float current_psi_H_1=0.f;
float current_psi_H_2=0.f;
float current_psi_H_5=0.f;
float current_psi_E_0=0.f;
float current_psi_E_1=0.f;
float current_psi_E_2=0.f;
float current_psi_E_5=0.f;
    {
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-0,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_5=0.f;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) {
const float derivative_0=Difference(p,0,2,1,z,y,x,SafeRead(p.field.inputs[5],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[5],z+(0),y+(1),x+(0)));
const float derivative_1=Difference(p,0,1,0,z,y,x,SafeRead(p.field.inputs[4],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[4],z+(1),y+(0),x+(0)));
const float derivative_2=Difference(p,0,0,0,z,y,x,SafeRead(p.field.inputs[3],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[3],z+(1),y+(0),x+(0)));
const float derivative_3=Difference(p,0,2,2,z,y,x,SafeRead(p.field.inputs[5],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[5],z+(0),y+(0),x+(1)));
const float derivative_4=Difference(p,0,1,2,z,y,x,SafeRead(p.field.inputs[4],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[4],z+(0),y+(0),x+(1)));
const float derivative_5=Difference(p,0,0,1,z,y,x,SafeRead(p.field.inputs[3],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[3],z+(0),y+(1),x+(0)));
const float stage0_0 = Profile(p,0,1,z,y,x);
const float stage0_1 = ReadPsi<false>(p,0,z,y,x);
const float stage0_2 = Profile(p,0,0,z,y,x);
const float stage0_3 = derivative_0;
const float stage0_4 = __fmul_rn(stage0_2, stage0_3);
const float stage0_5 = __fmaf_rn(stage0_0, stage0_1, stage0_4);
const float stage0_6 = Profile(p,1,1,z,y,x);
const float stage0_7 = ReadPsi<false>(p,1,z,y,x);
const float stage0_8 = Profile(p,1,0,z,y,x);
const float stage0_9 = derivative_1;
const float stage0_10 = __fmul_rn(stage0_8, stage0_9);
const float stage0_11 = __fmaf_rn(stage0_6, stage0_7, stage0_10);
const float stage0_12 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage0_13 = (-stage0_12);
const float stage0_14 = 1.0f;
const float stage0_15 = Profile(p,0,2,z,y,x);
const float stage0_16 = __fmaf_rn(stage0_3, stage0_15, stage0_5);
const float stage0_17 = __fmul_rn(stage0_14, stage0_16);
const float stage0_18 = -1.0f;
const float stage0_19 = Profile(p,1,2,z,y,x);
const float stage0_20 = __fmaf_rn(stage0_9, stage0_19, stage0_11);
const float stage0_21 = __fmul_rn(stage0_18, stage0_20);
const float stage0_22 = __fadd_rn(stage0_17, stage0_21);
const float stage0_23 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage0_24 = SafeRead(p.field.inputs[0],z,y,x);
const float stage0_25 = __fmul_rn(stage0_23, stage0_24);
const float stage0_26 = __fmaf_rn(stage0_13, stage0_22, stage0_25);
const float stage0_27 = Profile(p,2,1,z,y,x);
const float stage0_28 = ReadPsi<false>(p,2,z,y,x);
const float stage0_29 = Profile(p,2,0,z,y,x);
const float stage0_30 = derivative_2;
const float stage0_31 = __fmul_rn(stage0_29, stage0_30);
const float stage0_32 = __fmaf_rn(stage0_27, stage0_28, stage0_31);
const float stage0_33 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage0_34 = (-stage0_33);
const float stage0_35 = Profile(p,2,2,z,y,x);
const float stage0_36 = __fmaf_rn(stage0_30, stage0_35, stage0_32);
const float stage0_37 = __fmul_rn(stage0_14, stage0_36);
const float stage0_38 = derivative_3;
const float stage0_39 = __fmul_rn(stage0_18, stage0_38);
const float stage0_40 = __fadd_rn(stage0_37, stage0_39);
const float stage0_41 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage0_42 = SafeRead(p.field.inputs[1],z,y,x);
const float stage0_43 = __fmul_rn(stage0_41, stage0_42);
const float stage0_44 = __fmaf_rn(stage0_34, stage0_40, stage0_43);
const float stage0_45 = Profile(p,5,1,z,y,x);
const float stage0_46 = ReadPsi<false>(p,5,z,y,x);
const float stage0_47 = Profile(p,5,0,z,y,x);
const float stage0_48 = derivative_5;
const float stage0_49 = __fmul_rn(stage0_47, stage0_48);
const float stage0_50 = __fmaf_rn(stage0_45, stage0_46, stage0_49);
const float stage0_51 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage0_52 = (-stage0_51);
const float stage0_53 = derivative_4;
const float stage0_54 = __fmul_rn(stage0_14, stage0_53);
const float stage0_55 = Profile(p,5,2,z,y,x);
const float stage0_56 = __fmaf_rn(stage0_48, stage0_55, stage0_50);
const float stage0_57 = __fmul_rn(stage0_18, stage0_56);
const float stage0_58 = __fadd_rn(stage0_54, stage0_57);
const float stage0_59 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage0_60 = SafeRead(p.field.inputs[2],z,y,x);
const float stage0_61 = __fmul_rn(stage0_59, stage0_60);
const float stage0_62 = __fmaf_rn(stage0_52, stage0_58, stage0_61);
next_psi_0 = stage0_5;
next_psi_1 = stage0_11;
next_0 = stage0_26;
next_psi_2 = stage0_32;
next_1 = stage0_44;
next_psi_5 = stage0_50;
next_2 = stage0_62;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,0,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,0,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,0);
if(owned) {
Publish(observation,p.field.outputs[2],2,0,z,y,x,next_2);
}
}
}
if(wave>0 && qx>=0 && qx<32 && qy>=0 && qy<11) old_H_0=s[0+(0*11+qy+(0)-0)*32+qx-0];
if(wave>0 && qx>=0 && qx<32 && qy>=0 && qy<11) old_H_2=s[0+(1*11+qy+(0)-0)*32+qx-0];
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_0=next_psi_0;
current_psi_H_1=next_psi_1;
current_psi_H_2=next_psi_2;
current_psi_H_5=next_psi_5;
}
__syncthreads();
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[0+(0*11+qy+(0)-0)*32+qx-0]=current_H_0;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[0+(1*11+qy+(0)-0)*32+qx-0]=current_H_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-0,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_0;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_5=0.f;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[0+(1*11+qy+(-1)-0)*32+qx-0]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[0+(0*11+qy+(-1)-0)*32+qx-0]);
const float stage1_0 = Profile(p,0,1,z,y,x);
const float stage1_1 = ReadPsi<false>(p,0,z,y,x);
const float stage1_2 = Profile(p,0,0,z,y,x);
const float stage1_3 = derivative_0;
const float stage1_4 = __fmul_rn(stage1_2, stage1_3);
const float stage1_5 = __fmaf_rn(stage1_0, stage1_1, stage1_4);
const float stage1_6 = Profile(p,1,1,z,y,x);
const float stage1_7 = ReadPsi<false>(p,1,z,y,x);
const float stage1_8 = Profile(p,1,0,z,y,x);
const float stage1_9 = derivative_1;
const float stage1_10 = __fmul_rn(stage1_8, stage1_9);
const float stage1_11 = __fmaf_rn(stage1_6, stage1_7, stage1_10);
const float stage1_12 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage1_13 = 1.0f;
const float stage1_14 = Profile(p,0,2,z,y,x);
const float stage1_15 = __fmaf_rn(stage1_3, stage1_14, stage1_5);
const float stage1_16 = __fmul_rn(stage1_13, stage1_15);
const float stage1_17 = -1.0f;
const float stage1_18 = Profile(p,1,2,z,y,x);
const float stage1_19 = __fmaf_rn(stage1_9, stage1_18, stage1_11);
const float stage1_20 = __fmul_rn(stage1_17, stage1_19);
const float stage1_21 = __fadd_rn(stage1_16, stage1_20);
const float stage1_22 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage1_23 = SafeRead(p.field.inputs[0],z,y,x);
const float stage1_24 = __fmul_rn(stage1_22, stage1_23);
const float stage1_25 = __fmaf_rn(stage1_12, stage1_21, stage1_24);
const float stage1_26 = Profile(p,2,1,z,y,x);
const float stage1_27 = ReadPsi<false>(p,2,z,y,x);
const float stage1_28 = Profile(p,2,0,z,y,x);
const float stage1_29 = derivative_2;
const float stage1_30 = __fmul_rn(stage1_28, stage1_29);
const float stage1_31 = __fmaf_rn(stage1_26, stage1_27, stage1_30);
const float stage1_32 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage1_33 = Profile(p,2,2,z,y,x);
const float stage1_34 = __fmaf_rn(stage1_29, stage1_33, stage1_31);
const float stage1_35 = __fmul_rn(stage1_13, stage1_34);
const float stage1_36 = derivative_3;
const float stage1_37 = __fmul_rn(stage1_17, stage1_36);
const float stage1_38 = __fadd_rn(stage1_35, stage1_37);
const float stage1_39 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage1_40 = SafeRead(p.field.inputs[1],z,y,x);
const float stage1_41 = __fmul_rn(stage1_39, stage1_40);
const float stage1_42 = __fmaf_rn(stage1_32, stage1_38, stage1_41);
const float stage1_43 = Profile(p,5,1,z,y,x);
const float stage1_44 = ReadPsi<false>(p,5,z,y,x);
const float stage1_45 = Profile(p,5,0,z,y,x);
const float stage1_46 = derivative_5;
const float stage1_47 = __fmul_rn(stage1_45, stage1_46);
const float stage1_48 = __fmaf_rn(stage1_43, stage1_44, stage1_47);
const float stage1_49 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage1_50 = derivative_4;
const float stage1_51 = __fmul_rn(stage1_13, stage1_50);
const float stage1_52 = Profile(p,5,2,z,y,x);
const float stage1_53 = __fmaf_rn(stage1_46, stage1_52, stage1_48);
const float stage1_54 = __fmul_rn(stage1_17, stage1_53);
const float stage1_55 = __fadd_rn(stage1_51, stage1_54);
const float stage1_56 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage1_57 = SafeRead(p.field.inputs[2],z,y,x);
const float stage1_58 = __fmul_rn(stage1_56, stage1_57);
const float stage1_59 = __fmaf_rn(stage1_49, stage1_55, stage1_58);
next_psi_0 = stage1_5;
next_psi_1 = stage1_11;
next_0 = stage1_25;
next_psi_2 = stage1_31;
next_1 = stage1_42;
next_psi_5 = stage1_48;
next_2 = stage1_59;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,0,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,0);
if(owned) {
Publish(observation,p.field.outputs[1],4,0,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,0,z,y,x,next_2);
}
}
}
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_0=next_psi_0;
current_psi_E_1=next_psi_1;
current_psi_E_2=next_psi_2;
current_psi_E_5=next_psi_5;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-1,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<11) old_E_0=s[704+(0*10+qy+(0)-1)*32+qx-1];
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<11) old_E_1=previous_y_1;
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<11) old_E_2=s[704+(1*10+qy+(0)-1)*32+qx-1];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_5=0.f;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[704+(1*10+qy+(1)-1)*32+qx-1]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[704+(0*10+qy+(1)-1)*32+qx-1]);
const float stage2_0 = Profile(p,0,1,z,y,x);
const float stage2_1 = s[3584+(0*11+qy-0)*32+qx-0];
const float stage2_2 = Profile(p,0,0,z,y,x);
const float stage2_3 = derivative_0;
const float stage2_4 = __fmul_rn(stage2_2, stage2_3);
const float stage2_5 = __fmaf_rn(stage2_0, stage2_1, stage2_4);
const float stage2_6 = Profile(p,1,1,z,y,x);
const float stage2_7 = s[3584+(1*11+qy-0)*32+qx-0];
const float stage2_8 = Profile(p,1,0,z,y,x);
const float stage2_9 = derivative_1;
const float stage2_10 = __fmul_rn(stage2_8, stage2_9);
const float stage2_11 = __fmaf_rn(stage2_6, stage2_7, stage2_10);
const float stage2_12 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage2_13 = (-stage2_12);
const float stage2_14 = 1.0f;
const float stage2_15 = Profile(p,0,2,z,y,x);
const float stage2_16 = __fmaf_rn(stage2_3, stage2_15, stage2_5);
const float stage2_17 = __fmul_rn(stage2_14, stage2_16);
const float stage2_18 = -1.0f;
const float stage2_19 = Profile(p,1,2,z,y,x);
const float stage2_20 = __fmaf_rn(stage2_9, stage2_19, stage2_11);
const float stage2_21 = __fmul_rn(stage2_18, stage2_20);
const float stage2_22 = __fadd_rn(stage2_17, stage2_21);
const float stage2_23 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage2_24 = old_H_0;
const float stage2_25 = __fmul_rn(stage2_23, stage2_24);
const float stage2_26 = __fmaf_rn(stage2_13, stage2_22, stage2_25);
const float stage2_27 = Profile(p,2,1,z,y,x);
const float stage2_28 = s[3584+(2*11+qy-0)*32+qx-0];
const float stage2_29 = Profile(p,2,0,z,y,x);
const float stage2_30 = derivative_2;
const float stage2_31 = __fmul_rn(stage2_29, stage2_30);
const float stage2_32 = __fmaf_rn(stage2_27, stage2_28, stage2_31);
const float stage2_33 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage2_34 = (-stage2_33);
const float stage2_35 = Profile(p,2,2,z,y,x);
const float stage2_36 = __fmaf_rn(stage2_30, stage2_35, stage2_32);
const float stage2_37 = __fmul_rn(stage2_14, stage2_36);
const float stage2_38 = derivative_3;
const float stage2_39 = __fmul_rn(stage2_18, stage2_38);
const float stage2_40 = __fadd_rn(stage2_37, stage2_39);
const float stage2_41 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage2_42 = old_H_1;
const float stage2_43 = __fmul_rn(stage2_41, stage2_42);
const float stage2_44 = __fmaf_rn(stage2_34, stage2_40, stage2_43);
const float stage2_45 = Profile(p,5,1,z,y,x);
const float stage2_46 = s[3584+(3*11+qy-0)*32+qx-0];
const float stage2_47 = Profile(p,5,0,z,y,x);
const float stage2_48 = derivative_5;
const float stage2_49 = __fmul_rn(stage2_47, stage2_48);
const float stage2_50 = __fmaf_rn(stage2_45, stage2_46, stage2_49);
const float stage2_51 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage2_52 = (-stage2_51);
const float stage2_53 = derivative_4;
const float stage2_54 = __fmul_rn(stage2_14, stage2_53);
const float stage2_55 = Profile(p,5,2,z,y,x);
const float stage2_56 = __fmaf_rn(stage2_48, stage2_55, stage2_50);
const float stage2_57 = __fmul_rn(stage2_18, stage2_56);
const float stage2_58 = __fadd_rn(stage2_54, stage2_57);
const float stage2_59 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage2_60 = old_H_2;
const float stage2_61 = __fmul_rn(stage2_59, stage2_60);
const float stage2_62 = __fmaf_rn(stage2_52, stage2_58, stage2_61);
next_psi_0 = stage2_5;
next_psi_1 = stage2_11;
next_0 = stage2_26;
next_psi_2 = stage2_32;
next_1 = stage2_44;
next_psi_5 = stage2_50;
next_2 = stage2_62;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,1,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,1,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,1);
if(owned) {
Publish(observation,p.field.outputs[2],2,1,z,y,x,next_2);
}
}
}
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) previous_y_0=current_H_1;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[3584+(0*11+qy-0)*32+qx-0]=current_psi_H_0;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[3584+(1*11+qy-0)*32+qx-0]=current_psi_H_1;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[3584+(2*11+qy-0)*32+qx-0]=current_psi_H_2;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[3584+(3*11+qy-0)*32+qx-0]=current_psi_H_5;
if(wave>2 && qx>=1 && qx<31 && qy>=1 && qy<10) old_H_0=s[1344+(0*9+qy+(0)-1)*32+qx-1];
if(wave>2 && qx>=1 && qx<31 && qy>=1 && qy<10) old_H_2=s[1344+(1*9+qy+(0)-1)*32+qx-1];
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_0=next_psi_0;
current_psi_H_1=next_psi_1;
current_psi_H_2=next_psi_2;
current_psi_H_5=next_psi_5;
}
__syncthreads();
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[1344+(0*9+qy+(0)-1)*32+qx-1]=current_H_0;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[1344+(1*9+qy+(0)-1)*32+qx-1]=current_H_2;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[704+(0*10+qy+(0)-1)*32+qx-1]=current_E_0;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[704+(1*10+qy+(0)-1)*32+qx-1]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-1,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_2;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_5=0.f;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[1344+(1*9+qy+(-1)-1)*32+qx-1]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[1344+(0*9+qy+(-1)-1)*32+qx-1]);
const float stage3_0 = Profile(p,0,1,z,y,x);
const float stage3_1 = s[4992+(0*10+qy-1)*32+qx-1];
const float stage3_2 = Profile(p,0,0,z,y,x);
const float stage3_3 = derivative_0;
const float stage3_4 = __fmul_rn(stage3_2, stage3_3);
const float stage3_5 = __fmaf_rn(stage3_0, stage3_1, stage3_4);
const float stage3_6 = Profile(p,1,1,z,y,x);
const float stage3_7 = s[4992+(1*10+qy-1)*32+qx-1];
const float stage3_8 = Profile(p,1,0,z,y,x);
const float stage3_9 = derivative_1;
const float stage3_10 = __fmul_rn(stage3_8, stage3_9);
const float stage3_11 = __fmaf_rn(stage3_6, stage3_7, stage3_10);
const float stage3_12 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage3_13 = 1.0f;
const float stage3_14 = Profile(p,0,2,z,y,x);
const float stage3_15 = __fmaf_rn(stage3_3, stage3_14, stage3_5);
const float stage3_16 = __fmul_rn(stage3_13, stage3_15);
const float stage3_17 = -1.0f;
const float stage3_18 = Profile(p,1,2,z,y,x);
const float stage3_19 = __fmaf_rn(stage3_9, stage3_18, stage3_11);
const float stage3_20 = __fmul_rn(stage3_17, stage3_19);
const float stage3_21 = __fadd_rn(stage3_16, stage3_20);
const float stage3_22 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage3_23 = old_E_0;
const float stage3_24 = __fmul_rn(stage3_22, stage3_23);
const float stage3_25 = __fmaf_rn(stage3_12, stage3_21, stage3_24);
const float stage3_26 = Profile(p,2,1,z,y,x);
const float stage3_27 = s[4992+(2*10+qy-1)*32+qx-1];
const float stage3_28 = Profile(p,2,0,z,y,x);
const float stage3_29 = derivative_2;
const float stage3_30 = __fmul_rn(stage3_28, stage3_29);
const float stage3_31 = __fmaf_rn(stage3_26, stage3_27, stage3_30);
const float stage3_32 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage3_33 = Profile(p,2,2,z,y,x);
const float stage3_34 = __fmaf_rn(stage3_29, stage3_33, stage3_31);
const float stage3_35 = __fmul_rn(stage3_13, stage3_34);
const float stage3_36 = derivative_3;
const float stage3_37 = __fmul_rn(stage3_17, stage3_36);
const float stage3_38 = __fadd_rn(stage3_35, stage3_37);
const float stage3_39 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage3_40 = old_E_1;
const float stage3_41 = __fmul_rn(stage3_39, stage3_40);
const float stage3_42 = __fmaf_rn(stage3_32, stage3_38, stage3_41);
const float stage3_43 = Profile(p,5,1,z,y,x);
const float stage3_44 = s[4992+(3*10+qy-1)*32+qx-1];
const float stage3_45 = Profile(p,5,0,z,y,x);
const float stage3_46 = derivative_5;
const float stage3_47 = __fmul_rn(stage3_45, stage3_46);
const float stage3_48 = __fmaf_rn(stage3_43, stage3_44, stage3_47);
const float stage3_49 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage3_50 = derivative_4;
const float stage3_51 = __fmul_rn(stage3_13, stage3_50);
const float stage3_52 = Profile(p,5,2,z,y,x);
const float stage3_53 = __fmaf_rn(stage3_46, stage3_52, stage3_48);
const float stage3_54 = __fmul_rn(stage3_17, stage3_53);
const float stage3_55 = __fadd_rn(stage3_51, stage3_54);
const float stage3_56 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage3_57 = old_E_2;
const float stage3_58 = __fmul_rn(stage3_56, stage3_57);
const float stage3_59 = __fmaf_rn(stage3_49, stage3_55, stage3_58);
next_psi_0 = stage3_5;
next_psi_1 = stage3_11;
next_0 = stage3_25;
next_psi_2 = stage3_31;
next_1 = stage3_42;
next_psi_5 = stage3_48;
next_2 = stage3_59;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,1,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,1);
if(owned) {
Publish(observation,p.field.outputs[1],4,1,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,1,z,y,x,next_2);
}
}
}
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) previous_y_1=current_E_1;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[4992+(0*10+qy-1)*32+qx-1]=current_psi_E_0;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[4992+(1*10+qy-1)*32+qx-1]=current_psi_E_1;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[4992+(2*10+qy-1)*32+qx-1]=current_psi_E_2;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[4992+(3*10+qy-1)*32+qx-1]=current_psi_E_5;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_0=next_psi_0;
current_psi_E_1=next_psi_1;
current_psi_E_2=next_psi_2;
current_psi_E_5=next_psi_5;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-2,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<10) old_E_0=s[1920+(0*8+qy+(0)-2)*32+qx-2];
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<10) old_E_1=previous_y_3;
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<10) old_E_2=s[1920+(1*8+qy+(0)-2)*32+qx-2];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_5=0.f;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[1920+(1*8+qy+(1)-2)*32+qx-2]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[1920+(0*8+qy+(1)-2)*32+qx-2]);
const float stage4_0 = Profile(p,0,1,z,y,x);
const float stage4_1 = s[6272+(0*9+qy-1)*32+qx-1];
const float stage4_2 = Profile(p,0,0,z,y,x);
const float stage4_3 = derivative_0;
const float stage4_4 = __fmul_rn(stage4_2, stage4_3);
const float stage4_5 = __fmaf_rn(stage4_0, stage4_1, stage4_4);
const float stage4_6 = Profile(p,1,1,z,y,x);
const float stage4_7 = s[6272+(1*9+qy-1)*32+qx-1];
const float stage4_8 = Profile(p,1,0,z,y,x);
const float stage4_9 = derivative_1;
const float stage4_10 = __fmul_rn(stage4_8, stage4_9);
const float stage4_11 = __fmaf_rn(stage4_6, stage4_7, stage4_10);
const float stage4_12 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage4_13 = (-stage4_12);
const float stage4_14 = 1.0f;
const float stage4_15 = Profile(p,0,2,z,y,x);
const float stage4_16 = __fmaf_rn(stage4_3, stage4_15, stage4_5);
const float stage4_17 = __fmul_rn(stage4_14, stage4_16);
const float stage4_18 = -1.0f;
const float stage4_19 = Profile(p,1,2,z,y,x);
const float stage4_20 = __fmaf_rn(stage4_9, stage4_19, stage4_11);
const float stage4_21 = __fmul_rn(stage4_18, stage4_20);
const float stage4_22 = __fadd_rn(stage4_17, stage4_21);
const float stage4_23 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage4_24 = old_H_0;
const float stage4_25 = __fmul_rn(stage4_23, stage4_24);
const float stage4_26 = __fmaf_rn(stage4_13, stage4_22, stage4_25);
const float stage4_27 = Profile(p,2,1,z,y,x);
const float stage4_28 = s[6272+(2*9+qy-1)*32+qx-1];
const float stage4_29 = Profile(p,2,0,z,y,x);
const float stage4_30 = derivative_2;
const float stage4_31 = __fmul_rn(stage4_29, stage4_30);
const float stage4_32 = __fmaf_rn(stage4_27, stage4_28, stage4_31);
const float stage4_33 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage4_34 = (-stage4_33);
const float stage4_35 = Profile(p,2,2,z,y,x);
const float stage4_36 = __fmaf_rn(stage4_30, stage4_35, stage4_32);
const float stage4_37 = __fmul_rn(stage4_14, stage4_36);
const float stage4_38 = derivative_3;
const float stage4_39 = __fmul_rn(stage4_18, stage4_38);
const float stage4_40 = __fadd_rn(stage4_37, stage4_39);
const float stage4_41 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage4_42 = old_H_1;
const float stage4_43 = __fmul_rn(stage4_41, stage4_42);
const float stage4_44 = __fmaf_rn(stage4_34, stage4_40, stage4_43);
const float stage4_45 = Profile(p,5,1,z,y,x);
const float stage4_46 = s[6272+(3*9+qy-1)*32+qx-1];
const float stage4_47 = Profile(p,5,0,z,y,x);
const float stage4_48 = derivative_5;
const float stage4_49 = __fmul_rn(stage4_47, stage4_48);
const float stage4_50 = __fmaf_rn(stage4_45, stage4_46, stage4_49);
const float stage4_51 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage4_52 = (-stage4_51);
const float stage4_53 = derivative_4;
const float stage4_54 = __fmul_rn(stage4_14, stage4_53);
const float stage4_55 = Profile(p,5,2,z,y,x);
const float stage4_56 = __fmaf_rn(stage4_48, stage4_55, stage4_50);
const float stage4_57 = __fmul_rn(stage4_18, stage4_56);
const float stage4_58 = __fadd_rn(stage4_54, stage4_57);
const float stage4_59 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage4_60 = old_H_2;
const float stage4_61 = __fmul_rn(stage4_59, stage4_60);
const float stage4_62 = __fmaf_rn(stage4_52, stage4_58, stage4_61);
next_psi_0 = stage4_5;
next_psi_1 = stage4_11;
next_0 = stage4_26;
next_psi_2 = stage4_32;
next_1 = stage4_44;
next_psi_5 = stage4_50;
next_2 = stage4_62;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,2,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,2,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,2);
if(owned) {
Publish(observation,p.field.outputs[2],2,2,z,y,x,next_2);
}
}
}
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) previous_y_2=current_H_1;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[6272+(0*9+qy-1)*32+qx-1]=current_psi_H_0;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[6272+(1*9+qy-1)*32+qx-1]=current_psi_H_1;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[6272+(2*9+qy-1)*32+qx-1]=current_psi_H_2;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[6272+(3*9+qy-1)*32+qx-1]=current_psi_H_5;
if(wave>4 && qx>=2 && qx<30 && qy>=2 && qy<9) old_H_0=s[2432+(0*7+qy+(0)-2)*32+qx-2];
if(wave>4 && qx>=2 && qx<30 && qy>=2 && qy<9) old_H_2=s[2432+(1*7+qy+(0)-2)*32+qx-2];
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_0=next_psi_0;
current_psi_H_1=next_psi_1;
current_psi_H_2=next_psi_2;
current_psi_H_5=next_psi_5;
}
__syncthreads();
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[2432+(0*7+qy+(0)-2)*32+qx-2]=current_H_0;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[2432+(1*7+qy+(0)-2)*32+qx-2]=current_H_2;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[1920+(0*8+qy+(0)-2)*32+qx-2]=current_E_0;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[1920+(1*8+qy+(0)-2)*32+qx-2]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-2,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_4;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_5=0.f;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[2432+(1*7+qy+(-1)-2)*32+qx-2]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[2432+(0*7+qy+(-1)-2)*32+qx-2]);
const float stage5_0 = Profile(p,0,1,z,y,x);
const float stage5_1 = s[7424+(0*8+qy-2)*32+qx-2];
const float stage5_2 = Profile(p,0,0,z,y,x);
const float stage5_3 = derivative_0;
const float stage5_4 = __fmul_rn(stage5_2, stage5_3);
const float stage5_5 = __fmaf_rn(stage5_0, stage5_1, stage5_4);
const float stage5_6 = Profile(p,1,1,z,y,x);
const float stage5_7 = s[7424+(1*8+qy-2)*32+qx-2];
const float stage5_8 = Profile(p,1,0,z,y,x);
const float stage5_9 = derivative_1;
const float stage5_10 = __fmul_rn(stage5_8, stage5_9);
const float stage5_11 = __fmaf_rn(stage5_6, stage5_7, stage5_10);
const float stage5_12 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage5_13 = 1.0f;
const float stage5_14 = Profile(p,0,2,z,y,x);
const float stage5_15 = __fmaf_rn(stage5_3, stage5_14, stage5_5);
const float stage5_16 = __fmul_rn(stage5_13, stage5_15);
const float stage5_17 = -1.0f;
const float stage5_18 = Profile(p,1,2,z,y,x);
const float stage5_19 = __fmaf_rn(stage5_9, stage5_18, stage5_11);
const float stage5_20 = __fmul_rn(stage5_17, stage5_19);
const float stage5_21 = __fadd_rn(stage5_16, stage5_20);
const float stage5_22 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage5_23 = old_E_0;
const float stage5_24 = __fmul_rn(stage5_22, stage5_23);
const float stage5_25 = __fmaf_rn(stage5_12, stage5_21, stage5_24);
const float stage5_26 = Profile(p,2,1,z,y,x);
const float stage5_27 = s[7424+(2*8+qy-2)*32+qx-2];
const float stage5_28 = Profile(p,2,0,z,y,x);
const float stage5_29 = derivative_2;
const float stage5_30 = __fmul_rn(stage5_28, stage5_29);
const float stage5_31 = __fmaf_rn(stage5_26, stage5_27, stage5_30);
const float stage5_32 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage5_33 = Profile(p,2,2,z,y,x);
const float stage5_34 = __fmaf_rn(stage5_29, stage5_33, stage5_31);
const float stage5_35 = __fmul_rn(stage5_13, stage5_34);
const float stage5_36 = derivative_3;
const float stage5_37 = __fmul_rn(stage5_17, stage5_36);
const float stage5_38 = __fadd_rn(stage5_35, stage5_37);
const float stage5_39 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage5_40 = old_E_1;
const float stage5_41 = __fmul_rn(stage5_39, stage5_40);
const float stage5_42 = __fmaf_rn(stage5_32, stage5_38, stage5_41);
const float stage5_43 = Profile(p,5,1,z,y,x);
const float stage5_44 = s[7424+(3*8+qy-2)*32+qx-2];
const float stage5_45 = Profile(p,5,0,z,y,x);
const float stage5_46 = derivative_5;
const float stage5_47 = __fmul_rn(stage5_45, stage5_46);
const float stage5_48 = __fmaf_rn(stage5_43, stage5_44, stage5_47);
const float stage5_49 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage5_50 = derivative_4;
const float stage5_51 = __fmul_rn(stage5_13, stage5_50);
const float stage5_52 = Profile(p,5,2,z,y,x);
const float stage5_53 = __fmaf_rn(stage5_46, stage5_52, stage5_48);
const float stage5_54 = __fmul_rn(stage5_17, stage5_53);
const float stage5_55 = __fadd_rn(stage5_51, stage5_54);
const float stage5_56 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage5_57 = old_E_2;
const float stage5_58 = __fmul_rn(stage5_56, stage5_57);
const float stage5_59 = __fmaf_rn(stage5_49, stage5_55, stage5_58);
next_psi_0 = stage5_5;
next_psi_1 = stage5_11;
next_0 = stage5_25;
next_psi_2 = stage5_31;
next_1 = stage5_42;
next_psi_5 = stage5_48;
next_2 = stage5_59;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,2,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,2);
if(owned) {
Publish(observation,p.field.outputs[1],4,2,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,2,z,y,x,next_2);
}
}
}
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) previous_y_3=current_E_1;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[7424+(0*8+qy-2)*32+qx-2]=current_psi_E_0;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[7424+(1*8+qy-2)*32+qx-2]=current_psi_E_1;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[7424+(2*8+qy-2)*32+qx-2]=current_psi_E_2;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[7424+(3*8+qy-2)*32+qx-2]=current_psi_E_5;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_0=next_psi_0;
current_psi_E_1=next_psi_1;
current_psi_E_2=next_psi_2;
current_psi_E_5=next_psi_5;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-3,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<9) old_E_0=s[2880+(0*6+qy+(0)-3)*32+qx-3];
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<9) old_E_1=previous_y_5;
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<9) old_E_2=s[2880+(1*6+qy+(0)-3)*32+qx-3];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_5=0.f;
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<8) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[2880+(1*6+qy+(1)-3)*32+qx-3]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[2880+(0*6+qy+(1)-3)*32+qx-3]);
const float stage6_0 = Profile(p,0,1,z,y,x);
const float stage6_1 = s[8448+(0*7+qy-2)*32+qx-2];
const float stage6_2 = Profile(p,0,0,z,y,x);
const float stage6_3 = derivative_0;
const float stage6_4 = __fmul_rn(stage6_2, stage6_3);
const float stage6_5 = __fmaf_rn(stage6_0, stage6_1, stage6_4);
const float stage6_6 = Profile(p,1,1,z,y,x);
const float stage6_7 = s[8448+(1*7+qy-2)*32+qx-2];
const float stage6_8 = Profile(p,1,0,z,y,x);
const float stage6_9 = derivative_1;
const float stage6_10 = __fmul_rn(stage6_8, stage6_9);
const float stage6_11 = __fmaf_rn(stage6_6, stage6_7, stage6_10);
const float stage6_12 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage6_13 = (-stage6_12);
const float stage6_14 = 1.0f;
const float stage6_15 = Profile(p,0,2,z,y,x);
const float stage6_16 = __fmaf_rn(stage6_3, stage6_15, stage6_5);
const float stage6_17 = __fmul_rn(stage6_14, stage6_16);
const float stage6_18 = -1.0f;
const float stage6_19 = Profile(p,1,2,z,y,x);
const float stage6_20 = __fmaf_rn(stage6_9, stage6_19, stage6_11);
const float stage6_21 = __fmul_rn(stage6_18, stage6_20);
const float stage6_22 = __fadd_rn(stage6_17, stage6_21);
const float stage6_23 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage6_24 = old_H_0;
const float stage6_25 = __fmul_rn(stage6_23, stage6_24);
const float stage6_26 = __fmaf_rn(stage6_13, stage6_22, stage6_25);
const float stage6_27 = Profile(p,2,1,z,y,x);
const float stage6_28 = s[8448+(2*7+qy-2)*32+qx-2];
const float stage6_29 = Profile(p,2,0,z,y,x);
const float stage6_30 = derivative_2;
const float stage6_31 = __fmul_rn(stage6_29, stage6_30);
const float stage6_32 = __fmaf_rn(stage6_27, stage6_28, stage6_31);
const float stage6_33 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage6_34 = (-stage6_33);
const float stage6_35 = Profile(p,2,2,z,y,x);
const float stage6_36 = __fmaf_rn(stage6_30, stage6_35, stage6_32);
const float stage6_37 = __fmul_rn(stage6_14, stage6_36);
const float stage6_38 = derivative_3;
const float stage6_39 = __fmul_rn(stage6_18, stage6_38);
const float stage6_40 = __fadd_rn(stage6_37, stage6_39);
const float stage6_41 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage6_42 = old_H_1;
const float stage6_43 = __fmul_rn(stage6_41, stage6_42);
const float stage6_44 = __fmaf_rn(stage6_34, stage6_40, stage6_43);
const float stage6_45 = Profile(p,5,1,z,y,x);
const float stage6_46 = s[8448+(3*7+qy-2)*32+qx-2];
const float stage6_47 = Profile(p,5,0,z,y,x);
const float stage6_48 = derivative_5;
const float stage6_49 = __fmul_rn(stage6_47, stage6_48);
const float stage6_50 = __fmaf_rn(stage6_45, stage6_46, stage6_49);
const float stage6_51 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage6_52 = (-stage6_51);
const float stage6_53 = derivative_4;
const float stage6_54 = __fmul_rn(stage6_14, stage6_53);
const float stage6_55 = Profile(p,5,2,z,y,x);
const float stage6_56 = __fmaf_rn(stage6_48, stage6_55, stage6_50);
const float stage6_57 = __fmul_rn(stage6_18, stage6_56);
const float stage6_58 = __fadd_rn(stage6_54, stage6_57);
const float stage6_59 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage6_60 = old_H_2;
const float stage6_61 = __fmul_rn(stage6_59, stage6_60);
const float stage6_62 = __fmaf_rn(stage6_52, stage6_58, stage6_61);
next_psi_0 = stage6_5;
next_psi_1 = stage6_11;
next_0 = stage6_26;
next_psi_2 = stage6_32;
next_1 = stage6_44;
next_psi_5 = stage6_50;
next_2 = stage6_62;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,3,z,y,x,next_0);
static_cast<float *>(final.fields[0].data)[BeamzOffset3D<false>(final.fields[0],z,y,x)]=next_0;
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,3,z,y,x,next_1);
static_cast<float *>(final.fields[1].data)[BeamzOffset3D<false>(final.fields[1],z,y,x)]=next_1;
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,3);
if(owned) {
Publish(observation,p.field.outputs[2],2,3,z,y,x,next_2);
static_cast<float *>(final.fields[2].data)[BeamzOffset3D<false>(final.fields[2],z,y,x)]=next_2;
}
}
if(owned) WritePsi<false>(p,0,z,y,x,next_psi_0);
if(owned) WritePsi<false>(p,1,z,y,x,next_psi_1);
if(owned) WritePsi<false>(p,2,z,y,x,next_psi_2);
if(owned) WritePsi<false>(p,5,z,y,x,next_psi_5);
}
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) previous_y_4=current_H_1;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[8448+(0*7+qy-2)*32+qx-2]=current_psi_H_0;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[8448+(1*7+qy-2)*32+qx-2]=current_psi_H_1;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[8448+(2*7+qy-2)*32+qx-2]=current_psi_H_2;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[8448+(3*7+qy-2)*32+qx-2]=current_psi_H_5;
if(wave>6 && qx>=3 && qx<29 && qy>=3 && qy<8) old_H_0=s[3264+(0*5+qy+(0)-3)*32+qx-3];
if(wave>6 && qx>=3 && qx<29 && qy>=3 && qy<8) old_H_2=s[3264+(1*5+qy+(0)-3)*32+qx-3];
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<8) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_0=next_psi_0;
current_psi_H_1=next_psi_1;
current_psi_H_2=next_psi_2;
current_psi_H_5=next_psi_5;
}
__syncthreads();
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<8) s[3264+(0*5+qy+(0)-3)*32+qx-3]=current_H_0;
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<8) s[3264+(1*5+qy+(0)-3)*32+qx-3]=current_H_2;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[2880+(0*6+qy+(0)-3)*32+qx-3]=current_E_0;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[2880+(1*6+qy+(0)-3)*32+qx-3]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-3,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_6;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_5=0.f;
if(wave>=7 && qx>=4 && qx<29 && qy>=4 && qy<8) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[3264+(1*5+qy+(-1)-3)*32+qx-3]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[3264+(0*5+qy+(-1)-3)*32+qx-3]);
const float stage7_0 = Profile(p,0,1,z,y,x);
const float stage7_1 = s[9344+(0*6+qy-3)*32+qx-3];
const float stage7_2 = Profile(p,0,0,z,y,x);
const float stage7_3 = derivative_0;
const float stage7_4 = __fmul_rn(stage7_2, stage7_3);
const float stage7_5 = __fmaf_rn(stage7_0, stage7_1, stage7_4);
const float stage7_6 = Profile(p,1,1,z,y,x);
const float stage7_7 = s[9344+(1*6+qy-3)*32+qx-3];
const float stage7_8 = Profile(p,1,0,z,y,x);
const float stage7_9 = derivative_1;
const float stage7_10 = __fmul_rn(stage7_8, stage7_9);
const float stage7_11 = __fmaf_rn(stage7_6, stage7_7, stage7_10);
const float stage7_12 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage7_13 = 1.0f;
const float stage7_14 = Profile(p,0,2,z,y,x);
const float stage7_15 = __fmaf_rn(stage7_3, stage7_14, stage7_5);
const float stage7_16 = __fmul_rn(stage7_13, stage7_15);
const float stage7_17 = -1.0f;
const float stage7_18 = Profile(p,1,2,z,y,x);
const float stage7_19 = __fmaf_rn(stage7_9, stage7_18, stage7_11);
const float stage7_20 = __fmul_rn(stage7_17, stage7_19);
const float stage7_21 = __fadd_rn(stage7_16, stage7_20);
const float stage7_22 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage7_23 = old_E_0;
const float stage7_24 = __fmul_rn(stage7_22, stage7_23);
const float stage7_25 = __fmaf_rn(stage7_12, stage7_21, stage7_24);
const float stage7_26 = Profile(p,2,1,z,y,x);
const float stage7_27 = s[9344+(2*6+qy-3)*32+qx-3];
const float stage7_28 = Profile(p,2,0,z,y,x);
const float stage7_29 = derivative_2;
const float stage7_30 = __fmul_rn(stage7_28, stage7_29);
const float stage7_31 = __fmaf_rn(stage7_26, stage7_27, stage7_30);
const float stage7_32 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage7_33 = Profile(p,2,2,z,y,x);
const float stage7_34 = __fmaf_rn(stage7_29, stage7_33, stage7_31);
const float stage7_35 = __fmul_rn(stage7_13, stage7_34);
const float stage7_36 = derivative_3;
const float stage7_37 = __fmul_rn(stage7_17, stage7_36);
const float stage7_38 = __fadd_rn(stage7_35, stage7_37);
const float stage7_39 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage7_40 = old_E_1;
const float stage7_41 = __fmul_rn(stage7_39, stage7_40);
const float stage7_42 = __fmaf_rn(stage7_32, stage7_38, stage7_41);
const float stage7_43 = Profile(p,5,1,z,y,x);
const float stage7_44 = s[9344+(3*6+qy-3)*32+qx-3];
const float stage7_45 = Profile(p,5,0,z,y,x);
const float stage7_46 = derivative_5;
const float stage7_47 = __fmul_rn(stage7_45, stage7_46);
const float stage7_48 = __fmaf_rn(stage7_43, stage7_44, stage7_47);
const float stage7_49 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage7_50 = derivative_4;
const float stage7_51 = __fmul_rn(stage7_13, stage7_50);
const float stage7_52 = Profile(p,5,2,z,y,x);
const float stage7_53 = __fmaf_rn(stage7_46, stage7_52, stage7_48);
const float stage7_54 = __fmul_rn(stage7_17, stage7_53);
const float stage7_55 = __fadd_rn(stage7_51, stage7_54);
const float stage7_56 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage7_57 = old_E_2;
const float stage7_58 = __fmul_rn(stage7_56, stage7_57);
const float stage7_59 = __fmaf_rn(stage7_49, stage7_55, stage7_58);
next_psi_0 = stage7_5;
next_psi_1 = stage7_11;
next_0 = stage7_25;
next_psi_2 = stage7_31;
next_1 = stage7_42;
next_psi_5 = stage7_48;
next_2 = stage7_59;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,3,z,y,x,next_0);
static_cast<float *>(final.fields[3].data)[BeamzOffset3D<false>(final.fields[3],z,y,x)]=next_0;
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,3);
if(owned) {
Publish(observation,p.field.outputs[1],4,3,z,y,x,next_1);
static_cast<float *>(final.fields[4].data)[BeamzOffset3D<false>(final.fields[4],z,y,x)]=next_1;
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,3,z,y,x,next_2);
static_cast<float *>(final.fields[5].data)[BeamzOffset3D<false>(final.fields[5],z,y,x)]=next_2;
}
}
if(owned) WritePsi<false>(p,0,z,y,x,next_psi_0);
if(owned) WritePsi<false>(p,1,z,y,x,next_psi_1);
if(owned) WritePsi<false>(p,2,z,y,x,next_psi_2);
if(owned) WritePsi<false>(p,5,z,y,x,next_psi_5);
}
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) previous_y_5=current_E_1;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[9344+(0*6+qy-3)*32+qx-3]=current_psi_E_0;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[9344+(1*6+qy-3)*32+qx-3]=current_psi_E_1;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[9344+(2*6+qy-3)*32+qx-3]=current_psi_E_2;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[9344+(3*6+qy-3)*32+qx-3]=current_psi_E_5;
if(wave>=7 && qx>=4 && qx<29 && qy>=4 && qy<8) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_0=next_psi_0;
current_psi_E_1=next_psi_1;
current_psi_E_2=next_psi_2;
current_psi_E_5=next_psi_5;
}
}
    if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<8) previous_y_6=current_H_1;
  }
}


constexpr int DeepWarpD4M4F32ScalarHBytePackedSource20Threads=480;
constexpr int DeepWarpD4M4F32ScalarHBytePackedSource20SharedBytes=21504;
template<bool Packed> __global__ __launch_bounds__(480,1) void DeepWarpD4M4F32ScalarHBytePackedSource20(
    const __grid_constant__ PairCpmlPhase h,const __grid_constant__ PairCpmlPhase e,
    const __grid_constant__ PairFieldOutputs final,const __grid_constant__ FusedHSources hs,
    const __grid_constant__ FusedHSources es,const __grid_constant__ Observation observation,PairCpmlGrid grid) {
  if(blockDim.x!=480)return;
  extern __shared__ float s[];
  auto coordinate=[&](int axis,int block,int size) {const int low=(grid.high[axis]+size-1)/size;
    return (4&(1<<axis)) ? (block<low?block*size:grid.boundary[axis]+(block-low)*size) : grid.high[axis]+block*size;};
  auto end=[&](int axis,int block,int origin,int size) {const int low=(grid.high[axis]+size-1)/size;
    return min(origin+size,(4&(1<<axis))?(block<low?grid.high[axis]:grid.extent[axis]):grid.boundary[axis]);};
  const int ox=coordinate(2,blockIdx.x,25),oy=coordinate(1,blockIdx.y,8),oz=coordinate(0,blockIdx.z,64);
  const int end_x=end(2,blockIdx.x,ox,25),end_y=end(1,blockIdx.y,oy,8),end_z=end(0,blockIdx.z,oz,64);
  const int qx=threadIdx.x&31,qy=threadIdx.x>>5;
  __shared__ bool active_sources[6];
  if(threadIdx.x<6){const int c=threadIdx.x;const auto &g=c<3?hs.groups[c]:es.groups[c-3];
    const auto *starts=static_cast<const int *>(g.starts.data);bool active=false;
    for(int i=0;i<g.coefficients.dims[0];++i)
      active |= starts[3*i]<oz+68 && starts[3*i]+g.coefficients.dims[1]>oz-4 &&
                starts[3*i+1]<oy+12 && starts[3*i+1]+g.coefficients.dims[2]>oy-4 &&
                starts[3*i+2]<ox+29 && starts[3*i+2]+g.coefficients.dims[3]>ox-4;
    active_sources[c]=active;}
  __syncthreads();
  float previous_y_0=0.f;
float previous_y_1=0.f;
float previous_y_2=0.f;
float previous_y_3=0.f;
float previous_y_4=0.f;
float previous_y_5=0.f;
float previous_y_6=0.f;
float previous_psi_0_3=0.f;
float previous_psi_0_4=0.f;
float previous_psi_1_3=0.f;
float previous_psi_1_4=0.f;
float previous_psi_2_3=0.f;
float previous_psi_2_4=0.f;
float previous_psi_3_3=0.f;
float previous_psi_3_4=0.f;
float previous_psi_4_3=0.f;
float previous_psi_4_4=0.f;
float previous_psi_5_3=0.f;
float previous_psi_5_4=0.f;
  for(int wave=0;wave<end_z-oz+7;++wave){
    float current_H_0=0.f,old_H_0=0.f;
float current_H_1=0.f,old_H_1=0.f;
float current_H_2=0.f,old_H_2=0.f;
float current_E_0=0.f,old_E_0=0.f;
float current_E_1=0.f,old_E_1=0.f;
float current_E_2=0.f,old_E_2=0.f;
float current_psi_H_3=0.f;
float current_psi_H_4=0.f;
float current_psi_E_3=0.f;
float current_psi_E_4=0.f;
    {
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-0,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) {
const float derivative_0=Difference(p,0,2,1,z,y,x,SafeRead(p.field.inputs[5],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[5],z+(0),y+(1),x+(0)));
const float derivative_1=Difference(p,0,1,0,z,y,x,SafeRead(p.field.inputs[4],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[4],z+(1),y+(0),x+(0)));
const float derivative_2=Difference(p,0,0,0,z,y,x,SafeRead(p.field.inputs[3],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[3],z+(1),y+(0),x+(0)));
const float derivative_3=Difference(p,0,2,2,z,y,x,SafeRead(p.field.inputs[5],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[5],z+(0),y+(0),x+(1)));
const float derivative_4=Difference(p,0,1,2,z,y,x,SafeRead(p.field.inputs[4],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[4],z+(0),y+(0),x+(1)));
const float derivative_5=Difference(p,0,0,1,z,y,x,SafeRead(p.field.inputs[3],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[3],z+(0),y+(1),x+(0)));
const float stage0_0 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage0_1 = (-stage0_0);
const float stage0_2 = 1.0f;
const float stage0_3 = derivative_0;
const float stage0_4 = __fmul_rn(stage0_2, stage0_3);
const float stage0_5 = -1.0f;
const float stage0_6 = derivative_1;
const float stage0_7 = __fmul_rn(stage0_5, stage0_6);
const float stage0_8 = __fadd_rn(stage0_4, stage0_7);
const float stage0_9 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage0_10 = SafeRead(p.field.inputs[0],z,y,x);
const float stage0_11 = __fmul_rn(stage0_9, stage0_10);
const float stage0_12 = __fmaf_rn(stage0_1, stage0_8, stage0_11);
const float stage0_13 = Profile(p,3,1,z,y,x);
const float stage0_14 = ReadPsi<false>(p,3,z,y,x);
const float stage0_15 = Profile(p,3,0,z,y,x);
const float stage0_16 = derivative_3;
const float stage0_17 = __fmul_rn(stage0_15, stage0_16);
const float stage0_18 = __fmaf_rn(stage0_13, stage0_14, stage0_17);
const float stage0_19 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage0_20 = (-stage0_19);
const float stage0_21 = derivative_2;
const float stage0_22 = __fmul_rn(stage0_2, stage0_21);
const float stage0_23 = Profile(p,3,2,z,y,x);
const float stage0_24 = __fmaf_rn(stage0_16, stage0_23, stage0_18);
const float stage0_25 = __fmul_rn(stage0_5, stage0_24);
const float stage0_26 = __fadd_rn(stage0_22, stage0_25);
const float stage0_27 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage0_28 = SafeRead(p.field.inputs[1],z,y,x);
const float stage0_29 = __fmul_rn(stage0_27, stage0_28);
const float stage0_30 = __fmaf_rn(stage0_20, stage0_26, stage0_29);
const float stage0_31 = Profile(p,4,1,z,y,x);
const float stage0_32 = ReadPsi<false>(p,4,z,y,x);
const float stage0_33 = Profile(p,4,0,z,y,x);
const float stage0_34 = derivative_4;
const float stage0_35 = __fmul_rn(stage0_33, stage0_34);
const float stage0_36 = __fmaf_rn(stage0_31, stage0_32, stage0_35);
const float stage0_37 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage0_38 = (-stage0_37);
const float stage0_39 = Profile(p,4,2,z,y,x);
const float stage0_40 = __fmaf_rn(stage0_34, stage0_39, stage0_36);
const float stage0_41 = __fmul_rn(stage0_2, stage0_40);
const float stage0_42 = derivative_5;
const float stage0_43 = __fmul_rn(stage0_5, stage0_42);
const float stage0_44 = __fadd_rn(stage0_41, stage0_43);
const float stage0_45 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage0_46 = SafeRead(p.field.inputs[2],z,y,x);
const float stage0_47 = __fmul_rn(stage0_45, stage0_46);
const float stage0_48 = __fmaf_rn(stage0_38, stage0_44, stage0_47);
next_0 = stage0_12;
next_psi_3 = stage0_18;
next_1 = stage0_30;
next_psi_4 = stage0_36;
next_2 = stage0_48;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,0,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,0,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,0);
if(owned) {
Publish(observation,p.field.outputs[2],2,0,z,y,x,next_2);
}
}
}
if(wave>0 && qx>=0 && qx<32 && qy>=0 && qy<15) old_H_0=s[0+(0*15+qy+(0)-0)*32+qx-0];
if(wave>0 && qx>=0 && qx<32 && qy>=0 && qy<15) old_H_2=s[0+(1*15+qy+(0)-0)*32+qx-0];
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_3=next_psi_3;
current_psi_H_4=next_psi_4;
}
__syncthreads();
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) s[0+(0*15+qy+(0)-0)*32+qx-0]=current_H_0;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) s[0+(1*15+qy+(0)-0)*32+qx-0]=current_H_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-0,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_0;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[0+(1*15+qy+(-1)-0)*32+qx-0]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[0+(0*15+qy+(-1)-0)*32+qx-0]);
const float stage1_0 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage1_1 = 1.0f;
const float stage1_2 = derivative_0;
const float stage1_3 = __fmul_rn(stage1_1, stage1_2);
const float stage1_4 = -1.0f;
const float stage1_5 = derivative_1;
const float stage1_6 = __fmul_rn(stage1_4, stage1_5);
const float stage1_7 = __fadd_rn(stage1_3, stage1_6);
const float stage1_8 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage1_9 = SafeRead(p.field.inputs[0],z,y,x);
const float stage1_10 = __fmul_rn(stage1_8, stage1_9);
const float stage1_11 = __fmaf_rn(stage1_0, stage1_7, stage1_10);
const float stage1_12 = Profile(p,3,1,z,y,x);
const float stage1_13 = ReadPsi<false>(p,3,z,y,x);
const float stage1_14 = Profile(p,3,0,z,y,x);
const float stage1_15 = derivative_3;
const float stage1_16 = __fmul_rn(stage1_14, stage1_15);
const float stage1_17 = __fmaf_rn(stage1_12, stage1_13, stage1_16);
const float stage1_18 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage1_19 = derivative_2;
const float stage1_20 = __fmul_rn(stage1_1, stage1_19);
const float stage1_21 = Profile(p,3,2,z,y,x);
const float stage1_22 = __fmaf_rn(stage1_15, stage1_21, stage1_17);
const float stage1_23 = __fmul_rn(stage1_4, stage1_22);
const float stage1_24 = __fadd_rn(stage1_20, stage1_23);
const float stage1_25 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage1_26 = SafeRead(p.field.inputs[1],z,y,x);
const float stage1_27 = __fmul_rn(stage1_25, stage1_26);
const float stage1_28 = __fmaf_rn(stage1_18, stage1_24, stage1_27);
const float stage1_29 = Profile(p,4,1,z,y,x);
const float stage1_30 = ReadPsi<false>(p,4,z,y,x);
const float stage1_31 = Profile(p,4,0,z,y,x);
const float stage1_32 = derivative_4;
const float stage1_33 = __fmul_rn(stage1_31, stage1_32);
const float stage1_34 = __fmaf_rn(stage1_29, stage1_30, stage1_33);
const float stage1_35 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage1_36 = Profile(p,4,2,z,y,x);
const float stage1_37 = __fmaf_rn(stage1_32, stage1_36, stage1_34);
const float stage1_38 = __fmul_rn(stage1_1, stage1_37);
const float stage1_39 = derivative_5;
const float stage1_40 = __fmul_rn(stage1_4, stage1_39);
const float stage1_41 = __fadd_rn(stage1_38, stage1_40);
const float stage1_42 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage1_43 = SafeRead(p.field.inputs[2],z,y,x);
const float stage1_44 = __fmul_rn(stage1_42, stage1_43);
const float stage1_45 = __fmaf_rn(stage1_35, stage1_41, stage1_44);
next_0 = stage1_11;
next_psi_3 = stage1_17;
next_1 = stage1_28;
next_psi_4 = stage1_34;
next_2 = stage1_45;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,0,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,0);
if(owned) {
Publish(observation,p.field.outputs[1],4,0,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,0,z,y,x,next_2);
}
}
}
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_3=next_psi_3;
current_psi_E_4=next_psi_4;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-1,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<15) old_E_0=s[960+(0*14+qy+(0)-1)*32+qx-1];
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<15) old_E_1=previous_y_1;
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<15) old_E_2=s[960+(1*14+qy+(0)-1)*32+qx-1];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[960+(1*14+qy+(1)-1)*32+qx-1]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[960+(0*14+qy+(1)-1)*32+qx-1]);
const float stage2_0 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage2_1 = (-stage2_0);
const float stage2_2 = 1.0f;
const float stage2_3 = derivative_0;
const float stage2_4 = __fmul_rn(stage2_2, stage2_3);
const float stage2_5 = -1.0f;
const float stage2_6 = derivative_1;
const float stage2_7 = __fmul_rn(stage2_5, stage2_6);
const float stage2_8 = __fadd_rn(stage2_4, stage2_7);
const float stage2_9 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage2_10 = old_H_0;
const float stage2_11 = __fmul_rn(stage2_9, stage2_10);
const float stage2_12 = __fmaf_rn(stage2_1, stage2_8, stage2_11);
const float stage2_13 = Profile(p,3,1,z,y,x);
const float stage2_14 = previous_psi_0_3;
const float stage2_15 = Profile(p,3,0,z,y,x);
const float stage2_16 = derivative_3;
const float stage2_17 = __fmul_rn(stage2_15, stage2_16);
const float stage2_18 = __fmaf_rn(stage2_13, stage2_14, stage2_17);
const float stage2_19 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage2_20 = (-stage2_19);
const float stage2_21 = derivative_2;
const float stage2_22 = __fmul_rn(stage2_2, stage2_21);
const float stage2_23 = Profile(p,3,2,z,y,x);
const float stage2_24 = __fmaf_rn(stage2_16, stage2_23, stage2_18);
const float stage2_25 = __fmul_rn(stage2_5, stage2_24);
const float stage2_26 = __fadd_rn(stage2_22, stage2_25);
const float stage2_27 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage2_28 = old_H_1;
const float stage2_29 = __fmul_rn(stage2_27, stage2_28);
const float stage2_30 = __fmaf_rn(stage2_20, stage2_26, stage2_29);
const float stage2_31 = Profile(p,4,1,z,y,x);
const float stage2_32 = previous_psi_0_4;
const float stage2_33 = Profile(p,4,0,z,y,x);
const float stage2_34 = derivative_4;
const float stage2_35 = __fmul_rn(stage2_33, stage2_34);
const float stage2_36 = __fmaf_rn(stage2_31, stage2_32, stage2_35);
const float stage2_37 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage2_38 = (-stage2_37);
const float stage2_39 = Profile(p,4,2,z,y,x);
const float stage2_40 = __fmaf_rn(stage2_34, stage2_39, stage2_36);
const float stage2_41 = __fmul_rn(stage2_2, stage2_40);
const float stage2_42 = derivative_5;
const float stage2_43 = __fmul_rn(stage2_5, stage2_42);
const float stage2_44 = __fadd_rn(stage2_41, stage2_43);
const float stage2_45 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage2_46 = old_H_2;
const float stage2_47 = __fmul_rn(stage2_45, stage2_46);
const float stage2_48 = __fmaf_rn(stage2_38, stage2_44, stage2_47);
next_0 = stage2_12;
next_psi_3 = stage2_18;
next_1 = stage2_30;
next_psi_4 = stage2_36;
next_2 = stage2_48;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,1,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,1,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,1);
if(owned) {
Publish(observation,p.field.outputs[2],2,1,z,y,x,next_2);
}
}
}
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) previous_y_0=current_H_1;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) previous_psi_0_3=current_psi_H_3;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<15) previous_psi_0_4=current_psi_H_4;
if(wave>2 && qx>=1 && qx<31 && qy>=1 && qy<14) old_H_0=s[1856+(0*13+qy+(0)-1)*32+qx-1];
if(wave>2 && qx>=1 && qx<31 && qy>=1 && qy<14) old_H_2=s[1856+(1*13+qy+(0)-1)*32+qx-1];
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_3=next_psi_3;
current_psi_H_4=next_psi_4;
}
__syncthreads();
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) s[1856+(0*13+qy+(0)-1)*32+qx-1]=current_H_0;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) s[1856+(1*13+qy+(0)-1)*32+qx-1]=current_H_2;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) s[960+(0*14+qy+(0)-1)*32+qx-1]=current_E_0;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) s[960+(1*14+qy+(0)-1)*32+qx-1]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-1,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_2;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[1856+(1*13+qy+(-1)-1)*32+qx-1]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[1856+(0*13+qy+(-1)-1)*32+qx-1]);
const float stage3_0 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage3_1 = 1.0f;
const float stage3_2 = derivative_0;
const float stage3_3 = __fmul_rn(stage3_1, stage3_2);
const float stage3_4 = -1.0f;
const float stage3_5 = derivative_1;
const float stage3_6 = __fmul_rn(stage3_4, stage3_5);
const float stage3_7 = __fadd_rn(stage3_3, stage3_6);
const float stage3_8 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage3_9 = old_E_0;
const float stage3_10 = __fmul_rn(stage3_8, stage3_9);
const float stage3_11 = __fmaf_rn(stage3_0, stage3_7, stage3_10);
const float stage3_12 = Profile(p,3,1,z,y,x);
const float stage3_13 = previous_psi_1_3;
const float stage3_14 = Profile(p,3,0,z,y,x);
const float stage3_15 = derivative_3;
const float stage3_16 = __fmul_rn(stage3_14, stage3_15);
const float stage3_17 = __fmaf_rn(stage3_12, stage3_13, stage3_16);
const float stage3_18 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage3_19 = derivative_2;
const float stage3_20 = __fmul_rn(stage3_1, stage3_19);
const float stage3_21 = Profile(p,3,2,z,y,x);
const float stage3_22 = __fmaf_rn(stage3_15, stage3_21, stage3_17);
const float stage3_23 = __fmul_rn(stage3_4, stage3_22);
const float stage3_24 = __fadd_rn(stage3_20, stage3_23);
const float stage3_25 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage3_26 = old_E_1;
const float stage3_27 = __fmul_rn(stage3_25, stage3_26);
const float stage3_28 = __fmaf_rn(stage3_18, stage3_24, stage3_27);
const float stage3_29 = Profile(p,4,1,z,y,x);
const float stage3_30 = previous_psi_1_4;
const float stage3_31 = Profile(p,4,0,z,y,x);
const float stage3_32 = derivative_4;
const float stage3_33 = __fmul_rn(stage3_31, stage3_32);
const float stage3_34 = __fmaf_rn(stage3_29, stage3_30, stage3_33);
const float stage3_35 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage3_36 = Profile(p,4,2,z,y,x);
const float stage3_37 = __fmaf_rn(stage3_32, stage3_36, stage3_34);
const float stage3_38 = __fmul_rn(stage3_1, stage3_37);
const float stage3_39 = derivative_5;
const float stage3_40 = __fmul_rn(stage3_4, stage3_39);
const float stage3_41 = __fadd_rn(stage3_38, stage3_40);
const float stage3_42 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage3_43 = old_E_2;
const float stage3_44 = __fmul_rn(stage3_42, stage3_43);
const float stage3_45 = __fmaf_rn(stage3_35, stage3_41, stage3_44);
next_0 = stage3_11;
next_psi_3 = stage3_17;
next_1 = stage3_28;
next_psi_4 = stage3_34;
next_2 = stage3_45;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,1,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,1);
if(owned) {
Publish(observation,p.field.outputs[1],4,1,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,1,z,y,x,next_2);
}
}
}
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) previous_y_1=current_E_1;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) previous_psi_1_3=current_psi_E_3;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<15) previous_psi_1_4=current_psi_E_4;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_3=next_psi_3;
current_psi_E_4=next_psi_4;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-2,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<14) old_E_0=s[2688+(0*12+qy+(0)-2)*32+qx-2];
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<14) old_E_1=previous_y_3;
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<14) old_E_2=s[2688+(1*12+qy+(0)-2)*32+qx-2];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[2688+(1*12+qy+(1)-2)*32+qx-2]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[2688+(0*12+qy+(1)-2)*32+qx-2]);
const float stage4_0 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage4_1 = (-stage4_0);
const float stage4_2 = 1.0f;
const float stage4_3 = derivative_0;
const float stage4_4 = __fmul_rn(stage4_2, stage4_3);
const float stage4_5 = -1.0f;
const float stage4_6 = derivative_1;
const float stage4_7 = __fmul_rn(stage4_5, stage4_6);
const float stage4_8 = __fadd_rn(stage4_4, stage4_7);
const float stage4_9 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage4_10 = old_H_0;
const float stage4_11 = __fmul_rn(stage4_9, stage4_10);
const float stage4_12 = __fmaf_rn(stage4_1, stage4_8, stage4_11);
const float stage4_13 = Profile(p,3,1,z,y,x);
const float stage4_14 = previous_psi_2_3;
const float stage4_15 = Profile(p,3,0,z,y,x);
const float stage4_16 = derivative_3;
const float stage4_17 = __fmul_rn(stage4_15, stage4_16);
const float stage4_18 = __fmaf_rn(stage4_13, stage4_14, stage4_17);
const float stage4_19 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage4_20 = (-stage4_19);
const float stage4_21 = derivative_2;
const float stage4_22 = __fmul_rn(stage4_2, stage4_21);
const float stage4_23 = Profile(p,3,2,z,y,x);
const float stage4_24 = __fmaf_rn(stage4_16, stage4_23, stage4_18);
const float stage4_25 = __fmul_rn(stage4_5, stage4_24);
const float stage4_26 = __fadd_rn(stage4_22, stage4_25);
const float stage4_27 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage4_28 = old_H_1;
const float stage4_29 = __fmul_rn(stage4_27, stage4_28);
const float stage4_30 = __fmaf_rn(stage4_20, stage4_26, stage4_29);
const float stage4_31 = Profile(p,4,1,z,y,x);
const float stage4_32 = previous_psi_2_4;
const float stage4_33 = Profile(p,4,0,z,y,x);
const float stage4_34 = derivative_4;
const float stage4_35 = __fmul_rn(stage4_33, stage4_34);
const float stage4_36 = __fmaf_rn(stage4_31, stage4_32, stage4_35);
const float stage4_37 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage4_38 = (-stage4_37);
const float stage4_39 = Profile(p,4,2,z,y,x);
const float stage4_40 = __fmaf_rn(stage4_34, stage4_39, stage4_36);
const float stage4_41 = __fmul_rn(stage4_2, stage4_40);
const float stage4_42 = derivative_5;
const float stage4_43 = __fmul_rn(stage4_5, stage4_42);
const float stage4_44 = __fadd_rn(stage4_41, stage4_43);
const float stage4_45 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage4_46 = old_H_2;
const float stage4_47 = __fmul_rn(stage4_45, stage4_46);
const float stage4_48 = __fmaf_rn(stage4_38, stage4_44, stage4_47);
next_0 = stage4_12;
next_psi_3 = stage4_18;
next_1 = stage4_30;
next_psi_4 = stage4_36;
next_2 = stage4_48;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,2,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,2,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,2);
if(owned) {
Publish(observation,p.field.outputs[2],2,2,z,y,x,next_2);
}
}
}
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) previous_y_2=current_H_1;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) previous_psi_2_3=current_psi_H_3;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<14) previous_psi_2_4=current_psi_H_4;
if(wave>4 && qx>=2 && qx<30 && qy>=2 && qy<13) old_H_0=s[3456+(0*11+qy+(0)-2)*32+qx-2];
if(wave>4 && qx>=2 && qx<30 && qy>=2 && qy<13) old_H_2=s[3456+(1*11+qy+(0)-2)*32+qx-2];
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_3=next_psi_3;
current_psi_H_4=next_psi_4;
}
__syncthreads();
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) s[3456+(0*11+qy+(0)-2)*32+qx-2]=current_H_0;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) s[3456+(1*11+qy+(0)-2)*32+qx-2]=current_H_2;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) s[2688+(0*12+qy+(0)-2)*32+qx-2]=current_E_0;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) s[2688+(1*12+qy+(0)-2)*32+qx-2]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-2,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_4;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[3456+(1*11+qy+(-1)-2)*32+qx-2]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[3456+(0*11+qy+(-1)-2)*32+qx-2]);
const float stage5_0 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage5_1 = 1.0f;
const float stage5_2 = derivative_0;
const float stage5_3 = __fmul_rn(stage5_1, stage5_2);
const float stage5_4 = -1.0f;
const float stage5_5 = derivative_1;
const float stage5_6 = __fmul_rn(stage5_4, stage5_5);
const float stage5_7 = __fadd_rn(stage5_3, stage5_6);
const float stage5_8 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage5_9 = old_E_0;
const float stage5_10 = __fmul_rn(stage5_8, stage5_9);
const float stage5_11 = __fmaf_rn(stage5_0, stage5_7, stage5_10);
const float stage5_12 = Profile(p,3,1,z,y,x);
const float stage5_13 = previous_psi_3_3;
const float stage5_14 = Profile(p,3,0,z,y,x);
const float stage5_15 = derivative_3;
const float stage5_16 = __fmul_rn(stage5_14, stage5_15);
const float stage5_17 = __fmaf_rn(stage5_12, stage5_13, stage5_16);
const float stage5_18 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage5_19 = derivative_2;
const float stage5_20 = __fmul_rn(stage5_1, stage5_19);
const float stage5_21 = Profile(p,3,2,z,y,x);
const float stage5_22 = __fmaf_rn(stage5_15, stage5_21, stage5_17);
const float stage5_23 = __fmul_rn(stage5_4, stage5_22);
const float stage5_24 = __fadd_rn(stage5_20, stage5_23);
const float stage5_25 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage5_26 = old_E_1;
const float stage5_27 = __fmul_rn(stage5_25, stage5_26);
const float stage5_28 = __fmaf_rn(stage5_18, stage5_24, stage5_27);
const float stage5_29 = Profile(p,4,1,z,y,x);
const float stage5_30 = previous_psi_3_4;
const float stage5_31 = Profile(p,4,0,z,y,x);
const float stage5_32 = derivative_4;
const float stage5_33 = __fmul_rn(stage5_31, stage5_32);
const float stage5_34 = __fmaf_rn(stage5_29, stage5_30, stage5_33);
const float stage5_35 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage5_36 = Profile(p,4,2,z,y,x);
const float stage5_37 = __fmaf_rn(stage5_32, stage5_36, stage5_34);
const float stage5_38 = __fmul_rn(stage5_1, stage5_37);
const float stage5_39 = derivative_5;
const float stage5_40 = __fmul_rn(stage5_4, stage5_39);
const float stage5_41 = __fadd_rn(stage5_38, stage5_40);
const float stage5_42 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage5_43 = old_E_2;
const float stage5_44 = __fmul_rn(stage5_42, stage5_43);
const float stage5_45 = __fmaf_rn(stage5_35, stage5_41, stage5_44);
next_0 = stage5_11;
next_psi_3 = stage5_17;
next_1 = stage5_28;
next_psi_4 = stage5_34;
next_2 = stage5_45;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,2,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,2);
if(owned) {
Publish(observation,p.field.outputs[1],4,2,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,2,z,y,x,next_2);
}
}
}
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) previous_y_3=current_E_1;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) previous_psi_3_3=current_psi_E_3;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<14) previous_psi_3_4=current_psi_E_4;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_3=next_psi_3;
current_psi_E_4=next_psi_4;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-3,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<13) old_E_0=s[4160+(0*10+qy+(0)-3)*32+qx-3];
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<13) old_E_1=previous_y_5;
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<13) old_E_2=s[4160+(1*10+qy+(0)-3)*32+qx-3];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[4160+(1*10+qy+(1)-3)*32+qx-3]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[4160+(0*10+qy+(1)-3)*32+qx-3]);
const float stage6_0 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage6_1 = (-stage6_0);
const float stage6_2 = 1.0f;
const float stage6_3 = derivative_0;
const float stage6_4 = __fmul_rn(stage6_2, stage6_3);
const float stage6_5 = -1.0f;
const float stage6_6 = derivative_1;
const float stage6_7 = __fmul_rn(stage6_5, stage6_6);
const float stage6_8 = __fadd_rn(stage6_4, stage6_7);
const float stage6_9 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage6_10 = old_H_0;
const float stage6_11 = __fmul_rn(stage6_9, stage6_10);
const float stage6_12 = __fmaf_rn(stage6_1, stage6_8, stage6_11);
const float stage6_13 = Profile(p,3,1,z,y,x);
const float stage6_14 = previous_psi_4_3;
const float stage6_15 = Profile(p,3,0,z,y,x);
const float stage6_16 = derivative_3;
const float stage6_17 = __fmul_rn(stage6_15, stage6_16);
const float stage6_18 = __fmaf_rn(stage6_13, stage6_14, stage6_17);
const float stage6_19 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage6_20 = (-stage6_19);
const float stage6_21 = derivative_2;
const float stage6_22 = __fmul_rn(stage6_2, stage6_21);
const float stage6_23 = Profile(p,3,2,z,y,x);
const float stage6_24 = __fmaf_rn(stage6_16, stage6_23, stage6_18);
const float stage6_25 = __fmul_rn(stage6_5, stage6_24);
const float stage6_26 = __fadd_rn(stage6_22, stage6_25);
const float stage6_27 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage6_28 = old_H_1;
const float stage6_29 = __fmul_rn(stage6_27, stage6_28);
const float stage6_30 = __fmaf_rn(stage6_20, stage6_26, stage6_29);
const float stage6_31 = Profile(p,4,1,z,y,x);
const float stage6_32 = previous_psi_4_4;
const float stage6_33 = Profile(p,4,0,z,y,x);
const float stage6_34 = derivative_4;
const float stage6_35 = __fmul_rn(stage6_33, stage6_34);
const float stage6_36 = __fmaf_rn(stage6_31, stage6_32, stage6_35);
const float stage6_37 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage6_38 = (-stage6_37);
const float stage6_39 = Profile(p,4,2,z,y,x);
const float stage6_40 = __fmaf_rn(stage6_34, stage6_39, stage6_36);
const float stage6_41 = __fmul_rn(stage6_2, stage6_40);
const float stage6_42 = derivative_5;
const float stage6_43 = __fmul_rn(stage6_5, stage6_42);
const float stage6_44 = __fadd_rn(stage6_41, stage6_43);
const float stage6_45 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage6_46 = old_H_2;
const float stage6_47 = __fmul_rn(stage6_45, stage6_46);
const float stage6_48 = __fmaf_rn(stage6_38, stage6_44, stage6_47);
next_0 = stage6_12;
next_psi_3 = stage6_18;
next_1 = stage6_30;
next_psi_4 = stage6_36;
next_2 = stage6_48;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,3,z,y,x,next_0);
static_cast<float *>(final.fields[0].data)[BeamzOffset3D<false>(final.fields[0],z,y,x)]=next_0;
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,3,z,y,x,next_1);
static_cast<float *>(final.fields[1].data)[BeamzOffset3D<false>(final.fields[1],z,y,x)]=next_1;
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,3);
if(owned) {
Publish(observation,p.field.outputs[2],2,3,z,y,x,next_2);
static_cast<float *>(final.fields[2].data)[BeamzOffset3D<false>(final.fields[2],z,y,x)]=next_2;
}
}
if(owned) WritePsi<false>(p,3,z,y,x,next_psi_3);
if(owned) WritePsi<false>(p,4,z,y,x,next_psi_4);
}
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) previous_y_4=current_H_1;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) previous_psi_4_3=current_psi_H_3;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<13) previous_psi_4_4=current_psi_H_4;
if(wave>6 && qx>=3 && qx<29 && qy>=3 && qy<12) old_H_0=s[4800+(0*9+qy+(0)-3)*32+qx-3];
if(wave>6 && qx>=3 && qx<29 && qy>=3 && qy<12) old_H_2=s[4800+(1*9+qy+(0)-3)*32+qx-3];
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_3=next_psi_3;
current_psi_H_4=next_psi_4;
}
__syncthreads();
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) s[4800+(0*9+qy+(0)-3)*32+qx-3]=current_H_0;
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) s[4800+(1*9+qy+(0)-3)*32+qx-3]=current_H_2;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) s[4160+(0*10+qy+(0)-3)*32+qx-3]=current_E_0;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) s[4160+(1*10+qy+(0)-3)*32+qx-3]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-3,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_6;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
if(wave>=7 && qx>=4 && qx<29 && qy>=4 && qy<12) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[4800+(1*9+qy+(-1)-3)*32+qx-3]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[4800+(0*9+qy+(-1)-3)*32+qx-3]);
const float stage7_0 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage7_1 = 1.0f;
const float stage7_2 = derivative_0;
const float stage7_3 = __fmul_rn(stage7_1, stage7_2);
const float stage7_4 = -1.0f;
const float stage7_5 = derivative_1;
const float stage7_6 = __fmul_rn(stage7_4, stage7_5);
const float stage7_7 = __fadd_rn(stage7_3, stage7_6);
const float stage7_8 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage7_9 = old_E_0;
const float stage7_10 = __fmul_rn(stage7_8, stage7_9);
const float stage7_11 = __fmaf_rn(stage7_0, stage7_7, stage7_10);
const float stage7_12 = Profile(p,3,1,z,y,x);
const float stage7_13 = previous_psi_5_3;
const float stage7_14 = Profile(p,3,0,z,y,x);
const float stage7_15 = derivative_3;
const float stage7_16 = __fmul_rn(stage7_14, stage7_15);
const float stage7_17 = __fmaf_rn(stage7_12, stage7_13, stage7_16);
const float stage7_18 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage7_19 = derivative_2;
const float stage7_20 = __fmul_rn(stage7_1, stage7_19);
const float stage7_21 = Profile(p,3,2,z,y,x);
const float stage7_22 = __fmaf_rn(stage7_15, stage7_21, stage7_17);
const float stage7_23 = __fmul_rn(stage7_4, stage7_22);
const float stage7_24 = __fadd_rn(stage7_20, stage7_23);
const float stage7_25 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage7_26 = old_E_1;
const float stage7_27 = __fmul_rn(stage7_25, stage7_26);
const float stage7_28 = __fmaf_rn(stage7_18, stage7_24, stage7_27);
const float stage7_29 = Profile(p,4,1,z,y,x);
const float stage7_30 = previous_psi_5_4;
const float stage7_31 = Profile(p,4,0,z,y,x);
const float stage7_32 = derivative_4;
const float stage7_33 = __fmul_rn(stage7_31, stage7_32);
const float stage7_34 = __fmaf_rn(stage7_29, stage7_30, stage7_33);
const float stage7_35 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage7_36 = Profile(p,4,2,z,y,x);
const float stage7_37 = __fmaf_rn(stage7_32, stage7_36, stage7_34);
const float stage7_38 = __fmul_rn(stage7_1, stage7_37);
const float stage7_39 = derivative_5;
const float stage7_40 = __fmul_rn(stage7_4, stage7_39);
const float stage7_41 = __fadd_rn(stage7_38, stage7_40);
const float stage7_42 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage7_43 = old_E_2;
const float stage7_44 = __fmul_rn(stage7_42, stage7_43);
const float stage7_45 = __fmaf_rn(stage7_35, stage7_41, stage7_44);
next_0 = stage7_11;
next_psi_3 = stage7_17;
next_1 = stage7_28;
next_psi_4 = stage7_34;
next_2 = stage7_45;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,3,z,y,x,next_0);
static_cast<float *>(final.fields[3].data)[BeamzOffset3D<false>(final.fields[3],z,y,x)]=next_0;
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,3);
if(owned) {
Publish(observation,p.field.outputs[1],4,3,z,y,x,next_1);
static_cast<float *>(final.fields[4].data)[BeamzOffset3D<false>(final.fields[4],z,y,x)]=next_1;
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,3,z,y,x,next_2);
static_cast<float *>(final.fields[5].data)[BeamzOffset3D<false>(final.fields[5],z,y,x)]=next_2;
}
}
if(owned) WritePsi<false>(p,3,z,y,x,next_psi_3);
if(owned) WritePsi<false>(p,4,z,y,x,next_psi_4);
}
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) previous_y_5=current_E_1;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) previous_psi_5_3=current_psi_E_3;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<13) previous_psi_5_4=current_psi_E_4;
if(wave>=7 && qx>=4 && qx<29 && qy>=4 && qy<12) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_3=next_psi_3;
current_psi_E_4=next_psi_4;
}
}
    if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<12) previous_y_6=current_H_1;
  }
}


constexpr int DeepWarpD4M5F32ScalarHBytePackedSource20Threads=352;
constexpr int DeepWarpD4M5F32ScalarHBytePackedSource20SharedBytes=40448;
template<bool Packed> __global__ __launch_bounds__(352,1) void DeepWarpD4M5F32ScalarHBytePackedSource20(
    const __grid_constant__ PairCpmlPhase h,const __grid_constant__ PairCpmlPhase e,
    const __grid_constant__ PairFieldOutputs final,const __grid_constant__ FusedHSources hs,
    const __grid_constant__ FusedHSources es,const __grid_constant__ Observation observation,PairCpmlGrid grid) {
  if(blockDim.x!=352)return;
  extern __shared__ float s[];
  auto coordinate=[&](int axis,int block,int size) {const int low=(grid.high[axis]+size-1)/size;
    return (5&(1<<axis)) ? (block<low?block*size:grid.boundary[axis]+(block-low)*size) : grid.high[axis]+block*size;};
  auto end=[&](int axis,int block,int origin,int size) {const int low=(grid.high[axis]+size-1)/size;
    return min(origin+size,(5&(1<<axis))?(block<low?grid.high[axis]:grid.extent[axis]):grid.boundary[axis]);};
  const int ox=coordinate(2,blockIdx.x,25),oy=coordinate(1,blockIdx.y,4),oz=coordinate(0,blockIdx.z,64);
  const int end_x=end(2,blockIdx.x,ox,25),end_y=end(1,blockIdx.y,oy,4),end_z=end(0,blockIdx.z,oz,64);
  const int qx=threadIdx.x&31,qy=threadIdx.x>>5;
  __shared__ bool active_sources[6];
  if(threadIdx.x<6){const int c=threadIdx.x;const auto &g=c<3?hs.groups[c]:es.groups[c-3];
    const auto *starts=static_cast<const int *>(g.starts.data);bool active=false;
    for(int i=0;i<g.coefficients.dims[0];++i)
      active |= starts[3*i]<oz+68 && starts[3*i]+g.coefficients.dims[1]>oz-4 &&
                starts[3*i+1]<oy+8 && starts[3*i+1]+g.coefficients.dims[2]>oy-4 &&
                starts[3*i+2]<ox+29 && starts[3*i+2]+g.coefficients.dims[3]>ox-4;
    active_sources[c]=active;}
  __syncthreads();
  float previous_y_0=0.f;
float previous_y_1=0.f;
float previous_y_2=0.f;
float previous_y_3=0.f;
float previous_y_4=0.f;
float previous_y_5=0.f;
float previous_y_6=0.f;
  for(int wave=0;wave<end_z-oz+7;++wave){
    float current_H_0=0.f,old_H_0=0.f;
float current_H_1=0.f,old_H_1=0.f;
float current_H_2=0.f,old_H_2=0.f;
float current_E_0=0.f,old_E_0=0.f;
float current_E_1=0.f,old_E_1=0.f;
float current_E_2=0.f,old_E_2=0.f;
float current_psi_H_1=0.f;
float current_psi_H_2=0.f;
float current_psi_H_3=0.f;
float current_psi_H_4=0.f;
float current_psi_E_1=0.f;
float current_psi_E_2=0.f;
float current_psi_E_3=0.f;
float current_psi_E_4=0.f;
    {
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-0,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) {
const float derivative_0=Difference(p,0,2,1,z,y,x,SafeRead(p.field.inputs[5],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[5],z+(0),y+(1),x+(0)));
const float derivative_1=Difference(p,0,1,0,z,y,x,SafeRead(p.field.inputs[4],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[4],z+(1),y+(0),x+(0)));
const float derivative_2=Difference(p,0,0,0,z,y,x,SafeRead(p.field.inputs[3],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[3],z+(1),y+(0),x+(0)));
const float derivative_3=Difference(p,0,2,2,z,y,x,SafeRead(p.field.inputs[5],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[5],z+(0),y+(0),x+(1)));
const float derivative_4=Difference(p,0,1,2,z,y,x,SafeRead(p.field.inputs[4],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[4],z+(0),y+(0),x+(1)));
const float derivative_5=Difference(p,0,0,1,z,y,x,SafeRead(p.field.inputs[3],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[3],z+(0),y+(1),x+(0)));
const float stage0_0 = Profile(p,1,1,z,y,x);
const float stage0_1 = ReadPsi<false>(p,1,z,y,x);
const float stage0_2 = Profile(p,1,0,z,y,x);
const float stage0_3 = derivative_1;
const float stage0_4 = __fmul_rn(stage0_2, stage0_3);
const float stage0_5 = __fmaf_rn(stage0_0, stage0_1, stage0_4);
const float stage0_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage0_7 = (-stage0_6);
const float stage0_8 = 1.0f;
const float stage0_9 = derivative_0;
const float stage0_10 = __fmul_rn(stage0_8, stage0_9);
const float stage0_11 = -1.0f;
const float stage0_12 = Profile(p,1,2,z,y,x);
const float stage0_13 = __fmaf_rn(stage0_3, stage0_12, stage0_5);
const float stage0_14 = __fmul_rn(stage0_11, stage0_13);
const float stage0_15 = __fadd_rn(stage0_10, stage0_14);
const float stage0_16 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage0_17 = SafeRead(p.field.inputs[0],z,y,x);
const float stage0_18 = __fmul_rn(stage0_16, stage0_17);
const float stage0_19 = __fmaf_rn(stage0_7, stage0_15, stage0_18);
const float stage0_20 = Profile(p,2,1,z,y,x);
const float stage0_21 = ReadPsi<false>(p,2,z,y,x);
const float stage0_22 = Profile(p,2,0,z,y,x);
const float stage0_23 = derivative_2;
const float stage0_24 = __fmul_rn(stage0_22, stage0_23);
const float stage0_25 = __fmaf_rn(stage0_20, stage0_21, stage0_24);
const float stage0_26 = Profile(p,3,1,z,y,x);
const float stage0_27 = ReadPsi<false>(p,3,z,y,x);
const float stage0_28 = Profile(p,3,0,z,y,x);
const float stage0_29 = derivative_3;
const float stage0_30 = __fmul_rn(stage0_28, stage0_29);
const float stage0_31 = __fmaf_rn(stage0_26, stage0_27, stage0_30);
const float stage0_32 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage0_33 = (-stage0_32);
const float stage0_34 = Profile(p,2,2,z,y,x);
const float stage0_35 = __fmaf_rn(stage0_23, stage0_34, stage0_25);
const float stage0_36 = __fmul_rn(stage0_8, stage0_35);
const float stage0_37 = Profile(p,3,2,z,y,x);
const float stage0_38 = __fmaf_rn(stage0_29, stage0_37, stage0_31);
const float stage0_39 = __fmul_rn(stage0_11, stage0_38);
const float stage0_40 = __fadd_rn(stage0_36, stage0_39);
const float stage0_41 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage0_42 = SafeRead(p.field.inputs[1],z,y,x);
const float stage0_43 = __fmul_rn(stage0_41, stage0_42);
const float stage0_44 = __fmaf_rn(stage0_33, stage0_40, stage0_43);
const float stage0_45 = Profile(p,4,1,z,y,x);
const float stage0_46 = ReadPsi<false>(p,4,z,y,x);
const float stage0_47 = Profile(p,4,0,z,y,x);
const float stage0_48 = derivative_4;
const float stage0_49 = __fmul_rn(stage0_47, stage0_48);
const float stage0_50 = __fmaf_rn(stage0_45, stage0_46, stage0_49);
const float stage0_51 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage0_52 = (-stage0_51);
const float stage0_53 = Profile(p,4,2,z,y,x);
const float stage0_54 = __fmaf_rn(stage0_48, stage0_53, stage0_50);
const float stage0_55 = __fmul_rn(stage0_8, stage0_54);
const float stage0_56 = derivative_5;
const float stage0_57 = __fmul_rn(stage0_11, stage0_56);
const float stage0_58 = __fadd_rn(stage0_55, stage0_57);
const float stage0_59 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage0_60 = SafeRead(p.field.inputs[2],z,y,x);
const float stage0_61 = __fmul_rn(stage0_59, stage0_60);
const float stage0_62 = __fmaf_rn(stage0_52, stage0_58, stage0_61);
next_psi_1 = stage0_5;
next_0 = stage0_19;
next_psi_2 = stage0_25;
next_psi_3 = stage0_31;
next_1 = stage0_44;
next_psi_4 = stage0_50;
next_2 = stage0_62;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,0,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,0,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,0);
if(owned) {
Publish(observation,p.field.outputs[2],2,0,z,y,x,next_2);
}
}
}
if(wave>0 && qx>=0 && qx<32 && qy>=0 && qy<11) old_H_0=s[0+(0*11+qy+(0)-0)*32+qx-0];
if(wave>0 && qx>=0 && qx<32 && qy>=0 && qy<11) old_H_2=s[0+(1*11+qy+(0)-0)*32+qx-0];
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_1=next_psi_1;
current_psi_H_2=next_psi_2;
current_psi_H_3=next_psi_3;
current_psi_H_4=next_psi_4;
}
__syncthreads();
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[0+(0*11+qy+(0)-0)*32+qx-0]=current_H_0;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[0+(1*11+qy+(0)-0)*32+qx-0]=current_H_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-0,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_0;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[0+(1*11+qy+(-1)-0)*32+qx-0]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[0+(0*11+qy+(-1)-0)*32+qx-0]);
const float stage1_0 = Profile(p,1,1,z,y,x);
const float stage1_1 = ReadPsi<false>(p,1,z,y,x);
const float stage1_2 = Profile(p,1,0,z,y,x);
const float stage1_3 = derivative_1;
const float stage1_4 = __fmul_rn(stage1_2, stage1_3);
const float stage1_5 = __fmaf_rn(stage1_0, stage1_1, stage1_4);
const float stage1_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage1_7 = 1.0f;
const float stage1_8 = derivative_0;
const float stage1_9 = __fmul_rn(stage1_7, stage1_8);
const float stage1_10 = -1.0f;
const float stage1_11 = Profile(p,1,2,z,y,x);
const float stage1_12 = __fmaf_rn(stage1_3, stage1_11, stage1_5);
const float stage1_13 = __fmul_rn(stage1_10, stage1_12);
const float stage1_14 = __fadd_rn(stage1_9, stage1_13);
const float stage1_15 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage1_16 = SafeRead(p.field.inputs[0],z,y,x);
const float stage1_17 = __fmul_rn(stage1_15, stage1_16);
const float stage1_18 = __fmaf_rn(stage1_6, stage1_14, stage1_17);
const float stage1_19 = Profile(p,2,1,z,y,x);
const float stage1_20 = ReadPsi<false>(p,2,z,y,x);
const float stage1_21 = Profile(p,2,0,z,y,x);
const float stage1_22 = derivative_2;
const float stage1_23 = __fmul_rn(stage1_21, stage1_22);
const float stage1_24 = __fmaf_rn(stage1_19, stage1_20, stage1_23);
const float stage1_25 = Profile(p,3,1,z,y,x);
const float stage1_26 = ReadPsi<false>(p,3,z,y,x);
const float stage1_27 = Profile(p,3,0,z,y,x);
const float stage1_28 = derivative_3;
const float stage1_29 = __fmul_rn(stage1_27, stage1_28);
const float stage1_30 = __fmaf_rn(stage1_25, stage1_26, stage1_29);
const float stage1_31 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage1_32 = Profile(p,2,2,z,y,x);
const float stage1_33 = __fmaf_rn(stage1_22, stage1_32, stage1_24);
const float stage1_34 = __fmul_rn(stage1_7, stage1_33);
const float stage1_35 = Profile(p,3,2,z,y,x);
const float stage1_36 = __fmaf_rn(stage1_28, stage1_35, stage1_30);
const float stage1_37 = __fmul_rn(stage1_10, stage1_36);
const float stage1_38 = __fadd_rn(stage1_34, stage1_37);
const float stage1_39 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage1_40 = SafeRead(p.field.inputs[1],z,y,x);
const float stage1_41 = __fmul_rn(stage1_39, stage1_40);
const float stage1_42 = __fmaf_rn(stage1_31, stage1_38, stage1_41);
const float stage1_43 = Profile(p,4,1,z,y,x);
const float stage1_44 = ReadPsi<false>(p,4,z,y,x);
const float stage1_45 = Profile(p,4,0,z,y,x);
const float stage1_46 = derivative_4;
const float stage1_47 = __fmul_rn(stage1_45, stage1_46);
const float stage1_48 = __fmaf_rn(stage1_43, stage1_44, stage1_47);
const float stage1_49 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage1_50 = Profile(p,4,2,z,y,x);
const float stage1_51 = __fmaf_rn(stage1_46, stage1_50, stage1_48);
const float stage1_52 = __fmul_rn(stage1_7, stage1_51);
const float stage1_53 = derivative_5;
const float stage1_54 = __fmul_rn(stage1_10, stage1_53);
const float stage1_55 = __fadd_rn(stage1_52, stage1_54);
const float stage1_56 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage1_57 = SafeRead(p.field.inputs[2],z,y,x);
const float stage1_58 = __fmul_rn(stage1_56, stage1_57);
const float stage1_59 = __fmaf_rn(stage1_49, stage1_55, stage1_58);
next_psi_1 = stage1_5;
next_0 = stage1_18;
next_psi_2 = stage1_24;
next_psi_3 = stage1_30;
next_1 = stage1_42;
next_psi_4 = stage1_48;
next_2 = stage1_59;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,0,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,0);
if(owned) {
Publish(observation,p.field.outputs[1],4,0,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,0,z,y,x,next_2);
}
}
}
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_1=next_psi_1;
current_psi_E_2=next_psi_2;
current_psi_E_3=next_psi_3;
current_psi_E_4=next_psi_4;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-1,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<11) old_E_0=s[704+(0*10+qy+(0)-1)*32+qx-1];
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<11) old_E_1=previous_y_1;
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<11) old_E_2=s[704+(1*10+qy+(0)-1)*32+qx-1];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[704+(1*10+qy+(1)-1)*32+qx-1]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[704+(0*10+qy+(1)-1)*32+qx-1]);
const float stage2_0 = Profile(p,1,1,z,y,x);
const float stage2_1 = s[3584+(0*11+qy-0)*32+qx-0];
const float stage2_2 = Profile(p,1,0,z,y,x);
const float stage2_3 = derivative_1;
const float stage2_4 = __fmul_rn(stage2_2, stage2_3);
const float stage2_5 = __fmaf_rn(stage2_0, stage2_1, stage2_4);
const float stage2_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage2_7 = (-stage2_6);
const float stage2_8 = 1.0f;
const float stage2_9 = derivative_0;
const float stage2_10 = __fmul_rn(stage2_8, stage2_9);
const float stage2_11 = -1.0f;
const float stage2_12 = Profile(p,1,2,z,y,x);
const float stage2_13 = __fmaf_rn(stage2_3, stage2_12, stage2_5);
const float stage2_14 = __fmul_rn(stage2_11, stage2_13);
const float stage2_15 = __fadd_rn(stage2_10, stage2_14);
const float stage2_16 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage2_17 = old_H_0;
const float stage2_18 = __fmul_rn(stage2_16, stage2_17);
const float stage2_19 = __fmaf_rn(stage2_7, stage2_15, stage2_18);
const float stage2_20 = Profile(p,2,1,z,y,x);
const float stage2_21 = s[3584+(1*11+qy-0)*32+qx-0];
const float stage2_22 = Profile(p,2,0,z,y,x);
const float stage2_23 = derivative_2;
const float stage2_24 = __fmul_rn(stage2_22, stage2_23);
const float stage2_25 = __fmaf_rn(stage2_20, stage2_21, stage2_24);
const float stage2_26 = Profile(p,3,1,z,y,x);
const float stage2_27 = s[3584+(2*11+qy-0)*32+qx-0];
const float stage2_28 = Profile(p,3,0,z,y,x);
const float stage2_29 = derivative_3;
const float stage2_30 = __fmul_rn(stage2_28, stage2_29);
const float stage2_31 = __fmaf_rn(stage2_26, stage2_27, stage2_30);
const float stage2_32 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage2_33 = (-stage2_32);
const float stage2_34 = Profile(p,2,2,z,y,x);
const float stage2_35 = __fmaf_rn(stage2_23, stage2_34, stage2_25);
const float stage2_36 = __fmul_rn(stage2_8, stage2_35);
const float stage2_37 = Profile(p,3,2,z,y,x);
const float stage2_38 = __fmaf_rn(stage2_29, stage2_37, stage2_31);
const float stage2_39 = __fmul_rn(stage2_11, stage2_38);
const float stage2_40 = __fadd_rn(stage2_36, stage2_39);
const float stage2_41 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage2_42 = old_H_1;
const float stage2_43 = __fmul_rn(stage2_41, stage2_42);
const float stage2_44 = __fmaf_rn(stage2_33, stage2_40, stage2_43);
const float stage2_45 = Profile(p,4,1,z,y,x);
const float stage2_46 = s[3584+(3*11+qy-0)*32+qx-0];
const float stage2_47 = Profile(p,4,0,z,y,x);
const float stage2_48 = derivative_4;
const float stage2_49 = __fmul_rn(stage2_47, stage2_48);
const float stage2_50 = __fmaf_rn(stage2_45, stage2_46, stage2_49);
const float stage2_51 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage2_52 = (-stage2_51);
const float stage2_53 = Profile(p,4,2,z,y,x);
const float stage2_54 = __fmaf_rn(stage2_48, stage2_53, stage2_50);
const float stage2_55 = __fmul_rn(stage2_8, stage2_54);
const float stage2_56 = derivative_5;
const float stage2_57 = __fmul_rn(stage2_11, stage2_56);
const float stage2_58 = __fadd_rn(stage2_55, stage2_57);
const float stage2_59 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage2_60 = old_H_2;
const float stage2_61 = __fmul_rn(stage2_59, stage2_60);
const float stage2_62 = __fmaf_rn(stage2_52, stage2_58, stage2_61);
next_psi_1 = stage2_5;
next_0 = stage2_19;
next_psi_2 = stage2_25;
next_psi_3 = stage2_31;
next_1 = stage2_44;
next_psi_4 = stage2_50;
next_2 = stage2_62;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,1,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,1,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,1);
if(owned) {
Publish(observation,p.field.outputs[2],2,1,z,y,x,next_2);
}
}
}
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) previous_y_0=current_H_1;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[3584+(0*11+qy-0)*32+qx-0]=current_psi_H_1;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[3584+(1*11+qy-0)*32+qx-0]=current_psi_H_2;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[3584+(2*11+qy-0)*32+qx-0]=current_psi_H_3;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[3584+(3*11+qy-0)*32+qx-0]=current_psi_H_4;
if(wave>2 && qx>=1 && qx<31 && qy>=1 && qy<10) old_H_0=s[1344+(0*9+qy+(0)-1)*32+qx-1];
if(wave>2 && qx>=1 && qx<31 && qy>=1 && qy<10) old_H_2=s[1344+(1*9+qy+(0)-1)*32+qx-1];
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_1=next_psi_1;
current_psi_H_2=next_psi_2;
current_psi_H_3=next_psi_3;
current_psi_H_4=next_psi_4;
}
__syncthreads();
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[1344+(0*9+qy+(0)-1)*32+qx-1]=current_H_0;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[1344+(1*9+qy+(0)-1)*32+qx-1]=current_H_2;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[704+(0*10+qy+(0)-1)*32+qx-1]=current_E_0;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[704+(1*10+qy+(0)-1)*32+qx-1]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-1,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_2;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[1344+(1*9+qy+(-1)-1)*32+qx-1]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[1344+(0*9+qy+(-1)-1)*32+qx-1]);
const float stage3_0 = Profile(p,1,1,z,y,x);
const float stage3_1 = s[4992+(0*10+qy-1)*32+qx-1];
const float stage3_2 = Profile(p,1,0,z,y,x);
const float stage3_3 = derivative_1;
const float stage3_4 = __fmul_rn(stage3_2, stage3_3);
const float stage3_5 = __fmaf_rn(stage3_0, stage3_1, stage3_4);
const float stage3_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage3_7 = 1.0f;
const float stage3_8 = derivative_0;
const float stage3_9 = __fmul_rn(stage3_7, stage3_8);
const float stage3_10 = -1.0f;
const float stage3_11 = Profile(p,1,2,z,y,x);
const float stage3_12 = __fmaf_rn(stage3_3, stage3_11, stage3_5);
const float stage3_13 = __fmul_rn(stage3_10, stage3_12);
const float stage3_14 = __fadd_rn(stage3_9, stage3_13);
const float stage3_15 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage3_16 = old_E_0;
const float stage3_17 = __fmul_rn(stage3_15, stage3_16);
const float stage3_18 = __fmaf_rn(stage3_6, stage3_14, stage3_17);
const float stage3_19 = Profile(p,2,1,z,y,x);
const float stage3_20 = s[4992+(1*10+qy-1)*32+qx-1];
const float stage3_21 = Profile(p,2,0,z,y,x);
const float stage3_22 = derivative_2;
const float stage3_23 = __fmul_rn(stage3_21, stage3_22);
const float stage3_24 = __fmaf_rn(stage3_19, stage3_20, stage3_23);
const float stage3_25 = Profile(p,3,1,z,y,x);
const float stage3_26 = s[4992+(2*10+qy-1)*32+qx-1];
const float stage3_27 = Profile(p,3,0,z,y,x);
const float stage3_28 = derivative_3;
const float stage3_29 = __fmul_rn(stage3_27, stage3_28);
const float stage3_30 = __fmaf_rn(stage3_25, stage3_26, stage3_29);
const float stage3_31 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage3_32 = Profile(p,2,2,z,y,x);
const float stage3_33 = __fmaf_rn(stage3_22, stage3_32, stage3_24);
const float stage3_34 = __fmul_rn(stage3_7, stage3_33);
const float stage3_35 = Profile(p,3,2,z,y,x);
const float stage3_36 = __fmaf_rn(stage3_28, stage3_35, stage3_30);
const float stage3_37 = __fmul_rn(stage3_10, stage3_36);
const float stage3_38 = __fadd_rn(stage3_34, stage3_37);
const float stage3_39 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage3_40 = old_E_1;
const float stage3_41 = __fmul_rn(stage3_39, stage3_40);
const float stage3_42 = __fmaf_rn(stage3_31, stage3_38, stage3_41);
const float stage3_43 = Profile(p,4,1,z,y,x);
const float stage3_44 = s[4992+(3*10+qy-1)*32+qx-1];
const float stage3_45 = Profile(p,4,0,z,y,x);
const float stage3_46 = derivative_4;
const float stage3_47 = __fmul_rn(stage3_45, stage3_46);
const float stage3_48 = __fmaf_rn(stage3_43, stage3_44, stage3_47);
const float stage3_49 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage3_50 = Profile(p,4,2,z,y,x);
const float stage3_51 = __fmaf_rn(stage3_46, stage3_50, stage3_48);
const float stage3_52 = __fmul_rn(stage3_7, stage3_51);
const float stage3_53 = derivative_5;
const float stage3_54 = __fmul_rn(stage3_10, stage3_53);
const float stage3_55 = __fadd_rn(stage3_52, stage3_54);
const float stage3_56 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage3_57 = old_E_2;
const float stage3_58 = __fmul_rn(stage3_56, stage3_57);
const float stage3_59 = __fmaf_rn(stage3_49, stage3_55, stage3_58);
next_psi_1 = stage3_5;
next_0 = stage3_18;
next_psi_2 = stage3_24;
next_psi_3 = stage3_30;
next_1 = stage3_42;
next_psi_4 = stage3_48;
next_2 = stage3_59;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,1,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,1);
if(owned) {
Publish(observation,p.field.outputs[1],4,1,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,1,z,y,x,next_2);
}
}
}
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) previous_y_1=current_E_1;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[4992+(0*10+qy-1)*32+qx-1]=current_psi_E_1;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[4992+(1*10+qy-1)*32+qx-1]=current_psi_E_2;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[4992+(2*10+qy-1)*32+qx-1]=current_psi_E_3;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[4992+(3*10+qy-1)*32+qx-1]=current_psi_E_4;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_1=next_psi_1;
current_psi_E_2=next_psi_2;
current_psi_E_3=next_psi_3;
current_psi_E_4=next_psi_4;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-2,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<10) old_E_0=s[1920+(0*8+qy+(0)-2)*32+qx-2];
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<10) old_E_1=previous_y_3;
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<10) old_E_2=s[1920+(1*8+qy+(0)-2)*32+qx-2];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[1920+(1*8+qy+(1)-2)*32+qx-2]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[1920+(0*8+qy+(1)-2)*32+qx-2]);
const float stage4_0 = Profile(p,1,1,z,y,x);
const float stage4_1 = s[6272+(0*9+qy-1)*32+qx-1];
const float stage4_2 = Profile(p,1,0,z,y,x);
const float stage4_3 = derivative_1;
const float stage4_4 = __fmul_rn(stage4_2, stage4_3);
const float stage4_5 = __fmaf_rn(stage4_0, stage4_1, stage4_4);
const float stage4_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage4_7 = (-stage4_6);
const float stage4_8 = 1.0f;
const float stage4_9 = derivative_0;
const float stage4_10 = __fmul_rn(stage4_8, stage4_9);
const float stage4_11 = -1.0f;
const float stage4_12 = Profile(p,1,2,z,y,x);
const float stage4_13 = __fmaf_rn(stage4_3, stage4_12, stage4_5);
const float stage4_14 = __fmul_rn(stage4_11, stage4_13);
const float stage4_15 = __fadd_rn(stage4_10, stage4_14);
const float stage4_16 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage4_17 = old_H_0;
const float stage4_18 = __fmul_rn(stage4_16, stage4_17);
const float stage4_19 = __fmaf_rn(stage4_7, stage4_15, stage4_18);
const float stage4_20 = Profile(p,2,1,z,y,x);
const float stage4_21 = s[6272+(1*9+qy-1)*32+qx-1];
const float stage4_22 = Profile(p,2,0,z,y,x);
const float stage4_23 = derivative_2;
const float stage4_24 = __fmul_rn(stage4_22, stage4_23);
const float stage4_25 = __fmaf_rn(stage4_20, stage4_21, stage4_24);
const float stage4_26 = Profile(p,3,1,z,y,x);
const float stage4_27 = s[6272+(2*9+qy-1)*32+qx-1];
const float stage4_28 = Profile(p,3,0,z,y,x);
const float stage4_29 = derivative_3;
const float stage4_30 = __fmul_rn(stage4_28, stage4_29);
const float stage4_31 = __fmaf_rn(stage4_26, stage4_27, stage4_30);
const float stage4_32 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage4_33 = (-stage4_32);
const float stage4_34 = Profile(p,2,2,z,y,x);
const float stage4_35 = __fmaf_rn(stage4_23, stage4_34, stage4_25);
const float stage4_36 = __fmul_rn(stage4_8, stage4_35);
const float stage4_37 = Profile(p,3,2,z,y,x);
const float stage4_38 = __fmaf_rn(stage4_29, stage4_37, stage4_31);
const float stage4_39 = __fmul_rn(stage4_11, stage4_38);
const float stage4_40 = __fadd_rn(stage4_36, stage4_39);
const float stage4_41 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage4_42 = old_H_1;
const float stage4_43 = __fmul_rn(stage4_41, stage4_42);
const float stage4_44 = __fmaf_rn(stage4_33, stage4_40, stage4_43);
const float stage4_45 = Profile(p,4,1,z,y,x);
const float stage4_46 = s[6272+(3*9+qy-1)*32+qx-1];
const float stage4_47 = Profile(p,4,0,z,y,x);
const float stage4_48 = derivative_4;
const float stage4_49 = __fmul_rn(stage4_47, stage4_48);
const float stage4_50 = __fmaf_rn(stage4_45, stage4_46, stage4_49);
const float stage4_51 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage4_52 = (-stage4_51);
const float stage4_53 = Profile(p,4,2,z,y,x);
const float stage4_54 = __fmaf_rn(stage4_48, stage4_53, stage4_50);
const float stage4_55 = __fmul_rn(stage4_8, stage4_54);
const float stage4_56 = derivative_5;
const float stage4_57 = __fmul_rn(stage4_11, stage4_56);
const float stage4_58 = __fadd_rn(stage4_55, stage4_57);
const float stage4_59 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage4_60 = old_H_2;
const float stage4_61 = __fmul_rn(stage4_59, stage4_60);
const float stage4_62 = __fmaf_rn(stage4_52, stage4_58, stage4_61);
next_psi_1 = stage4_5;
next_0 = stage4_19;
next_psi_2 = stage4_25;
next_psi_3 = stage4_31;
next_1 = stage4_44;
next_psi_4 = stage4_50;
next_2 = stage4_62;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,2,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,2,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,2);
if(owned) {
Publish(observation,p.field.outputs[2],2,2,z,y,x,next_2);
}
}
}
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) previous_y_2=current_H_1;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[6272+(0*9+qy-1)*32+qx-1]=current_psi_H_1;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[6272+(1*9+qy-1)*32+qx-1]=current_psi_H_2;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[6272+(2*9+qy-1)*32+qx-1]=current_psi_H_3;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[6272+(3*9+qy-1)*32+qx-1]=current_psi_H_4;
if(wave>4 && qx>=2 && qx<30 && qy>=2 && qy<9) old_H_0=s[2432+(0*7+qy+(0)-2)*32+qx-2];
if(wave>4 && qx>=2 && qx<30 && qy>=2 && qy<9) old_H_2=s[2432+(1*7+qy+(0)-2)*32+qx-2];
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_1=next_psi_1;
current_psi_H_2=next_psi_2;
current_psi_H_3=next_psi_3;
current_psi_H_4=next_psi_4;
}
__syncthreads();
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[2432+(0*7+qy+(0)-2)*32+qx-2]=current_H_0;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[2432+(1*7+qy+(0)-2)*32+qx-2]=current_H_2;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[1920+(0*8+qy+(0)-2)*32+qx-2]=current_E_0;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[1920+(1*8+qy+(0)-2)*32+qx-2]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-2,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_4;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[2432+(1*7+qy+(-1)-2)*32+qx-2]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[2432+(0*7+qy+(-1)-2)*32+qx-2]);
const float stage5_0 = Profile(p,1,1,z,y,x);
const float stage5_1 = s[7424+(0*8+qy-2)*32+qx-2];
const float stage5_2 = Profile(p,1,0,z,y,x);
const float stage5_3 = derivative_1;
const float stage5_4 = __fmul_rn(stage5_2, stage5_3);
const float stage5_5 = __fmaf_rn(stage5_0, stage5_1, stage5_4);
const float stage5_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage5_7 = 1.0f;
const float stage5_8 = derivative_0;
const float stage5_9 = __fmul_rn(stage5_7, stage5_8);
const float stage5_10 = -1.0f;
const float stage5_11 = Profile(p,1,2,z,y,x);
const float stage5_12 = __fmaf_rn(stage5_3, stage5_11, stage5_5);
const float stage5_13 = __fmul_rn(stage5_10, stage5_12);
const float stage5_14 = __fadd_rn(stage5_9, stage5_13);
const float stage5_15 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage5_16 = old_E_0;
const float stage5_17 = __fmul_rn(stage5_15, stage5_16);
const float stage5_18 = __fmaf_rn(stage5_6, stage5_14, stage5_17);
const float stage5_19 = Profile(p,2,1,z,y,x);
const float stage5_20 = s[7424+(1*8+qy-2)*32+qx-2];
const float stage5_21 = Profile(p,2,0,z,y,x);
const float stage5_22 = derivative_2;
const float stage5_23 = __fmul_rn(stage5_21, stage5_22);
const float stage5_24 = __fmaf_rn(stage5_19, stage5_20, stage5_23);
const float stage5_25 = Profile(p,3,1,z,y,x);
const float stage5_26 = s[7424+(2*8+qy-2)*32+qx-2];
const float stage5_27 = Profile(p,3,0,z,y,x);
const float stage5_28 = derivative_3;
const float stage5_29 = __fmul_rn(stage5_27, stage5_28);
const float stage5_30 = __fmaf_rn(stage5_25, stage5_26, stage5_29);
const float stage5_31 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage5_32 = Profile(p,2,2,z,y,x);
const float stage5_33 = __fmaf_rn(stage5_22, stage5_32, stage5_24);
const float stage5_34 = __fmul_rn(stage5_7, stage5_33);
const float stage5_35 = Profile(p,3,2,z,y,x);
const float stage5_36 = __fmaf_rn(stage5_28, stage5_35, stage5_30);
const float stage5_37 = __fmul_rn(stage5_10, stage5_36);
const float stage5_38 = __fadd_rn(stage5_34, stage5_37);
const float stage5_39 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage5_40 = old_E_1;
const float stage5_41 = __fmul_rn(stage5_39, stage5_40);
const float stage5_42 = __fmaf_rn(stage5_31, stage5_38, stage5_41);
const float stage5_43 = Profile(p,4,1,z,y,x);
const float stage5_44 = s[7424+(3*8+qy-2)*32+qx-2];
const float stage5_45 = Profile(p,4,0,z,y,x);
const float stage5_46 = derivative_4;
const float stage5_47 = __fmul_rn(stage5_45, stage5_46);
const float stage5_48 = __fmaf_rn(stage5_43, stage5_44, stage5_47);
const float stage5_49 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage5_50 = Profile(p,4,2,z,y,x);
const float stage5_51 = __fmaf_rn(stage5_46, stage5_50, stage5_48);
const float stage5_52 = __fmul_rn(stage5_7, stage5_51);
const float stage5_53 = derivative_5;
const float stage5_54 = __fmul_rn(stage5_10, stage5_53);
const float stage5_55 = __fadd_rn(stage5_52, stage5_54);
const float stage5_56 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage5_57 = old_E_2;
const float stage5_58 = __fmul_rn(stage5_56, stage5_57);
const float stage5_59 = __fmaf_rn(stage5_49, stage5_55, stage5_58);
next_psi_1 = stage5_5;
next_0 = stage5_18;
next_psi_2 = stage5_24;
next_psi_3 = stage5_30;
next_1 = stage5_42;
next_psi_4 = stage5_48;
next_2 = stage5_59;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,2,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,2);
if(owned) {
Publish(observation,p.field.outputs[1],4,2,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,2,z,y,x,next_2);
}
}
}
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) previous_y_3=current_E_1;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[7424+(0*8+qy-2)*32+qx-2]=current_psi_E_1;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[7424+(1*8+qy-2)*32+qx-2]=current_psi_E_2;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[7424+(2*8+qy-2)*32+qx-2]=current_psi_E_3;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[7424+(3*8+qy-2)*32+qx-2]=current_psi_E_4;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_1=next_psi_1;
current_psi_E_2=next_psi_2;
current_psi_E_3=next_psi_3;
current_psi_E_4=next_psi_4;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-3,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<9) old_E_0=s[2880+(0*6+qy+(0)-3)*32+qx-3];
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<9) old_E_1=previous_y_5;
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<9) old_E_2=s[2880+(1*6+qy+(0)-3)*32+qx-3];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<8) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[2880+(1*6+qy+(1)-3)*32+qx-3]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[2880+(0*6+qy+(1)-3)*32+qx-3]);
const float stage6_0 = Profile(p,1,1,z,y,x);
const float stage6_1 = s[8448+(0*7+qy-2)*32+qx-2];
const float stage6_2 = Profile(p,1,0,z,y,x);
const float stage6_3 = derivative_1;
const float stage6_4 = __fmul_rn(stage6_2, stage6_3);
const float stage6_5 = __fmaf_rn(stage6_0, stage6_1, stage6_4);
const float stage6_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage6_7 = (-stage6_6);
const float stage6_8 = 1.0f;
const float stage6_9 = derivative_0;
const float stage6_10 = __fmul_rn(stage6_8, stage6_9);
const float stage6_11 = -1.0f;
const float stage6_12 = Profile(p,1,2,z,y,x);
const float stage6_13 = __fmaf_rn(stage6_3, stage6_12, stage6_5);
const float stage6_14 = __fmul_rn(stage6_11, stage6_13);
const float stage6_15 = __fadd_rn(stage6_10, stage6_14);
const float stage6_16 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage6_17 = old_H_0;
const float stage6_18 = __fmul_rn(stage6_16, stage6_17);
const float stage6_19 = __fmaf_rn(stage6_7, stage6_15, stage6_18);
const float stage6_20 = Profile(p,2,1,z,y,x);
const float stage6_21 = s[8448+(1*7+qy-2)*32+qx-2];
const float stage6_22 = Profile(p,2,0,z,y,x);
const float stage6_23 = derivative_2;
const float stage6_24 = __fmul_rn(stage6_22, stage6_23);
const float stage6_25 = __fmaf_rn(stage6_20, stage6_21, stage6_24);
const float stage6_26 = Profile(p,3,1,z,y,x);
const float stage6_27 = s[8448+(2*7+qy-2)*32+qx-2];
const float stage6_28 = Profile(p,3,0,z,y,x);
const float stage6_29 = derivative_3;
const float stage6_30 = __fmul_rn(stage6_28, stage6_29);
const float stage6_31 = __fmaf_rn(stage6_26, stage6_27, stage6_30);
const float stage6_32 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage6_33 = (-stage6_32);
const float stage6_34 = Profile(p,2,2,z,y,x);
const float stage6_35 = __fmaf_rn(stage6_23, stage6_34, stage6_25);
const float stage6_36 = __fmul_rn(stage6_8, stage6_35);
const float stage6_37 = Profile(p,3,2,z,y,x);
const float stage6_38 = __fmaf_rn(stage6_29, stage6_37, stage6_31);
const float stage6_39 = __fmul_rn(stage6_11, stage6_38);
const float stage6_40 = __fadd_rn(stage6_36, stage6_39);
const float stage6_41 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage6_42 = old_H_1;
const float stage6_43 = __fmul_rn(stage6_41, stage6_42);
const float stage6_44 = __fmaf_rn(stage6_33, stage6_40, stage6_43);
const float stage6_45 = Profile(p,4,1,z,y,x);
const float stage6_46 = s[8448+(3*7+qy-2)*32+qx-2];
const float stage6_47 = Profile(p,4,0,z,y,x);
const float stage6_48 = derivative_4;
const float stage6_49 = __fmul_rn(stage6_47, stage6_48);
const float stage6_50 = __fmaf_rn(stage6_45, stage6_46, stage6_49);
const float stage6_51 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage6_52 = (-stage6_51);
const float stage6_53 = Profile(p,4,2,z,y,x);
const float stage6_54 = __fmaf_rn(stage6_48, stage6_53, stage6_50);
const float stage6_55 = __fmul_rn(stage6_8, stage6_54);
const float stage6_56 = derivative_5;
const float stage6_57 = __fmul_rn(stage6_11, stage6_56);
const float stage6_58 = __fadd_rn(stage6_55, stage6_57);
const float stage6_59 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage6_60 = old_H_2;
const float stage6_61 = __fmul_rn(stage6_59, stage6_60);
const float stage6_62 = __fmaf_rn(stage6_52, stage6_58, stage6_61);
next_psi_1 = stage6_5;
next_0 = stage6_19;
next_psi_2 = stage6_25;
next_psi_3 = stage6_31;
next_1 = stage6_44;
next_psi_4 = stage6_50;
next_2 = stage6_62;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,3,z,y,x,next_0);
static_cast<float *>(final.fields[0].data)[BeamzOffset3D<false>(final.fields[0],z,y,x)]=next_0;
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,3,z,y,x,next_1);
static_cast<float *>(final.fields[1].data)[BeamzOffset3D<false>(final.fields[1],z,y,x)]=next_1;
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,3);
if(owned) {
Publish(observation,p.field.outputs[2],2,3,z,y,x,next_2);
static_cast<float *>(final.fields[2].data)[BeamzOffset3D<false>(final.fields[2],z,y,x)]=next_2;
}
}
if(owned) WritePsi<false>(p,1,z,y,x,next_psi_1);
if(owned) WritePsi<false>(p,2,z,y,x,next_psi_2);
if(owned) WritePsi<false>(p,3,z,y,x,next_psi_3);
if(owned) WritePsi<false>(p,4,z,y,x,next_psi_4);
}
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) previous_y_4=current_H_1;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[8448+(0*7+qy-2)*32+qx-2]=current_psi_H_1;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[8448+(1*7+qy-2)*32+qx-2]=current_psi_H_2;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[8448+(2*7+qy-2)*32+qx-2]=current_psi_H_3;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[8448+(3*7+qy-2)*32+qx-2]=current_psi_H_4;
if(wave>6 && qx>=3 && qx<29 && qy>=3 && qy<8) old_H_0=s[3264+(0*5+qy+(0)-3)*32+qx-3];
if(wave>6 && qx>=3 && qx<29 && qy>=3 && qy<8) old_H_2=s[3264+(1*5+qy+(0)-3)*32+qx-3];
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<8) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_1=next_psi_1;
current_psi_H_2=next_psi_2;
current_psi_H_3=next_psi_3;
current_psi_H_4=next_psi_4;
}
__syncthreads();
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<8) s[3264+(0*5+qy+(0)-3)*32+qx-3]=current_H_0;
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<8) s[3264+(1*5+qy+(0)-3)*32+qx-3]=current_H_2;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[2880+(0*6+qy+(0)-3)*32+qx-3]=current_E_0;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[2880+(1*6+qy+(0)-3)*32+qx-3]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-3,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_6;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
if(wave>=7 && qx>=4 && qx<29 && qy>=4 && qy<8) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[3264+(1*5+qy+(-1)-3)*32+qx-3]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[3264+(0*5+qy+(-1)-3)*32+qx-3]);
const float stage7_0 = Profile(p,1,1,z,y,x);
const float stage7_1 = s[9344+(0*6+qy-3)*32+qx-3];
const float stage7_2 = Profile(p,1,0,z,y,x);
const float stage7_3 = derivative_1;
const float stage7_4 = __fmul_rn(stage7_2, stage7_3);
const float stage7_5 = __fmaf_rn(stage7_0, stage7_1, stage7_4);
const float stage7_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage7_7 = 1.0f;
const float stage7_8 = derivative_0;
const float stage7_9 = __fmul_rn(stage7_7, stage7_8);
const float stage7_10 = -1.0f;
const float stage7_11 = Profile(p,1,2,z,y,x);
const float stage7_12 = __fmaf_rn(stage7_3, stage7_11, stage7_5);
const float stage7_13 = __fmul_rn(stage7_10, stage7_12);
const float stage7_14 = __fadd_rn(stage7_9, stage7_13);
const float stage7_15 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage7_16 = old_E_0;
const float stage7_17 = __fmul_rn(stage7_15, stage7_16);
const float stage7_18 = __fmaf_rn(stage7_6, stage7_14, stage7_17);
const float stage7_19 = Profile(p,2,1,z,y,x);
const float stage7_20 = s[9344+(1*6+qy-3)*32+qx-3];
const float stage7_21 = Profile(p,2,0,z,y,x);
const float stage7_22 = derivative_2;
const float stage7_23 = __fmul_rn(stage7_21, stage7_22);
const float stage7_24 = __fmaf_rn(stage7_19, stage7_20, stage7_23);
const float stage7_25 = Profile(p,3,1,z,y,x);
const float stage7_26 = s[9344+(2*6+qy-3)*32+qx-3];
const float stage7_27 = Profile(p,3,0,z,y,x);
const float stage7_28 = derivative_3;
const float stage7_29 = __fmul_rn(stage7_27, stage7_28);
const float stage7_30 = __fmaf_rn(stage7_25, stage7_26, stage7_29);
const float stage7_31 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage7_32 = Profile(p,2,2,z,y,x);
const float stage7_33 = __fmaf_rn(stage7_22, stage7_32, stage7_24);
const float stage7_34 = __fmul_rn(stage7_7, stage7_33);
const float stage7_35 = Profile(p,3,2,z,y,x);
const float stage7_36 = __fmaf_rn(stage7_28, stage7_35, stage7_30);
const float stage7_37 = __fmul_rn(stage7_10, stage7_36);
const float stage7_38 = __fadd_rn(stage7_34, stage7_37);
const float stage7_39 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage7_40 = old_E_1;
const float stage7_41 = __fmul_rn(stage7_39, stage7_40);
const float stage7_42 = __fmaf_rn(stage7_31, stage7_38, stage7_41);
const float stage7_43 = Profile(p,4,1,z,y,x);
const float stage7_44 = s[9344+(3*6+qy-3)*32+qx-3];
const float stage7_45 = Profile(p,4,0,z,y,x);
const float stage7_46 = derivative_4;
const float stage7_47 = __fmul_rn(stage7_45, stage7_46);
const float stage7_48 = __fmaf_rn(stage7_43, stage7_44, stage7_47);
const float stage7_49 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage7_50 = Profile(p,4,2,z,y,x);
const float stage7_51 = __fmaf_rn(stage7_46, stage7_50, stage7_48);
const float stage7_52 = __fmul_rn(stage7_7, stage7_51);
const float stage7_53 = derivative_5;
const float stage7_54 = __fmul_rn(stage7_10, stage7_53);
const float stage7_55 = __fadd_rn(stage7_52, stage7_54);
const float stage7_56 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage7_57 = old_E_2;
const float stage7_58 = __fmul_rn(stage7_56, stage7_57);
const float stage7_59 = __fmaf_rn(stage7_49, stage7_55, stage7_58);
next_psi_1 = stage7_5;
next_0 = stage7_18;
next_psi_2 = stage7_24;
next_psi_3 = stage7_30;
next_1 = stage7_42;
next_psi_4 = stage7_48;
next_2 = stage7_59;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,3,z,y,x,next_0);
static_cast<float *>(final.fields[3].data)[BeamzOffset3D<false>(final.fields[3],z,y,x)]=next_0;
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,3);
if(owned) {
Publish(observation,p.field.outputs[1],4,3,z,y,x,next_1);
static_cast<float *>(final.fields[4].data)[BeamzOffset3D<false>(final.fields[4],z,y,x)]=next_1;
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,3,z,y,x,next_2);
static_cast<float *>(final.fields[5].data)[BeamzOffset3D<false>(final.fields[5],z,y,x)]=next_2;
}
}
if(owned) WritePsi<false>(p,1,z,y,x,next_psi_1);
if(owned) WritePsi<false>(p,2,z,y,x,next_psi_2);
if(owned) WritePsi<false>(p,3,z,y,x,next_psi_3);
if(owned) WritePsi<false>(p,4,z,y,x,next_psi_4);
}
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) previous_y_5=current_E_1;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[9344+(0*6+qy-3)*32+qx-3]=current_psi_E_1;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[9344+(1*6+qy-3)*32+qx-3]=current_psi_E_2;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[9344+(2*6+qy-3)*32+qx-3]=current_psi_E_3;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[9344+(3*6+qy-3)*32+qx-3]=current_psi_E_4;
if(wave>=7 && qx>=4 && qx<29 && qy>=4 && qy<8) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_1=next_psi_1;
current_psi_E_2=next_psi_2;
current_psi_E_3=next_psi_3;
current_psi_E_4=next_psi_4;
}
}
    if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<8) previous_y_6=current_H_1;
  }
}


constexpr int DeepWarpD4M6F32ScalarHBytePackedSource20Threads=352;
constexpr int DeepWarpD4M6F32ScalarHBytePackedSource20SharedBytes=40448;
template<bool Packed> __global__ __launch_bounds__(352,1) void DeepWarpD4M6F32ScalarHBytePackedSource20(
    const __grid_constant__ PairCpmlPhase h,const __grid_constant__ PairCpmlPhase e,
    const __grid_constant__ PairFieldOutputs final,const __grid_constant__ FusedHSources hs,
    const __grid_constant__ FusedHSources es,const __grid_constant__ Observation observation,PairCpmlGrid grid) {
  if(blockDim.x!=352)return;
  extern __shared__ float s[];
  auto coordinate=[&](int axis,int block,int size) {const int low=(grid.high[axis]+size-1)/size;
    return (6&(1<<axis)) ? (block<low?block*size:grid.boundary[axis]+(block-low)*size) : grid.high[axis]+block*size;};
  auto end=[&](int axis,int block,int origin,int size) {const int low=(grid.high[axis]+size-1)/size;
    return min(origin+size,(6&(1<<axis))?(block<low?grid.high[axis]:grid.extent[axis]):grid.boundary[axis]);};
  const int ox=coordinate(2,blockIdx.x,25),oy=coordinate(1,blockIdx.y,4),oz=coordinate(0,blockIdx.z,64);
  const int end_x=end(2,blockIdx.x,ox,25),end_y=end(1,blockIdx.y,oy,4),end_z=end(0,blockIdx.z,oz,64);
  const int qx=threadIdx.x&31,qy=threadIdx.x>>5;
  __shared__ bool active_sources[6];
  if(threadIdx.x<6){const int c=threadIdx.x;const auto &g=c<3?hs.groups[c]:es.groups[c-3];
    const auto *starts=static_cast<const int *>(g.starts.data);bool active=false;
    for(int i=0;i<g.coefficients.dims[0];++i)
      active |= starts[3*i]<oz+68 && starts[3*i]+g.coefficients.dims[1]>oz-4 &&
                starts[3*i+1]<oy+8 && starts[3*i+1]+g.coefficients.dims[2]>oy-4 &&
                starts[3*i+2]<ox+29 && starts[3*i+2]+g.coefficients.dims[3]>ox-4;
    active_sources[c]=active;}
  __syncthreads();
  float previous_y_0=0.f;
float previous_y_1=0.f;
float previous_y_2=0.f;
float previous_y_3=0.f;
float previous_y_4=0.f;
float previous_y_5=0.f;
float previous_y_6=0.f;
  for(int wave=0;wave<end_z-oz+7;++wave){
    float current_H_0=0.f,old_H_0=0.f;
float current_H_1=0.f,old_H_1=0.f;
float current_H_2=0.f,old_H_2=0.f;
float current_E_0=0.f,old_E_0=0.f;
float current_E_1=0.f,old_E_1=0.f;
float current_E_2=0.f,old_E_2=0.f;
float current_psi_H_0=0.f;
float current_psi_H_3=0.f;
float current_psi_H_4=0.f;
float current_psi_H_5=0.f;
float current_psi_E_0=0.f;
float current_psi_E_3=0.f;
float current_psi_E_4=0.f;
float current_psi_E_5=0.f;
    {
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-0,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
float next_psi_5=0.f;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) {
const float derivative_0=Difference(p,0,2,1,z,y,x,SafeRead(p.field.inputs[5],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[5],z+(0),y+(1),x+(0)));
const float derivative_1=Difference(p,0,1,0,z,y,x,SafeRead(p.field.inputs[4],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[4],z+(1),y+(0),x+(0)));
const float derivative_2=Difference(p,0,0,0,z,y,x,SafeRead(p.field.inputs[3],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[3],z+(1),y+(0),x+(0)));
const float derivative_3=Difference(p,0,2,2,z,y,x,SafeRead(p.field.inputs[5],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[5],z+(0),y+(0),x+(1)));
const float derivative_4=Difference(p,0,1,2,z,y,x,SafeRead(p.field.inputs[4],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[4],z+(0),y+(0),x+(1)));
const float derivative_5=Difference(p,0,0,1,z,y,x,SafeRead(p.field.inputs[3],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[3],z+(0),y+(1),x+(0)));
const float stage0_0 = Profile(p,0,1,z,y,x);
const float stage0_1 = ReadPsi<false>(p,0,z,y,x);
const float stage0_2 = Profile(p,0,0,z,y,x);
const float stage0_3 = derivative_0;
const float stage0_4 = __fmul_rn(stage0_2, stage0_3);
const float stage0_5 = __fmaf_rn(stage0_0, stage0_1, stage0_4);
const float stage0_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage0_7 = (-stage0_6);
const float stage0_8 = 1.0f;
const float stage0_9 = Profile(p,0,2,z,y,x);
const float stage0_10 = __fmaf_rn(stage0_3, stage0_9, stage0_5);
const float stage0_11 = __fmul_rn(stage0_8, stage0_10);
const float stage0_12 = -1.0f;
const float stage0_13 = derivative_1;
const float stage0_14 = __fmul_rn(stage0_12, stage0_13);
const float stage0_15 = __fadd_rn(stage0_11, stage0_14);
const float stage0_16 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage0_17 = SafeRead(p.field.inputs[0],z,y,x);
const float stage0_18 = __fmul_rn(stage0_16, stage0_17);
const float stage0_19 = __fmaf_rn(stage0_7, stage0_15, stage0_18);
const float stage0_20 = Profile(p,3,1,z,y,x);
const float stage0_21 = ReadPsi<false>(p,3,z,y,x);
const float stage0_22 = Profile(p,3,0,z,y,x);
const float stage0_23 = derivative_3;
const float stage0_24 = __fmul_rn(stage0_22, stage0_23);
const float stage0_25 = __fmaf_rn(stage0_20, stage0_21, stage0_24);
const float stage0_26 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage0_27 = (-stage0_26);
const float stage0_28 = derivative_2;
const float stage0_29 = __fmul_rn(stage0_8, stage0_28);
const float stage0_30 = Profile(p,3,2,z,y,x);
const float stage0_31 = __fmaf_rn(stage0_23, stage0_30, stage0_25);
const float stage0_32 = __fmul_rn(stage0_12, stage0_31);
const float stage0_33 = __fadd_rn(stage0_29, stage0_32);
const float stage0_34 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage0_35 = SafeRead(p.field.inputs[1],z,y,x);
const float stage0_36 = __fmul_rn(stage0_34, stage0_35);
const float stage0_37 = __fmaf_rn(stage0_27, stage0_33, stage0_36);
const float stage0_38 = Profile(p,4,1,z,y,x);
const float stage0_39 = ReadPsi<false>(p,4,z,y,x);
const float stage0_40 = Profile(p,4,0,z,y,x);
const float stage0_41 = derivative_4;
const float stage0_42 = __fmul_rn(stage0_40, stage0_41);
const float stage0_43 = __fmaf_rn(stage0_38, stage0_39, stage0_42);
const float stage0_44 = Profile(p,5,1,z,y,x);
const float stage0_45 = ReadPsi<false>(p,5,z,y,x);
const float stage0_46 = Profile(p,5,0,z,y,x);
const float stage0_47 = derivative_5;
const float stage0_48 = __fmul_rn(stage0_46, stage0_47);
const float stage0_49 = __fmaf_rn(stage0_44, stage0_45, stage0_48);
const float stage0_50 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage0_51 = (-stage0_50);
const float stage0_52 = Profile(p,4,2,z,y,x);
const float stage0_53 = __fmaf_rn(stage0_41, stage0_52, stage0_43);
const float stage0_54 = __fmul_rn(stage0_8, stage0_53);
const float stage0_55 = Profile(p,5,2,z,y,x);
const float stage0_56 = __fmaf_rn(stage0_47, stage0_55, stage0_49);
const float stage0_57 = __fmul_rn(stage0_12, stage0_56);
const float stage0_58 = __fadd_rn(stage0_54, stage0_57);
const float stage0_59 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage0_60 = SafeRead(p.field.inputs[2],z,y,x);
const float stage0_61 = __fmul_rn(stage0_59, stage0_60);
const float stage0_62 = __fmaf_rn(stage0_51, stage0_58, stage0_61);
next_psi_0 = stage0_5;
next_0 = stage0_19;
next_psi_3 = stage0_25;
next_1 = stage0_37;
next_psi_4 = stage0_43;
next_psi_5 = stage0_49;
next_2 = stage0_62;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,0,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,0,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,0);
if(owned) {
Publish(observation,p.field.outputs[2],2,0,z,y,x,next_2);
}
}
}
if(wave>0 && qx>=0 && qx<32 && qy>=0 && qy<11) old_H_0=s[0+(0*11+qy+(0)-0)*32+qx-0];
if(wave>0 && qx>=0 && qx<32 && qy>=0 && qy<11) old_H_2=s[0+(1*11+qy+(0)-0)*32+qx-0];
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_0=next_psi_0;
current_psi_H_3=next_psi_3;
current_psi_H_4=next_psi_4;
current_psi_H_5=next_psi_5;
}
__syncthreads();
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[0+(0*11+qy+(0)-0)*32+qx-0]=current_H_0;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[0+(1*11+qy+(0)-0)*32+qx-0]=current_H_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-0,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_0;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
float next_psi_5=0.f;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[0+(1*11+qy+(-1)-0)*32+qx-0]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[0+(0*11+qy+(-1)-0)*32+qx-0]);
const float stage1_0 = Profile(p,0,1,z,y,x);
const float stage1_1 = ReadPsi<false>(p,0,z,y,x);
const float stage1_2 = Profile(p,0,0,z,y,x);
const float stage1_3 = derivative_0;
const float stage1_4 = __fmul_rn(stage1_2, stage1_3);
const float stage1_5 = __fmaf_rn(stage1_0, stage1_1, stage1_4);
const float stage1_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage1_7 = 1.0f;
const float stage1_8 = Profile(p,0,2,z,y,x);
const float stage1_9 = __fmaf_rn(stage1_3, stage1_8, stage1_5);
const float stage1_10 = __fmul_rn(stage1_7, stage1_9);
const float stage1_11 = -1.0f;
const float stage1_12 = derivative_1;
const float stage1_13 = __fmul_rn(stage1_11, stage1_12);
const float stage1_14 = __fadd_rn(stage1_10, stage1_13);
const float stage1_15 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage1_16 = SafeRead(p.field.inputs[0],z,y,x);
const float stage1_17 = __fmul_rn(stage1_15, stage1_16);
const float stage1_18 = __fmaf_rn(stage1_6, stage1_14, stage1_17);
const float stage1_19 = Profile(p,3,1,z,y,x);
const float stage1_20 = ReadPsi<false>(p,3,z,y,x);
const float stage1_21 = Profile(p,3,0,z,y,x);
const float stage1_22 = derivative_3;
const float stage1_23 = __fmul_rn(stage1_21, stage1_22);
const float stage1_24 = __fmaf_rn(stage1_19, stage1_20, stage1_23);
const float stage1_25 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage1_26 = derivative_2;
const float stage1_27 = __fmul_rn(stage1_7, stage1_26);
const float stage1_28 = Profile(p,3,2,z,y,x);
const float stage1_29 = __fmaf_rn(stage1_22, stage1_28, stage1_24);
const float stage1_30 = __fmul_rn(stage1_11, stage1_29);
const float stage1_31 = __fadd_rn(stage1_27, stage1_30);
const float stage1_32 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage1_33 = SafeRead(p.field.inputs[1],z,y,x);
const float stage1_34 = __fmul_rn(stage1_32, stage1_33);
const float stage1_35 = __fmaf_rn(stage1_25, stage1_31, stage1_34);
const float stage1_36 = Profile(p,4,1,z,y,x);
const float stage1_37 = ReadPsi<false>(p,4,z,y,x);
const float stage1_38 = Profile(p,4,0,z,y,x);
const float stage1_39 = derivative_4;
const float stage1_40 = __fmul_rn(stage1_38, stage1_39);
const float stage1_41 = __fmaf_rn(stage1_36, stage1_37, stage1_40);
const float stage1_42 = Profile(p,5,1,z,y,x);
const float stage1_43 = ReadPsi<false>(p,5,z,y,x);
const float stage1_44 = Profile(p,5,0,z,y,x);
const float stage1_45 = derivative_5;
const float stage1_46 = __fmul_rn(stage1_44, stage1_45);
const float stage1_47 = __fmaf_rn(stage1_42, stage1_43, stage1_46);
const float stage1_48 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage1_49 = Profile(p,4,2,z,y,x);
const float stage1_50 = __fmaf_rn(stage1_39, stage1_49, stage1_41);
const float stage1_51 = __fmul_rn(stage1_7, stage1_50);
const float stage1_52 = Profile(p,5,2,z,y,x);
const float stage1_53 = __fmaf_rn(stage1_45, stage1_52, stage1_47);
const float stage1_54 = __fmul_rn(stage1_11, stage1_53);
const float stage1_55 = __fadd_rn(stage1_51, stage1_54);
const float stage1_56 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage1_57 = SafeRead(p.field.inputs[2],z,y,x);
const float stage1_58 = __fmul_rn(stage1_56, stage1_57);
const float stage1_59 = __fmaf_rn(stage1_48, stage1_55, stage1_58);
next_psi_0 = stage1_5;
next_0 = stage1_18;
next_psi_3 = stage1_24;
next_1 = stage1_35;
next_psi_4 = stage1_41;
next_psi_5 = stage1_47;
next_2 = stage1_59;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,0,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,0);
if(owned) {
Publish(observation,p.field.outputs[1],4,0,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,0,z,y,x,next_2);
}
}
}
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_0=next_psi_0;
current_psi_E_3=next_psi_3;
current_psi_E_4=next_psi_4;
current_psi_E_5=next_psi_5;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-1,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<11) old_E_0=s[704+(0*10+qy+(0)-1)*32+qx-1];
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<11) old_E_1=previous_y_1;
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<11) old_E_2=s[704+(1*10+qy+(0)-1)*32+qx-1];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
float next_psi_5=0.f;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[704+(1*10+qy+(1)-1)*32+qx-1]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[704+(0*10+qy+(1)-1)*32+qx-1]);
const float stage2_0 = Profile(p,0,1,z,y,x);
const float stage2_1 = s[3584+(0*11+qy-0)*32+qx-0];
const float stage2_2 = Profile(p,0,0,z,y,x);
const float stage2_3 = derivative_0;
const float stage2_4 = __fmul_rn(stage2_2, stage2_3);
const float stage2_5 = __fmaf_rn(stage2_0, stage2_1, stage2_4);
const float stage2_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage2_7 = (-stage2_6);
const float stage2_8 = 1.0f;
const float stage2_9 = Profile(p,0,2,z,y,x);
const float stage2_10 = __fmaf_rn(stage2_3, stage2_9, stage2_5);
const float stage2_11 = __fmul_rn(stage2_8, stage2_10);
const float stage2_12 = -1.0f;
const float stage2_13 = derivative_1;
const float stage2_14 = __fmul_rn(stage2_12, stage2_13);
const float stage2_15 = __fadd_rn(stage2_11, stage2_14);
const float stage2_16 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage2_17 = old_H_0;
const float stage2_18 = __fmul_rn(stage2_16, stage2_17);
const float stage2_19 = __fmaf_rn(stage2_7, stage2_15, stage2_18);
const float stage2_20 = Profile(p,3,1,z,y,x);
const float stage2_21 = s[3584+(1*11+qy-0)*32+qx-0];
const float stage2_22 = Profile(p,3,0,z,y,x);
const float stage2_23 = derivative_3;
const float stage2_24 = __fmul_rn(stage2_22, stage2_23);
const float stage2_25 = __fmaf_rn(stage2_20, stage2_21, stage2_24);
const float stage2_26 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage2_27 = (-stage2_26);
const float stage2_28 = derivative_2;
const float stage2_29 = __fmul_rn(stage2_8, stage2_28);
const float stage2_30 = Profile(p,3,2,z,y,x);
const float stage2_31 = __fmaf_rn(stage2_23, stage2_30, stage2_25);
const float stage2_32 = __fmul_rn(stage2_12, stage2_31);
const float stage2_33 = __fadd_rn(stage2_29, stage2_32);
const float stage2_34 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage2_35 = old_H_1;
const float stage2_36 = __fmul_rn(stage2_34, stage2_35);
const float stage2_37 = __fmaf_rn(stage2_27, stage2_33, stage2_36);
const float stage2_38 = Profile(p,4,1,z,y,x);
const float stage2_39 = s[3584+(2*11+qy-0)*32+qx-0];
const float stage2_40 = Profile(p,4,0,z,y,x);
const float stage2_41 = derivative_4;
const float stage2_42 = __fmul_rn(stage2_40, stage2_41);
const float stage2_43 = __fmaf_rn(stage2_38, stage2_39, stage2_42);
const float stage2_44 = Profile(p,5,1,z,y,x);
const float stage2_45 = s[3584+(3*11+qy-0)*32+qx-0];
const float stage2_46 = Profile(p,5,0,z,y,x);
const float stage2_47 = derivative_5;
const float stage2_48 = __fmul_rn(stage2_46, stage2_47);
const float stage2_49 = __fmaf_rn(stage2_44, stage2_45, stage2_48);
const float stage2_50 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage2_51 = (-stage2_50);
const float stage2_52 = Profile(p,4,2,z,y,x);
const float stage2_53 = __fmaf_rn(stage2_41, stage2_52, stage2_43);
const float stage2_54 = __fmul_rn(stage2_8, stage2_53);
const float stage2_55 = Profile(p,5,2,z,y,x);
const float stage2_56 = __fmaf_rn(stage2_47, stage2_55, stage2_49);
const float stage2_57 = __fmul_rn(stage2_12, stage2_56);
const float stage2_58 = __fadd_rn(stage2_54, stage2_57);
const float stage2_59 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage2_60 = old_H_2;
const float stage2_61 = __fmul_rn(stage2_59, stage2_60);
const float stage2_62 = __fmaf_rn(stage2_51, stage2_58, stage2_61);
next_psi_0 = stage2_5;
next_0 = stage2_19;
next_psi_3 = stage2_25;
next_1 = stage2_37;
next_psi_4 = stage2_43;
next_psi_5 = stage2_49;
next_2 = stage2_62;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,1,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,1,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,1);
if(owned) {
Publish(observation,p.field.outputs[2],2,1,z,y,x,next_2);
}
}
}
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) previous_y_0=current_H_1;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[3584+(0*11+qy-0)*32+qx-0]=current_psi_H_0;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[3584+(1*11+qy-0)*32+qx-0]=current_psi_H_3;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[3584+(2*11+qy-0)*32+qx-0]=current_psi_H_4;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<11) s[3584+(3*11+qy-0)*32+qx-0]=current_psi_H_5;
if(wave>2 && qx>=1 && qx<31 && qy>=1 && qy<10) old_H_0=s[1344+(0*9+qy+(0)-1)*32+qx-1];
if(wave>2 && qx>=1 && qx<31 && qy>=1 && qy<10) old_H_2=s[1344+(1*9+qy+(0)-1)*32+qx-1];
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_0=next_psi_0;
current_psi_H_3=next_psi_3;
current_psi_H_4=next_psi_4;
current_psi_H_5=next_psi_5;
}
__syncthreads();
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[1344+(0*9+qy+(0)-1)*32+qx-1]=current_H_0;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[1344+(1*9+qy+(0)-1)*32+qx-1]=current_H_2;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[704+(0*10+qy+(0)-1)*32+qx-1]=current_E_0;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[704+(1*10+qy+(0)-1)*32+qx-1]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-1,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_2;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
float next_psi_5=0.f;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[1344+(1*9+qy+(-1)-1)*32+qx-1]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[1344+(0*9+qy+(-1)-1)*32+qx-1]);
const float stage3_0 = Profile(p,0,1,z,y,x);
const float stage3_1 = s[4992+(0*10+qy-1)*32+qx-1];
const float stage3_2 = Profile(p,0,0,z,y,x);
const float stage3_3 = derivative_0;
const float stage3_4 = __fmul_rn(stage3_2, stage3_3);
const float stage3_5 = __fmaf_rn(stage3_0, stage3_1, stage3_4);
const float stage3_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage3_7 = 1.0f;
const float stage3_8 = Profile(p,0,2,z,y,x);
const float stage3_9 = __fmaf_rn(stage3_3, stage3_8, stage3_5);
const float stage3_10 = __fmul_rn(stage3_7, stage3_9);
const float stage3_11 = -1.0f;
const float stage3_12 = derivative_1;
const float stage3_13 = __fmul_rn(stage3_11, stage3_12);
const float stage3_14 = __fadd_rn(stage3_10, stage3_13);
const float stage3_15 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage3_16 = old_E_0;
const float stage3_17 = __fmul_rn(stage3_15, stage3_16);
const float stage3_18 = __fmaf_rn(stage3_6, stage3_14, stage3_17);
const float stage3_19 = Profile(p,3,1,z,y,x);
const float stage3_20 = s[4992+(1*10+qy-1)*32+qx-1];
const float stage3_21 = Profile(p,3,0,z,y,x);
const float stage3_22 = derivative_3;
const float stage3_23 = __fmul_rn(stage3_21, stage3_22);
const float stage3_24 = __fmaf_rn(stage3_19, stage3_20, stage3_23);
const float stage3_25 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage3_26 = derivative_2;
const float stage3_27 = __fmul_rn(stage3_7, stage3_26);
const float stage3_28 = Profile(p,3,2,z,y,x);
const float stage3_29 = __fmaf_rn(stage3_22, stage3_28, stage3_24);
const float stage3_30 = __fmul_rn(stage3_11, stage3_29);
const float stage3_31 = __fadd_rn(stage3_27, stage3_30);
const float stage3_32 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage3_33 = old_E_1;
const float stage3_34 = __fmul_rn(stage3_32, stage3_33);
const float stage3_35 = __fmaf_rn(stage3_25, stage3_31, stage3_34);
const float stage3_36 = Profile(p,4,1,z,y,x);
const float stage3_37 = s[4992+(2*10+qy-1)*32+qx-1];
const float stage3_38 = Profile(p,4,0,z,y,x);
const float stage3_39 = derivative_4;
const float stage3_40 = __fmul_rn(stage3_38, stage3_39);
const float stage3_41 = __fmaf_rn(stage3_36, stage3_37, stage3_40);
const float stage3_42 = Profile(p,5,1,z,y,x);
const float stage3_43 = s[4992+(3*10+qy-1)*32+qx-1];
const float stage3_44 = Profile(p,5,0,z,y,x);
const float stage3_45 = derivative_5;
const float stage3_46 = __fmul_rn(stage3_44, stage3_45);
const float stage3_47 = __fmaf_rn(stage3_42, stage3_43, stage3_46);
const float stage3_48 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage3_49 = Profile(p,4,2,z,y,x);
const float stage3_50 = __fmaf_rn(stage3_39, stage3_49, stage3_41);
const float stage3_51 = __fmul_rn(stage3_7, stage3_50);
const float stage3_52 = Profile(p,5,2,z,y,x);
const float stage3_53 = __fmaf_rn(stage3_45, stage3_52, stage3_47);
const float stage3_54 = __fmul_rn(stage3_11, stage3_53);
const float stage3_55 = __fadd_rn(stage3_51, stage3_54);
const float stage3_56 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage3_57 = old_E_2;
const float stage3_58 = __fmul_rn(stage3_56, stage3_57);
const float stage3_59 = __fmaf_rn(stage3_48, stage3_55, stage3_58);
next_psi_0 = stage3_5;
next_0 = stage3_18;
next_psi_3 = stage3_24;
next_1 = stage3_35;
next_psi_4 = stage3_41;
next_psi_5 = stage3_47;
next_2 = stage3_59;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,1,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,1);
if(owned) {
Publish(observation,p.field.outputs[1],4,1,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,1,z,y,x,next_2);
}
}
}
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) previous_y_1=current_E_1;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[4992+(0*10+qy-1)*32+qx-1]=current_psi_E_0;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[4992+(1*10+qy-1)*32+qx-1]=current_psi_E_3;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[4992+(2*10+qy-1)*32+qx-1]=current_psi_E_4;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<11) s[4992+(3*10+qy-1)*32+qx-1]=current_psi_E_5;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_0=next_psi_0;
current_psi_E_3=next_psi_3;
current_psi_E_4=next_psi_4;
current_psi_E_5=next_psi_5;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-2,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<10) old_E_0=s[1920+(0*8+qy+(0)-2)*32+qx-2];
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<10) old_E_1=previous_y_3;
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<10) old_E_2=s[1920+(1*8+qy+(0)-2)*32+qx-2];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
float next_psi_5=0.f;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[1920+(1*8+qy+(1)-2)*32+qx-2]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[1920+(0*8+qy+(1)-2)*32+qx-2]);
const float stage4_0 = Profile(p,0,1,z,y,x);
const float stage4_1 = s[6272+(0*9+qy-1)*32+qx-1];
const float stage4_2 = Profile(p,0,0,z,y,x);
const float stage4_3 = derivative_0;
const float stage4_4 = __fmul_rn(stage4_2, stage4_3);
const float stage4_5 = __fmaf_rn(stage4_0, stage4_1, stage4_4);
const float stage4_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage4_7 = (-stage4_6);
const float stage4_8 = 1.0f;
const float stage4_9 = Profile(p,0,2,z,y,x);
const float stage4_10 = __fmaf_rn(stage4_3, stage4_9, stage4_5);
const float stage4_11 = __fmul_rn(stage4_8, stage4_10);
const float stage4_12 = -1.0f;
const float stage4_13 = derivative_1;
const float stage4_14 = __fmul_rn(stage4_12, stage4_13);
const float stage4_15 = __fadd_rn(stage4_11, stage4_14);
const float stage4_16 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage4_17 = old_H_0;
const float stage4_18 = __fmul_rn(stage4_16, stage4_17);
const float stage4_19 = __fmaf_rn(stage4_7, stage4_15, stage4_18);
const float stage4_20 = Profile(p,3,1,z,y,x);
const float stage4_21 = s[6272+(1*9+qy-1)*32+qx-1];
const float stage4_22 = Profile(p,3,0,z,y,x);
const float stage4_23 = derivative_3;
const float stage4_24 = __fmul_rn(stage4_22, stage4_23);
const float stage4_25 = __fmaf_rn(stage4_20, stage4_21, stage4_24);
const float stage4_26 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage4_27 = (-stage4_26);
const float stage4_28 = derivative_2;
const float stage4_29 = __fmul_rn(stage4_8, stage4_28);
const float stage4_30 = Profile(p,3,2,z,y,x);
const float stage4_31 = __fmaf_rn(stage4_23, stage4_30, stage4_25);
const float stage4_32 = __fmul_rn(stage4_12, stage4_31);
const float stage4_33 = __fadd_rn(stage4_29, stage4_32);
const float stage4_34 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage4_35 = old_H_1;
const float stage4_36 = __fmul_rn(stage4_34, stage4_35);
const float stage4_37 = __fmaf_rn(stage4_27, stage4_33, stage4_36);
const float stage4_38 = Profile(p,4,1,z,y,x);
const float stage4_39 = s[6272+(2*9+qy-1)*32+qx-1];
const float stage4_40 = Profile(p,4,0,z,y,x);
const float stage4_41 = derivative_4;
const float stage4_42 = __fmul_rn(stage4_40, stage4_41);
const float stage4_43 = __fmaf_rn(stage4_38, stage4_39, stage4_42);
const float stage4_44 = Profile(p,5,1,z,y,x);
const float stage4_45 = s[6272+(3*9+qy-1)*32+qx-1];
const float stage4_46 = Profile(p,5,0,z,y,x);
const float stage4_47 = derivative_5;
const float stage4_48 = __fmul_rn(stage4_46, stage4_47);
const float stage4_49 = __fmaf_rn(stage4_44, stage4_45, stage4_48);
const float stage4_50 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage4_51 = (-stage4_50);
const float stage4_52 = Profile(p,4,2,z,y,x);
const float stage4_53 = __fmaf_rn(stage4_41, stage4_52, stage4_43);
const float stage4_54 = __fmul_rn(stage4_8, stage4_53);
const float stage4_55 = Profile(p,5,2,z,y,x);
const float stage4_56 = __fmaf_rn(stage4_47, stage4_55, stage4_49);
const float stage4_57 = __fmul_rn(stage4_12, stage4_56);
const float stage4_58 = __fadd_rn(stage4_54, stage4_57);
const float stage4_59 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage4_60 = old_H_2;
const float stage4_61 = __fmul_rn(stage4_59, stage4_60);
const float stage4_62 = __fmaf_rn(stage4_51, stage4_58, stage4_61);
next_psi_0 = stage4_5;
next_0 = stage4_19;
next_psi_3 = stage4_25;
next_1 = stage4_37;
next_psi_4 = stage4_43;
next_psi_5 = stage4_49;
next_2 = stage4_62;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,2,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,2,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,2);
if(owned) {
Publish(observation,p.field.outputs[2],2,2,z,y,x,next_2);
}
}
}
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) previous_y_2=current_H_1;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[6272+(0*9+qy-1)*32+qx-1]=current_psi_H_0;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[6272+(1*9+qy-1)*32+qx-1]=current_psi_H_3;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[6272+(2*9+qy-1)*32+qx-1]=current_psi_H_4;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<10) s[6272+(3*9+qy-1)*32+qx-1]=current_psi_H_5;
if(wave>4 && qx>=2 && qx<30 && qy>=2 && qy<9) old_H_0=s[2432+(0*7+qy+(0)-2)*32+qx-2];
if(wave>4 && qx>=2 && qx<30 && qy>=2 && qy<9) old_H_2=s[2432+(1*7+qy+(0)-2)*32+qx-2];
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_0=next_psi_0;
current_psi_H_3=next_psi_3;
current_psi_H_4=next_psi_4;
current_psi_H_5=next_psi_5;
}
__syncthreads();
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[2432+(0*7+qy+(0)-2)*32+qx-2]=current_H_0;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[2432+(1*7+qy+(0)-2)*32+qx-2]=current_H_2;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[1920+(0*8+qy+(0)-2)*32+qx-2]=current_E_0;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[1920+(1*8+qy+(0)-2)*32+qx-2]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-2,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_4;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
float next_psi_5=0.f;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[2432+(1*7+qy+(-1)-2)*32+qx-2]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[2432+(0*7+qy+(-1)-2)*32+qx-2]);
const float stage5_0 = Profile(p,0,1,z,y,x);
const float stage5_1 = s[7424+(0*8+qy-2)*32+qx-2];
const float stage5_2 = Profile(p,0,0,z,y,x);
const float stage5_3 = derivative_0;
const float stage5_4 = __fmul_rn(stage5_2, stage5_3);
const float stage5_5 = __fmaf_rn(stage5_0, stage5_1, stage5_4);
const float stage5_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage5_7 = 1.0f;
const float stage5_8 = Profile(p,0,2,z,y,x);
const float stage5_9 = __fmaf_rn(stage5_3, stage5_8, stage5_5);
const float stage5_10 = __fmul_rn(stage5_7, stage5_9);
const float stage5_11 = -1.0f;
const float stage5_12 = derivative_1;
const float stage5_13 = __fmul_rn(stage5_11, stage5_12);
const float stage5_14 = __fadd_rn(stage5_10, stage5_13);
const float stage5_15 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage5_16 = old_E_0;
const float stage5_17 = __fmul_rn(stage5_15, stage5_16);
const float stage5_18 = __fmaf_rn(stage5_6, stage5_14, stage5_17);
const float stage5_19 = Profile(p,3,1,z,y,x);
const float stage5_20 = s[7424+(1*8+qy-2)*32+qx-2];
const float stage5_21 = Profile(p,3,0,z,y,x);
const float stage5_22 = derivative_3;
const float stage5_23 = __fmul_rn(stage5_21, stage5_22);
const float stage5_24 = __fmaf_rn(stage5_19, stage5_20, stage5_23);
const float stage5_25 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage5_26 = derivative_2;
const float stage5_27 = __fmul_rn(stage5_7, stage5_26);
const float stage5_28 = Profile(p,3,2,z,y,x);
const float stage5_29 = __fmaf_rn(stage5_22, stage5_28, stage5_24);
const float stage5_30 = __fmul_rn(stage5_11, stage5_29);
const float stage5_31 = __fadd_rn(stage5_27, stage5_30);
const float stage5_32 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage5_33 = old_E_1;
const float stage5_34 = __fmul_rn(stage5_32, stage5_33);
const float stage5_35 = __fmaf_rn(stage5_25, stage5_31, stage5_34);
const float stage5_36 = Profile(p,4,1,z,y,x);
const float stage5_37 = s[7424+(2*8+qy-2)*32+qx-2];
const float stage5_38 = Profile(p,4,0,z,y,x);
const float stage5_39 = derivative_4;
const float stage5_40 = __fmul_rn(stage5_38, stage5_39);
const float stage5_41 = __fmaf_rn(stage5_36, stage5_37, stage5_40);
const float stage5_42 = Profile(p,5,1,z,y,x);
const float stage5_43 = s[7424+(3*8+qy-2)*32+qx-2];
const float stage5_44 = Profile(p,5,0,z,y,x);
const float stage5_45 = derivative_5;
const float stage5_46 = __fmul_rn(stage5_44, stage5_45);
const float stage5_47 = __fmaf_rn(stage5_42, stage5_43, stage5_46);
const float stage5_48 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage5_49 = Profile(p,4,2,z,y,x);
const float stage5_50 = __fmaf_rn(stage5_39, stage5_49, stage5_41);
const float stage5_51 = __fmul_rn(stage5_7, stage5_50);
const float stage5_52 = Profile(p,5,2,z,y,x);
const float stage5_53 = __fmaf_rn(stage5_45, stage5_52, stage5_47);
const float stage5_54 = __fmul_rn(stage5_11, stage5_53);
const float stage5_55 = __fadd_rn(stage5_51, stage5_54);
const float stage5_56 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage5_57 = old_E_2;
const float stage5_58 = __fmul_rn(stage5_56, stage5_57);
const float stage5_59 = __fmaf_rn(stage5_48, stage5_55, stage5_58);
next_psi_0 = stage5_5;
next_0 = stage5_18;
next_psi_3 = stage5_24;
next_1 = stage5_35;
next_psi_4 = stage5_41;
next_psi_5 = stage5_47;
next_2 = stage5_59;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,2,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,2);
if(owned) {
Publish(observation,p.field.outputs[1],4,2,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,2,z,y,x,next_2);
}
}
}
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) previous_y_3=current_E_1;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[7424+(0*8+qy-2)*32+qx-2]=current_psi_E_0;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[7424+(1*8+qy-2)*32+qx-2]=current_psi_E_3;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[7424+(2*8+qy-2)*32+qx-2]=current_psi_E_4;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<10) s[7424+(3*8+qy-2)*32+qx-2]=current_psi_E_5;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_0=next_psi_0;
current_psi_E_3=next_psi_3;
current_psi_E_4=next_psi_4;
current_psi_E_5=next_psi_5;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-3,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<9) old_E_0=s[2880+(0*6+qy+(0)-3)*32+qx-3];
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<9) old_E_1=previous_y_5;
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<9) old_E_2=s[2880+(1*6+qy+(0)-3)*32+qx-3];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
float next_psi_5=0.f;
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<8) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[2880+(1*6+qy+(1)-3)*32+qx-3]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[2880+(0*6+qy+(1)-3)*32+qx-3]);
const float stage6_0 = Profile(p,0,1,z,y,x);
const float stage6_1 = s[8448+(0*7+qy-2)*32+qx-2];
const float stage6_2 = Profile(p,0,0,z,y,x);
const float stage6_3 = derivative_0;
const float stage6_4 = __fmul_rn(stage6_2, stage6_3);
const float stage6_5 = __fmaf_rn(stage6_0, stage6_1, stage6_4);
const float stage6_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage6_7 = (-stage6_6);
const float stage6_8 = 1.0f;
const float stage6_9 = Profile(p,0,2,z,y,x);
const float stage6_10 = __fmaf_rn(stage6_3, stage6_9, stage6_5);
const float stage6_11 = __fmul_rn(stage6_8, stage6_10);
const float stage6_12 = -1.0f;
const float stage6_13 = derivative_1;
const float stage6_14 = __fmul_rn(stage6_12, stage6_13);
const float stage6_15 = __fadd_rn(stage6_11, stage6_14);
const float stage6_16 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage6_17 = old_H_0;
const float stage6_18 = __fmul_rn(stage6_16, stage6_17);
const float stage6_19 = __fmaf_rn(stage6_7, stage6_15, stage6_18);
const float stage6_20 = Profile(p,3,1,z,y,x);
const float stage6_21 = s[8448+(1*7+qy-2)*32+qx-2];
const float stage6_22 = Profile(p,3,0,z,y,x);
const float stage6_23 = derivative_3;
const float stage6_24 = __fmul_rn(stage6_22, stage6_23);
const float stage6_25 = __fmaf_rn(stage6_20, stage6_21, stage6_24);
const float stage6_26 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage6_27 = (-stage6_26);
const float stage6_28 = derivative_2;
const float stage6_29 = __fmul_rn(stage6_8, stage6_28);
const float stage6_30 = Profile(p,3,2,z,y,x);
const float stage6_31 = __fmaf_rn(stage6_23, stage6_30, stage6_25);
const float stage6_32 = __fmul_rn(stage6_12, stage6_31);
const float stage6_33 = __fadd_rn(stage6_29, stage6_32);
const float stage6_34 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage6_35 = old_H_1;
const float stage6_36 = __fmul_rn(stage6_34, stage6_35);
const float stage6_37 = __fmaf_rn(stage6_27, stage6_33, stage6_36);
const float stage6_38 = Profile(p,4,1,z,y,x);
const float stage6_39 = s[8448+(2*7+qy-2)*32+qx-2];
const float stage6_40 = Profile(p,4,0,z,y,x);
const float stage6_41 = derivative_4;
const float stage6_42 = __fmul_rn(stage6_40, stage6_41);
const float stage6_43 = __fmaf_rn(stage6_38, stage6_39, stage6_42);
const float stage6_44 = Profile(p,5,1,z,y,x);
const float stage6_45 = s[8448+(3*7+qy-2)*32+qx-2];
const float stage6_46 = Profile(p,5,0,z,y,x);
const float stage6_47 = derivative_5;
const float stage6_48 = __fmul_rn(stage6_46, stage6_47);
const float stage6_49 = __fmaf_rn(stage6_44, stage6_45, stage6_48);
const float stage6_50 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage6_51 = (-stage6_50);
const float stage6_52 = Profile(p,4,2,z,y,x);
const float stage6_53 = __fmaf_rn(stage6_41, stage6_52, stage6_43);
const float stage6_54 = __fmul_rn(stage6_8, stage6_53);
const float stage6_55 = Profile(p,5,2,z,y,x);
const float stage6_56 = __fmaf_rn(stage6_47, stage6_55, stage6_49);
const float stage6_57 = __fmul_rn(stage6_12, stage6_56);
const float stage6_58 = __fadd_rn(stage6_54, stage6_57);
const float stage6_59 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage6_60 = old_H_2;
const float stage6_61 = __fmul_rn(stage6_59, stage6_60);
const float stage6_62 = __fmaf_rn(stage6_51, stage6_58, stage6_61);
next_psi_0 = stage6_5;
next_0 = stage6_19;
next_psi_3 = stage6_25;
next_1 = stage6_37;
next_psi_4 = stage6_43;
next_psi_5 = stage6_49;
next_2 = stage6_62;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,3,z,y,x,next_0);
static_cast<float *>(final.fields[0].data)[BeamzOffset3D<false>(final.fields[0],z,y,x)]=next_0;
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,3,z,y,x,next_1);
static_cast<float *>(final.fields[1].data)[BeamzOffset3D<false>(final.fields[1],z,y,x)]=next_1;
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,3);
if(owned) {
Publish(observation,p.field.outputs[2],2,3,z,y,x,next_2);
static_cast<float *>(final.fields[2].data)[BeamzOffset3D<false>(final.fields[2],z,y,x)]=next_2;
}
}
if(owned) WritePsi<false>(p,0,z,y,x,next_psi_0);
if(owned) WritePsi<false>(p,3,z,y,x,next_psi_3);
if(owned) WritePsi<false>(p,4,z,y,x,next_psi_4);
if(owned) WritePsi<false>(p,5,z,y,x,next_psi_5);
}
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) previous_y_4=current_H_1;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[8448+(0*7+qy-2)*32+qx-2]=current_psi_H_0;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[8448+(1*7+qy-2)*32+qx-2]=current_psi_H_3;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[8448+(2*7+qy-2)*32+qx-2]=current_psi_H_4;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<9) s[8448+(3*7+qy-2)*32+qx-2]=current_psi_H_5;
if(wave>6 && qx>=3 && qx<29 && qy>=3 && qy<8) old_H_0=s[3264+(0*5+qy+(0)-3)*32+qx-3];
if(wave>6 && qx>=3 && qx<29 && qy>=3 && qy<8) old_H_2=s[3264+(1*5+qy+(0)-3)*32+qx-3];
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<8) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_0=next_psi_0;
current_psi_H_3=next_psi_3;
current_psi_H_4=next_psi_4;
current_psi_H_5=next_psi_5;
}
__syncthreads();
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<8) s[3264+(0*5+qy+(0)-3)*32+qx-3]=current_H_0;
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<8) s[3264+(1*5+qy+(0)-3)*32+qx-3]=current_H_2;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[2880+(0*6+qy+(0)-3)*32+qx-3]=current_E_0;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[2880+(1*6+qy+(0)-3)*32+qx-3]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-3,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_6;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
float next_psi_5=0.f;
if(wave>=7 && qx>=4 && qx<29 && qy>=4 && qy<8) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[3264+(1*5+qy+(-1)-3)*32+qx-3]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[3264+(0*5+qy+(-1)-3)*32+qx-3]);
const float stage7_0 = Profile(p,0,1,z,y,x);
const float stage7_1 = s[9344+(0*6+qy-3)*32+qx-3];
const float stage7_2 = Profile(p,0,0,z,y,x);
const float stage7_3 = derivative_0;
const float stage7_4 = __fmul_rn(stage7_2, stage7_3);
const float stage7_5 = __fmaf_rn(stage7_0, stage7_1, stage7_4);
const float stage7_6 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage7_7 = 1.0f;
const float stage7_8 = Profile(p,0,2,z,y,x);
const float stage7_9 = __fmaf_rn(stage7_3, stage7_8, stage7_5);
const float stage7_10 = __fmul_rn(stage7_7, stage7_9);
const float stage7_11 = -1.0f;
const float stage7_12 = derivative_1;
const float stage7_13 = __fmul_rn(stage7_11, stage7_12);
const float stage7_14 = __fadd_rn(stage7_10, stage7_13);
const float stage7_15 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage7_16 = old_E_0;
const float stage7_17 = __fmul_rn(stage7_15, stage7_16);
const float stage7_18 = __fmaf_rn(stage7_6, stage7_14, stage7_17);
const float stage7_19 = Profile(p,3,1,z,y,x);
const float stage7_20 = s[9344+(1*6+qy-3)*32+qx-3];
const float stage7_21 = Profile(p,3,0,z,y,x);
const float stage7_22 = derivative_3;
const float stage7_23 = __fmul_rn(stage7_21, stage7_22);
const float stage7_24 = __fmaf_rn(stage7_19, stage7_20, stage7_23);
const float stage7_25 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage7_26 = derivative_2;
const float stage7_27 = __fmul_rn(stage7_7, stage7_26);
const float stage7_28 = Profile(p,3,2,z,y,x);
const float stage7_29 = __fmaf_rn(stage7_22, stage7_28, stage7_24);
const float stage7_30 = __fmul_rn(stage7_11, stage7_29);
const float stage7_31 = __fadd_rn(stage7_27, stage7_30);
const float stage7_32 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage7_33 = old_E_1;
const float stage7_34 = __fmul_rn(stage7_32, stage7_33);
const float stage7_35 = __fmaf_rn(stage7_25, stage7_31, stage7_34);
const float stage7_36 = Profile(p,4,1,z,y,x);
const float stage7_37 = s[9344+(2*6+qy-3)*32+qx-3];
const float stage7_38 = Profile(p,4,0,z,y,x);
const float stage7_39 = derivative_4;
const float stage7_40 = __fmul_rn(stage7_38, stage7_39);
const float stage7_41 = __fmaf_rn(stage7_36, stage7_37, stage7_40);
const float stage7_42 = Profile(p,5,1,z,y,x);
const float stage7_43 = s[9344+(3*6+qy-3)*32+qx-3];
const float stage7_44 = Profile(p,5,0,z,y,x);
const float stage7_45 = derivative_5;
const float stage7_46 = __fmul_rn(stage7_44, stage7_45);
const float stage7_47 = __fmaf_rn(stage7_42, stage7_43, stage7_46);
const float stage7_48 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage7_49 = Profile(p,4,2,z,y,x);
const float stage7_50 = __fmaf_rn(stage7_39, stage7_49, stage7_41);
const float stage7_51 = __fmul_rn(stage7_7, stage7_50);
const float stage7_52 = Profile(p,5,2,z,y,x);
const float stage7_53 = __fmaf_rn(stage7_45, stage7_52, stage7_47);
const float stage7_54 = __fmul_rn(stage7_11, stage7_53);
const float stage7_55 = __fadd_rn(stage7_51, stage7_54);
const float stage7_56 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage7_57 = old_E_2;
const float stage7_58 = __fmul_rn(stage7_56, stage7_57);
const float stage7_59 = __fmaf_rn(stage7_48, stage7_55, stage7_58);
next_psi_0 = stage7_5;
next_0 = stage7_18;
next_psi_3 = stage7_24;
next_1 = stage7_35;
next_psi_4 = stage7_41;
next_psi_5 = stage7_47;
next_2 = stage7_59;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,3,z,y,x,next_0);
static_cast<float *>(final.fields[3].data)[BeamzOffset3D<false>(final.fields[3],z,y,x)]=next_0;
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,3);
if(owned) {
Publish(observation,p.field.outputs[1],4,3,z,y,x,next_1);
static_cast<float *>(final.fields[4].data)[BeamzOffset3D<false>(final.fields[4],z,y,x)]=next_1;
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,3,z,y,x,next_2);
static_cast<float *>(final.fields[5].data)[BeamzOffset3D<false>(final.fields[5],z,y,x)]=next_2;
}
}
if(owned) WritePsi<false>(p,0,z,y,x,next_psi_0);
if(owned) WritePsi<false>(p,3,z,y,x,next_psi_3);
if(owned) WritePsi<false>(p,4,z,y,x,next_psi_4);
if(owned) WritePsi<false>(p,5,z,y,x,next_psi_5);
}
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) previous_y_5=current_E_1;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[9344+(0*6+qy-3)*32+qx-3]=current_psi_E_0;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[9344+(1*6+qy-3)*32+qx-3]=current_psi_E_3;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[9344+(2*6+qy-3)*32+qx-3]=current_psi_E_4;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<9) s[9344+(3*6+qy-3)*32+qx-3]=current_psi_E_5;
if(wave>=7 && qx>=4 && qx<29 && qy>=4 && qy<8) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_0=next_psi_0;
current_psi_E_3=next_psi_3;
current_psi_E_4=next_psi_4;
current_psi_E_5=next_psi_5;
}
}
    if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<8) previous_y_6=current_H_1;
  }
}


constexpr int DeepWarpD4M7F32ScalarHBytePackedSource20Threads=320;
constexpr int DeepWarpD4M7F32ScalarHBytePackedSource20SharedBytes=47104;
template<bool Packed> __global__ __launch_bounds__(320,1) void DeepWarpD4M7F32ScalarHBytePackedSource20(
    const __grid_constant__ PairCpmlPhase h,const __grid_constant__ PairCpmlPhase e,
    const __grid_constant__ PairFieldOutputs final,const __grid_constant__ FusedHSources hs,
    const __grid_constant__ FusedHSources es,const __grid_constant__ Observation observation,PairCpmlGrid grid) {
  if(blockDim.x!=320)return;
  extern __shared__ float s[];
  auto coordinate=[&](int axis,int block,int size) {const int low=(grid.high[axis]+size-1)/size;
    return (7&(1<<axis)) ? (block<low?block*size:grid.boundary[axis]+(block-low)*size) : grid.high[axis]+block*size;};
  auto end=[&](int axis,int block,int origin,int size) {const int low=(grid.high[axis]+size-1)/size;
    return min(origin+size,(7&(1<<axis))?(block<low?grid.high[axis]:grid.extent[axis]):grid.boundary[axis]);};
  const int ox=coordinate(2,blockIdx.x,25),oy=coordinate(1,blockIdx.y,3),oz=coordinate(0,blockIdx.z,64);
  const int end_x=end(2,blockIdx.x,ox,25),end_y=end(1,blockIdx.y,oy,3),end_z=end(0,blockIdx.z,oz,64);
  const int qx=threadIdx.x&31,qy=threadIdx.x>>5;
  __shared__ bool active_sources[6];
  if(threadIdx.x<6){const int c=threadIdx.x;const auto &g=c<3?hs.groups[c]:es.groups[c-3];
    const auto *starts=static_cast<const int *>(g.starts.data);bool active=false;
    for(int i=0;i<g.coefficients.dims[0];++i)
      active |= starts[3*i]<oz+68 && starts[3*i]+g.coefficients.dims[1]>oz-4 &&
                starts[3*i+1]<oy+7 && starts[3*i+1]+g.coefficients.dims[2]>oy-4 &&
                starts[3*i+2]<ox+29 && starts[3*i+2]+g.coefficients.dims[3]>ox-4;
    active_sources[c]=active;}
  __syncthreads();
  float previous_y_0=0.f;
float previous_y_1=0.f;
float previous_y_2=0.f;
float previous_y_3=0.f;
float previous_y_4=0.f;
float previous_y_5=0.f;
float previous_y_6=0.f;
  for(int wave=0;wave<end_z-oz+7;++wave){
    float current_H_0=0.f,old_H_0=0.f;
float current_H_1=0.f,old_H_1=0.f;
float current_H_2=0.f,old_H_2=0.f;
float current_E_0=0.f,old_E_0=0.f;
float current_E_1=0.f,old_E_1=0.f;
float current_E_2=0.f,old_E_2=0.f;
float current_psi_H_0=0.f;
float current_psi_H_1=0.f;
float current_psi_H_2=0.f;
float current_psi_H_3=0.f;
float current_psi_H_4=0.f;
float current_psi_H_5=0.f;
float current_psi_E_0=0.f;
float current_psi_E_1=0.f;
float current_psi_E_2=0.f;
float current_psi_E_3=0.f;
float current_psi_E_4=0.f;
float current_psi_E_5=0.f;
    {
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-0,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
float next_psi_5=0.f;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<10) {
const float derivative_0=Difference(p,0,2,1,z,y,x,SafeRead(p.field.inputs[5],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[5],z+(0),y+(1),x+(0)));
const float derivative_1=Difference(p,0,1,0,z,y,x,SafeRead(p.field.inputs[4],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[4],z+(1),y+(0),x+(0)));
const float derivative_2=Difference(p,0,0,0,z,y,x,SafeRead(p.field.inputs[3],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[3],z+(1),y+(0),x+(0)));
const float derivative_3=Difference(p,0,2,2,z,y,x,SafeRead(p.field.inputs[5],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[5],z+(0),y+(0),x+(1)));
const float derivative_4=Difference(p,0,1,2,z,y,x,SafeRead(p.field.inputs[4],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[4],z+(0),y+(0),x+(1)));
const float derivative_5=Difference(p,0,0,1,z,y,x,SafeRead(p.field.inputs[3],z+(0),y+(0),x+(0)),SafeRead(p.field.inputs[3],z+(0),y+(1),x+(0)));
const float stage0_0 = Profile(p,0,1,z,y,x);
const float stage0_1 = ReadPsi<false>(p,0,z,y,x);
const float stage0_2 = Profile(p,0,0,z,y,x);
const float stage0_3 = derivative_0;
const float stage0_4 = __fmul_rn(stage0_2, stage0_3);
const float stage0_5 = __fmaf_rn(stage0_0, stage0_1, stage0_4);
const float stage0_6 = Profile(p,1,1,z,y,x);
const float stage0_7 = ReadPsi<false>(p,1,z,y,x);
const float stage0_8 = Profile(p,1,0,z,y,x);
const float stage0_9 = derivative_1;
const float stage0_10 = __fmul_rn(stage0_8, stage0_9);
const float stage0_11 = __fmaf_rn(stage0_6, stage0_7, stage0_10);
const float stage0_12 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage0_13 = (-stage0_12);
const float stage0_14 = 1.0f;
const float stage0_15 = Profile(p,0,2,z,y,x);
const float stage0_16 = __fmaf_rn(stage0_3, stage0_15, stage0_5);
const float stage0_17 = __fmul_rn(stage0_14, stage0_16);
const float stage0_18 = -1.0f;
const float stage0_19 = Profile(p,1,2,z,y,x);
const float stage0_20 = __fmaf_rn(stage0_9, stage0_19, stage0_11);
const float stage0_21 = __fmul_rn(stage0_18, stage0_20);
const float stage0_22 = __fadd_rn(stage0_17, stage0_21);
const float stage0_23 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage0_24 = SafeRead(p.field.inputs[0],z,y,x);
const float stage0_25 = __fmul_rn(stage0_23, stage0_24);
const float stage0_26 = __fmaf_rn(stage0_13, stage0_22, stage0_25);
const float stage0_27 = Profile(p,2,1,z,y,x);
const float stage0_28 = ReadPsi<false>(p,2,z,y,x);
const float stage0_29 = Profile(p,2,0,z,y,x);
const float stage0_30 = derivative_2;
const float stage0_31 = __fmul_rn(stage0_29, stage0_30);
const float stage0_32 = __fmaf_rn(stage0_27, stage0_28, stage0_31);
const float stage0_33 = Profile(p,3,1,z,y,x);
const float stage0_34 = ReadPsi<false>(p,3,z,y,x);
const float stage0_35 = Profile(p,3,0,z,y,x);
const float stage0_36 = derivative_3;
const float stage0_37 = __fmul_rn(stage0_35, stage0_36);
const float stage0_38 = __fmaf_rn(stage0_33, stage0_34, stage0_37);
const float stage0_39 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage0_40 = (-stage0_39);
const float stage0_41 = Profile(p,2,2,z,y,x);
const float stage0_42 = __fmaf_rn(stage0_30, stage0_41, stage0_32);
const float stage0_43 = __fmul_rn(stage0_14, stage0_42);
const float stage0_44 = Profile(p,3,2,z,y,x);
const float stage0_45 = __fmaf_rn(stage0_36, stage0_44, stage0_38);
const float stage0_46 = __fmul_rn(stage0_18, stage0_45);
const float stage0_47 = __fadd_rn(stage0_43, stage0_46);
const float stage0_48 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage0_49 = SafeRead(p.field.inputs[1],z,y,x);
const float stage0_50 = __fmul_rn(stage0_48, stage0_49);
const float stage0_51 = __fmaf_rn(stage0_40, stage0_47, stage0_50);
const float stage0_52 = Profile(p,4,1,z,y,x);
const float stage0_53 = ReadPsi<false>(p,4,z,y,x);
const float stage0_54 = Profile(p,4,0,z,y,x);
const float stage0_55 = derivative_4;
const float stage0_56 = __fmul_rn(stage0_54, stage0_55);
const float stage0_57 = __fmaf_rn(stage0_52, stage0_53, stage0_56);
const float stage0_58 = Profile(p,5,1,z,y,x);
const float stage0_59 = ReadPsi<false>(p,5,z,y,x);
const float stage0_60 = Profile(p,5,0,z,y,x);
const float stage0_61 = derivative_5;
const float stage0_62 = __fmul_rn(stage0_60, stage0_61);
const float stage0_63 = __fmaf_rn(stage0_58, stage0_59, stage0_62);
const float stage0_64 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage0_65 = (-stage0_64);
const float stage0_66 = Profile(p,4,2,z,y,x);
const float stage0_67 = __fmaf_rn(stage0_55, stage0_66, stage0_57);
const float stage0_68 = __fmul_rn(stage0_14, stage0_67);
const float stage0_69 = Profile(p,5,2,z,y,x);
const float stage0_70 = __fmaf_rn(stage0_61, stage0_69, stage0_63);
const float stage0_71 = __fmul_rn(stage0_18, stage0_70);
const float stage0_72 = __fadd_rn(stage0_68, stage0_71);
const float stage0_73 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage0_74 = SafeRead(p.field.inputs[2],z,y,x);
const float stage0_75 = __fmul_rn(stage0_73, stage0_74);
const float stage0_76 = __fmaf_rn(stage0_65, stage0_72, stage0_75);
next_psi_0 = stage0_5;
next_psi_1 = stage0_11;
next_0 = stage0_26;
next_psi_2 = stage0_32;
next_psi_3 = stage0_38;
next_1 = stage0_51;
next_psi_4 = stage0_57;
next_psi_5 = stage0_63;
next_2 = stage0_76;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,0,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,0,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,0);
if(owned) {
Publish(observation,p.field.outputs[2],2,0,z,y,x,next_2);
}
}
}
if(wave>0 && qx>=0 && qx<32 && qy>=0 && qy<10) old_H_0=s[0+(0*10+qy+(0)-0)*32+qx-0];
if(wave>0 && qx>=0 && qx<32 && qy>=0 && qy<10) old_H_2=s[0+(1*10+qy+(0)-0)*32+qx-0];
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<10) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_0=next_psi_0;
current_psi_H_1=next_psi_1;
current_psi_H_2=next_psi_2;
current_psi_H_3=next_psi_3;
current_psi_H_4=next_psi_4;
current_psi_H_5=next_psi_5;
}
__syncthreads();
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<10) s[0+(0*10+qy+(0)-0)*32+qx-0]=current_H_0;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<10) s[0+(1*10+qy+(0)-0)*32+qx-0]=current_H_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-0,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_0;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
float next_psi_5=0.f;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<10) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[0+(1*10+qy+(-1)-0)*32+qx-0]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[0+(0*10+qy+(-1)-0)*32+qx-0]);
const float stage1_0 = Profile(p,0,1,z,y,x);
const float stage1_1 = ReadPsi<false>(p,0,z,y,x);
const float stage1_2 = Profile(p,0,0,z,y,x);
const float stage1_3 = derivative_0;
const float stage1_4 = __fmul_rn(stage1_2, stage1_3);
const float stage1_5 = __fmaf_rn(stage1_0, stage1_1, stage1_4);
const float stage1_6 = Profile(p,1,1,z,y,x);
const float stage1_7 = ReadPsi<false>(p,1,z,y,x);
const float stage1_8 = Profile(p,1,0,z,y,x);
const float stage1_9 = derivative_1;
const float stage1_10 = __fmul_rn(stage1_8, stage1_9);
const float stage1_11 = __fmaf_rn(stage1_6, stage1_7, stage1_10);
const float stage1_12 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage1_13 = 1.0f;
const float stage1_14 = Profile(p,0,2,z,y,x);
const float stage1_15 = __fmaf_rn(stage1_3, stage1_14, stage1_5);
const float stage1_16 = __fmul_rn(stage1_13, stage1_15);
const float stage1_17 = -1.0f;
const float stage1_18 = Profile(p,1,2,z,y,x);
const float stage1_19 = __fmaf_rn(stage1_9, stage1_18, stage1_11);
const float stage1_20 = __fmul_rn(stage1_17, stage1_19);
const float stage1_21 = __fadd_rn(stage1_16, stage1_20);
const float stage1_22 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage1_23 = SafeRead(p.field.inputs[0],z,y,x);
const float stage1_24 = __fmul_rn(stage1_22, stage1_23);
const float stage1_25 = __fmaf_rn(stage1_12, stage1_21, stage1_24);
const float stage1_26 = Profile(p,2,1,z,y,x);
const float stage1_27 = ReadPsi<false>(p,2,z,y,x);
const float stage1_28 = Profile(p,2,0,z,y,x);
const float stage1_29 = derivative_2;
const float stage1_30 = __fmul_rn(stage1_28, stage1_29);
const float stage1_31 = __fmaf_rn(stage1_26, stage1_27, stage1_30);
const float stage1_32 = Profile(p,3,1,z,y,x);
const float stage1_33 = ReadPsi<false>(p,3,z,y,x);
const float stage1_34 = Profile(p,3,0,z,y,x);
const float stage1_35 = derivative_3;
const float stage1_36 = __fmul_rn(stage1_34, stage1_35);
const float stage1_37 = __fmaf_rn(stage1_32, stage1_33, stage1_36);
const float stage1_38 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage1_39 = Profile(p,2,2,z,y,x);
const float stage1_40 = __fmaf_rn(stage1_29, stage1_39, stage1_31);
const float stage1_41 = __fmul_rn(stage1_13, stage1_40);
const float stage1_42 = Profile(p,3,2,z,y,x);
const float stage1_43 = __fmaf_rn(stage1_35, stage1_42, stage1_37);
const float stage1_44 = __fmul_rn(stage1_17, stage1_43);
const float stage1_45 = __fadd_rn(stage1_41, stage1_44);
const float stage1_46 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage1_47 = SafeRead(p.field.inputs[1],z,y,x);
const float stage1_48 = __fmul_rn(stage1_46, stage1_47);
const float stage1_49 = __fmaf_rn(stage1_38, stage1_45, stage1_48);
const float stage1_50 = Profile(p,4,1,z,y,x);
const float stage1_51 = ReadPsi<false>(p,4,z,y,x);
const float stage1_52 = Profile(p,4,0,z,y,x);
const float stage1_53 = derivative_4;
const float stage1_54 = __fmul_rn(stage1_52, stage1_53);
const float stage1_55 = __fmaf_rn(stage1_50, stage1_51, stage1_54);
const float stage1_56 = Profile(p,5,1,z,y,x);
const float stage1_57 = ReadPsi<false>(p,5,z,y,x);
const float stage1_58 = Profile(p,5,0,z,y,x);
const float stage1_59 = derivative_5;
const float stage1_60 = __fmul_rn(stage1_58, stage1_59);
const float stage1_61 = __fmaf_rn(stage1_56, stage1_57, stage1_60);
const float stage1_62 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage1_63 = Profile(p,4,2,z,y,x);
const float stage1_64 = __fmaf_rn(stage1_53, stage1_63, stage1_55);
const float stage1_65 = __fmul_rn(stage1_13, stage1_64);
const float stage1_66 = Profile(p,5,2,z,y,x);
const float stage1_67 = __fmaf_rn(stage1_59, stage1_66, stage1_61);
const float stage1_68 = __fmul_rn(stage1_17, stage1_67);
const float stage1_69 = __fadd_rn(stage1_65, stage1_68);
const float stage1_70 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage1_71 = SafeRead(p.field.inputs[2],z,y,x);
const float stage1_72 = __fmul_rn(stage1_70, stage1_71);
const float stage1_73 = __fmaf_rn(stage1_62, stage1_69, stage1_72);
next_psi_0 = stage1_5;
next_psi_1 = stage1_11;
next_0 = stage1_25;
next_psi_2 = stage1_31;
next_psi_3 = stage1_37;
next_1 = stage1_49;
next_psi_4 = stage1_55;
next_psi_5 = stage1_61;
next_2 = stage1_73;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,0,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,0);
if(owned) {
Publish(observation,p.field.outputs[1],4,0,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,0,z,y,x,next_2);
}
}
}
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<10) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_0=next_psi_0;
current_psi_E_1=next_psi_1;
current_psi_E_2=next_psi_2;
current_psi_E_3=next_psi_3;
current_psi_E_4=next_psi_4;
current_psi_E_5=next_psi_5;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-1,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<10) old_E_0=s[640+(0*9+qy+(0)-1)*32+qx-1];
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<10) old_E_1=previous_y_1;
if(wave>=2 && qx>=1 && qx<32 && qy>=1 && qy<10) old_E_2=s[640+(1*9+qy+(0)-1)*32+qx-1];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
float next_psi_5=0.f;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<9) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[640+(1*9+qy+(1)-1)*32+qx-1]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[640+(0*9+qy+(1)-1)*32+qx-1]);
const float stage2_0 = Profile(p,0,1,z,y,x);
const float stage2_1 = s[3136+(0*10+qy-0)*32+qx-0];
const float stage2_2 = Profile(p,0,0,z,y,x);
const float stage2_3 = derivative_0;
const float stage2_4 = __fmul_rn(stage2_2, stage2_3);
const float stage2_5 = __fmaf_rn(stage2_0, stage2_1, stage2_4);
const float stage2_6 = Profile(p,1,1,z,y,x);
const float stage2_7 = s[3136+(1*10+qy-0)*32+qx-0];
const float stage2_8 = Profile(p,1,0,z,y,x);
const float stage2_9 = derivative_1;
const float stage2_10 = __fmul_rn(stage2_8, stage2_9);
const float stage2_11 = __fmaf_rn(stage2_6, stage2_7, stage2_10);
const float stage2_12 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage2_13 = (-stage2_12);
const float stage2_14 = 1.0f;
const float stage2_15 = Profile(p,0,2,z,y,x);
const float stage2_16 = __fmaf_rn(stage2_3, stage2_15, stage2_5);
const float stage2_17 = __fmul_rn(stage2_14, stage2_16);
const float stage2_18 = -1.0f;
const float stage2_19 = Profile(p,1,2,z,y,x);
const float stage2_20 = __fmaf_rn(stage2_9, stage2_19, stage2_11);
const float stage2_21 = __fmul_rn(stage2_18, stage2_20);
const float stage2_22 = __fadd_rn(stage2_17, stage2_21);
const float stage2_23 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage2_24 = old_H_0;
const float stage2_25 = __fmul_rn(stage2_23, stage2_24);
const float stage2_26 = __fmaf_rn(stage2_13, stage2_22, stage2_25);
const float stage2_27 = Profile(p,2,1,z,y,x);
const float stage2_28 = s[3136+(2*10+qy-0)*32+qx-0];
const float stage2_29 = Profile(p,2,0,z,y,x);
const float stage2_30 = derivative_2;
const float stage2_31 = __fmul_rn(stage2_29, stage2_30);
const float stage2_32 = __fmaf_rn(stage2_27, stage2_28, stage2_31);
const float stage2_33 = Profile(p,3,1,z,y,x);
const float stage2_34 = s[3136+(3*10+qy-0)*32+qx-0];
const float stage2_35 = Profile(p,3,0,z,y,x);
const float stage2_36 = derivative_3;
const float stage2_37 = __fmul_rn(stage2_35, stage2_36);
const float stage2_38 = __fmaf_rn(stage2_33, stage2_34, stage2_37);
const float stage2_39 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage2_40 = (-stage2_39);
const float stage2_41 = Profile(p,2,2,z,y,x);
const float stage2_42 = __fmaf_rn(stage2_30, stage2_41, stage2_32);
const float stage2_43 = __fmul_rn(stage2_14, stage2_42);
const float stage2_44 = Profile(p,3,2,z,y,x);
const float stage2_45 = __fmaf_rn(stage2_36, stage2_44, stage2_38);
const float stage2_46 = __fmul_rn(stage2_18, stage2_45);
const float stage2_47 = __fadd_rn(stage2_43, stage2_46);
const float stage2_48 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage2_49 = old_H_1;
const float stage2_50 = __fmul_rn(stage2_48, stage2_49);
const float stage2_51 = __fmaf_rn(stage2_40, stage2_47, stage2_50);
const float stage2_52 = Profile(p,4,1,z,y,x);
const float stage2_53 = s[3136+(4*10+qy-0)*32+qx-0];
const float stage2_54 = Profile(p,4,0,z,y,x);
const float stage2_55 = derivative_4;
const float stage2_56 = __fmul_rn(stage2_54, stage2_55);
const float stage2_57 = __fmaf_rn(stage2_52, stage2_53, stage2_56);
const float stage2_58 = Profile(p,5,1,z,y,x);
const float stage2_59 = s[3136+(5*10+qy-0)*32+qx-0];
const float stage2_60 = Profile(p,5,0,z,y,x);
const float stage2_61 = derivative_5;
const float stage2_62 = __fmul_rn(stage2_60, stage2_61);
const float stage2_63 = __fmaf_rn(stage2_58, stage2_59, stage2_62);
const float stage2_64 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage2_65 = (-stage2_64);
const float stage2_66 = Profile(p,4,2,z,y,x);
const float stage2_67 = __fmaf_rn(stage2_55, stage2_66, stage2_57);
const float stage2_68 = __fmul_rn(stage2_14, stage2_67);
const float stage2_69 = Profile(p,5,2,z,y,x);
const float stage2_70 = __fmaf_rn(stage2_61, stage2_69, stage2_63);
const float stage2_71 = __fmul_rn(stage2_18, stage2_70);
const float stage2_72 = __fadd_rn(stage2_68, stage2_71);
const float stage2_73 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage2_74 = old_H_2;
const float stage2_75 = __fmul_rn(stage2_73, stage2_74);
const float stage2_76 = __fmaf_rn(stage2_65, stage2_72, stage2_75);
next_psi_0 = stage2_5;
next_psi_1 = stage2_11;
next_0 = stage2_26;
next_psi_2 = stage2_32;
next_psi_3 = stage2_38;
next_1 = stage2_51;
next_psi_4 = stage2_57;
next_psi_5 = stage2_63;
next_2 = stage2_76;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,1,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,1,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,1);
if(owned) {
Publish(observation,p.field.outputs[2],2,1,z,y,x,next_2);
}
}
}
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<10) previous_y_0=current_H_1;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<10) s[3136+(0*10+qy-0)*32+qx-0]=current_psi_H_0;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<10) s[3136+(1*10+qy-0)*32+qx-0]=current_psi_H_1;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<10) s[3136+(2*10+qy-0)*32+qx-0]=current_psi_H_2;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<10) s[3136+(3*10+qy-0)*32+qx-0]=current_psi_H_3;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<10) s[3136+(4*10+qy-0)*32+qx-0]=current_psi_H_4;
if(wave>=0 && qx>=0 && qx<32 && qy>=0 && qy<10) s[3136+(5*10+qy-0)*32+qx-0]=current_psi_H_5;
if(wave>2 && qx>=1 && qx<31 && qy>=1 && qy<9) old_H_0=s[1216+(0*8+qy+(0)-1)*32+qx-1];
if(wave>2 && qx>=1 && qx<31 && qy>=1 && qy<9) old_H_2=s[1216+(1*8+qy+(0)-1)*32+qx-1];
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<9) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_0=next_psi_0;
current_psi_H_1=next_psi_1;
current_psi_H_2=next_psi_2;
current_psi_H_3=next_psi_3;
current_psi_H_4=next_psi_4;
current_psi_H_5=next_psi_5;
}
__syncthreads();
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<9) s[1216+(0*8+qy+(0)-1)*32+qx-1]=current_H_0;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<9) s[1216+(1*8+qy+(0)-1)*32+qx-1]=current_H_2;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<10) s[640+(0*9+qy+(0)-1)*32+qx-1]=current_E_0;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<10) s[640+(1*9+qy+(0)-1)*32+qx-1]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-1,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_2;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
float next_psi_5=0.f;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<9) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[1216+(1*8+qy+(-1)-1)*32+qx-1]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[1216+(0*8+qy+(-1)-1)*32+qx-1]);
const float stage3_0 = Profile(p,0,1,z,y,x);
const float stage3_1 = s[5056+(0*9+qy-1)*32+qx-1];
const float stage3_2 = Profile(p,0,0,z,y,x);
const float stage3_3 = derivative_0;
const float stage3_4 = __fmul_rn(stage3_2, stage3_3);
const float stage3_5 = __fmaf_rn(stage3_0, stage3_1, stage3_4);
const float stage3_6 = Profile(p,1,1,z,y,x);
const float stage3_7 = s[5056+(1*9+qy-1)*32+qx-1];
const float stage3_8 = Profile(p,1,0,z,y,x);
const float stage3_9 = derivative_1;
const float stage3_10 = __fmul_rn(stage3_8, stage3_9);
const float stage3_11 = __fmaf_rn(stage3_6, stage3_7, stage3_10);
const float stage3_12 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage3_13 = 1.0f;
const float stage3_14 = Profile(p,0,2,z,y,x);
const float stage3_15 = __fmaf_rn(stage3_3, stage3_14, stage3_5);
const float stage3_16 = __fmul_rn(stage3_13, stage3_15);
const float stage3_17 = -1.0f;
const float stage3_18 = Profile(p,1,2,z,y,x);
const float stage3_19 = __fmaf_rn(stage3_9, stage3_18, stage3_11);
const float stage3_20 = __fmul_rn(stage3_17, stage3_19);
const float stage3_21 = __fadd_rn(stage3_16, stage3_20);
const float stage3_22 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage3_23 = old_E_0;
const float stage3_24 = __fmul_rn(stage3_22, stage3_23);
const float stage3_25 = __fmaf_rn(stage3_12, stage3_21, stage3_24);
const float stage3_26 = Profile(p,2,1,z,y,x);
const float stage3_27 = s[5056+(2*9+qy-1)*32+qx-1];
const float stage3_28 = Profile(p,2,0,z,y,x);
const float stage3_29 = derivative_2;
const float stage3_30 = __fmul_rn(stage3_28, stage3_29);
const float stage3_31 = __fmaf_rn(stage3_26, stage3_27, stage3_30);
const float stage3_32 = Profile(p,3,1,z,y,x);
const float stage3_33 = s[5056+(3*9+qy-1)*32+qx-1];
const float stage3_34 = Profile(p,3,0,z,y,x);
const float stage3_35 = derivative_3;
const float stage3_36 = __fmul_rn(stage3_34, stage3_35);
const float stage3_37 = __fmaf_rn(stage3_32, stage3_33, stage3_36);
const float stage3_38 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage3_39 = Profile(p,2,2,z,y,x);
const float stage3_40 = __fmaf_rn(stage3_29, stage3_39, stage3_31);
const float stage3_41 = __fmul_rn(stage3_13, stage3_40);
const float stage3_42 = Profile(p,3,2,z,y,x);
const float stage3_43 = __fmaf_rn(stage3_35, stage3_42, stage3_37);
const float stage3_44 = __fmul_rn(stage3_17, stage3_43);
const float stage3_45 = __fadd_rn(stage3_41, stage3_44);
const float stage3_46 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage3_47 = old_E_1;
const float stage3_48 = __fmul_rn(stage3_46, stage3_47);
const float stage3_49 = __fmaf_rn(stage3_38, stage3_45, stage3_48);
const float stage3_50 = Profile(p,4,1,z,y,x);
const float stage3_51 = s[5056+(4*9+qy-1)*32+qx-1];
const float stage3_52 = Profile(p,4,0,z,y,x);
const float stage3_53 = derivative_4;
const float stage3_54 = __fmul_rn(stage3_52, stage3_53);
const float stage3_55 = __fmaf_rn(stage3_50, stage3_51, stage3_54);
const float stage3_56 = Profile(p,5,1,z,y,x);
const float stage3_57 = s[5056+(5*9+qy-1)*32+qx-1];
const float stage3_58 = Profile(p,5,0,z,y,x);
const float stage3_59 = derivative_5;
const float stage3_60 = __fmul_rn(stage3_58, stage3_59);
const float stage3_61 = __fmaf_rn(stage3_56, stage3_57, stage3_60);
const float stage3_62 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage3_63 = Profile(p,4,2,z,y,x);
const float stage3_64 = __fmaf_rn(stage3_53, stage3_63, stage3_55);
const float stage3_65 = __fmul_rn(stage3_13, stage3_64);
const float stage3_66 = Profile(p,5,2,z,y,x);
const float stage3_67 = __fmaf_rn(stage3_59, stage3_66, stage3_61);
const float stage3_68 = __fmul_rn(stage3_17, stage3_67);
const float stage3_69 = __fadd_rn(stage3_65, stage3_68);
const float stage3_70 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage3_71 = old_E_2;
const float stage3_72 = __fmul_rn(stage3_70, stage3_71);
const float stage3_73 = __fmaf_rn(stage3_62, stage3_69, stage3_72);
next_psi_0 = stage3_5;
next_psi_1 = stage3_11;
next_0 = stage3_25;
next_psi_2 = stage3_31;
next_psi_3 = stage3_37;
next_1 = stage3_49;
next_psi_4 = stage3_55;
next_psi_5 = stage3_61;
next_2 = stage3_73;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,1,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,1);
if(owned) {
Publish(observation,p.field.outputs[1],4,1,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,1,z,y,x,next_2);
}
}
}
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<10) previous_y_1=current_E_1;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<10) s[5056+(0*9+qy-1)*32+qx-1]=current_psi_E_0;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<10) s[5056+(1*9+qy-1)*32+qx-1]=current_psi_E_1;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<10) s[5056+(2*9+qy-1)*32+qx-1]=current_psi_E_2;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<10) s[5056+(3*9+qy-1)*32+qx-1]=current_psi_E_3;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<10) s[5056+(4*9+qy-1)*32+qx-1]=current_psi_E_4;
if(wave>=1 && qx>=1 && qx<32 && qy>=1 && qy<10) s[5056+(5*9+qy-1)*32+qx-1]=current_psi_E_5;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<9) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_0=next_psi_0;
current_psi_E_1=next_psi_1;
current_psi_E_2=next_psi_2;
current_psi_E_3=next_psi_3;
current_psi_E_4=next_psi_4;
current_psi_E_5=next_psi_5;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-2,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<9) old_E_0=s[1728+(0*7+qy+(0)-2)*32+qx-2];
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<9) old_E_1=previous_y_3;
if(wave>=4 && qx>=2 && qx<31 && qy>=2 && qy<9) old_E_2=s[1728+(1*7+qy+(0)-2)*32+qx-2];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
float next_psi_5=0.f;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<8) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[1728+(1*7+qy+(1)-2)*32+qx-2]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[1728+(0*7+qy+(1)-2)*32+qx-2]);
const float stage4_0 = Profile(p,0,1,z,y,x);
const float stage4_1 = s[6784+(0*8+qy-1)*32+qx-1];
const float stage4_2 = Profile(p,0,0,z,y,x);
const float stage4_3 = derivative_0;
const float stage4_4 = __fmul_rn(stage4_2, stage4_3);
const float stage4_5 = __fmaf_rn(stage4_0, stage4_1, stage4_4);
const float stage4_6 = Profile(p,1,1,z,y,x);
const float stage4_7 = s[6784+(1*8+qy-1)*32+qx-1];
const float stage4_8 = Profile(p,1,0,z,y,x);
const float stage4_9 = derivative_1;
const float stage4_10 = __fmul_rn(stage4_8, stage4_9);
const float stage4_11 = __fmaf_rn(stage4_6, stage4_7, stage4_10);
const float stage4_12 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage4_13 = (-stage4_12);
const float stage4_14 = 1.0f;
const float stage4_15 = Profile(p,0,2,z,y,x);
const float stage4_16 = __fmaf_rn(stage4_3, stage4_15, stage4_5);
const float stage4_17 = __fmul_rn(stage4_14, stage4_16);
const float stage4_18 = -1.0f;
const float stage4_19 = Profile(p,1,2,z,y,x);
const float stage4_20 = __fmaf_rn(stage4_9, stage4_19, stage4_11);
const float stage4_21 = __fmul_rn(stage4_18, stage4_20);
const float stage4_22 = __fadd_rn(stage4_17, stage4_21);
const float stage4_23 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage4_24 = old_H_0;
const float stage4_25 = __fmul_rn(stage4_23, stage4_24);
const float stage4_26 = __fmaf_rn(stage4_13, stage4_22, stage4_25);
const float stage4_27 = Profile(p,2,1,z,y,x);
const float stage4_28 = s[6784+(2*8+qy-1)*32+qx-1];
const float stage4_29 = Profile(p,2,0,z,y,x);
const float stage4_30 = derivative_2;
const float stage4_31 = __fmul_rn(stage4_29, stage4_30);
const float stage4_32 = __fmaf_rn(stage4_27, stage4_28, stage4_31);
const float stage4_33 = Profile(p,3,1,z,y,x);
const float stage4_34 = s[6784+(3*8+qy-1)*32+qx-1];
const float stage4_35 = Profile(p,3,0,z,y,x);
const float stage4_36 = derivative_3;
const float stage4_37 = __fmul_rn(stage4_35, stage4_36);
const float stage4_38 = __fmaf_rn(stage4_33, stage4_34, stage4_37);
const float stage4_39 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage4_40 = (-stage4_39);
const float stage4_41 = Profile(p,2,2,z,y,x);
const float stage4_42 = __fmaf_rn(stage4_30, stage4_41, stage4_32);
const float stage4_43 = __fmul_rn(stage4_14, stage4_42);
const float stage4_44 = Profile(p,3,2,z,y,x);
const float stage4_45 = __fmaf_rn(stage4_36, stage4_44, stage4_38);
const float stage4_46 = __fmul_rn(stage4_18, stage4_45);
const float stage4_47 = __fadd_rn(stage4_43, stage4_46);
const float stage4_48 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage4_49 = old_H_1;
const float stage4_50 = __fmul_rn(stage4_48, stage4_49);
const float stage4_51 = __fmaf_rn(stage4_40, stage4_47, stage4_50);
const float stage4_52 = Profile(p,4,1,z,y,x);
const float stage4_53 = s[6784+(4*8+qy-1)*32+qx-1];
const float stage4_54 = Profile(p,4,0,z,y,x);
const float stage4_55 = derivative_4;
const float stage4_56 = __fmul_rn(stage4_54, stage4_55);
const float stage4_57 = __fmaf_rn(stage4_52, stage4_53, stage4_56);
const float stage4_58 = Profile(p,5,1,z,y,x);
const float stage4_59 = s[6784+(5*8+qy-1)*32+qx-1];
const float stage4_60 = Profile(p,5,0,z,y,x);
const float stage4_61 = derivative_5;
const float stage4_62 = __fmul_rn(stage4_60, stage4_61);
const float stage4_63 = __fmaf_rn(stage4_58, stage4_59, stage4_62);
const float stage4_64 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage4_65 = (-stage4_64);
const float stage4_66 = Profile(p,4,2,z,y,x);
const float stage4_67 = __fmaf_rn(stage4_55, stage4_66, stage4_57);
const float stage4_68 = __fmul_rn(stage4_14, stage4_67);
const float stage4_69 = Profile(p,5,2,z,y,x);
const float stage4_70 = __fmaf_rn(stage4_61, stage4_69, stage4_63);
const float stage4_71 = __fmul_rn(stage4_18, stage4_70);
const float stage4_72 = __fadd_rn(stage4_68, stage4_71);
const float stage4_73 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage4_74 = old_H_2;
const float stage4_75 = __fmul_rn(stage4_73, stage4_74);
const float stage4_76 = __fmaf_rn(stage4_65, stage4_72, stage4_75);
next_psi_0 = stage4_5;
next_psi_1 = stage4_11;
next_0 = stage4_26;
next_psi_2 = stage4_32;
next_psi_3 = stage4_38;
next_1 = stage4_51;
next_psi_4 = stage4_57;
next_psi_5 = stage4_63;
next_2 = stage4_76;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,2,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,2,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,2);
if(owned) {
Publish(observation,p.field.outputs[2],2,2,z,y,x,next_2);
}
}
}
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<9) previous_y_2=current_H_1;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<9) s[6784+(0*8+qy-1)*32+qx-1]=current_psi_H_0;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<9) s[6784+(1*8+qy-1)*32+qx-1]=current_psi_H_1;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<9) s[6784+(2*8+qy-1)*32+qx-1]=current_psi_H_2;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<9) s[6784+(3*8+qy-1)*32+qx-1]=current_psi_H_3;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<9) s[6784+(4*8+qy-1)*32+qx-1]=current_psi_H_4;
if(wave>=2 && qx>=1 && qx<31 && qy>=1 && qy<9) s[6784+(5*8+qy-1)*32+qx-1]=current_psi_H_5;
if(wave>4 && qx>=2 && qx<30 && qy>=2 && qy<8) old_H_0=s[2176+(0*6+qy+(0)-2)*32+qx-2];
if(wave>4 && qx>=2 && qx<30 && qy>=2 && qy<8) old_H_2=s[2176+(1*6+qy+(0)-2)*32+qx-2];
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<8) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_0=next_psi_0;
current_psi_H_1=next_psi_1;
current_psi_H_2=next_psi_2;
current_psi_H_3=next_psi_3;
current_psi_H_4=next_psi_4;
current_psi_H_5=next_psi_5;
}
__syncthreads();
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<8) s[2176+(0*6+qy+(0)-2)*32+qx-2]=current_H_0;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<8) s[2176+(1*6+qy+(0)-2)*32+qx-2]=current_H_2;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<9) s[1728+(0*7+qy+(0)-2)*32+qx-2]=current_E_0;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<9) s[1728+(1*7+qy+(0)-2)*32+qx-2]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-2,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_4;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
float next_psi_5=0.f;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<8) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[2176+(1*6+qy+(-1)-2)*32+qx-2]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[2176+(0*6+qy+(-1)-2)*32+qx-2]);
const float stage5_0 = Profile(p,0,1,z,y,x);
const float stage5_1 = s[8320+(0*7+qy-2)*32+qx-2];
const float stage5_2 = Profile(p,0,0,z,y,x);
const float stage5_3 = derivative_0;
const float stage5_4 = __fmul_rn(stage5_2, stage5_3);
const float stage5_5 = __fmaf_rn(stage5_0, stage5_1, stage5_4);
const float stage5_6 = Profile(p,1,1,z,y,x);
const float stage5_7 = s[8320+(1*7+qy-2)*32+qx-2];
const float stage5_8 = Profile(p,1,0,z,y,x);
const float stage5_9 = derivative_1;
const float stage5_10 = __fmul_rn(stage5_8, stage5_9);
const float stage5_11 = __fmaf_rn(stage5_6, stage5_7, stage5_10);
const float stage5_12 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage5_13 = 1.0f;
const float stage5_14 = Profile(p,0,2,z,y,x);
const float stage5_15 = __fmaf_rn(stage5_3, stage5_14, stage5_5);
const float stage5_16 = __fmul_rn(stage5_13, stage5_15);
const float stage5_17 = -1.0f;
const float stage5_18 = Profile(p,1,2,z,y,x);
const float stage5_19 = __fmaf_rn(stage5_9, stage5_18, stage5_11);
const float stage5_20 = __fmul_rn(stage5_17, stage5_19);
const float stage5_21 = __fadd_rn(stage5_16, stage5_20);
const float stage5_22 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage5_23 = old_E_0;
const float stage5_24 = __fmul_rn(stage5_22, stage5_23);
const float stage5_25 = __fmaf_rn(stage5_12, stage5_21, stage5_24);
const float stage5_26 = Profile(p,2,1,z,y,x);
const float stage5_27 = s[8320+(2*7+qy-2)*32+qx-2];
const float stage5_28 = Profile(p,2,0,z,y,x);
const float stage5_29 = derivative_2;
const float stage5_30 = __fmul_rn(stage5_28, stage5_29);
const float stage5_31 = __fmaf_rn(stage5_26, stage5_27, stage5_30);
const float stage5_32 = Profile(p,3,1,z,y,x);
const float stage5_33 = s[8320+(3*7+qy-2)*32+qx-2];
const float stage5_34 = Profile(p,3,0,z,y,x);
const float stage5_35 = derivative_3;
const float stage5_36 = __fmul_rn(stage5_34, stage5_35);
const float stage5_37 = __fmaf_rn(stage5_32, stage5_33, stage5_36);
const float stage5_38 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage5_39 = Profile(p,2,2,z,y,x);
const float stage5_40 = __fmaf_rn(stage5_29, stage5_39, stage5_31);
const float stage5_41 = __fmul_rn(stage5_13, stage5_40);
const float stage5_42 = Profile(p,3,2,z,y,x);
const float stage5_43 = __fmaf_rn(stage5_35, stage5_42, stage5_37);
const float stage5_44 = __fmul_rn(stage5_17, stage5_43);
const float stage5_45 = __fadd_rn(stage5_41, stage5_44);
const float stage5_46 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage5_47 = old_E_1;
const float stage5_48 = __fmul_rn(stage5_46, stage5_47);
const float stage5_49 = __fmaf_rn(stage5_38, stage5_45, stage5_48);
const float stage5_50 = Profile(p,4,1,z,y,x);
const float stage5_51 = s[8320+(4*7+qy-2)*32+qx-2];
const float stage5_52 = Profile(p,4,0,z,y,x);
const float stage5_53 = derivative_4;
const float stage5_54 = __fmul_rn(stage5_52, stage5_53);
const float stage5_55 = __fmaf_rn(stage5_50, stage5_51, stage5_54);
const float stage5_56 = Profile(p,5,1,z,y,x);
const float stage5_57 = s[8320+(5*7+qy-2)*32+qx-2];
const float stage5_58 = Profile(p,5,0,z,y,x);
const float stage5_59 = derivative_5;
const float stage5_60 = __fmul_rn(stage5_58, stage5_59);
const float stage5_61 = __fmaf_rn(stage5_56, stage5_57, stage5_60);
const float stage5_62 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage5_63 = Profile(p,4,2,z,y,x);
const float stage5_64 = __fmaf_rn(stage5_53, stage5_63, stage5_55);
const float stage5_65 = __fmul_rn(stage5_13, stage5_64);
const float stage5_66 = Profile(p,5,2,z,y,x);
const float stage5_67 = __fmaf_rn(stage5_59, stage5_66, stage5_61);
const float stage5_68 = __fmul_rn(stage5_17, stage5_67);
const float stage5_69 = __fadd_rn(stage5_65, stage5_68);
const float stage5_70 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage5_71 = old_E_2;
const float stage5_72 = __fmul_rn(stage5_70, stage5_71);
const float stage5_73 = __fmaf_rn(stage5_62, stage5_69, stage5_72);
next_psi_0 = stage5_5;
next_psi_1 = stage5_11;
next_0 = stage5_25;
next_psi_2 = stage5_31;
next_psi_3 = stage5_37;
next_1 = stage5_49;
next_psi_4 = stage5_55;
next_psi_5 = stage5_61;
next_2 = stage5_73;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,2,z,y,x,next_0);
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,2);
if(owned) {
Publish(observation,p.field.outputs[1],4,2,z,y,x,next_1);
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,2,z,y,x,next_2);
}
}
}
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<9) previous_y_3=current_E_1;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<9) s[8320+(0*7+qy-2)*32+qx-2]=current_psi_E_0;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<9) s[8320+(1*7+qy-2)*32+qx-2]=current_psi_E_1;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<9) s[8320+(2*7+qy-2)*32+qx-2]=current_psi_E_2;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<9) s[8320+(3*7+qy-2)*32+qx-2]=current_psi_E_3;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<9) s[8320+(4*7+qy-2)*32+qx-2]=current_psi_E_4;
if(wave>=3 && qx>=2 && qx<31 && qy>=2 && qy<9) s[8320+(5*7+qy-2)*32+qx-2]=current_psi_E_5;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<8) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_0=next_psi_0;
current_psi_E_1=next_psi_1;
current_psi_E_2=next_psi_2;
current_psi_E_3=next_psi_3;
current_psi_E_4=next_psi_4;
current_psi_E_5=next_psi_5;
}
}
{
const auto &p=h; const auto &sources=hs;
const int z=oz+wave-4-3,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<8) old_E_0=s[2560+(0*5+qy+(0)-3)*32+qx-3];
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<8) old_E_1=previous_y_5;
if(wave>=6 && qx>=3 && qx<30 && qy>=3 && qy<8) old_E_2=s[2560+(1*5+qy+(0)-3)*32+qx-3];
const float neighbor_0=__shfl_down_sync(0xffffffffu,old_E_0,1);
const float neighbor_1=__shfl_down_sync(0xffffffffu,old_E_1,1);
const float neighbor_2=__shfl_down_sync(0xffffffffu,old_E_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
float next_psi_5=0.f;
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<7) {
const float derivative_0=Difference(p,0,2,1,z,y,x,old_E_2,s[2560+(1*5+qy+(1)-3)*32+qx-3]);
const float derivative_1=Difference(p,0,1,0,z,y,x,old_E_1,current_E_1);
const float derivative_2=Difference(p,0,0,0,z,y,x,old_E_0,current_E_0);
const float derivative_3=Difference(p,0,2,2,z,y,x,old_E_2,neighbor_2);
const float derivative_4=Difference(p,0,1,2,z,y,x,old_E_1,neighbor_1);
const float derivative_5=Difference(p,0,0,1,z,y,x,old_E_0,s[2560+(0*5+qy+(1)-3)*32+qx-3]);
const float stage6_0 = Profile(p,0,1,z,y,x);
const float stage6_1 = s[9664+(0*6+qy-2)*32+qx-2];
const float stage6_2 = Profile(p,0,0,z,y,x);
const float stage6_3 = derivative_0;
const float stage6_4 = __fmul_rn(stage6_2, stage6_3);
const float stage6_5 = __fmaf_rn(stage6_0, stage6_1, stage6_4);
const float stage6_6 = Profile(p,1,1,z,y,x);
const float stage6_7 = s[9664+(1*6+qy-2)*32+qx-2];
const float stage6_8 = Profile(p,1,0,z,y,x);
const float stage6_9 = derivative_1;
const float stage6_10 = __fmul_rn(stage6_8, stage6_9);
const float stage6_11 = __fmaf_rn(stage6_6, stage6_7, stage6_10);
const float stage6_12 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[9].data)) : 0.f);
const float stage6_13 = (-stage6_12);
const float stage6_14 = 1.0f;
const float stage6_15 = Profile(p,0,2,z,y,x);
const float stage6_16 = __fmaf_rn(stage6_3, stage6_15, stage6_5);
const float stage6_17 = __fmul_rn(stage6_14, stage6_16);
const float stage6_18 = -1.0f;
const float stage6_19 = Profile(p,1,2,z,y,x);
const float stage6_20 = __fmaf_rn(stage6_9, stage6_19, stage6_11);
const float stage6_21 = __fmul_rn(stage6_18, stage6_20);
const float stage6_22 = __fadd_rn(stage6_17, stage6_21);
const float stage6_23 = (BufferContains(p.field.outputs[0],z,y,x) ? (*static_cast<const float *>(p.field.inputs[6].data)) : 0.f);
const float stage6_24 = old_H_0;
const float stage6_25 = __fmul_rn(stage6_23, stage6_24);
const float stage6_26 = __fmaf_rn(stage6_13, stage6_22, stage6_25);
const float stage6_27 = Profile(p,2,1,z,y,x);
const float stage6_28 = s[9664+(2*6+qy-2)*32+qx-2];
const float stage6_29 = Profile(p,2,0,z,y,x);
const float stage6_30 = derivative_2;
const float stage6_31 = __fmul_rn(stage6_29, stage6_30);
const float stage6_32 = __fmaf_rn(stage6_27, stage6_28, stage6_31);
const float stage6_33 = Profile(p,3,1,z,y,x);
const float stage6_34 = s[9664+(3*6+qy-2)*32+qx-2];
const float stage6_35 = Profile(p,3,0,z,y,x);
const float stage6_36 = derivative_3;
const float stage6_37 = __fmul_rn(stage6_35, stage6_36);
const float stage6_38 = __fmaf_rn(stage6_33, stage6_34, stage6_37);
const float stage6_39 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[10].data)) : 0.f);
const float stage6_40 = (-stage6_39);
const float stage6_41 = Profile(p,2,2,z,y,x);
const float stage6_42 = __fmaf_rn(stage6_30, stage6_41, stage6_32);
const float stage6_43 = __fmul_rn(stage6_14, stage6_42);
const float stage6_44 = Profile(p,3,2,z,y,x);
const float stage6_45 = __fmaf_rn(stage6_36, stage6_44, stage6_38);
const float stage6_46 = __fmul_rn(stage6_18, stage6_45);
const float stage6_47 = __fadd_rn(stage6_43, stage6_46);
const float stage6_48 = (BufferContains(p.field.outputs[1],z,y,x) ? (*static_cast<const float *>(p.field.inputs[7].data)) : 0.f);
const float stage6_49 = old_H_1;
const float stage6_50 = __fmul_rn(stage6_48, stage6_49);
const float stage6_51 = __fmaf_rn(stage6_40, stage6_47, stage6_50);
const float stage6_52 = Profile(p,4,1,z,y,x);
const float stage6_53 = s[9664+(4*6+qy-2)*32+qx-2];
const float stage6_54 = Profile(p,4,0,z,y,x);
const float stage6_55 = derivative_4;
const float stage6_56 = __fmul_rn(stage6_54, stage6_55);
const float stage6_57 = __fmaf_rn(stage6_52, stage6_53, stage6_56);
const float stage6_58 = Profile(p,5,1,z,y,x);
const float stage6_59 = s[9664+(5*6+qy-2)*32+qx-2];
const float stage6_60 = Profile(p,5,0,z,y,x);
const float stage6_61 = derivative_5;
const float stage6_62 = __fmul_rn(stage6_60, stage6_61);
const float stage6_63 = __fmaf_rn(stage6_58, stage6_59, stage6_62);
const float stage6_64 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[11].data)) : 0.f);
const float stage6_65 = (-stage6_64);
const float stage6_66 = Profile(p,4,2,z,y,x);
const float stage6_67 = __fmaf_rn(stage6_55, stage6_66, stage6_57);
const float stage6_68 = __fmul_rn(stage6_14, stage6_67);
const float stage6_69 = Profile(p,5,2,z,y,x);
const float stage6_70 = __fmaf_rn(stage6_61, stage6_69, stage6_63);
const float stage6_71 = __fmul_rn(stage6_18, stage6_70);
const float stage6_72 = __fadd_rn(stage6_68, stage6_71);
const float stage6_73 = (BufferContains(p.field.outputs[2],z,y,x) ? (*static_cast<const float *>(p.field.inputs[8].data)) : 0.f);
const float stage6_74 = old_H_2;
const float stage6_75 = __fmul_rn(stage6_73, stage6_74);
const float stage6_76 = __fmaf_rn(stage6_65, stage6_72, stage6_75);
next_psi_0 = stage6_5;
next_psi_1 = stage6_11;
next_0 = stage6_26;
next_psi_2 = stage6_32;
next_psi_3 = stage6_38;
next_1 = stage6_51;
next_psi_4 = stage6_57;
next_psi_5 = stage6_63;
next_2 = stage6_76;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],0,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],0,3,z,y,x,next_0);
static_cast<float *>(final.fields[0].data)[BeamzOffset3D<false>(final.fields[0],z,y,x)]=next_0;
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],0,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(owned) {
Publish(observation,p.field.outputs[1],1,3,z,y,x,next_1);
static_cast<float *>(final.fields[1].data)[BeamzOffset3D<false>(final.fields[1],z,y,x)]=next_1;
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],0,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(!constrained_2 && active_sources[2]) next_2=AddFusedHSource<2>(next_2,sources,z,y,x,3);
if(owned) {
Publish(observation,p.field.outputs[2],2,3,z,y,x,next_2);
static_cast<float *>(final.fields[2].data)[BeamzOffset3D<false>(final.fields[2],z,y,x)]=next_2;
}
}
if(owned) WritePsi<false>(p,0,z,y,x,next_psi_0);
if(owned) WritePsi<false>(p,1,z,y,x,next_psi_1);
if(owned) WritePsi<false>(p,2,z,y,x,next_psi_2);
if(owned) WritePsi<false>(p,3,z,y,x,next_psi_3);
if(owned) WritePsi<false>(p,4,z,y,x,next_psi_4);
if(owned) WritePsi<false>(p,5,z,y,x,next_psi_5);
}
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<8) previous_y_4=current_H_1;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<8) s[9664+(0*6+qy-2)*32+qx-2]=current_psi_H_0;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<8) s[9664+(1*6+qy-2)*32+qx-2]=current_psi_H_1;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<8) s[9664+(2*6+qy-2)*32+qx-2]=current_psi_H_2;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<8) s[9664+(3*6+qy-2)*32+qx-2]=current_psi_H_3;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<8) s[9664+(4*6+qy-2)*32+qx-2]=current_psi_H_4;
if(wave>=4 && qx>=2 && qx<30 && qy>=2 && qy<8) s[9664+(5*6+qy-2)*32+qx-2]=current_psi_H_5;
if(wave>6 && qx>=3 && qx<29 && qy>=3 && qy<7) old_H_0=s[2880+(0*4+qy+(0)-3)*32+qx-3];
if(wave>6 && qx>=3 && qx<29 && qy>=3 && qy<7) old_H_2=s[2880+(1*4+qy+(0)-3)*32+qx-3];
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<7) {
current_H_0=next_0;
current_H_1=next_1;
current_H_2=next_2;
current_psi_H_0=next_psi_0;
current_psi_H_1=next_psi_1;
current_psi_H_2=next_psi_2;
current_psi_H_3=next_psi_3;
current_psi_H_4=next_psi_4;
current_psi_H_5=next_psi_5;
}
__syncthreads();
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<7) s[2880+(0*4+qy+(0)-3)*32+qx-3]=current_H_0;
if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<7) s[2880+(1*4+qy+(0)-3)*32+qx-3]=current_H_2;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<8) s[2560+(0*5+qy+(0)-3)*32+qx-3]=current_E_0;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<8) s[2560+(1*5+qy+(0)-3)*32+qx-3]=current_E_2;
__syncthreads();
}
{
const auto &p=e; const auto &sources=es;
const int z=oz+wave-4-3,x=ox-4+qx,y=oy-4+qy;
const bool owned=x>=ox && x<end_x && y>=oy && y<end_y && z>=oz && z<end_z;
old_H_1=previous_y_6;
const float neighbor_0=__shfl_up_sync(0xffffffffu,current_H_0,1);
const float neighbor_1=__shfl_up_sync(0xffffffffu,current_H_1,1);
const float neighbor_2=__shfl_up_sync(0xffffffffu,current_H_2,1);
float next_0=0.f;
float next_1=0.f;
float next_2=0.f;
float next_psi_0=0.f;
float next_psi_1=0.f;
float next_psi_2=0.f;
float next_psi_3=0.f;
float next_psi_4=0.f;
float next_psi_5=0.f;
if(wave>=7 && qx>=4 && qx<29 && qy>=4 && qy<7) {
const float derivative_0=Difference(p,1,2,1,z,y,x,current_H_2,s[2880+(1*4+qy+(-1)-3)*32+qx-3]);
const float derivative_1=Difference(p,1,1,0,z,y,x,current_H_1,old_H_1);
const float derivative_2=Difference(p,1,0,0,z,y,x,current_H_0,old_H_0);
const float derivative_3=Difference(p,1,2,2,z,y,x,current_H_2,neighbor_2);
const float derivative_4=Difference(p,1,1,2,z,y,x,current_H_1,neighbor_1);
const float derivative_5=Difference(p,1,0,1,z,y,x,current_H_0,s[2880+(0*4+qy+(-1)-3)*32+qx-3]);
const float stage7_0 = Profile(p,0,1,z,y,x);
const float stage7_1 = s[10816+(0*5+qy-3)*32+qx-3];
const float stage7_2 = Profile(p,0,0,z,y,x);
const float stage7_3 = derivative_0;
const float stage7_4 = __fmul_rn(stage7_2, stage7_3);
const float stage7_5 = __fmaf_rn(stage7_0, stage7_1, stage7_4);
const float stage7_6 = Profile(p,1,1,z,y,x);
const float stage7_7 = s[10816+(1*5+qy-3)*32+qx-3];
const float stage7_8 = Profile(p,1,0,z,y,x);
const float stage7_9 = derivative_1;
const float stage7_10 = __fmul_rn(stage7_8, stage7_9);
const float stage7_11 = __fmaf_rn(stage7_6, stage7_7, stage7_10);
const float stage7_12 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[6].data),static_cast<int>(p.field.inputs[6].dims[0]),static_cast<const uint32_t *>(p.field.inputs[9].data),((z*static_cast<int>(p.field.outputs[0].dims[1])+y)*static_cast<int>(p.field.outputs[0].dims[2])+x)) : Read(p.field.inputs[9],z,y,x)) : 0.f);
const float stage7_13 = 1.0f;
const float stage7_14 = Profile(p,0,2,z,y,x);
const float stage7_15 = __fmaf_rn(stage7_3, stage7_14, stage7_5);
const float stage7_16 = __fmul_rn(stage7_13, stage7_15);
const float stage7_17 = -1.0f;
const float stage7_18 = Profile(p,1,2,z,y,x);
const float stage7_19 = __fmaf_rn(stage7_9, stage7_18, stage7_11);
const float stage7_20 = __fmul_rn(stage7_17, stage7_19);
const float stage7_21 = __fadd_rn(stage7_16, stage7_20);
const float stage7_22 = (BufferContains(p.field.outputs[0],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[6],z,y,x)) : 0.f);
const float stage7_23 = old_E_0;
const float stage7_24 = __fmul_rn(stage7_22, stage7_23);
const float stage7_25 = __fmaf_rn(stage7_12, stage7_21, stage7_24);
const float stage7_26 = Profile(p,2,1,z,y,x);
const float stage7_27 = s[10816+(2*5+qy-3)*32+qx-3];
const float stage7_28 = Profile(p,2,0,z,y,x);
const float stage7_29 = derivative_2;
const float stage7_30 = __fmul_rn(stage7_28, stage7_29);
const float stage7_31 = __fmaf_rn(stage7_26, stage7_27, stage7_30);
const float stage7_32 = Profile(p,3,1,z,y,x);
const float stage7_33 = s[10816+(3*5+qy-3)*32+qx-3];
const float stage7_34 = Profile(p,3,0,z,y,x);
const float stage7_35 = derivative_3;
const float stage7_36 = __fmul_rn(stage7_34, stage7_35);
const float stage7_37 = __fmaf_rn(stage7_32, stage7_33, stage7_36);
const float stage7_38 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[7].data),static_cast<int>(p.field.inputs[7].dims[0]),static_cast<const uint32_t *>(p.field.inputs[10].data),((z*static_cast<int>(p.field.outputs[1].dims[1])+y)*static_cast<int>(p.field.outputs[1].dims[2])+x)) : Read(p.field.inputs[10],z,y,x)) : 0.f);
const float stage7_39 = Profile(p,2,2,z,y,x);
const float stage7_40 = __fmaf_rn(stage7_29, stage7_39, stage7_31);
const float stage7_41 = __fmul_rn(stage7_13, stage7_40);
const float stage7_42 = Profile(p,3,2,z,y,x);
const float stage7_43 = __fmaf_rn(stage7_35, stage7_42, stage7_37);
const float stage7_44 = __fmul_rn(stage7_17, stage7_43);
const float stage7_45 = __fadd_rn(stage7_41, stage7_44);
const float stage7_46 = (BufferContains(p.field.outputs[1],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[7],z,y,x)) : 0.f);
const float stage7_47 = old_E_1;
const float stage7_48 = __fmul_rn(stage7_46, stage7_47);
const float stage7_49 = __fmaf_rn(stage7_38, stage7_45, stage7_48);
const float stage7_50 = Profile(p,4,1,z,y,x);
const float stage7_51 = s[10816+(4*5+qy-3)*32+qx-3];
const float stage7_52 = Profile(p,4,0,z,y,x);
const float stage7_53 = derivative_4;
const float stage7_54 = __fmul_rn(stage7_52, stage7_53);
const float stage7_55 = __fmaf_rn(stage7_50, stage7_51, stage7_54);
const float stage7_56 = Profile(p,5,1,z,y,x);
const float stage7_57 = s[10816+(5*5+qy-3)*32+qx-3];
const float stage7_58 = Profile(p,5,0,z,y,x);
const float stage7_59 = derivative_5;
const float stage7_60 = __fmul_rn(stage7_58, stage7_59);
const float stage7_61 = __fmaf_rn(stage7_56, stage7_57, stage7_60);
const float stage7_62 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? beamz::cuda::yee::PackedMaterialSource(static_cast<const float *>(p.field.inputs[8].data),static_cast<int>(p.field.inputs[8].dims[0]),static_cast<const uint32_t *>(p.field.inputs[11].data),((z*static_cast<int>(p.field.outputs[2].dims[1])+y)*static_cast<int>(p.field.outputs[2].dims[2])+x)) : Read(p.field.inputs[11],z,y,x)) : 0.f);
const float stage7_63 = Profile(p,4,2,z,y,x);
const float stage7_64 = __fmaf_rn(stage7_53, stage7_63, stage7_55);
const float stage7_65 = __fmul_rn(stage7_13, stage7_64);
const float stage7_66 = Profile(p,5,2,z,y,x);
const float stage7_67 = __fmaf_rn(stage7_59, stage7_66, stage7_61);
const float stage7_68 = __fmul_rn(stage7_17, stage7_67);
const float stage7_69 = __fadd_rn(stage7_65, stage7_68);
const float stage7_70 = (BufferContains(p.field.outputs[2],z,y,x) ? (Packed ? 1.f : Read(p.field.inputs[8],z,y,x)) : 0.f);
const float stage7_71 = old_E_2;
const float stage7_72 = __fmul_rn(stage7_70, stage7_71);
const float stage7_73 = __fmaf_rn(stage7_62, stage7_69, stage7_72);
next_psi_0 = stage7_5;
next_psi_1 = stage7_11;
next_0 = stage7_25;
next_psi_2 = stage7_31;
next_psi_3 = stage7_37;
next_1 = stage7_49;
next_psi_4 = stage7_55;
next_psi_5 = stage7_61;
next_2 = stage7_73;
if(BufferContains(p.field.outputs[0],z,y,x)) {
const bool constrained_0=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[0],1,0,p.metallic_edges,z,y,x); if(constrained_0) next_0=0.f;
if(owned) {
Publish(observation,p.field.outputs[0],3,3,z,y,x,next_0);
static_cast<float *>(final.fields[3].data)[BeamzOffset3D<false>(final.fields[3],z,y,x)]=next_0;
}
}
if(BufferContains(p.field.outputs[1],z,y,x)) {
const bool constrained_1=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[1],1,1,p.metallic_edges,z,y,x); if(constrained_1) next_1=0.f;
if(!constrained_1 && active_sources[4]) next_1=AddFusedHSource<1>(next_1,sources,z,y,x,3);
if(owned) {
Publish(observation,p.field.outputs[1],4,3,z,y,x,next_1);
static_cast<float *>(final.fields[4].data)[BeamzOffset3D<false>(final.fields[4],z,y,x)]=next_1;
}
}
if(BufferContains(p.field.outputs[2],z,y,x)) {
const bool constrained_2=p.metallic_edges && beamz::cuda::yee::PecConstrained(p.field.outputs[2],1,2,p.metallic_edges,z,y,x); if(constrained_2) next_2=0.f;
if(owned) {
Publish(observation,p.field.outputs[2],5,3,z,y,x,next_2);
static_cast<float *>(final.fields[5].data)[BeamzOffset3D<false>(final.fields[5],z,y,x)]=next_2;
}
}
if(owned) WritePsi<false>(p,0,z,y,x,next_psi_0);
if(owned) WritePsi<false>(p,1,z,y,x,next_psi_1);
if(owned) WritePsi<false>(p,2,z,y,x,next_psi_2);
if(owned) WritePsi<false>(p,3,z,y,x,next_psi_3);
if(owned) WritePsi<false>(p,4,z,y,x,next_psi_4);
if(owned) WritePsi<false>(p,5,z,y,x,next_psi_5);
}
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<8) previous_y_5=current_E_1;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<8) s[10816+(0*5+qy-3)*32+qx-3]=current_psi_E_0;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<8) s[10816+(1*5+qy-3)*32+qx-3]=current_psi_E_1;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<8) s[10816+(2*5+qy-3)*32+qx-3]=current_psi_E_2;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<8) s[10816+(3*5+qy-3)*32+qx-3]=current_psi_E_3;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<8) s[10816+(4*5+qy-3)*32+qx-3]=current_psi_E_4;
if(wave>=5 && qx>=3 && qx<30 && qy>=3 && qy<8) s[10816+(5*5+qy-3)*32+qx-3]=current_psi_E_5;
if(wave>=7 && qx>=4 && qx<29 && qy>=4 && qy<7) {
current_E_0=next_0;
current_E_1=next_1;
current_E_2=next_2;
current_psi_E_0=next_psi_0;
current_psi_E_1=next_psi_1;
current_psi_E_2=next_psi_2;
current_psi_E_3=next_psi_3;
current_psi_E_4=next_psi_4;
current_psi_E_5=next_psi_5;
}
}
    if(wave>=6 && qx>=3 && qx<29 && qy>=3 && qy<7) previous_y_6=current_H_1;
  }
}


}}} // namespace beamz::cuda::deep


int BeamzDeepD4AdmissionKind() { return 1; }
int BeamzEnqueueDeepWarpD4(void *opaque_stream,const BeamzProgramLaunch &program,
    const BeamzBuffer *final_fields,int step,const beamz::cuda::deep::Observation &observation) {
  const auto stream=static_cast<cudaStream_t>(opaque_stream);
  const auto &h=program.h_ab,&e=program.e_ab;
  if(!final_fields || !program.source_groups || program.source_group_count!=9 ||
     h.uniform_cpml_thickness!=12 || e.uniform_cpml_thickness!=12 ||
     h.nterms!=6 || e.nterms!=6 || h.metric_kind!=0 || e.metric_kind!=0 ||
     !HasPackedLosslessMaterial(e))return cudaErrorInvalidValue;
  for(int c=0;c<3;++c) {
    if(program.source_groups[c].coefficients.dims[0])return cudaErrorInvalidValue;
    if(h.inputs[6+c].rank!=0 || h.inputs[9+c].rank!=0)return cudaErrorInvalidValue;
  }
  for(int c=0;c<6;++c) {
    if(final_fields[c].element_type!=kBeamzF32)return cudaErrorInvalidValue;
    for(int q=0;q<3;++q)
      if(final_fields[c].data==h.inputs[q].data || final_fields[c].data==e.inputs[q].data)return cudaErrorInvalidValue;
    if(h.inputs[31+c].element_type!=kBeamzF32 || e.inputs[31+c].element_type!=kBeamzF32 ||
       h.outputs[3+c].element_type!=kBeamzF32 || e.outputs[3+c].element_type!=kBeamzF32 ||
       h.inputs[31+c].data==h.outputs[3+c].data || e.inputs[31+c].data==e.outputs[3+c].data)return cudaErrorInvalidValue;
  }
  PairFieldOutputs final{}; FusedHSources hs{},es{};hs.step=es.step=step;
  PairCpmlGrid grid{};
  for(int c=0;c<6;++c){final.fields[c]=final_fields[c];for(int a=0;a<3;++a)
    grid.extent[a]=std::max(grid.extent[a],static_cast<int>(final_fields[c].dims[a]));}
  for(int c=0;c<3;++c){hs.groups[c]=program.source_groups[3+c];es.groups[c]=program.source_groups[6+c];}
  for(int a=0;a<3;++a){int limit=static_cast<int>(h.outputs[0].dims[a]);
    for(int c=1;c<3;++c)limit=std::min(limit,static_cast<int>(h.outputs[c].dims[a]));
    if(limit<32)return cudaErrorInvalidValue;
    grid.field_high[a]=limit;grid.high[a]=16;grid.boundary[a]=limit-16;}
  const auto hp=MakePairCpmlPhase(h),ep=MakePairCpmlPhase(e);
  using namespace beamz::cuda::deep;

  int source_signature=0;
  for(int c=0;c<6;++c)if(program.source_groups[3+c].coefficients.dims[0]>0)source_signature|=1<<c;
  if(source_signature!=20)return cudaErrorInvalidValue;
  for(int c=0;c<3;++c) {
    if(h.inputs[6+c].element_type!=kBeamzF32 || h.inputs[9+c].element_type!=kBeamzF32 ||
       e.inputs[6+c].element_type!=kBeamzF32 || e.inputs[6+c].rank!=1 ||
       e.inputs[6+c].dims[0]<1 || e.inputs[6+c].dims[0]>2147483647LL ||
       e.inputs[9+c].element_type!=kBeamzS32 || e.inputs[9+c].rank!=1)return cudaErrorInvalidValue;
    int64_t volume=1;
    for(int a=0;a<3;++a) {
      const auto n=e.outputs[c].dims[a];
      if(n<1 || n>2147483647LL/volume)return cudaErrorInvalidValue;
      volume*=n;
    }
    if(e.inputs[9+c].dims[0]<(volume+3)/4)return cudaErrorInvalidValue;
  }
{ const int sizes[3]={64,8,25};int counts[3];bool active=true;
for(int a=0;a<3;++a){counts[a]=(0&(1<<a))?(grid.high[a]+sizes[a]-1)/sizes[a]+(grid.extent[a]-grid.boundary[a]+sizes[a]-1)/sizes[a]:(grid.boundary[a]-grid.high[a]+sizes[a]-1)/sizes[a];active &= counts[a]>0;}
if(active){DeepWarpD4M0F32ScalarHBytePackedSource20<true><<<dim3(counts[2],counts[1],counts[0]),DeepWarpD4M0F32ScalarHBytePackedSource20Threads,DeepWarpD4M0F32ScalarHBytePackedSource20SharedBytes,stream>>>(hp,ep,final,hs,es,observation,grid);
const auto error=cudaPeekAtLastError();if(error!=cudaSuccess)return error;}}
{ const int sizes[3]={64,8,25};int counts[3];bool active=true;
for(int a=0;a<3;++a){counts[a]=(1&(1<<a))?(grid.high[a]+sizes[a]-1)/sizes[a]+(grid.extent[a]-grid.boundary[a]+sizes[a]-1)/sizes[a]:(grid.boundary[a]-grid.high[a]+sizes[a]-1)/sizes[a];active &= counts[a]>0;}
if(active){DeepWarpD4M1F32ScalarHBytePackedSource20<true><<<dim3(counts[2],counts[1],counts[0]),DeepWarpD4M1F32ScalarHBytePackedSource20Threads,DeepWarpD4M1F32ScalarHBytePackedSource20SharedBytes,stream>>>(hp,ep,final,hs,es,observation,grid);
const auto error=cudaPeekAtLastError();if(error!=cudaSuccess)return error;}}
{ const int sizes[3]={64,8,25};int counts[3];bool active=true;
for(int a=0;a<3;++a){counts[a]=(2&(1<<a))?(grid.high[a]+sizes[a]-1)/sizes[a]+(grid.extent[a]-grid.boundary[a]+sizes[a]-1)/sizes[a]:(grid.boundary[a]-grid.high[a]+sizes[a]-1)/sizes[a];active &= counts[a]>0;}
if(active){DeepWarpD4M2F32ScalarHBytePackedSource20<true><<<dim3(counts[2],counts[1],counts[0]),DeepWarpD4M2F32ScalarHBytePackedSource20Threads,DeepWarpD4M2F32ScalarHBytePackedSource20SharedBytes,stream>>>(hp,ep,final,hs,es,observation,grid);
const auto error=cudaPeekAtLastError();if(error!=cudaSuccess)return error;}}
{ const int sizes[3]={64,4,25};int counts[3];bool active=true;
for(int a=0;a<3;++a){counts[a]=(3&(1<<a))?(grid.high[a]+sizes[a]-1)/sizes[a]+(grid.extent[a]-grid.boundary[a]+sizes[a]-1)/sizes[a]:(grid.boundary[a]-grid.high[a]+sizes[a]-1)/sizes[a];active &= counts[a]>0;}
if(active){DeepWarpD4M3F32ScalarHBytePackedSource20<true><<<dim3(counts[2],counts[1],counts[0]),DeepWarpD4M3F32ScalarHBytePackedSource20Threads,DeepWarpD4M3F32ScalarHBytePackedSource20SharedBytes,stream>>>(hp,ep,final,hs,es,observation,grid);
const auto error=cudaPeekAtLastError();if(error!=cudaSuccess)return error;}}
{ const int sizes[3]={64,8,25};int counts[3];bool active=true;
for(int a=0;a<3;++a){counts[a]=(4&(1<<a))?(grid.high[a]+sizes[a]-1)/sizes[a]+(grid.extent[a]-grid.boundary[a]+sizes[a]-1)/sizes[a]:(grid.boundary[a]-grid.high[a]+sizes[a]-1)/sizes[a];active &= counts[a]>0;}
if(active){DeepWarpD4M4F32ScalarHBytePackedSource20<true><<<dim3(counts[2],counts[1],counts[0]),DeepWarpD4M4F32ScalarHBytePackedSource20Threads,DeepWarpD4M4F32ScalarHBytePackedSource20SharedBytes,stream>>>(hp,ep,final,hs,es,observation,grid);
const auto error=cudaPeekAtLastError();if(error!=cudaSuccess)return error;}}
{ const int sizes[3]={64,4,25};int counts[3];bool active=true;
for(int a=0;a<3;++a){counts[a]=(5&(1<<a))?(grid.high[a]+sizes[a]-1)/sizes[a]+(grid.extent[a]-grid.boundary[a]+sizes[a]-1)/sizes[a]:(grid.boundary[a]-grid.high[a]+sizes[a]-1)/sizes[a];active &= counts[a]>0;}
if(active){DeepWarpD4M5F32ScalarHBytePackedSource20<true><<<dim3(counts[2],counts[1],counts[0]),DeepWarpD4M5F32ScalarHBytePackedSource20Threads,DeepWarpD4M5F32ScalarHBytePackedSource20SharedBytes,stream>>>(hp,ep,final,hs,es,observation,grid);
const auto error=cudaPeekAtLastError();if(error!=cudaSuccess)return error;}}
{ const int sizes[3]={64,4,25};int counts[3];bool active=true;
for(int a=0;a<3;++a){counts[a]=(6&(1<<a))?(grid.high[a]+sizes[a]-1)/sizes[a]+(grid.extent[a]-grid.boundary[a]+sizes[a]-1)/sizes[a]:(grid.boundary[a]-grid.high[a]+sizes[a]-1)/sizes[a];active &= counts[a]>0;}
if(active){DeepWarpD4M6F32ScalarHBytePackedSource20<true><<<dim3(counts[2],counts[1],counts[0]),DeepWarpD4M6F32ScalarHBytePackedSource20Threads,DeepWarpD4M6F32ScalarHBytePackedSource20SharedBytes,stream>>>(hp,ep,final,hs,es,observation,grid);
const auto error=cudaPeekAtLastError();if(error!=cudaSuccess)return error;}}
{ const int sizes[3]={64,3,25};int counts[3];bool active=true;
for(int a=0;a<3;++a){counts[a]=(7&(1<<a))?(grid.high[a]+sizes[a]-1)/sizes[a]+(grid.extent[a]-grid.boundary[a]+sizes[a]-1)/sizes[a]:(grid.boundary[a]-grid.high[a]+sizes[a]-1)/sizes[a];active &= counts[a]>0;}
if(active){DeepWarpD4M7F32ScalarHBytePackedSource20<true><<<dim3(counts[2],counts[1],counts[0]),DeepWarpD4M7F32ScalarHBytePackedSource20Threads,DeepWarpD4M7F32ScalarHBytePackedSource20SharedBytes,stream>>>(hp,ep,final,hs,es,observation,grid);
const auto error=cudaPeekAtLastError();if(error!=cudaSuccess)return error;}}
  return cudaSuccess;
}
int BeamzEnqueueDeepWarpDft4(void *opaque_stream,const BeamzProgramLaunch &program,
    int step,const beamz::cuda::deep::Observation &observation,const BeamzBuffer &remapped_indices) {
  if(!program.monitors || !program.monitors->monitor_count)return cudaSuccess;
  if(!observation.values || observation.key_count<1 || remapped_indices.rank!=4 ||
     remapped_indices.element_type!=kBeamzS32)return cudaErrorInvalidValue;
  auto h=program.h_ab,e=program.e_ab;auto monitors=*program.monitors;
  monitors.indices=remapped_indices;
  BeamzBuffer field{};field.rank=3;field.element_type=kBeamzF32;
  field.dims[0]=field.dims[1]=1;field.dims[2]=observation.key_count;
  field.row_stride=field.plane_stride=observation.key_count;
  for(int substep=0;substep<4;++substep){
    field.data=observation.values+substep*observation.key_count;
    for(int c=0;c<3;++c)h.outputs[c]=e.outputs[c]=field;
    const auto error=BeamzEnqueueDftGroups(static_cast<cudaStream_t>(opaque_stream),h,e,monitors,step+substep);
    if(error!=cudaSuccess)return error;
  }
  return cudaSuccess;
}
