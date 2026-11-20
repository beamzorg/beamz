#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include "abi_layout.h"
#include "kernels.h"
#include "launch.h"
#include "graph.h"
#include "yee_primitives.cuh"
#include "mstep_index.h"
#include "temporal/stencils/compact_full_edge.cuh"
#include "cell.cuh"
#include "native_runtime.h"
namespace beamz::cuda::compiled_native {
struct Source {
 const float* coefficients=nullptr;const float* waveforms=nullptr;
 const int32_t* starts=nullptr;const int32_t* current_step=nullptr;
 int count=0,nz=0,ny=0,nx=0,ntime=0;
};
struct Parameters {
 MstepCompactEdgeView phase[2];Source sources[6];
 int macro_step,physical_n[3];MstepIndexDecode index_decode;
};
__device__ __forceinline__ bool Contains(const MstepCompactField&f,int z,int y,int x){return z>=0&&y>=0&&x>=0&&z<f.z&&y<f.y&&x<f.x;}
__host__ __device__ __forceinline__ const MstepCompactField& InitialField(const Parameters&p,int field){return p.phase[field/3].own[field%3];}
__host__ __device__ __forceinline__ const MstepCompactField& FinalField(const Parameters&p,int field){return p.phase[field/3].output[field%3];}
template<int Phase>
__device__ __forceinline__ float ReadField(const Parameters&p,int field,int z,int y,int x){
 // The magnetic phase overwrites or publishes H before E reads it. During
 // adjoint replay the published H lives in its tape, apart from the old H.
 const auto&f=Phase==1&&field<3?FinalField(p,field):InitialField(p,field);
 return Contains(f,z,y,x)?f.data[z*f.plane+y*f.row+x]:0.f;
}
// Prepare proves canonical CPML12, scalar H coefficients and coincident source
// starts; unsupported descriptors use the shared general Program operation.
template<int Layout,bool Bulk=false> struct Environment {
 const Parameters&p;
 __device__ __forceinline__ bool Valid(int field,int z,int y,int x)const{
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
  if constexpr(Bulk)return false;
  const int family=slot/6,term=slot%6,axis=beamz::cuda::yee::CpmlAxis(term);
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
   const auto&psi=p.phase[index/6].psi[index%6];return kind==2?__ldg(psi.a+packed):kind==3?__ldg(psi.b+packed):__ldg(psi.inv+packed);
  }
  if(index<3)return kind==0?__ldg(p.phase[0].decay[index]):__ldg(p.phase[0].source[index]);
  const int c=index-3;const auto&v=p.phase[1];const int linear=(z*(p.physical_n[1]+(c!=1))+y)*(p.physical_n[2]+(c!=0))+x;
  if(kind==0)return Layout==3?__ldg(v.decay[c]):1.f;
  if constexpr(Layout==3) return __ldg(v.source[c]+linear);
  else if constexpr(Layout==2){const int h=__ldg(reinterpret_cast<const int32_t*>(v.codes[c])+(linear>>5));return __ldg(v.tables[c]+(h<0?~h:h+(linear&31)));}
  else return beamz::cuda::yee::PackedMaterialSource(v.tables[c],v.table_sizes[c],v.codes[c],linear);
 }
 __device__ __forceinline__ float Difference(int phase,int source,int axis,int z,int y,int x,float center,float adjacent)const{
  // H's forward wall guard is unreachable after the output support check.
  if(!Bulk && phase){const int coordinate=axis==0?z:axis==1?y:x;
   if(coordinate==0||coordinate==p.physical_n[axis])return 0.f;
  }
  const auto&v=p.phase[phase];
  const int coordinate=axis==0?z:axis==1?y:x;
  const float metric=v.metric_kind==0?v.inv:__ldg(v.metrics[axis]+(v.metric_kind==1?0:coordinate));
  return beamz::cuda::yee::ScaleYeeDifference(phase?__fsub_rn(center,adjacent):__fsub_rn(adjacent,center),metric);
 }
 __device__ __forceinline__ float Finish(int field,int version,int z,int y,int x,float value)const{
  const auto&s=p.sources[field];
  if(s.count){
   const int sz=z-s.starts[0],sy=y-s.starts[1],sx=x-s.starts[2];
   if(sz>=0&&sy>=0&&sx>=0&&sz<s.nz&&sy<s.ny&&sx<s.nx){
    int time=*s.current_step+p.macro_step+version-1;time=max(0,min(time,s.ntime-1));
    const int spatial=(sz*s.ny+sy)*s.nx+sx,volume=s.nz*s.ny*s.nx;
    // Single-source injection contracts; coincident scatter products round first.
    if(s.count==1)value=__fmaf_rn(__ldg(s.coefficients+spatial),__ldg(s.waveforms+time),value);
    else{
     #pragma unroll 4
     for(int source=0;source<s.count;++source)
      value=__fadd_rn(value,__fmul_rn(__ldg(s.coefficients+source*volume+spatial),__ldg(s.waveforms+source*s.ntime+time)));
    }
   }
  }
  return value;
 }
};
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

template <int Phase, int Layout, bool Bulk, int C>
__device__ __forceinline__ float Compute(const Parameters& p, int z, int y, int x) {
  Environment<Layout, Bulk> env{p};
  const auto read = [&] __device__(int field, int, int dz, int dy, int dx) {
    if constexpr (Bulk) {
      const auto& f = Phase == 1 && field < 3 ? FinalField(p, field) : InitialField(p, field);
      return f.data[(z + dz) * f.plane + (y + dy) * f.row + x + dx];
    }
    return ReadField<Phase>(p, field, z + dz, y + dy, x + dx);
  };
  return beamz::cuda::trace_native::AdvanceNativeCell<Phase, C, false>(env, read, 1, z, y, x);
}
template <int Phase, int C>
__device__ __forceinline__ void Store(const Parameters& p, int z, int y, int x, float value) {
  const auto& f = p.phase[Phase].output[C];
  f.data[z * f.plane + y * f.row + x] = value;
}
template <int Phase, int Layout>
__global__ __launch_bounds__(256) void OrdinaryPhase(const __grid_constant__ Parameters p) {
  const int nx = p.physical_n[2] + 1, ny = p.physical_n[1] + 1, nz = p.physical_n[0] + 1;
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= unsigned(nx * ny * nz)) return;
  const int z = MstepQuotient(i, p.index_decode.plane), r = int(i) - z * nx * ny;
  const int y = MstepQuotient(uint32_t(r), p.index_decode.x), x = r - y * nx;
  if (x >= 12 && x < nx - 13 && y >= 12 && y < ny - 13 && z >= 12 && z < nz - 13) {
    const float f0 = Compute<Phase, Layout, true, 0>(p, z, y, x);
    const float f1 = Compute<Phase, Layout, true, 1>(p, z, y, x);
    const float f2 = Compute<Phase, Layout, true, 2>(p, z, y, x);
    Store<Phase, 0>(p, z, y, x, f0);
    Store<Phase, 1>(p, z, y, x, f1);
    Store<Phase, 2>(p, z, y, x, f2);
  } else {
    Environment<Layout> env{p};
    if (env.Valid(Phase * 3, z, y, x))
      Store<Phase, 0>(p, z, y, x, Compute<Phase, Layout, false, 0>(p, z, y, x));
    if (env.Valid(Phase * 3 + 1, z, y, x))
      Store<Phase, 1>(p, z, y, x, Compute<Phase, Layout, false, 1>(p, z, y, x));
    if (env.Valid(Phase * 3 + 2, z, y, x))
      Store<Phase, 2>(p, z, y, x, Compute<Phase, Layout, false, 2>(p, z, y, x));
  }
}
inline bool PrepareOrdinary(const BeamzProgramLaunch&program,int step,Parameters*out,int*layout,
                            bool replay=false){
 if(step<0||!program.source_groups||program.source_group_count!=9||program.h_ab.tensor||program.e_ab.tensor)return false;
 *out={};int hl=0;
 const bool dense_e=program.e_ab.inputs[9].element_type==kBeamzF32;
 if(!MakeMstepCompactEdge(program.h_ab,&out->phase[0],&hl)||
    !MakeMstepCompactEdge(program.e_ab,&out->phase[1],layout,dense_e))return false;
 int minimum[3]{INT32_MAX,INT32_MAX,INT32_MAX};
 for(int field=0;field<6;++field){const auto&f=InitialField(*out,field);
  minimum[0]=std::min(minimum[0],f.z);minimum[1]=std::min(minimum[1],f.y);minimum[2]=std::min(minimum[2],f.x);
  if(f.row!=f.x||f.plane!=f.x*f.y||(!(replay&&field<3)&&FinalField(*out,field).data!=f.data))return false;
 }
 const int dz[6]{0,0,1,1,1,0},dy[6]{0,1,0,1,0,1},dx[6]{1,0,0,0,1,1};
 for(int field=0;field<6;++field){const auto&f=InitialField(*out,field);
  if(f.z!=minimum[0]+dz[field]||f.y!=minimum[1]+dy[field]||f.x!=minimum[2]+dx[field])return false;
 }
 int64_t cells=1;for(int a=0;a<3;++a){if(minimum[a]<=24)return false;out->physical_n[a]=minimum[a];cells*=minimum[a]+1;}
 if(cells>INT32_MAX)return false;
 for(int g=0;g<3;++g)if(program.source_groups[g].coefficients.dims[0])return false;
 for(int f=0;f<6;++f)if(!MakeSource(program.source_groups[3+f],&out->sources[f]))return false;
 for(int phase=0;phase<2;++phase)for(int term=0;term<6;++term){
  const auto&psi=out->phase[phase].psi[term];if(!(replay&&phase==1)&&psi.input!=psi.output)return false;
 }
 out->index_decode=MakeMstepIndexDecode(minimum[1]+1,minimum[2]+1);
 out->macro_step=step;return true;
}
template<int Layout>
inline cudaError_t EnqueueOrdinary(cudaStream_t stream,const Parameters&p,
                                   const BeamzProgramLaunch&program,int step,
                                   bool magnetic_only,
                                   const BeamzNativeStepEffects* effects){
 const auto effect=[&](int stage){
   return effects?effects->enqueue(stream,program,stage,step,effects->context):cudaSuccess;
 };
 auto status=effect(0);if(status!=cudaSuccess)return status;
 const int cells=(p.physical_n[0]+1)*(p.physical_n[1]+1)*(p.physical_n[2]+1),blocks=(cells+256-1)/256;
 OrdinaryPhase<0,Layout><<<blocks,256,0,stream>>>(p);
 status=cudaGetLastError();if(status!=cudaSuccess)return status;
 status=effect(1);if(status!=cudaSuccess||magnetic_only)return status;
 OrdinaryPhase<1,Layout><<<blocks,256,0,stream>>>(p);
 status=cudaGetLastError();if(status!=cudaSuccess)return status;
 status=effect(2);if(status!=cudaSuccess)return status;
 if(program.monitors){
   status=BeamzEnqueueDftGroups(stream,program.h_ab,program.e_ab,*program.monitors,step);
   if(status!=cudaSuccess)return status;
 }
 return effect(3);
}

cudaError_t EnqueueSelectedOrdinary(cudaStream_t stream, const Parameters& p,
                                   const BeamzProgramLaunch& program, int step,
                                   int layout, bool magnetic_only=false,
                                   const BeamzNativeStepEffects* effects=nullptr) {
  if (layout==1) return EnqueueOrdinary<1>(stream,p,program,step,magnetic_only,effects);
  if (layout==2) return EnqueueOrdinary<2>(stream,p,program,step,magnetic_only,effects);
  return EnqueueOrdinary<3>(stream,p,program,step,magnetic_only,effects);
}
}

bool BeamzEnqueueTracedReplayStep(void* opaque_stream, const BeamzProgramLaunch& program,
                                 int step, bool magnetic_only, cudaError_t* status) {
  using namespace beamz::cuda::compiled_native;
  Parameters replay;
  int layout = 0;
  if (!PrepareOrdinary(program, step, &replay, &layout, true)) return false;
  *status = EnqueueSelectedOrdinary(static_cast<cudaStream_t>(opaque_stream),
                                   replay, program, step, layout, magnetic_only);
  return true;
}

int BeamzLaunchTracedProgramEffects(void*opaque_stream,const BeamzProgramLaunch&program,
                                    const BeamzNativeStepEffects* effects){
 using namespace beamz::cuda::compiled_native;
 const auto validation=BeamzValidateNativeProgram(program);if(validation)return validation;
 Parameters initial;int layout=0;if(!PrepareOrdinary(program,0,&initial,&layout))
   return BeamzLaunchProgramEffects(opaque_stream,program,effects);
 const auto stream=static_cast<cudaStream_t>(opaque_stream);
 std::string key=BeamzGraphKey("trace-ordinary-d8631589fcbaea4163da01ab6167261c2a891fe8787a478aab4bc843f26a9f22",opaque_stream,program);
 if(effects)key+=effects->key;
 const auto enqueue=[&]()->cudaError_t{
  for(int step=0;step<program.nsteps;++step){auto p=initial;p.macro_step=step;
   const auto status=EnqueueSelectedOrdinary(stream,p,program,step,layout,false,effects);
   if(status!=cudaSuccess)return status;
  }return cudaSuccess;
 };
 return BeamzLaunchGraph(stream,key,(program.h_ab.cuda_flags&beamz::cuda::abi::kBeamzGraphCache)!=0,program.graph_cache_capacity,enqueue);
}

int BeamzLaunchTracedProgram(void* stream, const BeamzProgramLaunch& program) {
 return BeamzLaunchTracedProgramEffects(stream,program,nullptr);
}
