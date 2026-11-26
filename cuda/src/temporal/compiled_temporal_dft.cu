#include <cuda_runtime_api.h>
#include <cstdint>
#include "kernels.h"
#include "launch.h"
#include "host/observation/dft_work.h"
using beamz::cuda::temporal::kDftPointsPerBlock;
using beamz::cuda::temporal::kDftFrequencyBatch;
namespace {
struct DftFields {const float*values;int key_count;float dt;const int32_t*index_offsets;const void* remapped;int index_width;};

__device__ __forceinline__ float ObservationTime(
    const DftFields& fields, const BeamzDftGroupLaunch& monitors, int step_offset) {
  const int64_t step = static_cast<const int32_t*>(monitors.elapsed_steps.data)[0]
                      + static_cast<int64_t>(step_offset) + 1;
  return __fmaf_rn(static_cast<float>(step), fields.dt,
                  static_cast<const float*>(monitors.time.data)[0]);
}

__global__ void PrepareDftPhases(DftFields fields, BeamzDftGroupLaunch monitors,
                                 int step_offset) {
  const int frequency = blockIdx.x * blockDim.x + threadIdx.x;
  const int monitor = blockIdx.y;
  const int max_frequency_count =
      static_cast<int>(monitors.frequencies.dims[1]);
  if (monitor >= monitors.monitor_count || frequency >= max_frequency_count) {
    return;
  }
  const int frequency_offset = monitor * max_frequency_count + frequency;
  const int phase_offset = frequency_offset + blockIdx.z *
                                                  monitors.monitor_count *
                                                  max_frequency_count;
  step_offset += blockIdx.z;
  float window = 0.0f;
  float phase_sin = 0.0f;
  float phase_cos = 0.0f;
  const auto *counts = static_cast<const int32_t *>(monitors.counts.data);
  const int frequency_count = counts[5 * monitor];
  const int interval =
      counts[5 * monitor + 2] > 0 ? counts[5 * monitor + 2] : 1;
  const auto *codes = static_cast<const int32_t *>(monitors.codes.data);
  if (frequency < frequency_count && frequency_count > 0 &&
      frequency_count <= max_frequency_count &&
      (codes[2 * monitor] == 0 || codes[2 * monitor] == 1) &&
      (codes[2 * monitor + 1] == 0 || codes[2 * monitor + 1] == 1)) {
    const int64_t absolute_step =
        static_cast<int64_t>(
            static_cast<const int32_t *>(monitors.current_step.data)[0]) +
        step_offset;
    const float time = ObservationTime(fields, monitors, step_offset);
    const auto *windows = static_cast<const float *>(monitors.windows.data);
    const float start = windows[3 * monitor];
    const float end = windows[3 * monitor + 1];
    if (absolute_step % interval == 0 && time >= start && time <= end) {
      window = 1.0f;
      if (codes[2 * monitor] == 1 && isfinite(end) && end > start) {
        const float tau =
            fminf(fmaxf((time - start) / (end - start), 0.0f), 1.0f);
        window = 0.5f * (1.0f - cosf(6.2831853071795864769f * tau));
      }
      const float frequency_hz = static_cast<const float *>(
          monitors.frequencies.data)[frequency_offset];
      sincosf(6.2831853071795864769f * frequency_hz * time, &phase_sin,
              &phase_cos);
    }
  }
  static_cast<float *>(monitors.phase_window.data)[phase_offset] = window;
  static_cast<float *>(monitors.phase_sin.data)[phase_offset] = phase_sin;
  static_cast<float *>(monitors.phase_cos.data)[phase_offset] = phase_cos;
}


// The eight-corner gather is invariant throughout each time block.
struct EightNeighbors {
 int indices[8];float weights[8];
 __device__ void Load(const DftFields& fields,const BeamzDftGroupLaunch& monitors,int index_base,int base){
  if(fields.index_width==4){
   const auto* source=static_cast<const int4*>(fields.remapped)+index_base/4;
   #pragma unroll
   for(int i=0;i<2;++i){const int4 ix=source[i];
    indices[4*i]=ix.x;indices[4*i+1]=ix.y;indices[4*i+2]=ix.z;indices[4*i+3]=ix.w;}
  }else{
   const auto* bytes=static_cast<const unsigned char*>(fields.remapped)+size_t(index_base)*fields.index_width;
   const auto* words=reinterpret_cast<const uint2*>(bytes);
   #pragma unroll
   for(int group=0;group<2;++group){
    const uint2 low=words[group];
    const unsigned high=fields.index_width==3?
        reinterpret_cast<const unsigned*>(bytes+16)[group]:0;
    indices[4*group]=(low.x&65535u)|((high&255u)<<16);
    indices[4*group+1]=(low.x>>16)|(((high>>8)&255u)<<16);
    indices[4*group+2]=(low.y&65535u)|(((high>>16)&255u)<<16);
    indices[4*group+3]=(low.y>>16)|((high>>24)<<16);
   }
  }
  const float4* weight=static_cast<const float4*>(monitors.weights.data)+base/4;
  #pragma unroll
  for(int i=0;i<2;++i){
   const float4 w=weight[i];
   weights[4*i]=w.x;weights[4*i+1]=w.y;weights[4*i+2]=w.z;weights[4*i+3]=w.w;
  }
 }
 __device__ float Sample(const DftFields& f,int step)const{
  float sum=0.f;const float* values=f.values+size_t(step)*f.key_count;
  #pragma unroll
  for(int i=0;i<8;++i)if(unsigned(indices[i])<unsigned(f.key_count))sum+=values[indices[i]]*weights[i];
  return sum;
 }
};

template<bool Ragged,bool Eight=false>
__global__ void TemporalDftPacked(const __grid_constant__ DftFields fields,
 const __grid_constant__ BeamzDftGroupLaunch monitors,const uint64_t*locations,
 int macro,int depth,int raw_step_offset){
 const int offset=blockIdx.x*blockDim.x+threadIdx.x;
 if(offset>=monitors.dft_re.dims[0])return;
 const auto*counts=static_cast<const int32_t*>(monitors.counts.data);
 int monitor=0,nf=0,np=0;
 BeamzDftStorage storage{};
 for(;monitor<monitors.monitor_count;++monitor){nf=counts[5*monitor];np=counts[5*monitor+1];
  storage=BeamzDftStorageFor(monitors,monitor,nf,np);
  if(nf>0&&np>0&&offset>=storage.offset&&offset<storage.offset+storage.count)break;}
 if(monitor==monitors.monitor_count)return;
 const int vo=storage.offset;
 const int max_nf=monitors.frequencies.dims[1],max_np=BeamzDftPointCapacity(monitors.indices),interval=max(1,counts[5*monitor+2]),wo=counts[5*monitor+4];
 if(nf>max_nf||np>max_np||vo<0||wo<0||vo+storage.count>monitors.dft_re.dims[0]||vo+storage.count>monitors.dft_im.dims[0]||
  static_cast<int64_t>(wo)+nf>monitors.dft_weight.dims[0]||offset>=monitors.dft_im.dims[0])return;
 const int slot=(offset-vo)/(nf*np),frequency=((offset-vo)/np)%nf,point=(offset-vo)%np;
 const int component=BeamzDftComponent(monitors,monitor,slot,storage.compact);
 if(component<0)return;
 const bool weight_owner=slot==0&&point==0;
 float re=static_cast<const float*>(monitors.dft_re.data)[offset],im=static_cast<const float*>(monitors.dft_im.data)[offset];
 float weight=weight_owner?static_cast<const float*>(monitors.dft_weight.data)[wo+frequency]:0.f;
 const bool active=BeamzDftSlot(monitors,monitor,component,storage.compact)>=0;
 const int neighbors=BeamzDftNeighbors(monitors.indices),base=BeamzDftGatherBase(monitors.indices,monitor,component,point);
 const int index_base=fields.index_offsets?fields.index_offsets[monitor]+(component*np+point)*neighbors:base;
 EightNeighbors cache;if constexpr(Eight)if(active)cache.Load(fields,monitors,index_base,base);
 const int phase_stride=monitors.monitor_count*max_nf;
 const int mode=static_cast<const int32_t*>(monitors.codes.data)[2*monitor+1];
 const float length=static_cast<const float*>(monitors.windows.data)[3*monitor+2];
 for(int version=0;version<depth;++version){
  const int po=version*phase_stride+monitor*max_nf+frequency;
  const float window=static_cast<const float*>(monitors.phase_window.data)[po];
  if(window==0.f)continue;
  if(weight_owner)weight+=window;
  if(!active)continue;
  float sample=0.f;
  if constexpr(Eight)sample=cache.Sample(fields,raw_step_offset+version);else for(int neighbor=0;neighbor<neighbors;++neighbor){
   const int index=static_cast<const int32_t*>(monitors.indices.data)[index_base+neighbor];
   if(index>=0&&index<fields.key_count){
    size_t address=0;
    if constexpr(Ragged){const uint64_t row=__ldg(locations+index);
     address=unsigned(row)+((size_t(macro)&unsigned((row>>40)&15))*depth+version)*unsigned((row>>32)&255);
    }else address=size_t(raw_step_offset+version)*fields.key_count+index;
    sample+=fields.values[address]*static_cast<const float*>(monitors.weights.data)[base+neighbor];
   }
  }
  float scale=window;
  if(mode==1){if(!isfinite(length)||length<=0.f)continue;
   scale*=fields.dt*static_cast<float>(interval)*299792458.0f/length/sqrtf(6.2831853071795864769f);}
  re+=scale*sample*static_cast<const float*>(monitors.phase_cos.data)[po];
  im+=scale*sample*static_cast<const float*>(monitors.phase_sin.data)[po];
 }
 if(weight_owner)static_cast<float*>(monitors.dft_weight.data)[wo+frequency]=weight;
 static_cast<float*>(monitors.dft_re.data)[offset]=re;static_cast<float*>(monitors.dft_im.data)[offset]=im;
}

template<bool Ragged,int Batch,bool Eight=false>
__global__ void TemporalDftFrequencyReuse(const __grid_constant__ DftFields fields,
 const __grid_constant__ BeamzDftGroupLaunch monitors,const uint64_t*locations,
 int macro,int depth,int raw_step_offset){
 // Flatten only the point/frequency blocks each monitor actually owns.
 // A wide field monitor and a many-frequency port can share one launch.
 uint64_t tile=blockIdx.x+uint64_t(blockIdx.z)*gridDim.x;
 const int component=blockIdx.y;
 const auto*counts=static_cast<const int32_t*>(monitors.counts.data);
 static_assert(Batch==kDftFrequencyBatch);
 int monitor=0;
 beamz::cuda::temporal::DftWork work{};
 for(;monitor<monitors.monitor_count;++monitor){
  work=beamz::cuda::temporal::DftWorkFor(counts[5*monitor],counts[5*monitor+1]);
  if(tile<work.blocks)break;
  tile-=work.blocks;
 }
 if(monitor==monitors.monitor_count)return;
 const int point=int(tile%work.point_blocks)*kDftPointsPerBlock+threadIdx.x;
 const int first_frequency=int(tile/work.point_blocks)*Batch;
 const int nf=counts[5*monitor],np=counts[5*monitor+1],interval=max(1,counts[5*monitor+2]),wo=counts[5*monitor+4];
 const auto storage=BeamzDftStorageFor(monitors,monitor,nf,np);
 const int vo=storage.offset;
 const int max_nf=monitors.frequencies.dims[1],max_np=BeamzDftPointCapacity(monitors.indices);
 if(nf<1||nf>max_nf||np<1||np>max_np||point>=np||first_frequency>=nf||vo<0||wo<0||
  vo+storage.count>monitors.dft_re.dims[0]||vo+storage.count>monitors.dft_im.dims[0]||static_cast<int64_t>(wo)+nf>monitors.dft_weight.dims[0])return;
 const int phase_stride=monitors.monitor_count*max_nf;
 // Weight ownership is independent of the component mask. Preserve each
 // frequency's original rounded accumulator rather than adding a window sum.
 if(component==0&&point==0){
  #pragma unroll
  for(int k=0;k<Batch;++k){const int frequency=first_frequency+k;if(frequency<nf){
   float weight=static_cast<const float*>(monitors.dft_weight.data)[wo+frequency];
   for(int version=0;version<depth;++version){const float window=static_cast<const float*>(monitors.phase_window.data)[version*phase_stride+monitor*max_nf+frequency];if(window!=0.f)weight+=window;}
   static_cast<float*>(monitors.dft_weight.data)[wo+frequency]=weight;
  }}
 }
 const int slot=BeamzDftSlot(monitors,monitor,component,storage.compact);
 if(slot<0)return;
 const int neighbors=BeamzDftNeighbors(monitors.indices),base=BeamzDftGatherBase(monitors.indices,monitor,component,point);
 const int index_base=fields.index_offsets?fields.index_offsets[monitor]+(component*np+point)*neighbors:base;
 EightNeighbors cache;if constexpr(Eight)cache.Load(fields,monitors,index_base,base);
 const int mode=static_cast<const int32_t*>(monitors.codes.data)[2*monitor+1];
 const float length=static_cast<const float*>(monitors.windows.data)[3*monitor+2];
 if(mode==1&&(!isfinite(length)||length<=0.f))return;
 float re[Batch],im[Batch];
 #pragma unroll
 for(int k=0;k<Batch;++k){const int frequency=first_frequency+k;
  if(frequency<nf){const int offset=vo+(slot*nf+frequency)*np+point;re[k]=static_cast<const float*>(monitors.dft_re.data)[offset];im[k]=static_cast<const float*>(monitors.dft_im.data)[offset];}}
 for(int version=0;version<depth;++version){
  const int po=version*phase_stride+monitor*max_nf+first_frequency;
  const float window=static_cast<const float*>(monitors.phase_window.data)[po];if(window==0.f)continue;
  float sample=0.f;
  if constexpr(Eight)sample=cache.Sample(fields,raw_step_offset+version);else for(int neighbor=0;neighbor<neighbors;++neighbor){
   const int index=static_cast<const int32_t*>(monitors.indices.data)[index_base+neighbor];
   if(index>=0&&index<fields.key_count){
    size_t address=0;if constexpr(Ragged){const uint64_t row=__ldg(locations+index);
     address=unsigned(row)+((size_t(macro)&unsigned((row>>40)&15))*depth+version)*unsigned((row>>32)&255);
    }else address=size_t(raw_step_offset+version)*fields.key_count+index;
    sample+=fields.values[address]*static_cast<const float*>(monitors.weights.data)[base+neighbor];
   }
  }
  float scale=window;if(mode==1)scale*=fields.dt*static_cast<float>(interval)*299792458.0f/length/sqrtf(6.2831853071795864769f);
  #pragma unroll
  for(int k=0;k<Batch;++k)if(first_frequency+k<nf){
   re[k]+=scale*sample*static_cast<const float*>(monitors.phase_cos.data)[po+k];
   im[k]+=scale*sample*static_cast<const float*>(monitors.phase_sin.data)[po+k];
  }
 }
 #pragma unroll
 for(int k=0;k<Batch;++k){const int frequency=first_frequency+k;
  if(frequency<nf){const int offset=vo+(slot*nf+frequency)*np+point;static_cast<float*>(monitors.dft_re.data)[offset]=re[k];static_cast<float*>(monitors.dft_im.data)[offset]=im[k];}}
}

}
cudaError_t BeamzEnqueueTemporalDft(cudaStream_t stream,const BeamzLaunch&h,const BeamzLaunch&e,
 const BeamzDftGroupLaunch&input,int offset,int raw_step_offset,int depth,float*phase,const uint64_t*locations,const int32_t*index_offsets,const void* remapped,int index_width,uint64_t frequency_tiles){
 if(depth<1||depth>32||!phase||input.monitor_count<1)return cudaErrorInvalidValue;
 auto monitors=input;const size_t count=size_t(depth)*input.monitor_count*input.frequencies.dims[1];
 monitors.phase_window.data=phase;monitors.phase_sin.data=phase+count;monitors.phase_cos.data=phase+2*count;
 const DftFields fields{static_cast<const float*>(e.outputs[0].data),int(e.outputs[0].dims[2]),e.dt,index_offsets,remapped,index_width};
 PrepareDftPhases<<<dim3((monitors.frequencies.dims[1]+31)/32,monitors.monitor_count,depth),32,0,stream>>>(fields,monitors,offset);
 if(auto status=cudaPeekAtLastError();status!=cudaSuccess)return status;
 const int blocks=(monitors.dft_re.dims[0]+127)/128;
 if(blocks){
  if(monitors.frequencies.dims[1]>=kDftFrequencyBatch){
   if(!frequency_tiles)return cudaSuccess;
   const unsigned x=frequency_tiles>INT32_MAX?unsigned(INT32_MAX):unsigned(frequency_tiles);
   const uint64_t z=1+(frequency_tiles-1)/x;
   if(z>65535)return cudaErrorInvalidConfiguration;
   const dim3 reuse(x,6,unsigned(z));
   if(locations)TemporalDftFrequencyReuse<true,kDftFrequencyBatch><<<reuse,kDftPointsPerBlock,0,stream>>>(fields,monitors,locations,raw_step_offset/depth,depth,raw_step_offset);
   else if(BeamzDftNeighbors(monitors.indices)==8)TemporalDftFrequencyReuse<false,kDftFrequencyBatch,true><<<reuse,kDftPointsPerBlock,0,stream>>>(fields,monitors,nullptr,0,depth,raw_step_offset);
   else TemporalDftFrequencyReuse<false,kDftFrequencyBatch><<<reuse,kDftPointsPerBlock,0,stream>>>(fields,monitors,nullptr,0,depth,raw_step_offset);
  }else
  if(locations)TemporalDftPacked<true><<<blocks,128,0,stream>>>(fields,monitors,locations,raw_step_offset/depth,depth,raw_step_offset);
  else if(BeamzDftNeighbors(monitors.indices)==8)TemporalDftPacked<false,true><<<blocks,128,0,stream>>>(fields,monitors,nullptr,0,depth,raw_step_offset);
  else TemporalDftPacked<false><<<blocks,128,0,stream>>>(fields,monitors,nullptr,0,depth,raw_step_offset);
 }
 return cudaPeekAtLastError();
}
