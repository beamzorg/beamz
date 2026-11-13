// Standalone cross-SM publication contract test. NEVER runs automatically.
// nvcc -std=c++17 -arch=sm_86 scripts/cuda_persistent_fence_coalescing_stress.cu -o /tmp/beamz-fence-coalescing
// /tmp/beamz-fence-coalescing [mode=0] [iterations=10000]
// mode0: all1024 writers -> CTA barrier -> leader GPU release (NO fences).
// mode1: writers -> full syncwarp -> shared acq_rel RMW chain -> leader
//        shared acquire32 -> GPU release (NO fences, NO pre-release CTA join).
// mode2: CTA barrier -> leader-only threadfence -> GPU release, reference.
// mode3: all-writer threadfence -> CTA barrier -> GPU release, reference.
// Both CTAs use1024 threads, requiring one resident block/SM and >=2 SMs.
// Every writer's scalar even word shares lines with other writers and the
// consumer's odd-word stores. Epoch/ack prevents payload reuse while any
// reader is live; repeated two-slot payload reuse exercises stale data.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct alignas(16) State {
  uint32_t ready, ack, abort, sm[2];
  unsigned long long errors, checks;
  uint32_t first_epoch, first_tid, first_expected, first_actual;
};
__device__ __forceinline__ uint32_t Acq(const uint32_t* p){uint32_t v;asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(v):"l"(__cvta_generic_to_global(p)):"memory");return v;}
__device__ __forceinline__ uint32_t Relaxed(const uint32_t* p){uint32_t v;asm volatile("ld.relaxed.gpu.global.u32 %0,[%1];":"=r"(v):"l"(__cvta_generic_to_global(p)):"memory");return v;}
__device__ __forceinline__ void Rel(uint32_t* p,uint32_t v){asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(__cvta_generic_to_global(p)),"r"(v):"memory");}
__device__ __forceinline__ uint32_t CG(const uint32_t* p){uint32_t v;asm volatile("ld.global.cg.u32 %0,[%1];":"=r"(v):"l"(__cvta_generic_to_global(p)):"memory");return v;}
__device__ __forceinline__ void WB(uint32_t* p,uint32_t v){asm volatile("st.global.wb.u32 [%0],%1;"::"l"(__cvta_generic_to_global(p)),"r"(v):"memory");}
__device__ __forceinline__ uint32_t SharedAcquire(const uint32_t* p){uint32_t v;asm volatile("ld.acquire.cta.shared.u32 %0,[%1];":"=r"(v):"r"(uint32_t(__cvta_generic_to_shared(p))):"memory");return v;}
__device__ __forceinline__ void SharedRelease(uint32_t* p,uint32_t v){asm volatile("st.release.cta.shared.u32 [%0],%1;"::"r"(uint32_t(__cvta_generic_to_shared(p))),"r"(v):"memory");}
__device__ __forceinline__ void Arrive(uint32_t* p){uint32_t v;asm volatile("atom.acq_rel.cta.shared.add.u32 %0,[%1],1;":"=r"(v):"r"(uint32_t(__cvta_generic_to_shared(p))):"memory");}
__device__ __forceinline__ uint64_t Timer(){uint64_t t;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(t));return t;}
__device__ __forceinline__ uint32_t Symbol(uint32_t epoch,uint32_t tid){return (epoch*1664525u)^(tid*1013904223u)^0xa5f137d9u;}
__device__ void Error(State* s,uint32_t epoch,uint32_t tid,uint32_t expected,uint32_t actual){
  if(atomicAdd(&s->errors,1ULL)==0){s->first_epoch=epoch;s->first_tid=tid;s->first_expected=expected;s->first_actual=actual;}
}
__device__ bool Wait(State* s,const uint32_t* p,uint32_t epoch){const uint64_t start=Timer();for(;;){
  if(Relaxed(&s->abort))return false;if(Acq(p)>=epoch)return true;
  if(Timer()-start>5000000000ULL){Rel(&s->abort,1);return false;}__nanosleep(32);
}}

template<int Mode>
__global__ __launch_bounds__(1024,1) void Stress(uint32_t* payload,State* s,uint32_t iterations){
  __shared__ uint32_t arrivals,gate;
  __shared__ int aborted;
  const uint32_t tid=threadIdx.x;
  if(!tid){uint32_t sm;asm volatile("mov.u32 %0,%%smid;":"=r"(sm));s->sm[blockIdx.x]=sm;arrivals=gate=0;aborted=0;}
  __syncthreads();
  unsigned long long checks=0;
  for(uint32_t epoch=1;epoch<=iterations;++epoch){
    const uint32_t slot=(epoch&1u)*2048u;
    if(blockIdx.x==0){
      // Counter reset is exclusive and ordered before every writer's next
      // arrival. Prior iteration's ack/barrier retires both CTAs' readers.
      if(!tid){if(!Wait(s,&s->ack,epoch-1))aborted=1;arrivals=0;}
      __syncthreads();if(aborted)return;
      // Vary writer latency to prevent the leader from accidentally waiting
      // for everyone through instruction scheduling alone.
      if(((tid/32+epoch)&7u)==0)__nanosleep(128);
      WB(payload+slot+2*tid,Symbol(epoch,tid));
      if constexpr(Mode==3)__threadfence();
      if constexpr(Mode==1){
        __syncwarp(0xffffffffu);
        if(!(tid&31u))Arrive(&arrivals);
        if(!tid){
          const uint64_t start=Timer();
          while(SharedAcquire(&arrivals)!=32){if(Relaxed(&s->abort)||Timer()-start>5000000000ULL){Rel(&s->abort,1);aborted=1;break;}}
          if(!aborted)Rel(&s->ready,epoch);
        }
      }else{
        __syncthreads();
        if(!tid){if constexpr(Mode==2)__threadfence();Rel(&s->ready,epoch);}
      }
      // This barrier is AFTER publication/ack; it cannot justify the earlier
      // warp-mode Ready release, and reset cannot race any old arrival.
      if(!tid&&!aborted&&!Wait(s,&s->ack,epoch))aborted=1;
      __syncthreads();if(aborted)return;
    }else{
      if(!tid){if(!Wait(s,&s->ready,epoch))aborted=1;SharedRelease(&gate,epoch);}
      while(SharedAcquire(&gate)<epoch){}
      if(aborted)return;
      const uint32_t expected=Symbol(epoch,tid),actual=CG(payload+slot+2*tid);
      if(actual!=expected)Error(s,epoch,tid,expected,actual);++checks;
      // Different word on the SAME line, written from the other SM. No
      // overlapping word races, including at the next epoch's two-slot reuse.
      WB(payload+slot+2*tid+1,~expected);
      __syncthreads();
      if(!tid)Rel(&s->ack,epoch);
      __syncthreads();
    }
  }
  if(blockIdx.x==1)atomicAdd(&s->checks,checks);
}

static void Check(cudaError_t e,const char* op){if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",op,cudaGetErrorString(e));std::exit(2);}}
template<int Mode> static void Run(uint32_t* p,State* s,uint32_t iterations){
  int resident=0;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,Stress<Mode>,1024,0),"occupancy");
  if(resident!=1){std::fprintf(stderr,"Requires one block/SM, found%d\n",resident);std::exit(2);}
  Stress<Mode><<<2,1024>>>(p,s,iterations);
}
int main(int argc,char** argv){
  const int mode=argc>1?std::atoi(argv[1]):0;
  const uint32_t iterations=argc>2?uint32_t(std::strtoul(argv[2],nullptr,10)):10000;
  if(mode<0||mode>3||!iterations||iterations>1000000)return 2;
  int device=0;Check(cudaGetDevice(&device),"device");cudaDeviceProp properties{};Check(cudaGetDeviceProperties(&properties,device),"properties");
  if(properties.multiProcessorCount<2||properties.maxThreadsPerBlock<1024)return 2;
  uint32_t* payload=nullptr;State* state=nullptr;Check(cudaMalloc(&payload,4096*sizeof(uint32_t)),"payload");Check(cudaMalloc(&state,sizeof(State)),"state");
  Check(cudaMemset(payload,0,4096*sizeof(uint32_t)),"zero payload");Check(cudaMemset(state,0,sizeof(State)),"zero state");
  if(mode==0)Run<0>(payload,state,iterations);else if(mode==1)Run<1>(payload,state,iterations);else if(mode==2)Run<2>(payload,state,iterations);else Run<3>(payload,state,iterations);
  Check(cudaGetLastError(),"launch");Check(cudaDeviceSynchronize(),"synchronize");State report{};Check(cudaMemcpy(&report,state,sizeof(report),cudaMemcpyDeviceToHost),"download");
  const unsigned long long expected=1024ULL*iterations;
  const bool pass=!report.errors&&!report.abort&&report.checks==expected&&report.ready==iterations&&report.ack==iterations&&report.sm[0]!=report.sm[1];
  std::printf("{\"passed\":%s,\"mode\":%d,\"checks\":%llu,\"expected_checks\":%llu,\"errors\":%llu,\"abort\":%u,\"ready\":%u,\"ack\":%u,\"sm_ids\":[%u,%u],\"first_epoch\":%u,\"first_tid\":%u,\"first_expected\":%u,\"first_actual\":%u}\n",pass?"true":"false",mode,report.checks,expected,report.errors,report.abort,report.ready,report.ack,report.sm[0],report.sm[1],report.first_epoch,report.first_tid,report.first_expected,report.first_actual);
  Check(cudaFree(payload),"free payload");Check(cudaFree(state),"free state");return pass?0:1;
}
