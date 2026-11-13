// Standalone validation only. No BeamZ backend change, no automatic GPU run.
// Build: nvcc -std=c++17 -arch=sm_86 scripts/cuda_persistent_mixed_psi_stress.cu -o /tmp/beamz-mixed-psi-stress
// Run under the root agent's exclusive GPU queue: /tmp/beamz-mixed-psi-stress [iterations=10000] [rounds=64]
// Two resident 1024-thread CTAs must be on different SMs; only each leader works.
// Consumer primes target L1, producer replaces target, release/acquire handoff,
// consumer CG-loads fresh state and uses ordinary CA loads after its WB stores.
// Producer concurrently writes OTHER scalar words on the same 128-byte line.
// Acquire may invalidate the primed L1: this is a contract test, not a claim
// that stale lines survived acquire or that a cache hit was observed.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct alignas(16) State {
  uint32_t primed, ready, done, abort;
  unsigned long long errors, checks, neighbor_stores;
  uint32_t first_kind, first_expected, first_actual, sm[2];
};

__device__ __forceinline__ uint32_t Acq(const uint32_t* p) {
  uint32_t v; asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(v) : "l"(__cvta_generic_to_global(p)) : "memory"); return v;
}
__device__ __forceinline__ uint32_t Relaxed(const uint32_t* p) {
  uint32_t v; asm volatile("ld.relaxed.gpu.global.u32 %0,[%1];" : "=r"(v) : "l"(__cvta_generic_to_global(p)) : "memory"); return v;
}
__device__ __forceinline__ void Rel(uint32_t* p,uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0],%1;" :: "l"(__cvta_generic_to_global(p)),"r"(v) : "memory");
}
__device__ __forceinline__ uint32_t CA(const uint32_t* p) {
  uint32_t v; asm volatile("ld.global.ca.u32 %0,[%1];" : "=r"(v) : "l"(__cvta_generic_to_global(p)) : "memory"); return v;
}
__device__ __forceinline__ uint32_t CG(const uint32_t* p) {
  uint32_t v; asm volatile("ld.global.cg.u32 %0,[%1];" : "=r"(v) : "l"(__cvta_generic_to_global(p)) : "memory"); return v;
}
__device__ __forceinline__ void WB(uint32_t* p,uint32_t v) {
  asm volatile("st.global.wb.u32 [%0],%1;" :: "l"(__cvta_generic_to_global(p)),"r"(v) : "memory");
}
__device__ __forceinline__ uint64_t Timer() {
  uint64_t t; asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(t));return t;
}
__device__ void Error(State* s,uint32_t kind,uint32_t expected,uint32_t actual) {
  if(atomicAdd(&s->errors,1ULL)==0){s->first_kind=kind;s->first_expected=expected;s->first_actual=actual;}
}
__device__ bool Wait(State* s,const uint32_t* word,uint32_t epoch) {
  const uint64_t start=Timer();
  for(;;){if(Relaxed(&s->abort))return false;if(Acq(word)>=epoch)return true;
    if(Timer()-start>5000000000ULL){Rel(&s->abort,1);Error(s,9,epoch,Relaxed(word));return false;}}
}
__global__ __launch_bounds__(1024,1) void Stress(uint32_t* data,State* s,uint32_t iterations,uint32_t rounds) {
  if(threadIdx.x)return;
  uint32_t sm;asm volatile("mov.u32 %0,%%smid;":"=r"(sm));s->sm[blockIdx.x]=sm;
  if(blockIdx.x==0){
    unsigned long long stores=0;
    for(uint32_t i=1;i<=iterations;++i){
      if(!Wait(s,&s->primed,i))return;
      WB(data,i*17u);Rel(&s->ready,i);
      const uint64_t start=Timer();uint32_t k=1;
      while(Relaxed(&s->done)<i){
        if(Relaxed(&s->abort))return;
        WB(data+k,i^k);k=k==31?1:k+1;++stores;
        if(Timer()-start>5000000000ULL){Rel(&s->abort,1);Error(s,10,i,Relaxed(&s->done));return;}
      }
      // Acquire completes consumer->producer ownership handoff before reuse.
      if(!Wait(s,&s->done,i))return;
    }
    s->neighbor_stores=stores;
  }else{
    uint32_t prior=0;unsigned long long checks=0;
    for(uint32_t i=1;i<=iterations;++i){
      const uint32_t primer=CA(data);if(primer!=prior)Error(s,1,prior,primer);++checks;
      Rel(&s->primed,i);if(!Wait(s,&s->ready,i))return;
      uint32_t value=i*17u;
      const uint32_t fresh=CG(data);if(fresh!=value)Error(s,2,value,fresh);++checks;
      // This is the actual mixed-policy contract: a same-thread ordinary store
      // intervenes before EVERY continued ordinary load, even after a CG read.
      for(uint32_t j=0;j<rounds;++j){
        value=(value*1664525u)+1013904223u;WB(data,value);
        const uint32_t actual=CA(data);if(actual!=value)Error(s,3,value,actual);++checks;
      }
      prior=value;Rel(&s->done,i);
    }
    s->checks=checks;
  }
}

static void Check(cudaError_t status,const char* operation) {
  if(status!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",operation,cudaGetErrorString(status));std::exit(2);}
}
int main(int argc,char** argv) {
  const uint32_t iterations=argc>1?uint32_t(std::strtoul(argv[1],nullptr,10)):10000;
  const uint32_t rounds=argc>2?uint32_t(std::strtoul(argv[2],nullptr,10)):64;
  if(!iterations||iterations>1000000||!rounds||rounds>1000000)return 2;
  int device=0;Check(cudaGetDevice(&device),"get device");cudaDeviceProp properties{};
  Check(cudaGetDeviceProperties(&properties,device),"get properties");
  if(properties.multiProcessorCount<2||properties.maxThreadsPerBlock<1024){std::fprintf(stderr,"Requires >=2 SMs and 1024-thread blocks\n");return 2;}
  uint32_t* data=nullptr;State* state=nullptr;
  Check(cudaMalloc(&data,128),"allocate line");Check(cudaMalloc(&state,sizeof(State)),"allocate state");
  Check(cudaMemset(data,0,128),"zero line");Check(cudaMemset(state,0,sizeof(State)),"zero state");
  int resident=0;Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,Stress,1024,0),"occupancy");
  if(resident!=1){std::fprintf(stderr,"Requires one resident 1024-thread block/SM, found %d\n",resident);return 2;}
  Stress<<<2,1024>>>(data,state,iterations,rounds);Check(cudaGetLastError(),"launch");Check(cudaDeviceSynchronize(),"synchronize");
  State report{};Check(cudaMemcpy(&report,state,sizeof(report),cudaMemcpyDeviceToHost),"download state");
  const unsigned long long expected=static_cast<unsigned long long>(iterations)*(rounds+2);
  const bool pass=!report.errors&&!report.abort&&report.sm[0]!=report.sm[1]&&report.checks==expected&&report.neighbor_stores>0;
  std::printf("{\"passed\":%s,\"errors\":%llu,\"abort\":%u,\"checks\":%llu,\"expected_checks\":%llu,\"neighbor_stores\":%llu,\"sm_ids\":[%u,%u],\"first_kind\":%u,\"first_expected\":%u,\"first_actual\":%u,\"stale_L1_survival_proven\":false}\n",pass?"true":"false",report.errors,report.abort,report.checks,expected,report.neighbor_stores,report.sm[0],report.sm[1],report.first_kind,report.first_expected,report.first_actual);
  Check(cudaFree(data),"free line");Check(cudaFree(state),"free state");return pass?0:1;
}
