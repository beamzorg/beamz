// Opt-in diagnostic only. No field arithmetic or extra CTA barriers.
#pragma once
#ifndef MOVING_PHASE_PROFILE
#define MOVING_PHASE_PROFILE 0
#endif
namespace moving_phase_profile {
enum Phase {Init,HSetup,HCompute,HWait,Import,ImportWait,ESetup,ECompute,EWait,Publish,Loop,Count};
constexpr int Metadata=8,Stride=Metadata+Count;
#if MOVING_PHASE_PROFILE
struct Shared {
 volatile unsigned long long totals[Count],last,clock_begin,global_begin;
 volatile unsigned smid;
};
__device__ __forceinline__ unsigned long long GlobalTimer(){unsigned long long value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;}
__device__ __forceinline__ unsigned Smid(){unsigned value;asm volatile("mov.u32 %0, %%smid;":"=r"(value));return value;}
__device__ __forceinline__ void Begin(Shared&p){if(!threadIdx.x){const auto t=clock64();p.clock_begin=t;p.last=t;p.global_begin=GlobalTimer();p.smid=Smid();
#pragma unroll
 for(int i=0;i<Count;++i)p.totals[i]=0;
}}
__device__ __forceinline__ void Mark(Shared&p,Phase phase){if(!threadIdx.x){const auto now=clock64();p.totals[phase]=p.totals[phase]+now-p.last;p.last=now;}}
__device__ __forceinline__ void End(Shared&p,unsigned long long*records,int tile,int rank,int steps){if(!threadIdx.x){const auto end=clock64();const auto global_end=GlobalTimer();auto*out=records+static_cast<size_t>(tile)*Stride;out[0]=p.clock_begin;out[1]=end;out[2]=p.global_begin;out[3]=global_end;out[4]=p.smid;out[5]=rank;out[6]=tile;out[7]=steps;
#pragma unroll
 for(int i=0;i<Count;++i)out[Metadata+i]=p.totals[i];
}}
#endif
}
#if MOVING_PHASE_PROFILE
#define MOVING_PROFILE_DECLARE __shared__ moving_phase_profile::Shared moving_profile_state
#define MOVING_PROFILE_BEGIN() moving_phase_profile::Begin(moving_profile_state)
#define MOVING_PROFILE_MARK(phase) moving_phase_profile::Mark(moving_profile_state,moving_phase_profile::phase)
#define MOVING_PROFILE_END(g,tile,rank,steps) moving_phase_profile::End(moving_profile_state,(g).phase_profile,tile,rank,steps)
#else
#define MOVING_PROFILE_DECLARE
#define MOVING_PROFILE_BEGIN() ((void)0)
#define MOVING_PROFILE_MARK(phase) ((void)0)
#define MOVING_PROFILE_END(g,tile,rank,steps) ((void)0)
#endif
