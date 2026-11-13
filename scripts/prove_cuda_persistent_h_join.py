"""Prepare an actual H publication join with CPU atomic primitive shims.

Each CPU worker represents one already-converged CUDA warp and writes all32
lane tokens. The exact32-leader join/watchdog/publication code is retained.
The CUDA full-mask warp instruction itself is source-checked, not emulated.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

try:
    from scripts.prove_cuda_persistent_control import prepare as control_prepare
    from scripts.patch_cuda_native_persistent_h_publication_join import JOIN
except ModuleNotFoundError:
    from prove_cuda_persistent_control import prepare as control_prepare
    from patch_cuda_native_persistent_h_publication_join import JOIN


HARNESS=r'''
#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
namespace cuda {inline constexpr int thread_scope_device=0;template<class T,int>using atomic_ref=std::atomic_ref<T>;
inline constexpr auto memory_order_acquire=std::memory_order_acquire;
inline constexpr auto memory_order_release=std::memory_order_release;
inline constexpr auto memory_order_relaxed=std::memory_order_relaxed;}
#define __device__
#define __forceinline__ inline
static int atomicOr(int*w,int v){return std::atomic_ref<int>(*w).fetch_or(v);}
static void __nanosleep(unsigned){std::this_thread::yield();}
struct ThreadIndex{unsigned x;};thread_local ThreadIndex threadIdx{};
static void __syncwarp(uint32_t mask){if(mask!=0xffffffffu)throw std::logic_error("full-mask participation lost");std::atomic_thread_fence(std::memory_order_acq_rel);}
#include "actual_persistent_control.inc"
namespace beamz::cuda::moving_persistent {Cold shared_c{};volatile Cold&C(){return shared_c;}}
#include "actual_h_join.inc"
int main(){
 using namespace beamz::cuda::moving_persistent;
 try{
  size_t words_checked=0,positive_cases=0;
  for(int run=0;run<13;++run){
   const bool omitted=run==12;std::array<uint64_t,1024>words{};uint32_t ready=0;int32_t abort=0,bad=0;
   auto&c=shared_c;c={};c.ready=&ready;c.abort_global=&abort;c.bad=&bad;c.owner=0;
   c.timeout_ns=omitted?100000000ULL:1000000000ULL;
   std::atomic<bool>validated{false},invalid{false};std::vector<std::thread>threads;
   for(unsigned warp=0;warp<32;++warp){if(omitted&&warp==17)continue;
    threads.emplace_back([&,warp]{threadIdx.x=warp*32;
     std::this_thread::sleep_for(std::chrono::microseconds((31-warp)*11));
     for(unsigned lane=0;lane<32;++lane)words[warp*32+lane]=uint64_t(run+1)*2048+warp*32+lane;
     JoinHWriterWarpsAndPublish(3);
     if(!warp&&!omitted){
      if(Acquire(&ready)!=5||AcquireHWriterWarps()!=32)invalid=true;
      else{for(unsigned i=0;i<1024;++i)if(words[i]!=uint64_t(run+1)*2048+i)invalid=true;validated=true;}
     }
    });
   }
   for(auto&t:threads)t.join();
   if(omitted){if(!abort||!(bad&256)||ready!=0||c.h_arrivals!=31)throw std::logic_error("omitted writer warp was not rejected by actual watchdog");}
   else{if(abort||bad||invalid||!validated)throw std::logic_error("actual join released before all writer tokens");words_checked+=1024;++positive_cases;}
  }
  std::cout<<"{\"positive_cases\":"<<positive_cases<<",\"all_writer_words_checked\":"<<words_checked<<",\"omitted_warp_watchdog_rejected\":true,\"pass\":true}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}
}
'''


def prepare(source,output):
    source,output=source.resolve(),output.resolve()
    control=control_prepare(source,output)
    text=(source/'src/native_persistent.cuh').read_text()
    a=text.index('// TRACE_H_PUBLICATION_JOIN_BEGIN');b=text.index('// TRACE_H_PUBLICATION_JOIN_END',a)+len('// TRACE_H_PUBLICATION_JOIN_END\n')
    if text[a:b]!=JOIN:raise ValueError('Actual H join helpers changed')
    join=JOIN
    a=join.index('__device__ __forceinline__ void ArriveHWriterWarp()')
    b=join.index('__device__ __forceinline__ void JoinHWriterWarpsAndPublish(',a)
    primitives='''__device__ __forceinline__ void ArriveHWriterWarp(){std::atomic_ref<uint32_t>(const_cast<uint32_t&>(C().h_arrivals)).fetch_add(1,std::memory_order_acq_rel);}
__device__ __forceinline__ uint32_t AcquireHWriterWarps(){return std::atomic_ref<uint32_t>(const_cast<uint32_t&>(C().h_arrivals)).load(std::memory_order_acquire);}
'''
    join=join[:a]+primitives+join[b:]
    (output/'actual_h_join.inc').write_text('namespace beamz::cuda::moving_persistent {\n'+join+'}\n')
    (output/'actual-h-join.cc').write_text(HARNESS)
    command=['c++','-std=c++20','-O2','-pthread','-I'+str(output),str(output/'actual-h-join.cc'),'-o',str(output/'actual-h-join')]
    proof={'source_consumer_sha256':hashlib.sha256(text.encode()).hexdigest(),'actual_join_sha256':hashlib.sha256(JOIN.encode()).hexdigest(),
        'CPU_join_shim_sha256':hashlib.sha256((output/'actual_h_join.inc').read_bytes()).hexdigest(),'CPU_harness_sha256':hashlib.sha256(HARNESS.encode()).hexdigest(),
        'standalone_writer_CPU_fence':False,
        'control_proof':control,'compiler_command':command,'executed':False,
        'scope':'Exact join/watchdog/global publication control; only shared atomic primitives and GPU warp aggregate are CPU shims; no CUDA performance claim'}
    (output/'join-proof-plan.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--execute',action='store_true')
    a=p.parse_args();proof=prepare(a.source,a.output)
    if a.execute:
        r=subprocess.run(proof['compiler_command'],capture_output=True,text=True);(a.output/'join-compile.log').write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError('Actual join CPU compile failed')
        r=subprocess.run([proof['compiler_command'][-1]],capture_output=True,text=True,timeout=30);(a.output/'join-run.log').write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError('Actual join CPU proof rejected')
        proof.update(executed=True,result=json.loads(r.stdout));(a.output/'join-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'executed':proof['executed'],'result':proof.get('result'),'output':str(a.output)}))


if __name__=='__main__':main()
