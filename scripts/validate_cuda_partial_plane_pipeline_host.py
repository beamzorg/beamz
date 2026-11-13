"""Bind partial-step CUDA edits to the parent and stress actual queue helpers on CPU.

This does not model CUDA caches or replace the full GPU numerical gates. It
executes the staged queue/control source with C++ atomics and checks ownership,
parent readiness, tag publication, task uniqueness and forward progress.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_phase_cycle import block


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(candidate,parent):
    candidate,parent=Path(candidate).resolve(),Path(parent).resolve();src=candidate/'source/src'
    proof=json.loads((candidate/'partial-plane-compiler-proof.json').read_text())
    c=(src/'asymmetric_native.cuh').read_text();original=(parent/'src/asymmetric_native.cuh').read_text()
    if digest(src/'asymmetric_native.cuh')!=proof['consumer_sha256'] or digest(parent/'src/asymmetric_native.cuh')!=proof['parent_consumer_sha256']:
        raise ValueError('Explicit frozen consumer ancestry required')
    for name,sha in proof['helpers'].items():
        if digest(src/name)!=sha:raise ValueError('Staged control helper changed:'+name)
    for name in ('asymmetric_native_cell_generated.cuh','yee_cell_generated.cuh'):
        if (src/name).read_bytes()!=(parent/'src'/name).read_bytes():raise ValueError('Canonical arithmetic changed')
    for signature in ('void UniformBody(','void FixedCubeCachedBody(','void SourcePendingPairedBody('):
        body=block(c,signature)[2]
        if proof.get('outlined'):
            name=signature.split()[1].split('(')[0]
            clone=block(c,'void PlaneOutlined'+name+'(')[2].replace('void PlaneOutlined'+name+'(','void '+name+'(',1)
            if clone!=body:raise ValueError('Outlined native body differs from bound inline body:'+name)
        body=body.replace('const int tile=Pipeline?plane_pipeline::Tile():tiles[blockIdx.x],','const int tile=tiles[blockIdx.x],')
        body=body.replace('const int owner=Pipeline?plane_pipeline::Owner(p,tile):(rank&3)*p.frontier+p.slots[tile];','const int owner=(rank&3)*p.frontier+p.slots[tile];')
        body=body.replace('const int owner=Pipeline?plane_pipeline::Owner(p,producer):((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer];','const int owner=((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer];')
        body=body.replace('(Pipeline?__ldcg(p.owner_tags+owner):p.owner_tags[owner])','p.owner_tags[owner]')
        body=body.replace('\n if constexpr(Pipeline){if(!threadIdx.x)plane_pipeline::Started();}','')
        body=body.replace('\n  if constexpr(Pipeline){plane_pipeline::Skipped(prev);plane_pipeline::WaitParents(prev);if(plane_pipeline::Aborted())return;}','')
        body=body.replace('  if constexpr(Pipeline){plane_pipeline::WaitParents(n);if(plane_pipeline::Aborted())return;}\n','')
        body=body.replace(' if constexpr(Pipeline)plane_pipeline::Publish(n);\n ','')
        body=body.replace('ActiveAxes,CachedProfiles,Pipeline>env','ActiveAxes,CachedProfiles>env')
        body=body.replace('FixedCubeCachedEnvironment<N,ActiveAxes,Pipeline>env','FixedCubeCachedEnvironment<N,ActiveAxes>env')
        body=body.replace('constexpr int T=BlockT?BlockT:((Fast||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512),','constexpr int T=(Fast||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512,')
        if body!=block(original,signature)[2]:raise ValueError('Exact native arithmetic loop inverse failed:'+signature)
    for signature in ('float Finish(','FieldPacket ImportPacket(','void Observe(','int PsiOffset(','float Difference(','float Coefficient('):
        if block(c,signature)[2]!=block(original,signature)[2]:raise ValueError('Arithmetic/import helper changed:'+signature)
    directory=candidate/'host-partial-plane-proof';directory.mkdir(exist_ok=False)
    header=(src/'native_plane_pipeline_control.cuh').read_text()
    a,b,_=block(header,'volatile Cold& C()')
    header=header[:a]+'volatile Cold& C(){static thread_local Cold cold;return cold;}'+header[b:]
    a,b,_=block(header,'uint64_t Timer()')
    header=header[:a]+'uint64_t Timer(){return std::chrono::steady_clock::now().time_since_epoch().count();}'+header[b:]
    (directory/'control.h').write_text(header)
    code=r'''
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include "native_plane_pipeline.h"
#define __device__
#define __forceinline__ inline
struct ThreadIndex{int x=0;};thread_local ThreadIndex threadIdx;
void __nanosleep(int){std::this_thread::yield();}
void __threadfence(){std::atomic_thread_fence(std::memory_order_seq_cst);}
void __syncthreads(){std::atomic_thread_fence(std::memory_order_seq_cst);}
void atomicOr(int*p,int v){std::atomic_ref<int>(*p).fetch_or(v);}
namespace cuda{enum scope{thread_scope_device,thread_scope_block};
using std::memory_order_acquire;using std::memory_order_release;using std::memory_order_acq_rel;using std::memory_order_relaxed;
template<class T,scope S>using atomic_ref=std::atomic_ref<T>;}
struct Parameters{int ny,nx;};
#include "control.h"
int main(){using namespace plane_pipeline;size_t cases=0,tasks=0,steps=0;
for(int workers:{1,4,32,82})for(int depth:{4,16})for(int seed:{1,17,37}){
 BeamzNativePlanePipelineWorkspace q;q.nz=11;q.ny=9;q.nx=7;q.tile_count=q.nz*q.ny*q.nx;q.worker_count=workers;q.timeout_ns=30000000000ULL;
 std::vector<int>queue(q.tile_count),entry(q.tile_count),pending(q.tile_count),claimed(q.tile_count),ready(q.tile_count),planes(q.nz),done(q.tile_count),tags(3*q.ny*q.nx,-1),started(q.tile_count);
 int head=0,tail=1,abort=0,completed=0,starts=0,exited=0,bad=0;
 q.queue=queue.data();q.entry_ready=entry.data();q.pending=pending.data();q.claimed=claimed.data();q.ready=ready.data();q.plane_done=planes.data();
 q.head=&head;q.tail=&tail;q.abort=&abort;q.completed=&completed;q.started=&starts;q.exited=&exited;
 for(int t=0;t<q.tile_count;++t)pending[t]=(t/(q.ny*q.nx)>0)+((t/q.nx)%q.ny>0)+(t%q.nx>0);
 entry[0]=claimed[0]=1;Parameters p{q.ny,q.nx};std::vector<std::thread>threads;
 for(int worker=0;worker<workers;++worker)threads.emplace_back([&,worker]{
  auto&cold=const_cast<Cold&>(C());cold.q=q;cold.bad=&bad;cold.depth=depth;
  for(;;){Pop();if(Tile()<0)break;const int t=Tile(),z=t/(q.ny*q.nx),y=(t/q.nx)%q.ny,x=t%q.nx,owner=Owner(p,t);
   if(std::atomic_ref<int>(started[t]).fetch_add(1))Abort(8192);
   const int old=std::atomic_ref<int>(tags[owner]).load();
   if(old>=0){if(!Acquire(done.data()+old))Abort(16384);
    for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
     if(old/(q.ny*q.nx)+dz>=q.nz||(old/q.nx)%q.ny+dy>=q.ny||old%q.nx+dx>=q.nx)continue;
     if(!Acquire(done.data()+old+dz*q.ny*q.nx+dy*q.nx+dx))Abort(16384);}}
   Release(tags.data()+owner,t);Started();
   for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(z<dz||y<dy||x<dx)continue;
    const int parent=t-dz*q.ny*q.nx-dy*q.nx-dx;if(Acquire(tags.data()+Owner(p,parent))!=parent)Abort(32768);}
   for(int n=1;n<=depth&&!Aborted();++n){WaitParents(n);if(Aborted())break;
    if((t+n+worker+seed)%7==0)std::this_thread::yield();Publish(n);}
   if(Aborted())break;Release(done.data()+t,1);Complete();
  }
 });
 for(auto&thread:threads)thread.join();
 if(bad||abort||completed!=q.tile_count||starts!=q.tile_count||tail!=q.tile_count)throw std::runtime_error("Actual queue/control failed:"+std::to_string(bad));
 for(int t=0;t<q.tile_count;++t)if(done[t]!=1||started[t]!=1||ready[t]!=depth)throw std::runtime_error("Task/version lost");
 ++cases;tasks+=q.tile_count;steps+=size_t(q.tile_count)*depth;
}
std::cout<<"{\"cases\":"<<cases<<",\"tasks\":"<<tasks<<",\"step_waits\":"<<steps<<",\"passed\":true}\n";
}
'''
    cc=directory/'proof.cc';binary=directory/'proof';cc.write_text(code)
    subprocess.run(['g++','-std=c++20','-O2','-pthread','-fsanitize=address,undefined','-I'+str(directory),'-I'+str(src),str(cc),'-o',str(binary)],capture_output=True,text=True,check=True)
    run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=300,check=True)
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],helpers=proof['helpers'],actual_queue=json.loads(run.stdout),
                exact_native_loop_inverse=True,unchanged_arithmetic_import_helpers=True,canonical_headers_byte_identical=True,
                scope='Actual staged queue helpers under concurrent CPU workers and C++ atomic model; ASan/UBSan. CUDA memory visibility, full1024thread behavior, numerical values and performance require GPU gates.')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate,a.parent)))
