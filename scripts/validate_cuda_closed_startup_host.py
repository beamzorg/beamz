"""Exercise the actual staged late-start controller with a blocked diagonal."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_phase_cycle import block


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();candidate=a.candidate.resolve();src=candidate/'source/src'
    proof=json.loads((candidate/'host-closed-pipeline-proof/result.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if not proof['passed'] or sha(src/'native_closed_pipeline_control.cuh')!=proof['helpers']['native_closed_pipeline_control.cuh']:raise ValueError('Actual control source binding required')
    header=(candidate/'host-closed-pipeline-proof/actual/control.h').read_text()
    begin,end,_=block(header,'int Acquire(')
    header=header[:begin]+'''int Acquire(const int32_t*pointer){
      const int value=std::atomic_ref<int>(*const_cast<int*>(pointer)).load(std::memory_order_acquire);
      if(pointer==probe_address.load(std::memory_order_relaxed)&&value==0)probe_count.fetch_add(1,std::memory_order_relaxed);
      return value;}'''+header[end:]
    code=r'''
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include "native_closed_pipeline.h"
#define __device__
#define __forceinline__ inline
struct ThreadIndex{int x=0;};thread_local ThreadIndex threadIdx;
void __nanosleep(int){std::this_thread::yield();}
void __threadfence(){std::atomic_thread_fence(std::memory_order_seq_cst);}
void __syncthreads(){std::atomic_thread_fence(std::memory_order_seq_cst);}
void atomicOr(int*p,int v){std::atomic_ref<int>(*p).fetch_or(v);}
template<class T>T __ldg(const T*p){return *p;}
template<class T>T __ldcg(const T*p){return *p;}
using std::min;
std::atomic<const int*>probe_address{nullptr};std::atomic<int>probe_count{0};
struct Parameters{int nz,ny,nx;};
#include "control.h"
int main(){using namespace closed_pipeline;
 std::vector<int>ready(27,1);const int tile=25,diagonal=12;ready[diagonal]=0;probe_address.store(ready.data()+diagonal);
 BeamzNativeClosedPipelineWorkspace q;q.ready=ready.data();int bad=0,abort=0;q.abort=&abort;
 std::atomic<bool>finished{false};std::atomic<int>floor{-1};Parameters p{3,3,3};
 std::thread worker([&]{auto&cold=const_cast<Cold&>(C());cold.q=q;cold.bad=&bad;cold.tile=tile;cold.depth=4;cold.abort_local=0;cold.parent_ready=0;
  WaitParents(p,1,true);floor.store(C().parent_ready);finished.store(true,std::memory_order_release);});
 const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
 while(probe_count.load()<2&&!finished.load(std::memory_order_acquire)&&std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
 const bool blocked=probe_count.load()>=2&&!finished.load(std::memory_order_acquire);
 Release(ready.data()+diagonal,1);worker.join();
 if(!blocked||!finished.load()||floor.load()!=1||bad||abort)throw std::runtime_error("Late startup did not wait for its diagonal");
 std::cout<<"{\"passed\":true,\"actual_diagonal_zero_polls\":"<<probe_count.load()<<",\"late_start_blocked_until_diagonal_published\":true,\"scope\":\"Actual staged controller with CPUmatchingPTXprimitiveemulation; noFP/GPUclaim\"}\n";
}
'''
    if proof.get('paired_handoff'):
        code=code.replace('q.abort=&abort;','q.abort=&abort;q.original_count=27;q.node_plane=9;')
        code=code.replace('cold.tile=tile;','cold.tile=16; // Group0 original tile25 in supernode16.\n')
    out=candidate/'startup-controller-proof-v1';out.mkdir(exist_ok=False);rows=[]
    for name in ('actual','missing-startup-diagonal'):
        folder=out/name;folder.mkdir();h=header
        if name!='actual':
            if h.count('if(startup){')!=1:raise ValueError('Explicit late-start selector required')
            h=h.replace('if(startup){','if(false){')
        (folder/'control.h').write_text(h);cc=folder/'proof.cc';cc.write_text(code);binary=folder/'proof'
        subprocess.run(['g++','-std=c++20','-O2','-pthread','-fsanitize=address,undefined','-I'+str(folder),'-I'+str(src),str(cc),'-o',str(binary)],capture_output=True,text=True,check=True)
        run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=10)
        if (run.returncode!=0 if name=='actual' else run.returncode==0):raise RuntimeError(name+run.stdout+run.stderr)
        rows.append(dict(name=name,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
    result=dict(passed=True,control_sha256=proof['helpers']['native_closed_pipeline_control.cuh'],rows=rows)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
