"""Build actual cache-retirement helpers into an isolated GPU visibility test."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


EARLY_ACK=r'''
namespace closed_stress {
// The consumer has no live values in versions1..Depth-1. It must ACK before
// READY; the producer then becomes responsible for destructive retirement.
__global__ __launch_bounds__(1024,1) void EarlyAckStress(Parameters p,BeamzNativeClosedPipelineWorkspace q,
 int*entered,int*acknowledged,int*bad){
 using namespace closed_pipeline;
 if(!threadIdx.x){auto&cold=const_cast<Cold&>(C());cold.q=q;cold.bad=bad;cold.depth=4;cold.tile=blockIdx.x;cold.parent_ready=0;cold.abort_local=0;for(int e=0;e<7;++e)cold.acked_step[e]=0;}
 __syncthreads();
 if(!blockIdx.x){
  InitializePackets(p);__threadfence();__syncthreads();
  if(!threadIdx.x){Release(entered,1);const uint64_t start=Timer();while(!Acquire(acknowledged)){if(Timer()-start>q.timeout_ns){Abort(256);break;}__nanosleep(32);}}
  __syncthreads();
  for(int n=1;n<=4;++n){
   for(int i=threadIdx.x;i<Words;i+=1024)p.trails[(n-1)*Words+i]=__uint_as_float(0xa5b60000u^uint32_t(i));
   PublishPacket(p,n);
  }
 }else{
  if(!threadIdx.x){const uint64_t start=Timer();while(!Acquire(entered)){if(Timer()-start>q.timeout_ns){Abort(256);break;}__nanosleep(32);}}
  __syncthreads();AckPacketsUpTo(p,3);__syncthreads();if(!threadIdx.x)Release(acknowledged,1);
 }
}
}
static void early_ack_regression(){
 using namespace closed_stress;Allocations memory;BeamzNativeClosedPipelineWorkspace q;
 q.owner_slots=memory.copy({0,1});q.packet_refs=memory.ints(8);q.ready=memory.ints(2);q.abort=memory.ints(1);
 int*entered=memory.ints(1),*acknowledged=memory.ints(1),*bad=memory.ints(1);
 Parameters p{1,1,2,static_cast<float*>(memory.bytes(8ULL*Words*4))};
 check(cudaMemset(entered,0,4),"early handshake reset");check(cudaMemset(acknowledged,0,4),"early ACK reset");check(cudaMemset(bad,0,4),"early failure reset");check(cudaMemset(q.abort,0,4),"early abort reset");
 check(cudaFuncSetAttribute(EarlyAckStress,cudaFuncAttributeMaxDynamicSharedMemorySize,closed_pipeline::SharedBytes),"early shared capacity");
 void*args[]={&p,&q,&entered,&acknowledged,&bad};
 check(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(EarlyAckStress),dim3(2),dim3(1024),args,closed_pipeline::SharedBytes,nullptr),"early cooperative launch");check(cudaDeviceSynchronize(),"early cooperative synchronization");
 int result=0,refs[4];check(cudaMemcpy(&result,bad,4,cudaMemcpyDeviceToHost),"early result");check(cudaMemcpy(refs,q.packet_refs,16,cudaMemcpyDeviceToHost),"early counters");
 if(result||refs[0]!=16||refs[1]!=16||refs[2]!=16)throw std::runtime_error("Early ACK/producer READY handshake failed");
}
'''


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();src=a.candidate.resolve()/'source/src';out=a.output.resolve();out.mkdir(exist_ok=False)
    proof=json.loads((a.candidate/'closed-pipeline-compiler-proof.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if not proof.get('dead_face_discard'):raise ValueError('Explicit cache-retirement candidate required')
    for name,value in proof['helpers'].items():
        if sha(src/name)!=value:raise ValueError('Staged helper changed')
    reset=block((src/'native_closed_pipeline_kernel.cuh').read_text(),'void AsymmetricClosedReset(')[2].replace('void AsymmetricClosedReset(','__global__ void ResetStress(',1)
    code=Path(__file__).with_name('cuda_closed_pipeline_memory_stress.cu').read_text().replace('// BEAMZ_CLOSED_ACTUAL_RESET_INSERT',reset)
    code=once(code,'struct Parameters{int nz,ny,nx;};','struct Parameters{int nz,ny,nx;float*trails=nullptr;};\nconstexpr int SurfacePoints=735;')
    code=once(code,'#include "native_closed_pipeline_control.cuh"','#include "native_closed_pipeline_control.cuh"\n#include "native_dead_face_control.cuh"')
    code=once(code,'  if(!threadIdx.x)tags[owner]=','  InitializePackets(p);\n  if(!threadIdx.x)tags[owner]=')
    code=once(code,'   uint32_t*target=','   __syncthreads();AckPacketsUpTo(p,step);\n   uint32_t*target=')
    code=once(code,'   Publish(step);','   PublishPacket(p,step);')
    code=once(code,'  __syncthreads();if(!threadIdx.x&&!Aborted())Complete();','  AckPacketsUpTo(p,Depth-1);__syncthreads();if(!threadIdx.x&&!Aborted())Complete();')
    code=once(code,' q.next=memory.ints(count);',' q.packet_refs=memory.ints(size_t(plan.owner_count)*Depth);\n q.next=memory.ints(count);')
    code=once(code,'Parameters physics{native.counts[0],native.counts[1],native.counts[2]};','Parameters physics{native.counts[0],native.counts[1],native.counts[2],reinterpret_cast<float*>(payload)};')
    if proof.get('single_kernel'):code=once(code,'BeamzBuildNativeClosedCohorts(native,4);','BeamzBuildNativeClosedCohorts(native,4,true);')
    if proof.get('queue_fifo'):code=once(code,'q.cohort=c;q.cohort_tasks=','q.cohort=c;q.cohort_offset=plan.cohort_offsets[c];q.cohort_tasks=')
    code=once(code,'int main(){',EARLY_ACK+'\nint main(){')
    code=once(code,' startup_regression();int epoch=0',' startup_regression();early_ack_regression();int epoch=0')
    code=once(code,'\\\"late_startup_positive_and_negative_passed\\\":true,','\\\"early_ACK_before_READY_passed\\\":true,\\\"late_startup_positive_and_negative_passed\\\":true,')
    source=out/'stress.cu';source.write_text(code);binary=out/'stress';root=Path('/home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root');nvcc=root/'opt/cuda/bin/nvcc'
    command=[str(nvcc),'-std=c++17','-arch=sm_86','-lineinfo','-Xptxas=-v','-I'+str(src),str(source),str(src/'native_moving.cc'),'-o',str(binary)]
    with (out/'build.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('Retirement stress compile failed; inspect '+str(out/'build.log'))
    result=dict(binary=str(binary),sha256=sha(binary),source_sha256=sha(source),helpers=proof['helpers'],expected_stress_entries=5,command=command,GPU_executed=False,scope='Actual packet-refcount, full-line discard, unique-node queue and typed ready helpers. Synthetic packet data only.')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
