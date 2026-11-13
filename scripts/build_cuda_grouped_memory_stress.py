"""Build actual named-group queue visibility and private-frame isolation stress."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


FRAME=r'''
__global__ __launch_bounds__(1024,1) void FrameStress(int*bad){
 using namespace closed_pipeline;
 uint32_t*frame=reinterpret_cast<uint32_t*>(Frame());
 uint32_t*profiles=reinterpret_cast<uint32_t*>(CommonProfiles());
 if(threadIdx.x<576)profiles[threadIdx.x]=0xcafe0000u^uint32_t(threadIdx.x);
 __syncthreads();
 const int g=Group(),tid=LocalThread();
 for(int epoch=0;epoch<3+4*g;++epoch){
  if(!tid&&((epoch+g)&1))__nanosleep(1000);
  for(int i=tid;i<3*4096;i+=512)frame[i]=Pattern(epoch,int(blockIdx.x)*2+g,7,i);
  Join();
  for(int i=tid;i<3*4096;i+=512){const int read=(i*73+41)%(3*4096);
   if(frame[read]!=Pattern(epoch,int(blockIdx.x)*2+g,7,read))atomicOr(bad,4);
  }
  for(int i=tid;i<576;i+=512)if(profiles[i]!=(0xcafe0000u^uint32_t(i)))atomicOr(bad,8);
  Join();
 }
}
'''


def build(candidate,output):
    candidate=Path(candidate).resolve();src=candidate/'source/src';out=Path(output).resolve();out.mkdir(exist_ok=False)
    proof=json.loads((candidate/'closed-pipeline-compiler-proof.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if not proof.get('phase_grouped_worker'):raise ValueError('Actual grouped worker source required')
    for name,value in proof['helpers'].items():
        if sha(src/name)!=value:raise ValueError('Changed bound helper:'+name)
    reset=block((src/'native_closed_pipeline_kernel.cuh').read_text(),'void AsymmetricClosedReset(')[2].replace('void AsymmetricClosedReset(','__global__ void ResetStress(',1)
    code=Path(__file__).with_name('cuda_closed_pipeline_memory_stress.cu').read_text().replace('// BEAMZ_CLOSED_ACTUAL_RESET_INSERT',reset)
    code=once(code,'BeamzBuildNativeClosedCohorts(native,4);','BeamzBuildNativeClosedCohorts(native,4,true);')
    a,b,packet=block(code,'void PacketStress(')
    packet=packet.replace('threadIdx.x','LocalThread()').replace('__syncthreads();','Join();').replace('word+=1024','word+=512')
    packet=once(packet,' using namespace closed_pipeline;',' using namespace closed_pipeline;\n if(int(blockIdx.x)*2+Group()>=q.worker_count)return;')
    code=code[:a]+packet+code[b:]
    code=once(code,'PacketStress<Depth><<<q.worker_count,1024,','PacketStress<Depth><<<(q.worker_count+1)/2,1024,')
    a,b,late=block(code,'void LateStartupStress(');late=late.replace('__syncthreads();','Join();').replace('word+=1024','word+=512');code=code[:a]+late+code[b:]
    code=once(code,'__launch_bounds__(1024,1) void LateStartupStress','__launch_bounds__(512,1) void LateStartupStress')
    code=once(code,'dim3(2),dim3(1024),','dim3(2),dim3(512),')
    code=once(code,'} // namespace closed_stress',FRAME+'\n} // namespace closed_stress')
    code=once(code,' startup_regression();int epoch=0,cases=0;',r''' startup_regression();
 {
  Allocations frame_memory;int*bad=frame_memory.ints(1);check(cudaMemset(bad,0,4),"frame diagnostic reset");
  check(cudaFuncSetAttribute(closed_stress::FrameStress,cudaFuncAttributeMaxDynamicSharedMemorySize,closed_stress::closed_pipeline::SharedBytes),"frame capacity");
  closed_stress::FrameStress<<<4,1024,closed_stress::closed_pipeline::SharedBytes>>>(bad);check(cudaGetLastError(),"frame launch");check(cudaDeviceSynchronize(),"frame joins");
  int result=0;check(cudaMemcpy(&result,bad,4,cudaMemcpyDeviceToHost),"frame result");if(result)throw std::runtime_error("Independent frame/barrier/profile violation:"+std::to_string(result));
 }
 int epoch=0,cases=0;''')
    code=once(code,'for(int workers:{1,4,sms})','for(int workers:{1,3,4,sms,2*sms+1})')
    code=once(code,'\"packet_word_reads\":','\"packet_word_reads\":') if False else code
    source=out/'stress.cu';source.write_text(code);binary=out/'stress';nvcc=Path('/home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root/opt/cuda/bin/nvcc')
    command=[str(nvcc),'-std=c++17','-arch=sm_86','-lineinfo','-Xptxas=-v','-I'+str(src),str(source),str(src/'native_moving.cc'),'-o',str(binary)]
    with (out/'build.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('Grouped stress compile failed:'+str(out/'build.log'))
    result=dict(binary=str(binary),sha256=sha(binary),source_sha256=sha(source),helpers=proof['helpers'],expected_stress_entries=5,command=command,GPU_executed=False,
                scope='Actual grouped control/queue release-acquire visibility and 4blocks disjoint12288-word field frames, common576-word immutableprofiles, distinct group iteration counts3/7, odd logicalworker counts1/3/2SM+1 and controlled clipped startup. Synthetic, not FDTD.')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.candidate,a.output)))
