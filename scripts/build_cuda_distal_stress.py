"""Build a GPU audit of causal last-use retirement with reader-completion probes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();src=a.candidate.resolve()/'source/src';out=a.output.resolve();out.mkdir(exist_ok=False)
    proof=json.loads((a.candidate/'closed-pipeline-compiler-proof.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if not proof.get('distal_disposal'):raise ValueError('Explicit causal-disposal candidate required')
    for name,value in proof['helpers'].items():
        if sha(src/name)!=value:raise ValueError('Staged helper changed')
    reset=block((src/'native_closed_pipeline_kernel.cuh').read_text(),'void AsymmetricClosedReset(')[2].replace('void AsymmetricClosedReset(','__global__ void ResetStress(',1)
    code=Path(__file__).with_name('cuda_closed_pipeline_memory_stress.cu').read_text().replace('// BEAMZ_CLOSED_ACTUAL_RESET_INSERT',reset)
    code=once(code,'struct Parameters{int nz,ny,nx;};','struct Parameters{int nz,ny,nx;float*trails=nullptr;};\nconstexpr int BZ=7,BY=15,BX=31,SurfacePoints=735;')
    code=once(code,'#include "native_closed_pipeline_control.cuh"','#include "native_closed_pipeline_control.cuh"\n#include "native_distal_packet.cuh"')
    code=once(code,'uint32_t*payload,uint64_t*tags,int32_t*bad,int epoch){','uint32_t*payload,uint64_t*tags,int32_t*bad,int epoch,int32_t*reader_done){')
    code=once(code,'   uint32_t*target=',r'''   __syncthreads();
   if(!threadIdx.x)for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;
    if(z>=dz&&y>=dy&&x>=dx)atomicOr(reader_done+(size_t(t-dz*p.ny*p.nx-dy*p.nx-dx)*Depth+step-1),1<<(bits-1));}
   __syncthreads();
   if(!threadIdx.x&&z&&y&&x&&step<Depth&&step<=min(z*BZ-1,min(y*BY-1,x*BX-1))){
    const int producer=t-p.ny*p.nx-p.nx-1;
    if(Acquire(reader_done+size_t(producer)*Depth+step-1)!=127)atomicOr(bad,4096);
   }
   RetireDistalPacket(p,step);
   uint32_t*target=''')
    code=once(code,'Parameters physics{native.counts[0],native.counts[1],native.counts[2]};',
              'Parameters physics{native.counts[0],native.counts[1],native.counts[2],reinterpret_cast<float*>(payload)};auto*reader_done=memory.ints(size_t(count)*Depth);')
    code=once(code,'  ResetStress<<<','  check(cudaMemset(reader_done,0,size_t(count)*Depth*4),"reset read-completion probes");\n  ResetStress<<<')
    code=once(code,'(physics,q,payload,tags,bad,epoch);','(physics,q,payload,tags,bad,epoch,reader_done);')
    if proof.get('single_kernel'):code=once(code,'BeamzBuildNativeClosedCohorts(native,4);','BeamzBuildNativeClosedCohorts(native,4,true);')
    if proof.get('queue_fifo'):code=once(code,'q.cohort=c;q.cohort_tasks=','q.cohort=c;q.cohort_offset=plan.cohort_offsets[c];q.cohort_tasks=')
    code=once(code,'\\\"late_startup_positive_and_negative_passed\\\":true,','\\\"causal_all7_reader_completion_probes_passed\\\":true,\\\"late_startup_positive_and_negative_passed\\\":true,')
    source=out/'stress.cu';source.write_text(code);binary=out/'stress';nvcc=Path('/home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root/opt/cuda/bin/nvcc')
    command=[str(nvcc),'-std=c++17','-arch=sm_86','-lineinfo','-Xptxas=-v','-I'+str(src),str(source),str(src/'native_moving.cc'),'-o',str(binary)]
    with (out/'build.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('Causal-disposal stress compile failed; inspect '+str(out/'build.log'))
    result=dict(binary=str(binary),sha256=sha(binary),source_sha256=sha(source),helpers=proof['helpers'],expected_stress_entries=4,command=command,GPU_executed=False,scope='Actual counter-free whole-line discard and typed scheduler; debug reader-completion probes observe all7 readers before disposal. Synthetic data, no FDTD throughput claim.')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
