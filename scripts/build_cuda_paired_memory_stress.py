"""Build actual paired CUDA queue/reservation and H/E frame visibility stress."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


def build(candidate,output):
    candidate=Path(candidate).resolve();src=candidate/'source/src';out=Path(output).resolve();out.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    proof=json.loads((candidate/'paired-handoff-compiler-proof.json').read_text())
    for name,value in proof['helpers'].items():
        if sha(src/name)!=value:raise ValueError('Actual helper changed:'+name)
    code=Path(__file__).with_name('cuda_paired_pipeline_memory_stress.cu').read_text();kernel=(src/'native_closed_pipeline_kernel.cuh').read_text()
    reset=block(kernel,'void AsymmetricClosedReset(')[2].replace('void AsymmetricClosedReset(','__global__ void ResetStress(',1);code=once(code,'// ACTUAL_RESET',reset)
    handoff=(src/'native_paired_handoff.cuh').read_text();handoff=handoff[:handoff.index('template<int Depth,int T,bool Electric>')];code=once(code,'// ACTUAL_HANDOFF_OPERATIONS',handoff)
    worker=block(kernel,'void AsymmetricClosedStep(')[2];worker=once(worker,'void AsymmetricClosedStep(','void PacketStress(')
    worker=once(worker,'const __grid_constant__ BeamzNativeClosedPipelineWorkspace q){','const __grid_constant__ BeamzNativeClosedPipelineWorkspace q,uint32_t*payload,int epoch){')
    worker=once(worker,'GroupStageProfiles(p,CommonProfiles());','for(int i=threadIdx.x;i<576;i+=1024)reinterpret_cast<uint32_t*>(CommonProfiles())[i]=uint32_t(i);')
    window='''    const int oz=(tile/(p.ny*p.nx))*BZ,oy=((tile/p.nx)%p.ny)*BY,ox=(tile%p.nx)*BX;
    C().first=max(1,max(oz-p.phase[0].own[0].z,max(oy-p.phase[0].own[0].y,ox-p.phase[0].own[1].x)));
    C().last=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));'''
    worker=once(worker,window,'    C().first=1;C().last=Depth;')
    worker=once(worker,f'PhaseRebindingBody<Depth,CubeN,{proof["phase_threads"]}>(p);','SyntheticBody<Depth>(p,payload,epoch);')
    code=once(code,'// ACTUAL_RESERVATION_WORKER','template<int Depth,int Policy,int CubeN=0>\n__global__ __launch_bounds__(1024,1) '+worker)
    original=Path(__file__).with_name('cuda_closed_pipeline_memory_stress.cu').read_text();late=block(original,'void LateStartupStress(')[2]
    late=late.replace('cold.tile=blockIdx.x?25:12;','cold.tile=blockIdx.x?16:12;').replace('__syncthreads();','Join();').replace('word+=1024','word+=512')
    code=once(code,'// ACTUAL_LATE_STARTUP_WORKER','__global__ __launch_bounds__(512,1) '+late)
    startup=block(original,'void startup_regression(')[2].replace('using namespace closed_stress;','using namespace paired_stress;').replace('q.tile_count=27;','q.tile_count=27;q.original_count=27;q.node_plane=9;').replace('dim3(2),dim3(1024),','dim3(2),dim3(512),')
    code=once(code,'// CONTROLLED_STARTUP_HOST',startup)
    if proof.get('phase_threads')==256:
        code=re.sub(r'\b(?:512|1024)\b',lambda m:{'512':'256','1024':'512'}[m[0]],code)
        code=once(code,'if(threadIdx.x<576)profiles[threadIdx.x]=uint32_t(threadIdx.x)^0xcafe0000u;','for(int i=threadIdx.x;i<576;i+=512)profiles[i]=uint32_t(i)^0xcafe0000u;')
    source=out/'stress.cu';source.write_text(code);binary=out/'stress';nvcc=Path('/home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root/opt/cuda/bin/nvcc')
    command=[str(nvcc),'-std=c++17','-arch=sm_86','-lineinfo','-Xptxas=-v','-I'+str(src),str(source),str(src/'native_moving.cc'),'-o',str(binary)]
    with (out/'build.log').open('w') as log:run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if run.returncode:raise RuntimeError('Paired GPU stress build failed:'+str(out/'build.log'))
    result=dict(binary=str(binary),sha256=sha(binary),source_sha256=sha(source),helpers=proof['helpers'],expected_stress_entries=5,command=command,GPU_executed=False,
                reservation_binding='Actual worker with physics body replaced by synthetic family payload; windows set1..Depth to isolate all-node queue visibility. Separate actual-H/E-helper frame test has producer1..14/consumer3..16 windows and producer-lastE. Controlled actual all7bootstrap regression retained.',scope='Build only; CUDA synthetic acquire/release/ACK visibility, not FDTD arithmetic or throughput')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.candidate,a.output)))
