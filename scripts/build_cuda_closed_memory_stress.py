"""Build, without executing, a source-bound closed-pipeline GPU stress."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_phase_cycle import block


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();candidate=a.candidate.resolve();src=candidate/'source/src';out=a.output.resolve();out.mkdir(exist_ok=False)
    proof=json.loads((candidate/'closed-pipeline-compiler-proof.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for name,value in proof['helpers'].items():
        if sha(src/name)!=value:raise ValueError('Staged helper changed')
    reset=block((src/'native_closed_pipeline_kernel.cuh').read_text(),'void AsymmetricClosedReset(')[2].replace('void AsymmetricClosedReset(','__global__ void ResetStress(',1)
    code=Path(__file__).with_name('cuda_closed_pipeline_memory_stress.cu').read_text().replace('// BEAMZ_CLOSED_ACTUAL_RESET_INSERT',reset)
    if proof.get('single_kernel'):
        anchor='BeamzBuildNativeClosedCohorts(native,4);'
        if code.count(anchor)!=1:raise ValueError('Explicit actual global owner planner binding required')
        code=code.replace(anchor,'BeamzBuildNativeClosedCohorts(native,4,true);')
    if proof.get('queue_fifo'):
        anchor='q.cohort=c;q.cohort_tasks='
        if code.count(anchor)!=1:raise ValueError('Explicit FIFO cohort offset required')
        code=code.replace(anchor,'q.cohort=c;q.cohort_offset=plan.cohort_offsets[c];q.cohort_tasks=')
    if proof.get('phase_rebinding') and proof.get('phase_threads')==512:
        a,b,kernel=block(code,'void PacketStress(');kernel=kernel.replace('word+=1024','word+=512');code=code[:a]+kernel+code[b:]
        code=code.replace('__launch_bounds__(1024,1) void PacketStress','__launch_bounds__(512,2) void PacketStress')
        code=code.replace('PacketStress<Depth><<<q.worker_count,1024,','PacketStress<Depth><<<q.worker_count,512,')
        code=code.replace('for(int workers:{1,4,sms})','for(int workers:{1,4,sms,2*sms})')
    source=out/'stress.cu';source.write_text(code);binary=out/'stress'
    root=Path('/home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root');nvcc=root/'opt/cuda/bin/nvcc'
    command=[str(nvcc),'-std=c++17','-arch=sm_86','-lineinfo','-Xptxas=-v','-I'+str(src),str(source),str(src/'native_moving.cc'),'-o',str(binary)]
    with (out/'build.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('Standalone stress compile failed; inspect '+str(out/'build.log'))
    result=dict(binary=str(binary),sha256=sha(binary),source_sha256=sha(source),helpers=proof['helpers'],expected_stress_entries=4,command=command,GPU_executed=False,scope='Build only; acquireGPUcampaignlock/checkdeadline before explicitexecution')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
