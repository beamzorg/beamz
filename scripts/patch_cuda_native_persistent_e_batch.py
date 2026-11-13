"""Batch explicit E publications; next H join publishes the preceding E prefix.

All-thread H fences remain unconditional. The H(n+1) writer join therefore
orders every preceding E(n) writer before odd counter2n+1 is released.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

try:
    from scripts.patch_cuda_native_persistent import sha,once
except ModuleNotFoundError:
    from patch_cuda_native_persistent import sha,once


def e_batch(code,group):
    if group not in (2,4):raise ValueError('E publication group2/4 only')
    if '// TRACE_H_PUBLICATION_JOIN_BEGIN' not in code:raise ValueError('True H publication join required')
    code=once(code,'__device__ __forceinline__ void PublishAndWait(int step,int last){',
        f'constexpr int EPublicationSteps={group};\n'+
        '__device__ __forceinline__ bool ShouldPublishE(int step,int last){return step%EPublicationSteps==0||step==last;}\n'+
        '__device__ __forceinline__ void PublishAndWait(int step,int last){')
    fence='  __threadfence();\n  __syncthreads(); // RLpair'
    code=once(code,fence,'  if(ShouldPublishE(n,Fast?Depth:window[1]))__threadfence();\n  __syncthreads(); // RLpair')
    code=once(code,'  if(!threadIdx.x)PublishAndWait(2*n,2*(Fast?Depth:window[1]));',
        '  if(!threadIdx.x&&ShouldPublishE(n,Fast?Depth:window[1]))PublishAndWait(2*n,2*(Fast?Depth:window[1]));')
    # The one H fence still executes by ALL lanes, not only H surface writers.
    if '  __threadfence(); // Every H/psiH/observationH/trailH writer publishes before R0.' not in code:
        raise ValueError('Unconditional all-writer H fence required')
    return code


def patch(source,output,group):
    source,output=source.resolve(),output.resolve()
    if output.exists() or source==output or source in output.parents or output in source.parents:
        raise ValueError('New disjoint private source required')
    old=(source/'src/native_persistent.cuh').read_text();new=e_batch(old,group)
    shutil.copytree(source,output);(output/'src/native_persistent.cuh').write_text(new)
    proof={'parent_persistent_consumer_sha256':sha(old.encode()),'persistent_consumer_sha256':sha(new.encode()),
        'patcher_sha256':sha(Path(__file__).read_bytes()),'E_publication_steps':group,'expected_entries':65,
        'workspace_abi_changed':False,'H_fence_all_writers_unconditional':True,
        'fact':'H(n+1) all-writer fence+warp/CTAjoin orders every prior E(n) store before releaseOdd2n+1, so ready>=2n remains valid',
        'last_window':'explicit E fence/release mandatory at windowlast, even unaligned; completion remains a separate stamp',
        'barriers_unchanged':old.count('__syncthreads()')==new.count('__syncthreads()'),
        'requires':['fenced-prefix oracle','omitted-last-E and H-only-fence negatives','actual controller proof','resources','all30 exact','sanitizers','paired timings'],'gpu_executed':False}
    (output.parent/'persistent-E-batch-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--E-group',type=int,required=True,choices=(2,4))
    a=p.parse_args();print(json.dumps(patch(a.source,a.output,a.E_group)))
