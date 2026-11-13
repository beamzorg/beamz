"""Copy-only compile-time grouping of persistent release fences/readiness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.patch_cuda_native_persistent_specialize import patch_specialized
    from scripts.patch_cuda_native_persistent import patch_copy, sha, once
except ModuleNotFoundError:
    from patch_cuda_native_persistent_specialize import patch_specialized
    from patch_cuda_native_persistent import patch_copy, sha, once


def batch(code,group):
    if group not in (1,2,4,16):raise ValueError('Publication group must be 1/2/4/16')
    code=once(code,'__device__ __forceinline__ void PublishAndWait(int step,int last){',
        f'constexpr int PublicationSteps={group};\n'+
        '__device__ __forceinline__ bool ShouldPublish(int step,int last){return step%PublicationSteps==0||step==last;}\n'+
        '__device__ __forceinline__ void PublishAndWait(int step,int last){')
    code=once(code,' Release(C().ready+C().owner,uint32_t(step));',
        ' if(ShouldPublish(step,last))Release(C().ready+C().owner,uint32_t(step));')
    # No CTA barriers removed. At each grouped publication every writer orders
    # all its preceding stores; readiness remains release-before-upstream-wait.
    first='  __threadfence();\n  __syncthreads(); // RLpair'
    code=once(code,first,'  if(ShouldPublish(n,window[1]))__threadfence();\n  __syncthreads(); // RLpair')
    second='  __threadfence();\n  __syncthreads(); // B3'
    if second in code:
        code=once(code,second,'  if(ShouldPublish(n,Fast?Depth:window[1]))__threadfence();\n  __syncthreads(); // B3')
    return code


def patch_batch(source,output,group=4,specialized=True):
    proof=(patch_specialized if specialized else patch_copy)(source,output)
    path=output/'src/native_persistent.cuh';old=path.read_text();new=batch(old,group)
    path.write_text(new)
    proof.update({'publication_steps':group,'grouped_specialized':specialized,
        'unbatched_consumer_sha256':sha(old.encode()),'persistent_consumer_sha256':sha(new.encode()),
        'batch_patcher_sha256':sha(Path(__file__).read_bytes()),'workspace_abi_changed':False,
        'fence_policy':'every writer before existing retire barrier iff n%G==0 or last; all prior stores published',
        'readiness_policy':'release n only if publication; upstream wait remains exact n+1 every iteration; end-window mandatory',
        'requires':['CPU grouped controller and actual control proof G1/2/4/16','zero resources','all30 exact first/replay','sanitizers','paired timings']})
    (output.parent/'persistent-batched-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--publication-steps',type=int,default=4,choices=(1,2,4,16));p.add_argument('--p5-only',action='store_true')
    a=p.parse_args();print(json.dumps(patch_batch(a.source.resolve(),a.output.resolve(),a.publication_steps,not a.p5_only)))
