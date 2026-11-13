"""Retain an acquired upstream readiness floor until this tile completes.

The physical slot WAR proof keeps every upstream owner stable throughout this
consumer's lifetime. An acquired ready minimum therefore authorizes all later
requests at or below that floor without another tag/ready polling round.
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


def floor_cache(code):
    code=once(code,' uint32_t epoch;',' uint32_t epoch,upstream_floor;\n int depth;')
    code=once(code,'__device__ __forceinline__ void Claim(){\n if(PollAbort())return;',
        '__device__ __forceinline__ void Claim(){\n if(PollAbort())return;\n C().upstream_floor=0;')
    code=once(code,'__device__ __forceinline__ bool WaitProducer(int producer,int owner,int step){',
        '__device__ __forceinline__ uint32_t WaitProducer(int producer,int owner,int step){')
    code=once(code,'ready=Acquire(C().ready+owner);if(ready>=uint32_t(step))return true;',
        'ready=Acquire(C().ready+owner);if(ready>=uint32_t(step))return ready;')
    code=once(code,'__device__ __forceinline__ void WaitUpstream(int step){',
        '''__device__ __forceinline__ void WaitUpstream(int step){
 if(PollAbort()||uint32_t(step)<=C().upstream_floor)return;
 uint32_t floor=C().depth;''')
    code=once(code,'   if(!WaitProducer(producer,owner,step))return;',
        '   const uint32_t seen=WaitProducer(producer,owner,step);if(C().abort_local)return;floor=seen<floor?seen:floor;')
    code=once(code,'\n }\n}\n__device__ __forceinline__ void PublishAndWait(',
        '\n }\n C().upstream_floor=floor;\n}\n__device__ __forceinline__ void PublishAndWait(') if 'constexpr int PublicationSteps=' not in code else once(code,
        '\n }\n}\nconstexpr int PublicationSteps=',
        '\n }\n C().upstream_floor=floor;\n}\nconstexpr int PublicationSteps=')
    code=once(code,'C().epoch=Acquire(p.epoch);C().tile=-1;',
        'C().epoch=Acquire(p.epoch);C().depth=Depth;C().upstream_floor=0;C().tile=-1;')
    return code


def patch(source,output):
    source,output=source.resolve(),output.resolve()
    if output.exists() or source==output or source in output.parents or output in source.parents:
        raise ValueError('New disjoint copied source required')
    old=(source/'src/native_persistent.cuh').read_text()
    if 'if(ready!=previous_ready)' not in old:raise ValueError('Requires hardened relaxed-v2 controller')
    if sha((source/'src/native_persistent.h').read_bytes())!='00b85f62ad6128a8d391128a6cc1f6d2459daf9e8fa45a91fff7c3877d44160f':
        raise ValueError('Frozen persistent workspace ABI required')
    new=floor_cache(old);shutil.copytree(source,output);(output/'src/native_persistent.cuh').write_text(new)
    proof={'parent_persistent_consumer_sha256':sha(old.encode()),'persistent_consumer_sha256':sha(new.encode()),
        'patcher_sha256':sha(Path(__file__).read_bytes()),'workspace_abi_changed':False,'expected_entries':65,
        'cache':'per-CTA cold min final-acquired parent ready, reset at every Claim; Depth if no parents',
        'proof':'Every parent physical slot reuse waits ALL old consumers including current tile completion, so acquired ready never invalidates inside tile',
        'requires':['actual controller cached-floor proof','slot WAR negative','zero resources','all30 exact','sanitizers','paired timings'],'gpu_executed':False}
    (output.parent/'persistent-upstream-floor-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(patch(a.source,a.output)))
