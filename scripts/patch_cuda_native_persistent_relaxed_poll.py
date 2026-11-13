"""Copy-only atomic polling policy: relaxed failures, acquire on success.

Mutable payload remains .cg. Release publications and final successful acquire
checks are retained; only failed/progress polling avoids acquire invalidation.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    from scripts.patch_cuda_native_persistent import sha,once
    from scripts.patch_cuda_permanent_family_frontier import PARENT_HASHES
except ModuleNotFoundError:
    from patch_cuda_native_persistent import sha,once
    from patch_cuda_permanent_family_frontier import PARENT_HASHES


def relaxed_poll(code):
    code=once(code,'template<class T>__device__ __forceinline__ void Release(T*word,T value){',
        '''template<class T>__device__ __forceinline__ T Relaxed(const T*word){
 ::cuda::atomic_ref<T,::cuda::thread_scope_device>a(*const_cast<T*>(word));return a.load(::cuda::memory_order_relaxed);
}
template<class T>__device__ __forceinline__ void Release(T*word,T value){''')
    code=once(code,'if(Acquire(C().abort_global)){','if(Relaxed(C().abort_global)){')
    code=once(code,
        'for(;;){if(PollAbort())return false;const uint32_t seen=Acquire(C().completed+tile);\n  if(seen==C().epoch)return true;',
        '''for(;;){if(PollAbort())return false;const uint32_t seen=Relaxed(C().completed+tile);
  if(seen==C().epoch&&Acquire(C().completed+tile)==C().epoch)return true;''')
    code=once(code,'const uint64_t tag=Acquire(C().tags+owner);','const uint64_t tag=Relaxed(C().tags+owner);')
    code=once(code,'  if(tag==expected){',
        '  if(tag!=previous_tag){previous_tag=tag;last=Timer();}\n  if(tag==expected){')
    code=once(code,'  }else if(tag!=previous_tag){previous_tag=tag;last=Timer();}','  }')
    code=once(code,
        'const uint32_t ready=Acquire(C().ready+owner);\n   if(ready>=uint32_t(step))return true;',
        '''uint32_t ready=Relaxed(C().ready+owner);
   // Speculative relaxed observations NEVER authorize payload reads. Final
   // acquire-tag precedes final acquire-ready, ordering the holder's reset.
   if(ready>=uint32_t(step)&&Acquire(C().tags+owner)==expected){
    ready=Acquire(C().ready+owner);if(ready>=uint32_t(step))return true;
   }''')
    code=once(code,'if(ready>previous_ready){previous_ready=ready;last=Timer();}',
        'if(ready!=previous_ready){previous_ready=ready;last=Timer();}')
    return code


def patch(source,output):
    source,output=source.resolve(),output.resolve()
    if output.exists() or source==output or source in output.parents or output in source.parents:
        raise ValueError('New disjoint private source required')
    old=(source/'src/native_persistent.cuh').read_text()
    if sha((source/'src/native_persistent.h').read_bytes())!='00b85f62ad6128a8d391128a6cc1f6d2459daf9e8fa45a91fff7c3877d44160f':
        raise ValueError('Frozen persistent workspace ABI required')
    asymmetric=(source/'src/asymmetric_native.cuh').read_text()
    suffix='\n#include "native_persistent.cuh"\n'
    if not asymmetric.endswith(suffix) or sha(asymmetric[:-len(suffix)].encode())!=PARENT_HASHES['asymmetric_native.cuh']:
        raise ValueError('Byte-identical selected52865 old consumer required')
    new=relaxed_poll(old)
    shutil.copytree(source,output);(output/'src/native_persistent.cuh').write_text(new)
    proof={'parent_persistent_consumer_sha256':sha(old.encode()),'persistent_consumer_sha256':sha(new.encode()),
        'patcher_sha256':sha(Path(__file__).read_bytes()),'workspace_abi_changed':False,
        'old_consumer_unchanged':True,'expected_entries':65,'gpu_executed':False,
        'policy':'atomic relaxed progress/failure polls; final successful acquire completion, tag THEN ready; all release stores unchanged',
        'scope':'Payload .cg and FP/source/observer statements unchanged; abort only control',
        'requires':['actual controller proof','source/acquire-order proof','zero resources','all30 first/replay exact','sanitizers','paired timings']}
    (output.parent/'persistent-relaxed-poll-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(patch(a.source,a.output)))
