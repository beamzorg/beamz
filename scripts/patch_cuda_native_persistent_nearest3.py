"""Full-step-only transitive readiness: nearest3 later, ALL7 at startup.

Exact accepted parent has cumulative full-step publication AFTER its current
upstream wait. Early H/E publication cannot use this pass. Host9b and the
workspace ABI are copied byte-for-byte; all7 old-consumer WAR waits remain.
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


PARENT='f76604eede410c12d37ae0a5e0759be72f3c2c05cae6910cb9716922483bc742'
HOST='9b8afea28037f21e893dd74153e407c6d1cea32bc34f3ed45f4d37775ab6f015'
ABI='00b85f62ad6128a8d391128a6cc1f6d2459daf9e8fa45a91fff7c3877d44160f'


def nearest3(code):
    if sha(code.encode())!=PARENT:raise ValueError('Exact fullstep cumulative f766 parent only; early phase-ready is forbidden')
    if 'TRACE_PHASE_READY_COUNTER' in code or 'TRACE_IMPORT_GATE' in code:
        raise ValueError('Early phase publications do not imply current upstream completion')
    code=once(code,'__device__ __forceinline__ void WaitUpstream(int step){',
        '// TRACE_NEAREST3_LATER_ONLY: fullstep publication; startup always ALL7.\n'+
        'template<bool NearestOnly=false>\n__device__ __forceinline__ void WaitUpstream(int step){')
    a=code.index('template<bool NearestOnly=false>');b=code.index('constexpr int PublicationSteps=',a)
    wait=code[a:b]
    wait=once(wait,'for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;',
        'for(int neighbor=1;neighbor<(NearestOnly?4:8);++neighbor){const int bits=NearestOnly?(1<<(neighbor-1)):neighbor;const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;')
    code=code[:a]+wait+code[b:]
    code=once(code,'if(step<last)WaitUpstream(step+1);','if(step<last)WaitUpstream<true>(step+1);')
    if code.count('WaitUpstream(window[0]);')!=2:raise ValueError('Expected BOTH paired and Uniform startup waits')
    code=code.replace('WaitUpstream(window[0]);','WaitUpstream<false>(window[0]);')
    code=once(code,'WaitUpstream(1);','WaitUpstream<false>(1);')
    return code


def patch(source,output):
    source,output=source.resolve(),output.resolve()
    if output.exists() or source==output or source in output.parents or output in source.parents:
        raise ValueError('New disjoint private source required')
    old=(source/'src/native_persistent.cuh').read_text()
    if sha((source/'src/native_extension.cc').read_bytes())!=HOST:raise ValueError('Root9b native host must remain exact')
    if sha((source/'src/native_persistent.h').read_bytes())!=ABI:raise ValueError('Frozen persistent host/workspace ABI required')
    new=nearest3(old);shutil.copytree(source,output);(output/'src/native_persistent.cuh').write_text(new)
    proof={'parent_consumer_sha256':PARENT,'parent_module_sha256':'eb229e671d91948ff1b76906c54e80165ffaa98c2d3cd08d1ff9fae56204a323',
        'consumer_sha256':sha(new.encode()),'host_sha256':HOST,'workspace_header_sha256':ABI,'patcher_sha256':sha(Path(__file__).read_bytes()),
        'contract':'ALL7 startup atfirst>=1, includinglatefirst; ONLYlater nextstep uses nearestaxes1/2/4. ALL7 WAR and unconditional base tag validation retained.',
        'geometry':'For supported diagonal field payload atactive consumer n, every intermediate box on an upwind-axis path isactive n; fullstepready carries transitive readiness/cumulativevisibility.',
        'mutable_state':'Fresh psi/observer ownership needs priorversionn-1. Active face atn or its completed priorversionn-1 carries the diagonal prior-state publication; later-only ensurescurrentfirst<=n-1.',
        'not_applicable':'Early H/E phase-ready releases precede current upstream completion; pass rejects all other source hashes.',
        'CUDA_body_FP_joins_and_host_unchanged':True,'expected_entries':65,'GPU_executed':False,
        'requires':['supported-payload/intermediatebox proof','all7 startup/latefirst regression','rejected-v5 negative','actual controller matrix','root8B build seal','all30 exact GPU gates andscreen']}
    (output.parent/'nearest3-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(patch(a.source,a.output)))
