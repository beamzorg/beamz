"""Specialize policy while preserving the paired P5 storage schedule.

This alternative avoids introducing the register-heavier UniformBody at all.
Every selected body retains paired storage, original five barriers and ordered
scalar graph; only captured policy/support/common-E facts are specialized.
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


def paired_specialize(code,include_noio=False):
    if 'void PersistentUniformBody' in code:raise ValueError('P5-only persistent source required')
    code=once(code,'int tile,owner,abort_local;','int tile,owner,abort_local,body_kind;')
    code=once(code,'struct PersistentEnvironment:Environment<1,false,true,5,false,20,false>{',
        'template<bool Fast,int Policy,bool UniformE>\nstruct PersistentPairedEnvironment:Environment<1,Fast,true,Policy,UniformE,20,false>{')
    code=once(code,'using Base=Environment<1,false,true,5,false,20,false>;',
        'using Base=Environment<1,Fast,true,Policy,UniformE,20,false>;')
    code=once(code,'PersistentEnvironment(const Parameters&physics,uint8_t facts)',
        'PersistentPairedEnvironment(const Parameters&physics,uint8_t facts)')
    code=once(code,'template<int Depth>\n__device__ __forceinline__ void PersistentPairedBody(const Parameters&p){\n constexpr int Layout=1,Policy=5,BlockT=1024,SourceMask=20;\n constexpr bool Fast=false,IsotropicH=true,UniformE=false,PlanePair=false;',
        'template<int Depth,bool Fast,int Policy,bool UniformE>\n__device__ __forceinline__ void PersistentPairedBody(const Parameters&p){\n constexpr int Layout=1,BlockT=1024,SourceMask=20;\n constexpr bool IsotropicH=true,PlanePair=false;')
    phase_ready='// TRACE_PHASE_READY_COUNTER:' in code
    previous_start='// TRACE_STARTUP_PREVIOUS_VERSION' in code
    fast_start=0 if previous_start else 1
    code=once(code,' if(!threadIdx.x)Claim();\n __syncthreads();',
        f' if(!threadIdx.x){{Claim();if constexpr(Fast){{if(!C().abort_local)WaitUpstream({fast_start});}}}}\n __syncthreads();')
    code=once(code,'const uint8_t flags=Policy==0?0:Policy==2?1:Policy==3?16:Policy==4?17:p.flags[tile];',
        'const uint8_t flags=p.flags[tile];')
    code=once(code,'PersistentEnvironment env{p,flags};',
        'PersistentPairedEnvironment<Fast,Policy,UniformE> env{p,flags};')
    if not phase_ready:
        code=once(code,'if(ShouldPublish(n,window[1]))__threadfence();',
            'if(ShouldPublish(n,Fast?Depth:window[1]))__threadfence();')
        code=once(code,'PublishAndWait(n,window[1]);','PublishAndWait(n,Fast?Depth:window[1]);')
    else:
        code=once(code,'PublishAndWait(2*n-1,2*window[1]);','PublishAndWait(2*n-1,2*(Fast?Depth:window[1]));')
        code=once(code,'PublishAndWait(2*n,2*window[1]);','PublishAndWait(2*n,2*(Fast?Depth:window[1]));')
    selector='!(facts&63)?0:(!(facts&17)&&(facts&64))?1:2' if include_noio else '!(facts&63)?0:2'
    code=once(code,'else{const int job=atomicAdd(C().next,1);C().tile=job<C().count?C().jobs[job]:-1;}',
        'else{const int job=atomicAdd(C().next,1);C().tile=job<C().count?C().jobs[job]:-1;\n'+
        f'    if(C().tile>=0){{const uint8_t facts=p.flags[C().tile];C().body_kind={selector};}}\n   }}')
    dispatch='  if(C().body_kind==0)PersistentPairedBody<Depth,true,0,false>(p);\n'
    if include_noio:dispatch+='  else if(C().body_kind==1)PersistentPairedBody<Depth,false,1,true>(p);\n'
    dispatch+='  else PersistentPairedBody<Depth,false,5,false>(p);'
    code=once(code,'  PersistentPairedBody<Depth>(p);',dispatch)
    return code


def patch(source,output,include_noio=False):
    source,output=source.resolve(),output.resolve()
    if output.exists() or source==output or source in output.parents or output in source.parents:
        raise ValueError('New disjoint private source required')
    old=(source/'src/native_persistent.cuh').read_text();new=paired_specialize(old,include_noio)
    shutil.copytree(source,output);(output/'src/native_persistent.cuh').write_text(new)
    proof={'parent_persistent_consumer_sha256':sha(old.encode()),'persistent_consumer_sha256':sha(new.encode()),
        'patcher_sha256':sha(Path(__file__).read_bytes()),'workspace_abi_changed':False,'expected_entries':65,
        'storage':'ALL branches original paired P5 storage and five barriers; no UniformBody',
        'policies':[0,1,5] if include_noio else [0,5],'NoIO_commonE_specialized':include_noio,
        'facts':'bulk low6==0 proves complete supported trajectory; NoIO low6 policy1 AND captured commonEbit64',
        'requires':['zero resources','actual control proof','all30 exact','sanitizers','paired timings'],'gpu_executed':False}
    (output.parent/'persistent-paired-specialization-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--noio',action='store_true')
    a=p.parse_args();print(json.dumps(patch(a.source,a.output,a.noio)))
