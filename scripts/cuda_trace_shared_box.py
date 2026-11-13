"""Combine smaller cache ownership with a two-barrier all-shared field schedule.

The seed body is the previously validated all-shared schedule. The new design
fits two complete H/E caches per SM, leaving registers available for arithmetic
and optional future CPML histories. Optional profile staging is disabled so it
does not cross the two-block shared-memory capacity boundary.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,block,sha

SEED=ROOT/'.cache/cuda-trace-five-hour-2026-11-13-1059/shared-fields-v1'


def stage(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:
        raise ValueError('New disjoint copied source required')
    shape=json.loads((source.parent/'box-proof.json').read_text())
    if shape['owned_zyx']!=[7,7,31] or shape['threads']!=512:
        raise ValueError('Initial experiment admits only7x7x31/512 mapping')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=shape['consumer_sha256']:raise ValueError('Copied box identity changed')
    seed=(SEED/'source/src/asymmetric_native.cuh').read_text()
    manifest=json.loads((SEED/'manifest.json').read_text())
    if sha(seed.encode())!=manifest['source_hashes']['src/asymmetric_native.cuh']:
        raise ValueError('Validated seed schedule identity changed')
    _,_,body=block(seed,'void PermanentFamilyBody(')
    body=body[body.index('{')+1:-1]
    body=body.replace('constexpr int T=(Fast||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512,Slots=PAD/T,ZStep=T/512;',
                      'constexpr int T=ResearchThreads,Slots=PAD/T,ZStep=T/(CY*CX);')
    body=body.replace(' static_assert(T==1024,"Permanent-family frontier requires admitted1024-thread owner mapping");','')
    body=body.replace('threadIdx.x/512','threadIdx.x/(CY*CX)')
    body=body.replace('&7','&(CZ-1)').replace('&15','&(CY-1)')
    body=body.replace('zs==7?','zs==(CZ-1)?').replace('sy==15?','sy==(CY-1)?')
    if 'constexpr int T=ResearchThreads' not in body:raise ValueError('Seed thread mapping changed')
    # Separate family clocks retain the known rematerialization boundary.
    text=original
    for signature in ('void UniformBody(','void FixedCubeCachedBody('):
        begin,end,old=block(text,signature)
        prefix=old[:old.index('{')+1]
        newbody=body
        if 'FixedCube' in signature:
            newbody='\n constexpr int Layout=1,Policy=1;constexpr bool Fast=false,IsotropicH=true,UniformE=true;'+body
            env='Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes>env{p,flags};'
            if newbody.count(env)!=1:raise ValueError('Seed environment changed')
            newbody=newbody.replace(env,'FixedCubeCachedEnvironment<N,ActiveAxes>env{p,flags,nullptr};')
        text=text[:begin]+prefix+newbody+'}'+text[end:]
    text=text.replace('Environment<1,false,true,1,true,-1,false,ActiveAxes,true>',
                      'Environment<1,false,true,1,true,-1,false,ActiveAxes,false>')
    text=text.replace('if(kind>=2)return ReadIndependentProfileCache(profiles,index,kind,Packed(index,z,y,x));',
                      'if(kind>=2){const auto&psi=p.phase[index/6].psi[index%6];const int packed=Packed(index,z,y,x);return kind==2?__ldg(psi.a+packed):kind==3?__ldg(psi.b+packed):__ldg(psi.inv+packed);}')
    text=text.replace('3ULL*PAD*sizeof(float)+8*sizeof(size_t)+3*sizeof(int)+6ULL*LowXPoints*sizeof(float)',
                      '6ULL*PAD*sizeof(float)+8*sizeof(size_t)+4*sizeof(int)+6ULL*LowXPoints*sizeof(float)')
    text=text.replace('ProfileCachedSharedBytes=SharedBytes+864ULL*sizeof(float)','ProfileCachedSharedBytes=SharedBytes')
    text=text.replace('FixedCubeSharedBytes=SharedBytes+864ULL*sizeof(float)','FixedCubeSharedBytes=SharedBytes')
    text=text.replace('FixedCubeSharedBytes==29284','FixedCubeSharedBytes==50408')
    text=text.replace('SharedBytes==25828&&ProfileCachedSharedBytes==29284','SharedBytes==50408&&ProfileCachedSharedBytes==50408')
    # The body has no optional shared-profile staging or private field arrays.
    if 'float h[3][Slots],e[3][Slots]' in text:raise ValueError('Private field array remains')
    shutil.copytree(source,output)
    (output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.two-resident-shared-cache.v1',parent_consumer_sha256=sha(original.encode()),
               consumer_sha256=sha(text.encode()),seed_consumer_sha256=sha(seed.encode()),
               producer_sha256=sha(Path(__file__).read_bytes()),threads=512,owned_zyx=[7,7,31],
               shared_bytes=50408,two_blocks_shared_bytes=100816,sm86_shared_capacity=102400,
               persistent_private_field_words=0,barriers_per_step=2,profiles_independent_global=True,
               source_and_DFT_unchanged=True,expected_kernels=56,GPU_executed=False)
    (output.parent/'shared-box-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(stage(a.source,a.output)))
