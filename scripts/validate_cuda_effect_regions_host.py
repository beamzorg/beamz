"""Validate region-body ancestry and actual emitted effect environments on CPU."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


def validate(candidate):
    candidate=Path(candidate).resolve();src=candidate/'source/src';proof=json.loads((candidate/'closed-pipeline-compiler-proof.json').read_text());c=(src/'asymmetric_native.cuh').read_text();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if not proof.get('effect_regions') or sha(src/'asymmetric_native.cuh')!=proof['consumer_sha256']:raise ValueError('Bound effect-region compiler source required')
    for name,value in proof['helpers'].items():
        if sha(src/name)!=value:raise ValueError('Staged helper changed')
    body=block(c,'void TracedEffectRegionsBody(')[2]
    declarations=[
        'TracedRuntimeCoefficient<Environment<1,true,true,2,false,-1,false,-1,false,true>>env{{p,flags,nullptr}};',
        'TracedRuntimeCoefficient<Environment<1,true,true,4,false,20,true,-1,false,true>>env{{p,flags,nullptr}};',
        'TracedBoundaryEffects<CubeN>env{{{p,flags,profiles}}};']
    for phase,r in enumerate(proof['effect_region_lowering']):
        inner=r['original'][1:-1];branches=[]
        for i,decl in enumerate(declarations):
            condition='if(!(flags&62u))' if i==0 else 'else if(!(flags&46u))' if i==1 else 'else'
            if proof.get('effect_program_specialization') and i==2:
                options=[]
                for bits,axes in ((2,1),(4,2),(8,4),(None,-1)):
                    test=f'if((flags&14u)=={bits}u)' if bits==2 else f'else if((flags&14u)=={bits}u)' if bits else 'else'
                    options.append(test+'{'+f'TracedBoundaryEffects<CubeN,{axes},true>env{{{{p,flags,profiles}}}};'+inner+'}')
                emitted='\n'.join(options)
            elif proof.get('effect_program_specialization') and phase==1:
                effect=2 if i==0 else 4;mask='-1,false' if i==0 else '20,true'
                def typed(uniform):return f'Environment<1,true,true,{effect},{uniform},{mask},-1,false,true>env{{p,flags,nullptr}};'
                emitted='if(flags&64u){'+typed('true')+inner+'}else{'+typed('false')+inner+'}'
            else:emitted=decl+inner
            branches.append(condition+'{'+emitted+'}')
        expected='\n'.join(branches)
        if r['lowered']!=expected:raise ValueError('Only complete identical H/E regions may specialize')
        body=once(body,expected,r['original'])
    body=once(body,'void TracedEffectRegionsBody(const Parameters&p){\n constexpr int Layout=1,Policy=5,BlockT=1024,ActiveAxes=-1;constexpr bool Fast=false,IsotropicH=true,UniformE=false,CachedProfiles=true,Pipeline=true;',
              'void UniformBody(const Parameters&p,const int32_t*tiles){')
    body=once(body,' constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);\n\n',
              ' constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);\n static_assert(!CachedProfiles||(Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE),"Profile staging only certified NoIO1024");\n')
    body=once(body,'const int tile=closed_pipeline::Tile(),','const int tile=Pipeline?closed_pipeline::Tile():tiles[blockIdx.x],')
    body=once(body,' const uint8_t flags=p.flags[tile];\n\n',
              ' const uint8_t flags=Policy==0?0:Policy==2?1:Policy==3?16:Policy==4?17:p.flags[tile];\n Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,CachedProfiles,Pipeline>env{p,flags,profiles};\n')
    if body!=block(c,'void UniformBody(')[2]:raise ValueError('Exact field-layout/body inverse failed')
    primitives=(src/'yee_primitives.cuh').read_text();parts=[]
    for signature in ('constexpr int CpmlAxis(','float ScaleYeeDifference(','float PackedMaterialSource('):
        a,b,v=block(primitives,signature)
        if 'PackedMaterial' in signature:v='template<class Index>\n'+v
        parts.append(v)
    environment=block(c,'struct Environment {')[2]+';'
    a=c.index('struct Environment {');environment=c[c.rfind('template<',0,a):a]+environment
    fixed=block(c,'struct FixedCubeCachedEnvironment:')[2]+';'
    a=c.index('struct FixedCubeCachedEnvironment:');fixed=c[c.rfind('template<',0,a):a]+fixed
    methods='\n'.join(block(c,s)[2] for s in ('int IndependentProfileCacheIndex(','float ReadIndependentProfileCache('))
    code=Path(__file__).with_name('prove_cuda_effect_regions_host.cc').read_text().replace('// BEAMZ_EFFECT_CANONICAL_PRIMITIVES','\n'.join(parts)).replace('// BEAMZ_EFFECT_ACTUAL_ENVIRONMENTS',methods+'\n'+environment+'\n'+fixed)
    out=candidate/'host-effect-regions-proof';out.mkdir(exist_ok=False);source=out/'proof.cc';source.write_text(code);binary=out/'proof'
    defines=['-DBEAMZ_EFFECT_PROGRAM_UNIFORM_GENERAL=1'] if proof.get('effect_program_specialization') else []
    command=['g++','-std=c++20','-O2','-ffp-contract=off','-fsanitize=address,undefined',*defines,'-I'+str(src),str(source),str(src/'native_moving.cc'),'-o',str(binary)]
    built=subprocess.run(command,capture_output=True,text=True);(out/'build.log').write_text(built.stdout+built.stderr)
    if built.returncode:raise RuntimeError('Effect host compile failed; inspect '+str(out/'build.log'))
    run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=180);(out/'run.log').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError('Effect method/fact proof failed; inspect '+str(out/'run.log'))
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],helpers=proof['helpers'],field_body_exact_inverse=True,all_import_publication_and_barrier_sites_preserved=True,actual_methods=json.loads(run.stdout),command=command)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate)))
