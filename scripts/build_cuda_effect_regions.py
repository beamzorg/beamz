"""Stage a single ready worker with traced H/E effect-region specialization."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_cuda_closed_pipeline import stage as closed_stage
from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


def stage(source,output,expected,*,retirement=False,fifo=False,specialize_program=False,distal_retirement=False):
    if retirement and distal_retirement:raise ValueError('One explicit retirement protocol required')
    output=Path(output).resolve()
    if retirement:
        from scripts.build_cuda_dead_face_pipeline import stage as retirement_stage
        proof=retirement_stage(source,output,expected,single_kernel=True)
    else:proof=closed_stage(source,output,expected,front_startup=True,single_kernel=True)
    src=output/'src'
    if distal_retirement:
        from scripts.cuda_native_distal_disposal import bind
        bind(output,proof)
    c=(src/'asymmetric_native.cuh').read_text();a,b,body=block(c,'void UniformBody(')
    body=once(body,'void UniformBody(const Parameters&p,const int32_t*tiles){',
              'void TracedEffectRegionsBody(const Parameters&p){\n constexpr int Layout=1,Policy=5,BlockT=1024,ActiveAxes=-1;constexpr bool Fast=false,IsotropicH=true,UniformE=false,CachedProfiles=true,Pipeline=true;')
    body=once(body,' static_assert(!CachedProfiles||(Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE),"Profile staging only certified NoIO1024");','')
    body=once(body,'const int tile=Pipeline?closed_pipeline::Tile():tiles[blockIdx.x],','const int tile=closed_pipeline::Tile(),')
    body=once(body,'const uint8_t flags=Policy==0?0:Policy==2?1:Policy==3?16:Policy==4?17:p.flags[tile];','const uint8_t flags=p.flags[tile];')
    body=once(body,' Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,CachedProfiles,Pipeline>env{p,flags,profiles};','')
    # Both arithmetic regions have identical register ownership before/after
    # their specialization. Imports/publication/barriers occur only once.
    regions=[]
    for phase in range(2):
        marker='{int step;asm volatile("mov.b32 %0,%1;":'
        search=0 if not regions else regions[-1]['end']
        begin=body.index(marker,search);_,end,region=block(body[begin:],marker)
        end+=begin
        declarations=[
            'TracedRuntimeCoefficient<Environment<1,true,true,2,false,-1,false,-1,false,true>>env{{p,flags,nullptr}};',
            'TracedRuntimeCoefficient<Environment<1,true,true,4,false,20,true,-1,false,true>>env{{p,flags,nullptr}};',
            'TracedBoundaryEffects<CubeN>env{{{p,flags,profiles}}};']
        inner=region[1:-1];generated=[]
        for i,decl in enumerate(declarations):
            prefix='if(!(flags&62u))' if i==0 else 'else if(!(flags&46u))' if i==1 else 'else'
            if specialize_program and i==2:
                alternatives=[]
                for bits,axes in ((2,1),(4,2),(8,4),(None,-1)):
                    branch=f'if((flags&14u)=={bits}u)' if bits==2 else f'else if((flags&14u)=={bits}u)' if bits else 'else'
                    alternatives.append(branch+'{'+f'TracedBoundaryEffects<CubeN,{axes},true>env{{{{p,flags,profiles}}}};'+inner+'}')
                emitted='\n'.join(alternatives)
            elif specialize_program and phase==1:
                effect=2 if i==0 else 4;mask='-1,false' if i==0 else '20,true'
                def typed(uniform):return f'Environment<1,true,true,{effect},{uniform},{mask},-1,false,true>env{{p,flags,nullptr}};'
                emitted='if(flags&64u){'+typed('true')+inner+'}else{'+typed('false')+inner+'}'
            else:emitted=decl+inner
            generated.append(prefix+'{'+emitted+'}')
        regions.append(dict(begin=begin,end=end,phase=phase,original=region,lowered='\n'.join(generated)))
    for r in reversed(regions):body=body[:r['begin']]+r['lowered']+body[r['end']:]
    clone='template<int Depth,int CubeN>\n__device__ __forceinline__ '+body+'\n'
    marker='#include "native_closed_pipeline_kernel.cuh"'
    c=once(c,marker,'#include "native_effect_regions.cuh"\n'+clone+'\n'+marker)
    (src/'asymmetric_native.cuh').write_text(c)
    k=(src/'native_closed_pipeline_kernel.cuh').read_text()
    k=once(k,'  else ClosedDispatch<Depth,Policy,CubeN>(p);','  else TracedEffectRegionsBody<Depth,CubeN>(p);');(src/'native_closed_pipeline_kernel.cuh').write_text(k)
    helper=Path(__file__).with_name('cuda_native_effect_regions.cuh').read_bytes();(src/'native_effect_regions.cuh').write_bytes(helper)
    proof.update(effect_regions=True,consumer_sha256=hashlib.sha256(c.encode()).hexdigest(),effect_region_lowering=regions,
                 field_layout='One Uniform native1024 register/shared layout; imports/window/publication occur once',
                 effect_partitions=['No CPML/source/nativewall (optional observations)','No CPML/nativewall (source20 and optional observations)','General CPML/nativewall/source20/observations'],
                 coefficient_fact='Captured per-tile flag64 selects identical common E coefficients; other points read canonical packed material coefficients')
    proof['effect_program_specialization']=specialize_program
    for name in (*proof['helpers'],'native_effect_regions.cuh'):proof['helpers'][name]=hashlib.sha256((src/name).read_bytes()).hexdigest()
    if fifo:
        from scripts.cuda_native_fifo_control import bind
        bind(output,proof)
    (output.parent/'closed-pipeline-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return {k:v for k,v in proof.items() if k not in ('body_bindings','effect_region_lowering')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);p.add_argument('--retirement',action='store_true');p.add_argument('--fifo',action='store_true');p.add_argument('--specialize-program',action='store_true');p.add_argument('--distal-retirement',action='store_true');a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256,retirement=a.retirement,fifo=a.fifo,specialize_program=a.specialize_program,distal_retirement=a.distal_retirement)))
