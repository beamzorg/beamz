"""Add a typed proper-cycle source specialization to the current frozen engine.

The coordinate/storage pass lives in cuda_cyclic_fixture.py and preserves world
observation IDs. This binding admits the corresponding Z-normal source class;
it keeps the existing source20 entry bodies and all NoIO/bulk kernels unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.cuda_trace_phase_cycle import block
from scripts.patch_cuda_cyclic_source import FACT10


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Unique cyclic architecture binding required: '+old[:100])
    return text.replace(old,new,1)


def declaration(text,signature):
    start,end,_=block(text,signature)
    start=text.rfind('template<',0,start)
    return text[start:end]


def stage(source,output,expected):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or source in output.parents or output in source.parents:raise ValueError('New disjoint research source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if hashlib.sha256(original.encode()).hexdigest()!=expected:raise ValueError('Explicit frozen parent identity required')
    text=once(original,'SourceMask==20&&(Policy==3||Policy==4||Policy==5)',
                       '(SourceMask==20||SourceMask==10)&&(Policy==3||Policy==4||Policy==5)')
    start,end,old=block(text,'if constexpr(PlanePair)')
    body=old[old.index('{')+1:-1]
    plane10='''
     if constexpr(SourceMask==10){
      if(z==s.starts[0]&&(Policy!=5||(y>=12&&x>=12&&y<12+s.ny&&x<12+s.nx))){
       int time=*s.current_step+p.macro_step+version-1;time=max(0,min(time,s.ntime-1));
       const int offset=(y-12)*s.nx+(x-12);
       value=__fadd_rn(value,__fmul_rn(__ldg(s.coefficients+offset),__ldg(s.waveforms+time)));
       value=__fadd_rn(value,__fmul_rn(__ldg(s.coefficients+s.ny*s.nx+offset),__ldg(s.waveforms+s.ntime+time)));
      }
     }else{
'''
    text=text[:start]+'if constexpr(PlanePair){'+plane10+body+'\n     }\n    }'+text[end:]
    pieces=[declaration(original,'void ResearchSourceFacts('),
            declaration(original,'void AsymmetricSourcePending20('),
            declaration(original,'void AsymmetricSourcePlanePair20(')]
    added='\n'.join(pieces).replace('ResearchSourceFacts','ResearchSourceFacts10').replace('AsymmetricSourcePending20','AsymmetricSourcePending10').replace('AsymmetricSourcePlanePair20','AsymmetricSourcePlanePair10').replace(',1024,20',',1024,10')
    text=once(text,'inline bool MakeSource(',added+'\n\ninline bool MakeSource(')
    start,end,launch=block(text,'inline cudaError_t LaunchSourcePending(')
    launch=launch[launch.index('{')+1:-1]
    condition='if(BeamzResearchSourcePendingEnabled&&BeamzNativeMovingSourceMask20Class(Layout,IsotropicH,presence))'
    substart,subend,sub=block(launch,condition)
    dispatch=sub[sub.index('{')+1:-1].replace('AsymmetricSourcePlanePair20','AsymmetricSourcePlanePair10').replace('AsymmetricSourcePending20','AsymmetricSourcePending10')
    anchor=condition+'{'
    text=once(text,anchor,anchor+'\n   if(presence==10u){'+dispatch+'\n   }\n')
    # Existing entry bodies remain source-identical; only a new template
    # specialization sees SourceMask10 and its ordered source-address adapter.
    for signature in ('void UniformBody(','void FixedCubeCachedBody(',
                      'void SourcePendingPairedBody(','void AsymmetricSourcePending20(',
                      'void AsymmetricSourcePlanePair20(','void Observe('):
        if block(original,signature)[2]!=block(text,signature)[2]:raise ValueError('Existing body changed: '+signature)
    header=(source/'src/native_moving.h').read_text()
    header=once(header,' return layout==1&&isotropic&&presence==20u;',
                       ' return layout==1&&isotropic&&(presence==20u||presence==10u);')
    header=once(header,'// CUDA consumer owns the private epoch increment',FACT10+'\n// CUDA consumer owns the private epoch increment')
    extension=(source/'src/native_extension.cc').read_text()
    old='if(BeamzNativeSourcePlanePairFact(validate.program.source_groups,validate.program.source_group_count,plane_starts,minimum,moving_workspace_.source_presence_mask))moving_workspace_.data_flags|=8u;'
    new=old.replace('))moving_workspace_',')||BeamzNativeSourcePlanePairFact10(validate.program.source_groups,validate.program.source_group_count,plane_starts,minimum,moving_workspace_.source_presence_mask))moving_workspace_')
    extension=once(extension,old,new)
    extension=once(extension,'NB_MODULE(_cuda_native, module) {','''NB_MODULE(_cuda_native, module) {
 module.attr("moving_source_mask10_generated")=true;
 module.attr("moving_cyclic_world_ordered_observation")=true;
''')
    shutil.copytree(source,output)
    (output/'src/asymmetric_native.cuh').write_text(text)
    (output/'src/native_moving.h').write_text(header)
    (output/'src/native_extension.cc').write_text(extension)
    proof=dict(schema='beamz.current-cyclic-source-lowering.v1',parent_consumer_sha256=expected,
               consumer_sha256=hashlib.sha256(text.encode()).hexdigest(),
               expected_kernels=80,existing_entry_bodies_unchanged=True,source_mask=10,
               requires='Typed proper-cycle storage transform; world raw observation IDs and independent inverse-world output comparisons',
               ordered_source_rounding='Two independent RN products and sequential RN adds',
               CUDA_executed=False,generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output.parent/'cyclic-layout-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True)
    a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
