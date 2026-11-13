"""Specialize traced coincident plane sources inside CPML/mixed tiles.

The existing host fact proves two coincident sources, one X plane and the
transverse [12,N-12) rectangle. Unlike the old CPML-free specialization, this
lowering retains explicit Z/Y bounds. It preserves both split-round additions,
independent waveform/coefficients and the absolute phase clock.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

from scripts.cuda_trace_column_schedule import PARENT,PARENT_SHA
from scripts.cuda_trace_phase_cycle import ROOT,block,sha


def transform(text):
    old='static_assert(!PlanePair||(SourceMask==20&&(Policy==3||Policy==4)),"Plane source requires exact CPML-free class");'
    new='static_assert(!PlanePair||(SourceMask==20&&(Policy==3||Policy==4||Policy==5)),"Plane source requires certified coincident topology; Policy5 keeps transverse bounds");'
    if text.count(old)!=1:raise ValueError('Exact old plane-source admission required')
    text=text.replace(old,new)
    old='if(x==s.starts[2]){'
    new='if(x==s.starts[2]&&(Policy!=5||(z>=12&&y>=12&&z<12+s.nz&&y<12+s.ny))){'
    if text.count(old)!=1:raise ValueError('Exact plane callback predicate required')
    text=text.replace(old,new)
    old='template<int Depth,bool UniformE>__device__ __forceinline__ void ResearchSourceFacts('
    text=text.replace(old,'template<int Depth,bool UniformE,bool PlanePair=false>__device__ __forceinline__ void ResearchSourceFacts(')
    start,end,facts=block(text,'void ResearchSourceFacts(')
    if facts.count(',20,false,')!=5:raise ValueError('All five independent CPML axis variants required')
    facts=facts.replace(',20,false,',',20,PlanePair,')
    text=text[:start]+facts+text[end:]
    start,end,entry=block(text,'void AsymmetricSourcePlanePair20(')
    if 'const int32_t*ids)' not in entry or entry.count('SourcePendingPairedBody<')!=3:
        raise ValueError('Exact source-plane wrapper required')
    entry=entry[:entry.index('{')+1]+'''
 static_assert(Policy==3||Policy==4||Policy==5,"Certified coincident source-plane policies");
 if constexpr(Policy==5){
  if(p.flags[ids[blockIdx.x]]&64)ResearchSourceFacts<Depth,true,true>(p,ids);
  else ResearchSourceFacts<Depth,false,true>(p,ids);
 }else if constexpr(Policy==3){
  if(p.flags[ids[blockIdx.x]]&64)SourcePendingPairedBody<Depth,1,false,true,Policy,true,1024,20,true>(p,ids);
  else SourcePendingPairedBody<Depth,1,false,true,Policy,false,1024,20,true>(p,ids);
 }else SourcePendingPairedBody<Depth,1,false,true,Policy,false,1024,20,true>(p,ids);
}'''
    text=text[:start]+entry+text[end:]
    start,end,launch=block(text,'inline cudaError_t LaunchSourcePending(')
    if launch.count('if constexpr(Policy==3||Policy==4)')!=1:
        raise ValueError('Exact source launch admission required')
    launch=launch.replace('if constexpr(Policy==3||Policy==4)','if constexpr(Policy==3||Policy==4||Policy==5)')
    launch=launch.replace('cudaFuncAttributeMaxDynamicSharedMemorySize,SourcePendingSharedBytes',
                          'cudaFuncAttributeMaxDynamicSharedMemorySize,ResearchSourceSharedBytes<Policy>')
    launch=launch.replace('<<<count,1024,SourcePendingSharedBytes,',
                          '<<<count,1024,ResearchSourceSharedBytes<Policy>,')
    text=text[:start]+launch+text[end:]
    return text


def stage(source,output,expected):
    source,output=Path(source).resolve(),Path(output).resolve()
    if (output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents
            or source in output.parents or output in source.parents):raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=expected:raise ValueError('Explicit immutable parent hash required')
    text=transform(original)
    shutil.copytree(source,output)
    (output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.trace-bounded-source-plane.v1',parent_source=str(source),parent_consumer_sha256=expected,
                consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
                admission='Existing invocation-wide SourcePlanePairFact and SourceMask20; Policy5 retains full transverse bounds',
                source_bounds='x == field-specific plane && 12 <= z < 12+nz && 12 <= y < 12+ny',
                independent_cpml_profiles=True,ordered_source_additions=2,barriers_unchanged=True,
                generic_source_fallback_retained=True,expected_kernels=68,GPU_executed=False)
    (output.parent/'bounded-source-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=PARENT)
    p.add_argument('--expected-consumer-sha256',default=PARENT_SHA)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
