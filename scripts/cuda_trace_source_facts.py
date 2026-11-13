"""Lower traced CPML-axis facts and independent profiles in source tiles.

The source-bearing Policy5 phase pipeline and chronological source/DFT nodes
stay intact. Axis masks select a compile-time variant before any tile work;
optional independent coefficient staging reuses its existing startup barrier.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,sha,block


def stage(source,output,mode):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    if mode not in ('axes','profiles','both'):raise ValueError('Explicit source fact lowering required')
    axes=mode in ('axes','both');cached=mode in ('profiles','both');text=original
    template='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int BlockT=0,int SourceMask=-1,bool PlanePair=false>'
    assert text.count(template)==1
    text=text.replace(template,template[:-1]+',int ActiveAxes=-1,bool CachedProfiles=false>')
    start,end,body=block(text,'void SourcePendingPairedBody(')
    anchor='float*low_stage=reinterpret_cast<float*>(bases+8)+3;'
    assert body.count(anchor)==1
    body=body.replace(anchor,anchor+'\n float*profiles=CachedProfiles?low_stage+3*LowXPoints:nullptr;\n if constexpr(CachedProfiles)StageIndependentCpmlProfiles(p,profiles);')
    old='Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair>env{p,flags};'
    assert body.count(old)==1
    body=body.replace(old,'Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair,ActiveAxes,CachedProfiles>env{p,flags,profiles};')
    text=text[:start]+body+text[end:]
    helper='template<int Depth,bool UniformE>__device__ __forceinline__ void ResearchSourceFacts(const Parameters&p,const int32_t*ids){\n'
    def call(mask):return f'SourcePendingPairedBody<Depth,1,false,true,5,UniformE,1024,20,false,{mask},{str(cached).lower()}>(p,ids);return;'
    if axes:
        helper+=' switch(p.flags[ids[blockIdx.x]]&14){\n'
        for mask in (0,1,2,4):helper+=f' case {mask*2}:'+call(mask)+'\n'
        helper+=' default:'+call(-1)+'\n }\n'
    else:helper+=' '+call(-1)+'\n'
    helper+='}\n'
    marker='constexpr size_t SourcePendingSharedBytes='
    assert text.count(marker)==1;text=text.replace(marker,helper+'\n'+marker)
    start,end,body=block(text,'void AsymmetricSourcePending20(')
    for uniform in ('true','false'):
        old=f'SourcePendingPairedBody<Depth,1,false,true,Policy,{uniform},1024,20>(p,ids);'
        assert body.count(old)==(1 if uniform=='true' else 2)
        body=body.replace(old,f'{{if constexpr(Policy==5)ResearchSourceFacts<Depth,{uniform}>(p,ids);else {old}}}')
    text=text[:start]+body+text[end:]
    if cached:
        anchor='static_assert(SourcePendingSharedBytes==58680,"Proved source-only small cache layout");'
        assert text.count(anchor)==1
        text=text.replace(anchor,anchor+'\n template<int Policy>constexpr size_t ResearchSourceSharedBytes=SourcePendingSharedBytes+(Policy==5?864ULL*sizeof(float):0);\n static_assert(ResearchSourceSharedBytes<5> == 62136,"Source profiles retain the64KiB carveout class");')
        old='status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SourcePendingSharedBytes);'
        assert text.count(old)==1;text=text.replace(old,'status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,ResearchSourceSharedBytes<Policy>);')
        old='AsymmetricSourcePending20<Depth,Policy><<<count,1024,SourcePendingSharedBytes,stream>>>'
        assert text.count(old)==1;text=text.replace(old,'AsymmetricSourcePending20<Depth,Policy><<<count,1024,ResearchSourceSharedBytes<Policy>,stream>>>')
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.traced-source-facts.v1',mode=mode,axis_masks=[0,1,2,4,-1] if axes else [-1],
               independent_profile_words=864 if cached else 0,profile_equality_assumptions=False,
               policy5_shared_bytes=62136 if cached else 58680,added_barriers=0,
               source_order_and_scalar_FP_unchanged=True,parent_sha256=PARENT_CONSUMER,
               consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),expected_kernels=66,GPU_executed=False)
    (output.parent/'source-facts-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=('axes','profiles','both'),required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.mode)))
