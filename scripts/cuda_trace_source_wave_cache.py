"""Hoist CTA-uniform source waveform reads using source-effect lifetimes.

Only the certified bounded Policy5 plane path uses this cache. Its clocks and
waveforms are immutable throughout a moving macro. The existing startup barrier
publishes the first four values; RLpair retires all H/E source consumers before
thread0 stages the next values, and RE publishes them. No new barrier is added.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

from scripts.cuda_trace_phase_cycle import ROOT,block,sha
from scripts.cuda_phase_schedule_ir import Event,check_schedule


def prove_wave_lifetime(*,early_overwrite=False,omit_publication=False):
    events=[]
    words=frozenset(('wave',i) for i in range(4))
    for n in range(3):
        events += [Event(f'H{n} source',reads=frozenset(('wave',i) for i in (0,1))),
                   Event(f'E{n} source',reads=frozenset(('wave',i) for i in (2,3)))]
        if not early_overwrite:events += [Event('RLpair',barrier=True)]
        events += [Event('stage next waves',writes=words)]
        if early_overwrite:events += [Event('RLpair',barrier=True)]
        if not omit_publication:events += [Event('RE',barrier=True)]
    return check_schedule(events)


def stage(source,output,expected):
    source,output=Path(source).resolve(),Path(output).resolve()
    if (output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents
            or source in output.parents or output in source.parents):raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=expected:raise ValueError('Explicit immutable parent required')
    if 'Policy!=5||(z>=12&&y>=12' not in original:raise ValueError('Bounded source parent required')
    text=original
    old='bool CachedProfiles=false> struct Environment'
    if text.count(old)!=1:raise ValueError('Exact environment template required')
    text=text.replace(old,'bool CachedProfiles=false,bool CachedWaves=false> struct Environment')
    old='const Parameters&p;uint8_t flags;const float*profiles=nullptr;'
    if text.count(old)!=1:raise ValueError('Exact environment fields required')
    text=text.replace(old,old+'const float*source_waves=nullptr;')
    start,end,finish=block(text,'float Finish(')
    # Only the plane branch uses the cached values; generic fallbacks retain
    # their own independent source loops and mutable clock reads.
    pstart,pend,plane=block(finish,'if constexpr(PlanePair)')
    if plane.count('__ldg(s.waveforms+time)')!=1 or plane.count('__ldg(s.waveforms+s.ntime+time)')!=1:
        raise ValueError('Both split-round waveform operands required')
    plane=plane.replace('__ldg(s.waveforms+time)',
                         '(CachedWaves?source_waves[field==2?0:2]:__ldg(s.waveforms+time))')
    plane=plane.replace('__ldg(s.waveforms+s.ntime+time)',
                         '(CachedWaves?source_waves[field==2?1:3]:__ldg(s.waveforms+s.ntime+time))')
    finish=finish[:pstart]+plane+finish[pend:]
    text=text[:start]+finish+text[end:]
    marker='// Two-plane publication groups; use after the reviewed base SourcePendingBody helpers.'
    helper='''
__device__ __forceinline__ void StageSourceWaves(const Parameters&p,float*waves,int version){
 #pragma unroll
 for(int family=0;family<2;++family){const auto&s=p.sources[2+2*family];
  int time=*s.current_step+p.macro_step+version-1;time=max(0,min(time,s.ntime-1));
  waves[2*family]=__ldg(s.waveforms+time);waves[2*family+1]=__ldg(s.waveforms+s.ntime+time);
 }
}
'''
    text=text.replace(marker,helper+'\n'+marker)
    old='bool PlanePair=false,int ActiveAxes=-1,bool CachedProfiles=false>\n__device__ __forceinline__ void SourcePendingPairedBody('
    if text.count(old)!=1:raise ValueError('Exact source pipeline template required')
    text=text.replace(old,'bool PlanePair=false,int ActiveAxes=-1,bool CachedProfiles=false,bool CachedWaves=false>\n__device__ __forceinline__ void SourcePendingPairedBody(')
    start,end,body=block(text,'void SourcePendingPairedBody(')
    body=body.replace('float*profiles=CachedProfiles?low_stage+3*LowXPoints:nullptr;',
        'float*profiles=CachedProfiles?low_stage+3*LowXPoints:nullptr;\n float*waves=CachedWaves?low_stage+3*LowXPoints+864:nullptr;\n static_assert(!CachedWaves||(Policy==5&&PlanePair&&CachedProfiles),"Certified bounded source and profile allocation required");')
    old='if(!threadIdx.x)*phase_step=first;__syncthreads();'
    if body.count(old)!=1:raise ValueError('Unique source startup publication required')
    body=body.replace(old,'if(!threadIdx.x){*phase_step=first;if constexpr(CachedWaves)StageSourceWaves(p,waves,first);}\n __syncthreads();')
    old='Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair,ActiveAxes,CachedProfiles>env{p,flags,profiles};'
    body=body.replace(old,'Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair,ActiveAxes,CachedProfiles,CachedWaves>env{p,flags,profiles,waves};')
    old='if(!threadIdx.x)*phase_step=n+1;'
    if body.count(old)!=1:raise ValueError('Unique post-RLpair phase update required')
    if body.index('// RLpair retires')>body.index(old):raise ValueError('Source consumers not retired before cache overwrite')
    body=body.replace(old,'if(!threadIdx.x){*phase_step=n+1;if constexpr(CachedWaves){if(n<(Fast?Depth:window[1]))StageSourceWaves(p,waves,n+1);}}')
    text=text[:start]+body+text[end:]
    start,end,facts=block(text,'void ResearchSourceFacts(')
    if facts.count(',true>(p,ids)')!=5:raise ValueError('Five source profile variants required')
    facts=facts.replace(',true>(p,ids)',',true,PlanePair>(p,ids)')
    text=text[:start]+facts+text[end:]
    start,end,launch=block(text,'inline cudaError_t LaunchSourcePending(')
    # Only the plane-specialized launch allocates the extra four values.
    old='const auto specialized=AsymmetricSourcePlanePair20<Depth,Policy>;'
    if launch.count(old)!=1:raise ValueError('Exact plane-source launch required')
    launch=launch.replace(old,old+'\n     constexpr size_t WaveSharedBytes=ResearchSourceSharedBytes<Policy>+(Policy==5?4*sizeof(float):0);')
    first=launch.index('constexpr size_t WaveSharedBytes=')
    stop=launch.index('return cudaGetLastError();',first)
    region=launch[first:stop].replace('MaxDynamicSharedMemorySize,ResearchSourceSharedBytes<Policy>',
                                      'MaxDynamicSharedMemorySize,WaveSharedBytes').replace(
                                      '<<<count,1024,ResearchSourceSharedBytes<Policy>,','<<<count,1024,WaveSharedBytes,')
    launch=launch[:first]+region+launch[stop:]
    text=text[:start]+launch+text[end:]
    shutil.copytree(source,output)
    (output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.trace-uniform-source-wave-cache.v1',parent_source=str(source),
                parent_consumer_sha256=expected,consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
                lifetime_proof=prove_wave_lifetime(),added_shared_bytes=16,added_barriers=0,
                admission='Certified Policy5 plane pair; source clocks/waveforms immutable during macro',
                expected_kernels=68,GPU_executed=False)
    (output.parent/'source-wave-cache-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--expected-consumer-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
