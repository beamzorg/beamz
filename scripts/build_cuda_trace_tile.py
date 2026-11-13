"""Bind a freshly generated temporal tile to the complete native runtime.

Only the stable environment/ABI and launch names are retained from the frozen
parent. All three numerical compute-loop bodies are replaced by generated
topology code, including source and CPML paths. Installed sources are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from scripts.cuda_trace_phase_cycle import ROOT,block,sha
from scripts.cuda_trace_tile_codegen import generate
from scripts.cuda_trace_tile_topology import TileTopology,verify_topology


def replace_body(text,signature,body):
    start,end,old=block(text,signature)
    return text[:start]+old[:old.index('{')+1]+'\n'+body+'\n}'+text[end:]


def replace_last_bounds(prefix,bound):
    start=prefix.rfind('__launch_bounds__(')
    if start<0:raise ValueError('Launch bounds missing')
    opening=prefix.index('(',start);depth=0
    for end in range(opening,len(prefix)):
        if prefix[end]=='(':depth+=1
        elif prefix[end]==')':
            depth-=1
            if not depth:return prefix[:start]+f'__launch_bounds__({bound},1)'+prefix[end+1:]
    raise ValueError('Unterminated launch bounds')


def stage(source,output,expected,cache=(16,16,16),bulk_groups=4,base_groups=2,*,shared_e_sources=False,source_groups=None,generic_source_groups=None,inline_cells=False):
    source,output=Path(source).resolve(),Path(output).resolve()
    if (output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents
            or source in output.parents or output in source.parents):
        raise ValueError('New disjoint research source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=expected:raise ValueError('Explicit frozen parent hash required')
    generated,proof=generate(cache,bulk_groups,base_groups,shared_e_sources=shared_e_sources,source_groups=source_groups,generic_source_groups=generic_source_groups,inline_cells=inline_cells)
    bulk,base,src,generic_src=proof['bulk'],proof['base'],proof['source'],proof['generic_source'];cz,cy,cx=cache
    words=max(p['shared_words'] for p in (bulk,base,src,generic_src));shared=4*words+76;cached=shared+3456
    source_shared=(3*cz*cy*cx+max(src['shared_words'],generic_src['shared_words'])//2)*4+76 if shared_e_sources else shared
    if max(cached,source_shared+3456)>101376:raise ValueError('Generated provider storage exceeds RTX3090 block capacity')
    text=original
    old='constexpr int BZ=7,BY=15,BX=31,CZ=8,CY=16,CX=32,Threads=512;'
    if text.count(old)!=1:raise ValueError('Reviewed native ABI geometry declaration required')
    text=text.replace(old,f'constexpr int BZ={cz-1},BY={cy-1},BX={cx-1},CZ={cz},CY={cy},CX={cx},Threads={base["threads"]};')
    start=text.index('constexpr size_t SharedBytes=')
    end=text.index('struct Source {',start)
    text=text[:start]+f'''constexpr size_t SharedBytes={shared},ProfileCachedSharedBytes={cached},FixedCubeSharedBytes={cached},TraceSourceFrameBytes={source_shared};
static_assert(BEAMZ_NATIVE_ASYMMETRIC_SWIZZLE==0,"Generated topology owns its lane/storage mapping");
'''+text[end:]
    # Callback observation masks are physical8x16 bins and must stay unchanged.
    text=text.replace('z=WrapZ(z);y&=15;x&=31;','z=WrapZ(z);y&=(CY-1);x&=(CX-1);')
    text=text.replace('if(threadIdx.x<512)for(int i=threadIdx.x;i<864;i+=512)',
                      'for(int i=threadIdx.x;i<864;i+=blockDim.x)')
    marker='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool CachedProfiles=false>'
    if text.count(marker)!=1:raise ValueError('Stable native callback/compute boundary required')
    text=text.replace(marker,'#include "trace_tile_generated.cuh"\n\n'+marker)
    text=replace_body(text,'void UniformBody(','''
 constexpr int Groups=((Fast&&Layout==1)||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))?TraceBulkGroups:TraceBaseGroups;
 extern __shared__ float frame[];
 float*profiles=CachedProfiles?frame+SharedBytes/sizeof(float):nullptr;
 const int tile=tiles[blockIdx.x];
 const uint8_t flags=Policy==0?0:Policy==2?1:Policy==3?16:Policy==4?17:p.flags[tile];
 Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,CachedProfiles>env{p,flags,profiles};
 RunTraceTile<Depth,Layout,Fast,CachedProfiles,Groups>(p,tiles,env);
''')
    text=replace_body(text,'void FixedCubeCachedBody(','''
 extern __shared__ float frame[];
 const auto flags=p.flags[tiles[blockIdx.x]];
 FixedCubeCachedEnvironment<N,ActiveAxes>env{p,flags,frame+SharedBytes/sizeof(float)};
 RunTraceTile<Depth,1,false,true,TraceBulkGroups>(p,tiles,env);
''')
    source_wrapper='''
 constexpr int Groups=PlanePair?TraceSourceGroups:TraceGenericSourceGroups;
 extern __shared__ float frame[];
 const auto flags=p.flags[tiles[blockIdx.x]];
 Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair,ActiveAxes,CachedProfiles>env{p,flags,CachedProfiles?frame+TraceSourceFrameBytes/sizeof(float):nullptr};
 RunTraceTile<Depth,Layout,Fast,CachedProfiles,Groups>(p,tiles,env);
'''
    if shared_e_sources:
        source_wrapper=source_wrapper.replace('RunTraceTile<','RunTraceTileSharedE<')
        source_wrapper=source_wrapper.replace('TraceSourceFrameBytes/sizeof(float)',
                                              '(3*PAD+TraceTileStorage<Groups>::Words/2+19)')
    text=replace_body(text,'void SourcePendingPairedBody(',source_wrapper)
    # Match every entry's real thread mapping and launch bounds, rather than
    # inheriting the1024 suffix in historical entry names as a semantic fact.
    for signature,bound in [
        ('void Asymmetric(','((Fast&&Layout==1)?TraceBulkThreads:TraceBaseThreads)'),
        ('void AsymmetricNoIOFixedCube(','TraceBulkThreads'),
        ('void AsymmetricNoIOUniform1024(','TraceBulkThreads'),
        ('void AsymmetricObserverUniform1024(','TraceBulkThreads'),
        ('void AsymmetricSourcePending20(','TraceGenericSourceThreads'),
        ('void AsymmetricSourcePlanePair20(','TraceSourceThreads')]:
        start,_,_=block(text,signature);prefix=text[:start]
        text=replace_last_bounds(prefix,bound)+text[start:]
    text=text.replace('<<<count,(Fast?1024:512),SharedBytes,',
                      '<<<count,((Fast&&Layout==1)?TraceBulkThreads:TraceBaseThreads),SharedBytes,')
    for memory in ('FixedCubeSharedBytes','ProfileCachedSharedBytes','SharedBytes'):
        text=text.replace(f'<<<uniform_count,1024,{memory},',f'<<<uniform_count,TraceBulkThreads,{memory},')
    text=text.replace('<<<count,1024,ResearchSourceSharedBytes<Policy>,',
                      '<<<count,TraceSourceThreads,ResearchSourceSharedBytes<Policy>,')
    text=text.replace('AsymmetricSourcePending20<Depth,Policy><<<count,TraceSourceThreads,',
                      'AsymmetricSourcePending20<Depth,Policy><<<count,TraceGenericSourceThreads,')
    start=text.index('constexpr size_t SourcePendingSharedBytes=')
    end=text.index('template<int Depth,int Policy>',start)
    text=text[:start]+'''constexpr size_t SourcePendingSharedBytes=TraceSourceFrameBytes;
template<int Policy>constexpr size_t ResearchSourceSharedBytes=SourcePendingSharedBytes+(Policy==5?864ULL*sizeof(float):0);
'''+text[end:]
    _,_,old_observe=block(original,'void Observe(')
    _,_,new_observe=block(text,'void Observe(')
    if old_observe!=new_observe:raise ValueError('Physical observation addressing changed')
    # Generated shape and host ownership must agree in every plan/workspace.
    header=(source/'src/native_moving.h').read_text()
    for old,new in [('kOwnedZ=7,kOwnedY=15,kOwnedX=31',f'kOwnedZ={cz-1},kOwnedY={cy-1},kOwnedX={cx-1}'),
                    ('owned[3]{7,15,31}',f'owned[3]{{{cz-1},{cy-1},{cx-1}}}'),
                    ('owned{7,15,31}',f'owned{{{cz-1},{cy-1},{cx-1}}}')]:
        if header.count(old)!=1:raise ValueError('Unique host ownership declaration required: '+old)
        header=header.replace(old,new)
    shutil.copytree(source,output)
    (output/'src/asymmetric_native.cuh').write_text(text)
    (output/'src/native_moving.h').write_text(header)
    (output/'src/trace_tile_generated.cuh').write_text(generated)
    extension=(output/'src/native_extension.cc').read_text()
    extension=extension.replace('result["moving_source_pending_barriers"]=5;',
                                'result["moving_source_pending_barriers"]=2;')
    extension=extension.replace('result["moving_source_pending_shared_bytes"]=58680;result["moving_original_shared_bytes"]=51748;',
                                 f'result["moving_source_pending_shared_bytes"]={source_shared};result["moving_original_shared_bytes"]={shared};')
    extension=extension.replace('module.attr("moving_source_pending_barriers")=5;',
                                'module.attr("moving_source_pending_barriers")=2;')
    extension=extension.replace('module.attr("moving_source_pending_mask20_1024")=true;',
                                f'module.attr("moving_source_pending_mask20_1024")={str(generic_src["threads"]==1024).lower()};')
    for field in ('moving_observer_uniform_1024','moving_noio_uniform_1024'):
        extension=extension.replace(f'module.attr("{field}")=true;',f'module.attr("{field}")={str(bulk["threads"]==1024).lower()};')
    extension=extension.replace('module.attr("moving_noio_uniform_threads")=1024;',
                                f'module.attr("moving_noio_uniform_threads")={bulk["threads"]};')
    extension=extension.replace('NB_MODULE(_cuda_native, module) {',f'''NB_MODULE(_cuda_native, module) {{
 module.attr("research_cache_shape")=nb::make_tuple({cz},{cy},{cx});
 module.attr("moving_trace_generated")=true;
 module.attr("moving_source_pending_threads")={generic_src['threads']};
 module.attr("moving_source_plane_pair_threads")={src['threads']};
''')
    (output/'src/native_extension.cc').write_text(extension)
    result=dict(proof,parent_source=str(source),parent_consumer_sha256=expected,
                consumer_sha256=sha(text.encode()),host_header_sha256=sha(header.encode()),
                shared_bytes=shared,cached_shared_bytes=cached,source_shared_bytes=source_shared,expected_kernels=68,
                generator_sha256=sha(Path(__file__).read_bytes()),
                physics_header_unchanged=True,source_specialization_retained=True,
                source_barriers_per_step=2)
    (output.parent/'tile-compiler-proof.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-consumer-sha256',required=True)
    p.add_argument('--cache',type=int,nargs=3,default=(16,16,16))
    p.add_argument('--bulk-groups',type=int,default=4)
    p.add_argument('--base-groups',type=int,default=2)
    p.add_argument('--source-groups',type=int)
    p.add_argument('--generic-source-groups',type=int)
    p.add_argument('--shared-e-sources',action='store_true')
    p.add_argument('--inline-cells',action='store_true')
    a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256,
                                          tuple(a.cache),a.bulk_groups,a.base_groups,
                                          shared_e_sources=a.shared_e_sources,source_groups=a.source_groups,
                                          generic_source_groups=a.generic_source_groups,inline_cells=a.inline_cells)))
