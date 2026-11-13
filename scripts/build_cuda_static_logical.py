"""Stage fixed logical Yee cells and full profiles in traced dead E planes.

Preserve the frozen rank scheduler, native four-rank packets, policy launches,
thread counts and numerical cell header. Only field dataflow/storage changes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from scripts.build_cuda_partial_plane_pipeline import once
from scripts.build_cuda_phase_rebinding import H_OLD,E_OLD
from scripts.cuda_trace_phase_cycle import block


def lower_region(region,phase):
    if phase==0:
        region=once(region,'const int hz=(-step-1)&7,hy=(-step-1)&15,hx=(-step-1)&31,low=-step-1;','const int low=-step-1;')
        region=once(region,'const int dy=(sy-hy)&15,dx=(sx-hx)&31,y=oy+low+dy,x=ox+low+dx;','const int y=oy+low+sy,x=ox+low+sx;')
        region=once(region,'z=oz+low+((zs-hz)&7)','z=oz+low+zs');region=once(region,'if(zs!=hz&&sy!=hy&&sx!=hx)','if(zs>0&&sy>0&&sx>0)')
        region=once(region,H_OLD,'return frame[StaticReadSlot<0>(field,zs,sy,sx,dz,ddy,ddx)];');region=once(region,'h[c][j]=value;','result[c][j]=value;')
    else:
        region=once(region,'const int ehz=(-step-1)&7,ehy=(-step-1)&15,ehx=(-step-1)&31,elow=-step-1;','const int elow=-step-1;')
        region=once(region,'const int y=oy+elow+((sy-ehy)&15),x=ox+elow+((sx-ehx)&31);','const int y=oy+elow+sy,x=ox+elow+sx;')
        region=once(region,'z=oz+elow+((zs-ehz)&7)','z=oz+elow+zs');region=once(region,'if(zs!=ehz&&sy!=ehy&&sx!=ehx)','if(zs>0&&sy>0&&sx>0)')
        region=once(region,E_OLD,'return frame[StaticReadSlot<1>(field,zs,sy,sx,dz,ddy,ddx)];');region=once(region,'e[c][j]=value;','result[c][j]=value;')
    return region


def stage(source,output,expected,*,legacy_general_source=True):
    source,output=Path(source).resolve(),Path(output).resolve();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if output.exists() or source==output or source in output.parents or output in source.parents:raise ValueError('New disjoint private source tree required')
    if sha(source/'src/asymmetric_native.cuh')!=expected:raise ValueError('Exact frozen native source required')
    shutil.copytree(source,output);src=output/'src';c=(src/'asymmetric_native.cuh').read_text();original=c
    native=block(c,'void UniformBody(')[2];body=Path(__file__).with_name('cuda_native_static_logical.cuh').read_text();regions=[];position=0
    for phase in range(2):
        marker='{int step;asm volatile("mov.b32 %0,%1;":';a=native.index(marker,position);_,b,region=block(native[a:],marker);position=a+b
        lowered=lower_region(region,phase);body=once(body,'// BEAMZ_STATIC_'+('H' if phase==0 else 'E')+'_REGION',lowered);regions.append(dict(phase=phase,original=region,emitted=lowered))
    # Inline the output bank's publication, so its address never escapes to an
    # array-reference function boundary before compiler scalar replacement.
    publish=block(body,'void StaticPublishOwned(')[2];inside=publish[publish.index('{')+1:-1]
    declaration=' constexpr int Slots=PAD/T,ZStep=T/(CY*CX);const int sx=threadIdx.x&(CX-1),sy=(threadIdx.x/CX)&(CY-1),zbase=T==512?0:threadIdx.x/(CY*CX);'
    inside=once(inside,declaration,'');publications=[]
    for phase in range(2):
        emitted='{'+re.sub(r'\bPhase\b',str(phase),inside).replace('values[c][j]','result[c][j]')+'}'
        body=once(body,f'StaticPublishOwned<Depth,Layout,T,{phase}>(p,frame,bases,result,n,oz,oy,ox);',emitted);publications.append(dict(phase=phase,source=publish,emitted=emitted))
    uniform='''void UniformBody(const Parameters&p,const int32_t*tiles){
 constexpr int T=(Fast||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512;
 StaticLogicalBody<Depth,Layout,Fast,IsotropicH,Policy,UniformE,ActiveAxes,CachedProfiles,0,-1,false,T>(p,tiles);
}'''
    fixed='''void FixedCubeCachedBody(const Parameters&p,const int32_t*tiles){
 StaticLogicalBody<Depth,1,false,true,1,true,ActiveAxes,true,N,-1,false,1024>(p,tiles);
}'''
    pending='''void SourcePendingPairedBody(const Parameters&p,const int32_t*tiles){
 constexpr int T=BlockT?BlockT:((Layout==1&&Policy==1&&IsotropicH&&UniformE)?1024:512);
 StaticLogicalBody<Depth,Layout,Fast,IsotropicH,Policy,UniformE,ActiveAxes,CachedProfiles,0,SourceMask,PlanePair,T>(p,tiles);
}'''
    legacy=None
    if legacy_general_source:
        old_pending=block(original,'void SourcePendingPairedBody(')[2];position=original.index('void SourcePendingPairedBody(');template_start=original.rfind('template<',0,position)
        legacy=original[template_start:position]+old_pending.replace('void SourcePendingPairedBody(','void StaticLegacySourcePendingBody(',1)
        pending='''void SourcePendingPairedBody(const Parameters&p,const int32_t*tiles){
 if constexpr(Policy==5&&!PlanePair)StaticLegacySourcePendingBody<Depth,Layout,Fast,IsotropicH,Policy,UniformE,BlockT,SourceMask,PlanePair,ActiveAxes,CachedProfiles>(p,tiles);
 else{
  constexpr int T=BlockT?BlockT:((Layout==1&&Policy==1&&IsotropicH&&UniformE)?1024:512);
  StaticLogicalBody<Depth,Layout,Fast,IsotropicH,Policy,UniformE,ActiveAxes,CachedProfiles,0,SourceMask,PlanePair,T>(p,tiles);
 }
}'''
    wrappers=[]
    for signature,replacement in (('void UniformBody(',uniform),('void FixedCubeCachedBody(',fixed),('void SourcePendingPairedBody(',pending)):
        a,b,old=block(c,signature);c=c[:a]+replacement+c[b:];wrappers.append(dict(signature=signature,original=old,emitted=replacement))
    if legacy:
        position=c.index('void SourcePendingPairedBody(');template_start=c.rfind('template<',0,position);c=c[:template_start]+legacy+'\n'+c[template_start:]
    anchor='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool CachedProfiles=false>'
    c=once(c,anchor,'#include "native_static_logical.cuh"\n'+anchor)
    start=c.index('constexpr size_t SharedBytes=');end=c.index('struct Source {',start)
    c=c[:start]+'''constexpr size_t SharedBytes=98432,ProfileCachedSharedBytes=SharedBytes,FixedCubeSharedBytes=SharedBytes;
static_assert(6ULL*PAD*sizeof(float)+8*sizeof(size_t)+2*sizeof(int)<=SharedBytes&&SharedBytes<=101376,"Fixed logical fields and profile holes fit SM86");
'''+c[end:]
    c=once(c,'constexpr size_t SourcePendingSharedBytes=(3ULL*PAD+2048)*sizeof(float)+8*sizeof(size_t)+3*sizeof(int)+3ULL*LowXPoints*sizeof(float);','constexpr size_t SourcePendingSharedBytes=SharedBytes;')
    c=once(c,'static_assert(SourcePendingSharedBytes==58680,"Proved source-only small cache layout");','static_assert(SourcePendingSharedBytes==98432,"Fixed logical source layout");')
    c=once(c,'template<int Policy>constexpr size_t ResearchSourceSharedBytes=SourcePendingSharedBytes+(Policy==5?864ULL*sizeof(float):0);','template<int Policy>constexpr size_t ResearchSourceSharedBytes=SharedBytes;')
    c=once(c,'static_assert(ResearchSourceSharedBytes<5> == 62136,"Source profiles retain the64KiB carveout class");','static_assert(ResearchSourceSharedBytes<5> == 98432,"Profiles occupy only unread Eplanes");')
    (src/'asymmetric_native.cuh').write_text(c);(src/'native_static_logical.cuh').write_text(body)
    e=(src/'native_extension.cc').read_text();e=once(e,'NB_MODULE(_cuda_native, module) {','NB_MODULE(_cuda_native, module) {\n module.attr("moving_static_logical_cells")=true;module.attr("moving_static_shared_bytes")=98432;module.attr("moving_static_profile_words")=864;module.attr("moving_static_reserved_E_words")=896;');(src/'native_extension.cc').write_text(e)
    proof=dict(schema='beamz.static-logical-native-compiler.v1',static_logical=True,legacy_general_source=legacy_general_source,legacy_source_binding=legacy,expected_kernels=68,parent_consumer_sha256=expected,consumer_sha256=sha(src/'asymmetric_native.cuh'),extension_sha256=sha(src/'native_extension.cc'),helpers={'native_static_logical.cuh':sha(src/'native_static_logical.cuh')},region_bindings=regions,publication_bindings=publications,wrapper_bindings=wrappers,
               shared_bytes=98432,profile_words=864,reserved_E_words=896,profile_cache='OriginalCachedProfiles/FixedCube predicates unchanged, all864 independent words embedded in unread Ecomponentnormalhighplanes. Cachedfalse retains originalglobalprofileloads.',
               numerics='Frozen scalar cell header/primitives and Environment/FixedCube methods unchanged; only shared field read routes, outputplacement and cachedprofilelookup change. Fullsource/observation finish and originalUniformE/ActiveAxes/SourceMask/PlanePair facts.',
               scheduling='Frozen fourrankowners, native rank/policyparallel launches/threadcounts and fullkernel readiness; no persistentqueue/newownerallocation. Foursharedreaderretirement/publication joins, one3*Slots temporarybank, fixedlogicalghosts and affinephysicalcoordinates/surfaceindices.',
               parent_native_moving_sha256=sha(src/'native_moving.cc'),parent_native_header_sha256=sha(src/'native_moving.h'),GPU_executed=False)
    (output.parent/'static-logical-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return {k:v for k,v in proof.items() if k not in ('region_bindings','publication_bindings','wrapper_bindings')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);p.add_argument('--all-static',action='store_true');a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256,legacy_general_source=not a.all_static)))
