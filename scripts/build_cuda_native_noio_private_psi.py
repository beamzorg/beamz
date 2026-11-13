"""Stage native single-axis CPML with phase register reuse and private psi.

This changes dataflow on existing certified NoIO1024 dispatches only. Frozen
rank/policy scheduling and all source/bulk/observer/general paths are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from scripts.build_cuda_partial_plane_pipeline import once
from scripts.build_cuda_phase_rebinding import H_OLD,H_NEW,E_OLD,E_NEW
from scripts.cuda_native_phase_state_fact import generate_guard
from scripts.cuda_trace_phase_cycle import block
from scripts.patch_cuda_traced_lifetime_private_psi import POINT


def region(phase,original,phase_mask=3):
    lowered=once(original,H_OLD,H_NEW) if phase==0 else once(original,E_OLD,E_NEW)
    lowered=re.sub(r'\b(?:h|e)(?=\[)','state',lowered)
    if not phase_mask&(1<<phase):return lowered
    prefix='h' if phase==0 else 'e';clock='' if phase==0 else 'e';hz,hy,hx=(clock+s for s in ('hz','hy','hx'))
    anchor=f'    if(zs!={hz}&&sy!={hy}&&sx!={hx}){{\n     const auto read='
    inserted=f'''    if(zs!={hz}&&sy!={hy}&&sx!={hx}){{
     const int dz_owned=(zs-{hz})&7,dy_owned=(sy-{hy})&15,dx_owned=(sx-{hx})&31;
     PrivatePsiHistoryPoint<decltype(env)> point_env(env,history.{prefix}a[j],history.{prefix}b[j],n==first||dz_owned==1||dy_owned==1||dx_owned==1,n==Depth||dz_owned==BZ||dy_owned==BY||dx_owned==BX);
     const auto read='''
    lowered=once(lowered,anchor,inserted)
    for c in range(3):lowered=once(lowered,f'AdvanceNativeCell<{phase},{c},false>(env,read,n,z,y,x)',f'AdvanceNativeCell<{phase},{c},false>(point_env,read,n,z,y,x)')
    return lowered


def stage(source,output,expected,*,phase_mask=3):
    source,output=Path(source).resolve(),Path(output).resolve();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if output.exists() or source==output or source in output.parents or output in source.parents:raise ValueError('New disjoint source tree required')
    if sha(source/'src/asymmetric_native.cuh')!=expected:raise ValueError('Frozen native parent required')
    shutil.copytree(source,output);src=output/'src';c=(src/'asymmetric_native.cuh').read_text();old=c;native=block(c,'void UniformBody(')[2];position=0;bindings=[]
    body=Path(__file__).with_name('cuda_native_noio_private_psi.cuh').read_text();body=once(body,'// BEAMZ_PRIVATE_PSI_POINT',POINT)
    for phase in range(2):
        marker='{int step;asm volatile("mov.b32 %0,%1;":';a=native.index(marker,position);_,b,r=block(native[a:],marker);position=a+b;emitted=region(phase,r,phase_mask)
        body=once(body,'// BEAMZ_NOIO_PSI_'+('H' if phase==0 else 'E')+'_REGION',emitted);bindings.append(dict(phase=phase,original=r,emitted=emitted))
    # NativeSession's canonical program key already includes the actual input
    # pointers/shapes captured here. No independent user selector is introduced.
    guard=generate_guard();(src/'runtime_private_psi_state_fact.h').write_text(guard['source']);c=once(c,'#include "native_moving.h"','#include "native_moving.h"\n#include "runtime_private_psi_state_fact.h"')
    c=once(c,'int nz,ny,nx,frontier,macro_step;','int nz,ny,nx,frontier,macro_step;bool private_psi_independent=false;')
    anchor='*out={};int h_layout=0;';addition='''*out={};int h_layout=0;
 BeamzBuffer cache_state[12]{},cache_profile[36]{};
 for(int family=0;family<2;++family){const auto&v=family?e:h;for(int term=0;term<6;++term){cache_state[6*family+term]=v.inputs[31+term];for(int kind=0;kind<3;++kind)cache_profile[18*family+3*term+kind]=v.inputs[13+3*term+kind];}}
 out->private_psi_independent=BeamzNativeIndependentPhaseState(cache_state,cache_profile);''';c=once(c,anchor,addition)
    wrappers=[]
    for signature,legacy,call,condition in (
        ('void UniformBody(','NativeNoioLegacyUniformBody','NativeNoioPrivatePsiBody<Depth,0,ActiveAxes>(p,tiles);','Layout==1&&!Fast&&IsotropicH&&Policy==1&&UniformE&&CachedProfiles&&(ActiveAxes==1||ActiveAxes==2||ActiveAxes==4)'),
        ('void FixedCubeCachedBody(','NativeNoioLegacyFixedCubeBody','NativeNoioPrivatePsiBody<Depth,N,ActiveAxes>(p,tiles);','ActiveAxes==1||ActiveAxes==2||ActiveAxes==4')):
        a,b,original=block(c,signature);start=c.rfind('template<',0,a);template=c[start:a];args='Depth,Layout,Fast,IsotropicH,Policy,UniformE,ActiveAxes,CachedProfiles' if 'Uniform' in signature else 'Depth,N,ActiveAxes'
        declaration=original[:original.index('{')];replacement=declaration+'{\n if constexpr('+condition+'){if(p.private_psi_independent){'+call+'return;}}\n '+legacy+'<'+args+'>(p,tiles);\n}'
        c=c[:start]+template+original.replace(signature,'void '+legacy+'(',1)+'\n'+template+replacement+c[b:];wrappers.append(dict(signature=signature,original=original,emitted=replacement))
    prototype='template<int Depth,int CubeN,int ActiveAxes>\n__device__ __forceinline__ void NativeNoioPrivatePsiBody(const Parameters&p,const int32_t*tiles);\n'
    anchor='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1,bool CachedProfiles=false>'
    c=once(c,anchor+'\n__device__ __forceinline__ void NativeNoioLegacyUniformBody',prototype+anchor+'\n__device__ __forceinline__ void NativeNoioLegacyUniformBody')
    anchor='template<int Depth,int N,int ActiveAxes=-1>\n__device__ __forceinline__ void NativeNoioLegacyFixedCubeBody'
    c=once(c,anchor,'#include "native_noio_private_psi.cuh"\n'+anchor)
    (src/'asymmetric_native.cuh').write_text(c);(src/'native_noio_private_psi.cuh').write_text(body)
    e=(src/'native_extension.cc').read_text();e=once(e,'NB_MODULE(_cuda_native, module) {','NB_MODULE(_cuda_native, module) {\n module.attr("moving_native_noio_private_psi")=true;module.attr("moving_native_noio_private_psi_floats")=16;');(src/'native_extension.cc').write_text(e)
    proof=dict(schema='beamz.native-noio-phase-private-psi.v1',expected_kernels=68,parent_consumer_sha256=expected,consumer_sha256=sha(src/'asymmetric_native.cuh'),extension_sha256=sha(src/'native_extension.cc'),helpers={name:sha(src/name) for name in ('native_noio_private_psi.cuh','runtime_private_psi_state_fact.h')},region_bindings=bindings,wrapper_bindings=wrappers,prepare_guard_original='*out={};int h_layout=0;',prepare_guard_emitted=addition,private_field_floats=12,private_psi_floats=8*phase_mask.bit_count(),phase_mask=phase_mask,shared_bytes=52680,dispatch_allocation_bytes=55204,
               state_ownership='Exact original first-owned reload and highface/final publish protocol, singleaxis disjoint state/profile bytes. Source/observer/general bodies frozen, original native fourrank owners and rank/policy overlap unchanged.',GPU_executed=False)
    (output.parent/'native-noio-private-psi-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return {k:v for k,v in proof.items() if k not in ('region_bindings','wrapper_bindings')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);p.add_argument('--phase-mask',type=int,choices=(1,2,3),default=3);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256,phase_mask=a.phase_mask)))
