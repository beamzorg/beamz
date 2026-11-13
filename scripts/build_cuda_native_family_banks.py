"""Separate native E/H shared banks while retaining the optimized point loops.

The native warp map and private scalar arrays are unchanged. Distinct banks
remove shared overwrite hazards, replacing four phase joins with two. Ten of
twelve independent CPML terms are cached in bulk/NoIO; the remaining two use
their original global arrays. Source paired kernels retain their old layout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.cuda_trace_phase_cycle import block


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Unique native family-bank anchor required:'+old[:100])
    return text.replace(old,new,1)


def stage(source,output,expected):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or source in output.parents or output in source.parents:raise ValueError('New disjoint research source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if hashlib.sha256(original.encode()).hexdigest()!=expected:raise ValueError('Explicit frozen native parent required')
    c=original
    start=c.index('constexpr size_t SharedBytes=');end=c.index('struct Source {',start)
    c=c[:start]+'''constexpr size_t SharedBytes=6ULL*PAD*sizeof(float)+8*sizeof(size_t)+3*sizeof(int);
constexpr size_t ProfileCachedSharedBytes=SharedBytes+720ULL*sizeof(float),FixedCubeSharedBytes=ProfileCachedSharedBytes;
static_assert(SharedBytes==98380&&ProfileCachedSharedBytes==101260,"Separate native families and10independent cached terms");
'''+c[end:]
    a,b,stage_profiles=block(c,'void StageIndependentCpmlProfiles(')
    partial=stage_profiles.replace('void StageIndependentCpmlProfiles(','void StagePartialIndependentCpmlProfiles(').replace('i<864','i<720')
    c=c[:b]+'\n__device__ __forceinline__ '+partial+'\n'+c[b:]
    c=once(c,'bool CachedProfiles=false> struct Environment {','bool CachedProfiles=false,int ProfileTerms=12> struct Environment {')
    c=once(c,'if constexpr(CachedProfiles)return ReadIndependentProfileCache(profiles,index,kind,packed);',
           'if constexpr(CachedProfiles){if(index<ProfileTerms)return ReadIndependentProfileCache(profiles,index,kind,packed);}')
    c=once(c,'else{const auto&psi=p.phase[index/6].psi[index%6];return kind==2?__ldg(psi.a+packed):kind==3?__ldg(psi.b+packed):__ldg(psi.inv+packed);}',
           '{const auto&psi=p.phase[index/6].psi[index%6];return kind==2?__ldg(psi.a+packed):kind==3?__ldg(psi.b+packed):__ldg(psi.inv+packed);}')
    base='Environment<1,false,true,1,true,-1,false,ActiveAxes,true>'
    if c.count(base)!=2:raise ValueError('Two fixed-environment base references required')
    c=c.replace(base,'Environment<1,false,true,1,true,-1,false,ActiveAxes,true,10>')
    c=once(c,'if(kind>=2)return ReadIndependentProfileCache(profiles,index,kind,Packed(index,z,y,x));',
           'if(kind>=2&&index<10)return ReadIndependentProfileCache(profiles,index,kind,Packed(index,z,y,x));')
    edits=[]
    for signature in ('void UniformBody(','void FixedCubeCachedBody('):
        begin,end,body=block(c,signature);before=body
        body=once(body,'bases=reinterpret_cast<size_t*>(frame+3*PAD);','bases=reinterpret_cast<size_t*>(frame+6*PAD);')
        body=once(body,' float*low_stage=frame+3*PAD+8*sizeof(size_t)/sizeof(float)+3;','')
        body=once(body,'if constexpr(CachedProfiles)StageIndependentCpmlProfiles(p,profiles);',
                  'if constexpr(CachedProfiles)StagePartialIndependentCpmlProfiles(p,profiles);')
        if signature=='void UniformBody(':
            body=once(body,'Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,CachedProfiles>env',
                      'Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,CachedProfiles,10>env')
        # Preserve H/E mathematics verbatim. Replace only the old ghost import,
        # low-X staging and overwrite joins between the two numerical phases.
        h_end=body.index('  const int hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31,low=-n-1;')
        e_begin=body.index('  {int step;asm volatile',h_end)
        old_phase=body[h_end:e_begin]
        new_phase='''  const int hz=(-n-1)&7,hy=(-n-1)&15,hx=(-n-1)&31,low=-n-1;
  const int dy=(sy-hy)&15,dx=(sx-hx)&31;
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;
   if(zs==hz||sy==hy||sx==hx){const auto v=ImportPacket<Depth,SupportSafe,(Layout==1)>(p,bases,n,oz,oy,ox,(zs-hz)&7,dy,dx);
    h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;
   }
   #pragma unroll
   for(int c=0;c<3;++c)frame[(3+c)*PAD+Slot(zs,sy,sx)]=h[c][j];
  }
  __syncthreads(); // F1 publishes H; separate E bank retires old E readers.
'''
        body=body[:h_end]+new_phase+body[e_begin:]
        body=once(body,'return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];',
                  'return frame[(3+field)*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];')
        a,b,old_retire=block(body,'if(n<(Fast?Depth:window[1]))')
        new_retire='''if(n<(Fast?Depth:window[1])){
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;
   #pragma unroll
   for(int c=0;c<3;++c)frame[c*PAD+Slot(zs,sy,sx)]=e[c][j];
  }
  __syncthreads(); // F2 publishes E and retires all current H readers.
  }'''
        body=body[:a]+new_retire+body[b:]
        if any(x in body for x in ('low_stage','StageRegisterLow','RetireRegisterLow')):raise ValueError('Obsolete aliased low-X storage remains')
        edits.append(dict(signature=signature,before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=hashlib.sha256(body.encode()).hexdigest(),old_phase=old_phase,new_phase=new_phase,old_retire=old_retire,new_retire=new_retire))
        c=c[:begin]+body+c[end:]
    # The paired source phase layout and all of its cached864-profile bindings
    # remain byte-identical; the environment's default ProfileTerms stays12.
    if block(c,'void SourcePendingPairedBody(')[2]!=block(original,'void SourcePendingPairedBody(')[2]:raise ValueError('Source layout changed')
    e=(source/'src/native_extension.cc').read_text()
    e=once(e,'result["moving_original_shared_bytes"]=51748;','result["moving_original_shared_bytes"]=98380;')
    e=once(e,'NB_MODULE(_cuda_native, module) {','NB_MODULE(_cuda_native, module) {\n module.attr("moving_native_family_banks")=true;module.attr("moving_native_family_bank_cached_terms")=10;')
    shutil.copytree(source,output)
    (output/'src/asymmetric_native.cuh').write_text(c);(output/'src/native_extension.cc').write_text(e)
    proof=dict(schema='beamz.native-separate-family-banks.v1',parent_consumer_sha256=expected,consumer_sha256=hashlib.sha256(c.encode()).hexdigest(),
               extension_sha256=hashlib.sha256(e.encode()).hexdigest(),expected_kernels=68,shared_bytes=98380,cached_shared_bytes=101260,
               unchanged_source_paired_body=True,cached_terms=10,source_cached_terms=12,barriers_per_step={'native_bulk_noio':2,'paired_source':5},
               bindings=edits,canonical_arithmetic_headers_unchanged=True,GPU_executed=False,
               scope='Separate E/H shared homes with unchanged native warp map, private values, point arithmetic, source/DFT/global ownership; profiles10cached2global in bulk/NoIO only')
    (output.parent/'native-family-banks-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
