"""Combine one-bank phase rebinding with full independent CPML profile caching.

This is a new immutable architecture, not a retuning of the two-CTA experiment.
It trades two small blocks for one 1024-thread block and twelve private field
values per thread, preserving all 864 independent profile words in shared.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_cuda_phase_rebinding import stage as phase_stage
from scripts.build_cuda_partial_plane_pipeline import once


ENVIRONMENT = r'''
template<int CubeN,int Policy,bool UniformE,int Axes=-1>
struct CachedPhaseEnvironment:PhaseEnvironment<CubeN,Policy,UniformE,Axes>{
 using Base=PhaseEnvironment<CubeN,Policy,UniformE,Axes>;
 __device__ __forceinline__ CachedPhaseEnvironment(const Parameters&p,uint8_t flags,const float*profiles):Base(p,flags){this->profiles=profiles;}
 __device__ __forceinline__ float Coefficient(int kind,int index,int z,int y,int x)const{
  if(kind>=2){int packed;
   if constexpr((Policy==1||Policy==5)&&CubeN>0){FixedCubeCachedEnvironment<CubeN,Axes,true>e{this->p,this->flags,nullptr};packed=e.Packed(index,z,y,x);}
   else packed=Base::Packed(index,z,y,x);
   return ReadIndependentProfileCache(this->profiles,index,kind,packed);
  }
  return Base::Coefficient(kind,index,z,y,x);
 }
};
'''


def stage(source,output,expected):
    output=Path(output).resolve();phase_stage(source,output,expected);src=output/'src'
    proof=json.loads((output.parent/'phase-rebinding-compiler-proof.json').read_text())
    body=(src/'native_phase_rebinding.cuh').read_text()
    body=once(body,'template<int Depth,int CubeN,int T=512>','template<int Depth,int CubeN,int T=1024>')
    body=once(body,'T==512,"First prototype preserves native geometry and doubles resident CTA opportunities"','T==1024,"Cached profile prototype preserves native geometry and uses twelve private field floats"')
    body=once(body,'volatile int*window=reinterpret_cast<volatile int*>(bases+8);','volatile int*window=reinterpret_cast<volatile int*>(bases+8);float*profiles=reinterpret_cast<float*>(bases+8)+2;')
    body=once(body,'__syncthreads();const uint8_t flags=p.flags[tile];','const uint8_t flags=p.flags[tile];if(flags&14u)StageIndependentCpmlProfiles(p,profiles);__syncthreads();')
    for binding in proof['phase_region_bindings']:
        old=binding['emitted'];new=old.replace('PhaseEnvironment<','CachedPhaseEnvironment<').replace('env{p,flags};','env{p,flags,profiles};')
        body=once(body,old,new);binding['emitted']=new
    (src/'native_phase_rebinding.cuh').write_text(body)
    (src/'native_cached_phase_environment.cuh').write_text(ENVIRONMENT)
    c=(src/'asymmetric_native.cuh').read_text();c=once(c,'#include "native_phase_environment.cuh"','#include "native_phase_environment.cuh"\n#include "native_cached_phase_environment.cuh"');(src/'asymmetric_native.cuh').write_text(c)
    control=(src/'native_closed_pipeline_control.cuh').read_text().replace('constexpr size_t ColdOffset=49280;','constexpr size_t ColdOffset=52736;');(src/'native_closed_pipeline_control.cuh').write_text(control)
    kernel=(src/'native_closed_pipeline_kernel.cuh').read_text()
    kernel=once(kernel,'__launch_bounds__(512,2)','__launch_bounds__(1024,1)');kernel=once(kernel,'PhaseRebindingBody<Depth,CubeN,512>(p)','PhaseRebindingBody<Depth,CubeN,1024>(p)')
    kernel=once(kernel,'closed_pipeline::SharedBytes==49504','closed_pipeline::SharedBytes==52960')
    kernel=once(kernel,'3*PAD*sizeof(float)+8*sizeof(size_t)+2*sizeof(int),','3*PAD*sizeof(float)+8*sizeof(size_t)+2*sizeof(int)+864*sizeof(float),')
    kernel=once(kernel,'<<<q.worker_count,512,','<<<q.worker_count,1024,');(src/'native_closed_pipeline_kernel.cuh').write_text(kernel)
    e=(src/'native_extension.cc').read_text();e=once(e,'closed.max_workers*=2;','');e=once(e,'module.attr("moving_phase_rebinding_threads")=512;','module.attr("moving_phase_rebinding_threads")=1024;');e=once(e,'module.attr("moving_phase_rebinding_shared_bytes")=49504;','module.attr("moving_phase_rebinding_shared_bytes")=52960;module.attr("moving_phase_full_profile_cache")=true;');(src/'native_extension.cc').write_text(e)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    proof.update(phase_full_profile_cache=True,phase_profile_words=864,phase_threads=1024,phase_resident_target=1,phase_shared_bytes=52960,consumer_sha256=sha(src/'asymmetric_native.cuh'),extension_sha256=sha(src/'native_extension.cc'))
    for name in (*proof['helpers'],'native_cached_phase_environment.cuh'):proof['helpers'][name]=sha(src/name)
    for name in ('closed-pipeline-compiler-proof.json','phase-rebinding-compiler-proof.json'):(output.parent/name).write_text(json.dumps(proof,indent=2)+'\n')
    return {k:v for k,v in proof.items() if k not in ('body_bindings','phase_region_bindings')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
