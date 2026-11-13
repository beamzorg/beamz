"""Keep three CPML recurrence colors in shared memory, fourth direct global.

The fourth color is intentionally recomputed from canonical global state to
free four long-lived register values. All original scalar updates are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from scripts.build_cuda_native_noio_shared_psi import stage as shared_stage,shared_region
from scripts.build_cuda_partial_plane_pipeline import once

POINT=r'''
template<class Base,int Axes>struct NativeNoioPartialSharedPoint:Base{
 float&retained;bool first,flush;
 __device__ __forceinline__ NativeNoioPartialSharedPoint(Base&base,float&value,bool fresh,bool publish):Base(base),retained(value),first(fresh),flush(publish){}
 __device__ __forceinline__ float LoadLocal(int slot,int z,int y,int x)const{
  if(slot%6!=NativeNoioCompactTerm<Axes>(1)||first)return Base::LoadLocal(slot,z,y,x);
  return retained;
 }
 __device__ __forceinline__ void StoreLocal(int slot,int z,int y,int x,float value)const{
  if(slot%6==NativeNoioCompactTerm<Axes>(1)){retained=value;if(flush)Base::StoreLocal(slot,z,y,x,value);}
  else Base::StoreLocal(slot,z,y,x,value);
 }
};
'''


def partial_region(phase,original):
    text=shared_region(phase,original)
    if phase==0:text=once(text,'PrivatePsiHistoryPoint<decltype(env)> point_env(env,private_hpsi[j],psi_cache[Slot(zs,sy,sx)],','NativeNoioPartialSharedPoint<decltype(env),ActiveAxes> point_env(env,psi_cache[Slot(zs,sy,sx)],')
    return text


def stage(source,output,expected):
    shared_stage(source,output,expected);output=Path(output).resolve();src=output/'src';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();proof=json.loads((output.parent/'native-noio-private-psi-compiler-proof.json').read_text());body=(src/'native_noio_private_psi.cuh').read_text();bindings=[]
    for binding in proof['region_bindings']:
        emitted=partial_region(binding['phase'],binding['original']);body=once(body,binding['emitted'],emitted);bindings.append(dict(**{k:v for k,v in binding.items() if k!='emitted'},emitted=emitted))
    body=once(body,'float state[3][Slots],private_hpsi[Slots];','float state[3][Slots];');body=once(body,'#include "native_noio_compact_profiles.cuh"','#include "native_noio_compact_profiles.cuh"\n'+POINT);(src/'native_noio_private_psi.cuh').write_text(body)
    e=(src/'native_extension.cc').read_text();e=once(e,'module.attr("moving_native_noio_private_psi_floats")=4;','module.attr("moving_native_noio_private_psi_floats")=0;module.attr("moving_native_noio_partial_shared_psi")=true;');(src/'native_extension.cc').write_text(e)
    proof.update(schema='beamz.native-noio-partial-shared-psi.v1',partial_shared_psi=True,private_psi_floats=0,cached_terms=3,direct_global_terms=1,region_bindings=bindings,extension_sha256=sha(src/'native_extension.cc'),helpers={name:sha(src/name) for name in proof['helpers']});(output.parent/'native-noio-private-psi-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return {k:v for k,v in proof.items() if k not in ('region_bindings','wrapper_bindings','original_wrapper_bindings','noio_launcher_original','noio_launcher_emitted')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
