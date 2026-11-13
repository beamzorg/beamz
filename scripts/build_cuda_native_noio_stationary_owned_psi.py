"""Keep compressed shared CPML slots stationary in physical owned coordinates.

Each owned box maps bijectively modulo7x15x31, rather than8x16x32 resident
coordinates. Point-local old state has only one reader. Entering cells reload
global state and may replace only an outgoing physical point's old slot.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from scripts.build_cuda_native_noio_owned_psi import stage as owned_stage,owned_region
from scripts.build_cuda_partial_plane_pipeline import once


def stationary_region(phase,original):
    text=owned_region(phase,original)
    old=f'NativeOwnedPsiPoint<decltype(env)> point_env(env,psi_a[j],psi_b[j],psi_cache+{phase*2}*NativeOwnedPsiPoints,NativeOwnedPsiIndex(dz_owned-1,dy_owned-1,dx_owned-1),'
    new=f'NativeStationaryOwnedPsiPoint<decltype(env)> point_env(env,psi_cache+{phase*2}*NativeOwnedPsiPoints,NativeStationaryOwnedIndex(dz_owned,dy_owned,dx_owned,psi_z,psi_y,psi_x),'
    return once(text,old,new)


def stage(source,output,expected):
    owned_stage(source,output,expected);output=Path(output).resolve();src=output/'src';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();proof=json.loads((output.parent/'native-noio-private-psi-compiler-proof.json').read_text());body=(src/'native_noio_private_psi.cuh').read_text();bindings=[]
    body=once(body,'#include "native_noio_owned_psi.cuh"','#include "native_noio_stationary_owned_psi.cuh"');body=once(body,'float state[3][Slots],psi_a[Slots],psi_b[Slots];','float state[3][Slots];')
    body=once(body,'for(int n=first;n<=last;++n){','for(int n=first;n<=last;++n){\n  const int psi_z=NativeStationaryOwnedShift<BZ>(n),psi_y=NativeStationaryOwnedShift<BY>(n),psi_x=NativeStationaryOwnedShift<BX>(n);')
    initial='''#pragma unroll
  for(int j=0;j<Slots;++j){psi_a[j]=0.f;psi_b[j]=0.f;}
  '''
    if body.count(initial)!=2:raise ValueError('Two phase psi initializations required')
    body=body.replace(initial,'')
    for binding in proof['region_bindings']:
        phase=binding['phase'];emitted=stationary_region(phase,binding['original']);body=once(body,binding['emitted'],emitted);bindings.append(dict(**{k:v for k,v in binding.items() if k!='emitted'},emitted=emitted))
        publication=f'const int psi_index=NativeOwnedPsiIndex(dz,dy,dx);psi_cache[{phase*2}*NativeOwnedPsiPoints+psi_index]=psi_a[j];psi_cache[{phase*2+1}*NativeOwnedPsiPoints+psi_index]=psi_b[j];'
        body=once(body,publication,'// Each point-local psi recurrence already published to its unique stationary slot.')
    (src/'native_noio_private_psi.cuh').write_text(body);(src/'native_noio_stationary_owned_psi.cuh').write_bytes(Path(__file__).with_name('cuda_native_noio_stationary_owned_psi.cuh').read_bytes());e=(src/'native_extension.cc').read_text();e=once(e,'module.attr("moving_native_noio_temporary_psi_floats")=8;','module.attr("moving_native_noio_temporary_psi_floats")=0;module.attr("moving_native_noio_stationary_owned_psi")=true;');(src/'native_extension.cc').write_text(e)
    proof.update(schema='beamz.native-noio-stationary-owned-psi.v1',stationary_owned_psi=True,temporary_psi_floats=0,psi_storage='Physical coordinates modulo OWNED7x15x31. One current point per slot, immediate pointlocal stores safe; incoming faces alone reload. No psi temporary outputs or separate publication loops.',region_bindings=bindings,extension_sha256=sha(src/'native_extension.cc'),helpers={name:sha(src/name) for name in (*proof['helpers'],'native_noio_stationary_owned_psi.cuh')});(output.parent/'native-noio-private-psi-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return {k:v for k,v in proof.items() if k not in ('region_bindings','wrapper_bindings','original_wrapper_bindings','noio_launcher_original','noio_launcher_emitted')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
