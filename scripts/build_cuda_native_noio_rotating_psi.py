"""Trade explicit register rotations for one point of emitted arithmetic code.

The stream loop never dynamically indexes a private register array: its front
point is always index0, followed by a constant-index rotate. A complete cycle
restores original point order before shared publication. Canonical cells remain
unchanged. This is an architectural code-size/liveness experiment, not a win.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from scripts.build_cuda_native_noio_owned_psi import stage as owned_stage,owned_region
from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block

ROTATE=r'''
    #pragma unroll
    for(int c=0;c<3;++c){const float old=state[c][0];
     #pragma unroll
     for(int k=0;k<Slots-1;++k)state[c][k]=state[c][k+1];
     state[c][Slots-1]=old;
    }
    const float old_a=psi_a[0],old_b=psi_b[0];
    #pragma unroll
    for(int k=0;k<Slots-1;++k){psi_a[k]=psi_a[k+1];psi_b[k]=psi_b[k+1];}
    psi_a[Slots-1]=old_a;psi_b[Slots-1]=old_b;
'''


def rotating_region(phase,original):
    text=owned_region(phase,original);text=once(text,'#pragma unroll\n   for(int j=0;j<Slots;++j)','#pragma unroll 1\n   for(int j=0;j<Slots;++j)')
    text=re.sub(r'(state\[[^\]]+\]|psi_[ab])\[j\]',r'\1[0]',text)
    a,b,loop=block(text,'for(int j=0;j<Slots;++j)');return text[:a]+loop[:-1]+ROTATE+'}'+text[b:]


def stage(source,output,expected):
    owned_stage(source,output,expected);output=Path(output).resolve();src=output/'src';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();proof=json.loads((output.parent/'native-noio-private-psi-compiler-proof.json').read_text());body=(src/'native_noio_private_psi.cuh').read_text();bindings=[]
    for binding in proof['region_bindings']:
        emitted=rotating_region(binding['phase'],binding['original']);body=once(body,binding['emitted'],emitted);bindings.append(dict(**{k:v for k,v in binding.items() if k!='emitted'},emitted=emitted))
    (src/'native_noio_private_psi.cuh').write_text(body);e=(src/'native_extension.cc').read_text();e=once(e,'module.attr("moving_native_noio_temporary_psi_floats")=8;','module.attr("moving_native_noio_temporary_psi_floats")=8;module.attr("moving_native_noio_rotating_point_code")=true;');(src/'native_extension.cc').write_text(e)
    proof.update(schema='beamz.native-noio-rotating-point-owned-psi.v1',rotating_point_code=True,private_indexing='One scalar arithmetic point body, front register index0, constant-index complete rotations, original point order restored after Slots iterations, all ghosts rotate too.',region_bindings=bindings,extension_sha256=sha(src/'native_extension.cc'),helpers={name:sha(src/name) for name in proof['helpers']});(output.parent/'native-noio-private-psi-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return {k:v for k,v in proof.items() if k not in ('region_bindings','wrapper_bindings','original_wrapper_bindings','noio_launcher_original','noio_launcher_emitted')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
