"""Bind counter-free, causally proved packet last use to actual import joins."""
from __future__ import annotations
import hashlib
from pathlib import Path
from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


def bind(output,proof):
    src=Path(output)/'src';p=src/'asymmetric_native.cuh';c=p.read_text()
    c=once(c,'#include "native_closed_pipeline_control.cuh"','#include "native_closed_pipeline_control.cuh"\n#include "native_distal_packet.cuh"')
    for signature in ('void UniformBody(','void FixedCubeCachedBody(','void SourcePendingPairedBody('):
        a,b,body=block(c,signature)
        if 'Paired' in signature:
            i=body.index('__syncthreads(); // Ppair');end=body.index('\n',i)
            body=body[:end]+'\n   if constexpr(Pipeline){if(pair==0)closed_pipeline::RetireDistalPacket(p,n);}\n'+body[end:]
        else:
            anchor='__syncthreads(); // B1 retires all old-E frame readers and publishes low imports.'
            body=once(body,anchor,anchor+'\n  if constexpr(Pipeline)closed_pipeline::RetireDistalPacket(p,n);')
        c=c[:a]+body+c[b:]
    p.write_text(c);helper=Path(__file__).with_name('cuda_native_distal_packet.cuh').read_bytes();(src/'native_distal_packet.cuh').write_bytes(helper)
    proof['helpers']['native_distal_packet.cuh']=hashlib.sha256(helper).hexdigest();proof['consumer_sha256']=hashlib.sha256(c.encode()).hexdigest()
    proof['distal_disposal']=True;proof['distal_protocol']='One distal reader, after all its global imports; all8corner-neighbor tasks active at n by monotone windows; 3parent n publication DAG retires other6 readers. No counters or producer changes; outermost producer packets retained. Fully contained128B lines only; owner reuse still waits all7readers COMPLETE.'
