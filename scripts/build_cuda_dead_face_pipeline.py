"""Stage trace-proven last-use cache retirement on a fresh closed pipeline."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_cuda_closed_pipeline import stage as closed_stage
from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_trace_phase_cycle import block


def stage(source,output,expected,*,single_kernel=False,outline_policies=False):
    output=Path(output).resolve()
    proof=closed_stage(source,output,expected,front_startup=True,single_kernel=single_kernel,outline_policies=outline_policies)
    src=output/'src';c=(src/'asymmetric_native.cuh').read_text()
    c=once(c,'#include "native_closed_pipeline_control.cuh"','#include "native_closed_pipeline_control.cuh"\n#include "native_dead_face_control.cuh"')
    c=once(c,'!q.next||!q.pending||','!q.packet_refs||!q.next||!q.pending||')
    bindings=[]
    for signature in ('void UniformBody(','void FixedCubeCachedBody(','void SourcePendingPairedBody('):
        a,b,body=block(c,signature)
        body=once(body,' if(!threadIdx.x){const int owner=',
                  ' if constexpr(Pipeline)closed_pipeline::InitializePackets(p);\n if(!threadIdx.x){const int owner=')
        if 'Paired' in signature:
            anchor='__syncthreads(); // Ppair'
            i=body.index(anchor);e=body.index('\n',i)
            body=body[:e]+'\n   if constexpr(Pipeline){if(pair==0)closed_pipeline::AckPacketsUpTo(p,n);}\n'+body[e:]
        else:
            anchor='__syncthreads(); // B1 retires all old-E frame readers and publishes low imports.'
            body=once(body,anchor,anchor+'\n  if constexpr(Pipeline)closed_pipeline::AckPacketsUpTo(p,n);')
        body=once(body,'closed_pipeline::Publish(n);','closed_pipeline::PublishPacket(p,n);')
        bindings.append(dict(signature=signature,after_sha256=hashlib.sha256(body.encode()).hexdigest()))
        c=c[:a]+body+c[b:]
    (src/'asymmetric_native.cuh').write_text(c)
    h=(src/'native_closed_pipeline.h').read_text()
    h=once(h,' int32_t*next=nullptr,',' int32_t*packet_refs=nullptr;\n int32_t*next=nullptr,');(src/'native_closed_pipeline.h').write_text(h)
    h=(src/'native_closed_pipeline_control.cuh').read_text()
    h=once(h,'int tile,depth,abort_local,parent_ready;','int tile,depth,abort_local,parent_ready,acked_step[7];')
    h=once(h,'C().tile=-1;C().abort_local=0;C().parent_ready=0;','C().tile=-1;C().abort_local=0;C().parent_ready=0;for(int e=0;e<7;++e)C().acked_step[e]=0;');(src/'native_closed_pipeline_control.cuh').write_text(h)
    k=(src/'native_closed_pipeline_kernel.cuh').read_text()
    k=once(k,'  __syncthreads();\n  if(!threadIdx.x&&!Aborted())Complete();',
             '  if(!Aborted())AckPacketsUpTo(p,Depth-1);\n  __syncthreads();\n  if(!threadIdx.x&&!Aborted())Complete();');(src/'native_closed_pipeline_kernel.cuh').write_text(k)
    e=(src/'native_extension.cc').read_text()
    e=once(e,'+observation_bytes+closed_bytes_bound;', '+observation_bytes+closed_bytes_bound+owners*depth*sizeof(int32_t);')
    e=once(e,'+moving_closed_plan_.CapacityBytes();','+moving_closed_plan_.CapacityBytes()+allocated_closed_owners*depth*sizeof(int32_t);')
    e=once(e,'    closed.next=static_cast<int32_t*>','    closed.packet_refs=static_cast<int32_t*>(upload(nullptr,allocated_closed_owners*depth*sizeof(int32_t)));\n    closed.next=static_cast<int32_t*>')
    e=once(e,'const void*closed_pointers[]={closed.next,','const void*closed_pointers[]={closed.packet_refs,closed.next,')
    e=once(e,'module.attr("moving_closed_pipeline_compiled")=true;','module.attr("moving_closed_pipeline_compiled")=true;module.attr("moving_dead_face_discard_compiled")=true;')
    e=once(e,'    result["moving_closed_pipeline_metadata_bytes"]=',
             '    result["moving_dead_face_discard_active"]=moving_closed_active_;result["moving_dead_face_refcount_bytes"]=size_t(std::max(size_t(moving_plan_.epochs)*moving_plan_.max_frontier,size_t(moving_closed_plan_.owner_count)))*moving_plan_.depth*sizeof(int32_t);\n    result["moving_closed_pipeline_metadata_bytes"]=')
    (src/'native_extension.cc').write_text(e)
    control=Path(__file__).with_name('cuda_native_dead_face_control.cuh').read_bytes();(src/'native_dead_face_control.cuh').write_bytes(control)
    proof.update(dead_face_discard=True,discard_bindings=bindings,consumer_sha256=hashlib.sha256(c.encode()).hexdigest(),extension_sha256=hashlib.sha256(e.encode()).hexdigest(),
                 discard_protocol='All7nominal-reader acqrel ACKs after all global imports; READY-bit handshake joins producer publication; whole128Blines fully inside private nonterminal rows only. CTA completion joins all disposal before owner reuse.')
    for name in (*proof['helpers'],'native_dead_face_control.cuh'):proof['helpers'][name]=hashlib.sha256((src/name).read_bytes()).hexdigest()
    (output.parent/'closed-pipeline-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);p.add_argument('--single-kernel',action='store_true');p.add_argument('--outline-policies',action='store_true');a=p.parse_args();r=stage(a.source,a.output,a.expected_consumer_sha256,single_kernel=a.single_kernel,outline_policies=a.outline_policies);print(json.dumps({k:v for k,v in r.items() if k!='body_bindings'}))
