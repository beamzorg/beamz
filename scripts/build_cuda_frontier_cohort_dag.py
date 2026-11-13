"""Remove rank joins using the physical frontier's three-parent ownership DAG.

The packed field frontier has no history-slot reuse hazards. This lowering keeps
all native class-specialized kernels and their launch mappings, and replaces
only cross-task ownership metadata and host completion dependencies.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.cuda_trace_phase_cycle import block
from scripts.patch_cuda_native_cohort_dag import transform


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Unique frontier DAG anchor required:'+old[:100])
    return text.replace(old,new,1)


def stage(source,output,expected):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or source in output.parents or output in source.parents:raise ValueError('New disjoint research source required')
    c=(source/'src/asymmetric_native.cuh').read_text()
    if hashlib.sha256(c.encode()).hexdigest()!=expected:raise ValueError('Explicit immutable packed frontier required')
    if 'PhysicalFrontierIndex' not in c or 'PublishPhysicalFrontier' not in c:raise ValueError('Physical frontier ownership proof required')
    device_edits=[]
    for signature in ('void UniformBody(','void FixedCubeCachedBody(','void SourcePendingPairedBody('):
        begin,end,body=block(c,signature);old=body
        a,b,_=block(body,'if(!threadIdx.x){const int owner=')
        body=body[:a]+body[b:]
        a,b,_=block(body,'if(threadIdx.x<8){const int bits=')
        body=body[:a]+body[b:]
        epoch=' const uint64_t epoch=uint64_t(*p.epoch)<<32;'
        body=once(body,epoch,'')
        if 'owner_tags' in body or 'bases[' in body:raise ValueError('Obsolete history ownership escaped removal')
        # Mathematical operations, barriers and physical publications are kept.
        device_edits.append(dict(signature=signature,before_sha256=hashlib.sha256(old.encode()).hexdigest(),after_sha256=hashlib.sha256(body.encode()).hexdigest()))
        c=c[:begin]+body+c[end:]
    e=(source/'src/native_extension.cc').read_text();h=(source/'src/native_moving.h').read_text()
    schedule=Path('scripts/cuda_native_cohort_schedule.inc').read_text()
    for policy in (3,4,5):
        schedule=once(schedule,f'case {policy}:status=Launch<Depth,Layout,false,IsotropicH,{policy}>(branch,p,w.tile_ids_device+start,count);break;',
                      f'case {policy}:status=LaunchSourcePending<Depth,Layout,IsotropicH,{policy}>(branch,p,w.tile_ids_device+start,count,w.source_presence_mask,bool(w.data_flags&8u));break;')
    c,e,h=transform(c,e,h,schedule)
    header=Path('scripts/cuda_native_cohort_dependencies.h').read_text()
    start=header.index(' // Follow actual physical owner reuse.')
    end=header.index(' for(const auto&deps:predecessors)',start)
    header=header[:start]+''' // Physical field frontier: no history-slot WAW/WAR edges exist.
 // Axial completed parents imply all seven lower-corner producers and the
 // componentwise physical-value last-reader order.
'''+header[end:]
    header=once(header,'for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;',
                'for(int bits:{1,2,4}){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;')
    header=header.replace('cohort graph requires a prepared six-policy rank partition','physical frontier DAG requires a prepared six-policy partition')
    for name in ('c','e'):
        value=locals()[name]
        value=value.replace('dependency_dag_v1','physical_frontier_three_parent_v1')
        if name=='c':c=value
        else:e=value
    shutil.copytree(source,output)
    for name,value in [('asymmetric_native.cuh',c),('native_extension.cc',e),('native_moving.h',h),('native_cohort_dependencies.h',header)]:
        (output/'src'/name).write_text(value)
    proof=dict(schema='beamz.physical-frontier-cohort-dag.v1',parent_consumer_sha256=expected,
               consumer_sha256=hashlib.sha256(c.encode()).hexdigest(),extension_sha256=hashlib.sha256(e.encode()).hexdigest(),
               header_sha256=hashlib.sha256(h.encode()).hexdigest(),dependency_builder_sha256=hashlib.sha256(header.encode()).hexdigest(),
               device_edits=device_edits,expected_kernels=69,history_slot_metadata_accesses=False,
               specialization='Existing native bulk, certified NoIO, observers and paired source dispatch retained',
               order='Three immediate axial predecessor completions plus class stream FIFO; final join before chronological DFT and next macro',
               scope='Architecture staged only; actual graph reachability, CUDA physics and performance must be checked separately')
    (output.parent/'frontier-cohort-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
