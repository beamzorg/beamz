"""Remove shared E publication once its last in-CTA consumer is outside the window.

The full-private-field body has already computed/observed each sourced E value
in registers. Shared E is only for the next H phase. At the final actual local
iteration there is no next H reader, so both the aliased overwrite and its two
reader-retirement/publication barriers are dead. Global trails/final fields
and every source/observer effect remain in their existing order.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

from scripts.cuda_trace_column_schedule import PARENT, PARENT_SHA
from scripts.cuda_trace_phase_cycle import ROOT, block, sha
from scripts.cuda_space_time_ir import yee_operator


def prove_exit(*, shared_observer=False):
    if shared_observer:
        raise ValueError('A deferred shared-memory observer keeps the final E cache live')
    operator=yee_operator(cpml_axes=(0,1,2))
    shared_read_versions=[]
    for update in operator.updates:
        for read in update.reads:
            # A shifted E dependency reads the shared cache. Same-point own
            # fields use private registers in this admitted lowering.
            if read.field.startswith('E') and any(read.offset):
                if update.output.field.startswith('E') or read.version!=0:
                    raise ValueError('An unsupported current-E cache reader exists')
                shared_read_versions.append(read.version)
    if not shared_read_versions:
        raise ValueError('No canonical shared E readers found')
    checks=0
    for depth in (4,16):
        for first in range(1,depth+1):
            for last in range(first,depth+1):
                # A publication of E_n is needed exactly when H_(n+1)
                # belongs to this CTA's admitted local execution window.
                for n in range(first,last+1):
                    readers=[v for v in range(n+1,last+1) if v-1==n]
                    assert bool(readers)==(n<last)
                    checks+=1
    return dict(window_iteration_checks=checks,last_shared_E_has_no_in_CTA_reader=True,
                retained=['All private fields','Global trails','Final fields','Sources','Raw monitors','DFT'],
                guard='CTA-uniform n < (Fast ? Depth : window[1])',
                removed_per_nonempty_CTA=dict(shared_component_stores=3*4096,block_barriers=2),
                restrictions=['Full-private H/E bodies only','No deferred shared-memory observer'])


def transform_body(body):
    start=body.index('  __syncthreads(); // B3 retires current-H frame readers before E overwrite.')
    marker='  __syncthreads(); // B4 publishes E and the next phase counter.'
    end=body.index(marker,start)+len(marker)
    old=body[start:end]
    if old.count('__syncthreads()')!=2 or old.count('frame[c*PAD+')!=1:
        raise ValueError('Exact reviewed final shared-publication region required')
    return body[:start]+'  if(n<(Fast?Depth:window[1])){\n'+old+'\n  }'+body[end:]


def stage(source,output,expected):
    source,output=Path(source).resolve(),Path(output).resolve()
    if (output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents
            or source in output.parents or output in source.parents):
        raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=expected:
        raise ValueError('Explicit immutable parent hash required')
    text=original
    for signature in ('void UniformBody(','void FixedCubeCachedBody('):
        start,end,body=block(text,signature)
        text=text[:start]+transform_body(body)+text[end:]
    shutil.copytree(source,output)
    (output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.trace-shared-cache-exit.v1',proof=prove_exit(),
               parent_source=str(source),parent_consumer_sha256=expected,
               consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
               source_paired_pipeline_unchanged=True,expected_kernels=66,GPU_executed=False)
    (output.parent/'cache-exit-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=PARENT)
    p.add_argument('--expected-consumer-sha256',default=PARENT_SHA)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
