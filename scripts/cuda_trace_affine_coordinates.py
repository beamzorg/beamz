"""Strength-reduce periodic owner-to-physical coordinate expressions.

For a power-of-two cache extent C, low=-step-1 and head=low&(C-1):
 low+((slot-head)&(C-1)) == slot-((slot+step+1)&~(C-1)).
This exact integer identity retains every physical point and field lifetime.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,sha,block


def prove():
    checks=0
    for c in (8,16,32):
        for step in range(1,130):
            for slot in range(c):
                low=-step-1;head=low&(c-1)
                assert low+((slot-head)&(c-1))==slot-((slot+step+1)&~(c-1))
                assert ((slot-head)&(c-1))==((slot+step+1)&(c-1))
                checks+=1
    return dict(periodic_coordinate_checks=checks,steps=129,extents=[8,16,32])


def transform(body):
    replacements={
      'dy=(sy-hy)&15,dx=(sx-hx)&31,y=oy+low+dy,x=ox+low+dx;':
      'dy=(sy+step+1)&15,dx=(sx+step+1)&31,y=oy+sy-((sy+step+1)&~15),x=ox+sx-((sx+step+1)&~31);',
      'z=oz+low+((zs-hz)&7);':'z=oz+zs-((zs+step+1)&~7);',
      'y=oy+elow+((sy-ehy)&15),x=ox+elow+((sx-ehx)&31);':
      'y=oy+sy-((sy+step+1)&~15),x=ox+sx-((sx+step+1)&~31);',
      'z=oz+elow+((zs-ehz)&7);':'z=oz+zs-((zs+step+1)&~7);',
      'const int y=oy+low+dy,x=ox+low+dx;':
      'const int y=oy+sy-((sy+n+1)&~15),x=ox+sx-((sx+n+1)&~31);',
      'z=oz+low+dz,surface=Surface(':'z=oz+zs-((zs+n+1)&~7),surface=Surface(',
    }
    for before,after in replacements.items():
        if body.count(before)!=1:raise ValueError('Exact reviewed coordinate site required: '+before)
        body=body.replace(before,after)
    return body


def stage(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    text=original
    for signature in ('void UniformBody(','void FixedCubeCachedBody('):
        start,end,body=block(text,signature);text=text[:start]+transform(body)+text[end:]
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.trace-affine-coordinates.v1',proof=prove(),parent_sha256=PARENT_CONSUMER,
               consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
               scalar_cell_and_FP_unchanged=True,expected_kernels=66,GPU_executed=False)
    (output.parent/'affine-coordinate-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output)))
