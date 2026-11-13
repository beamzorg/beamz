"""Compose disjoint, explicitly named scalar-consumer research changes.

Every parent is compared with the exact frozen baseline. Three-way conflicts
and changes to any other native source are rejected. A composition requires its
own resource, full-state and performance gates; parent ratios are never multiplied.
"""
from __future__ import annotations
import argparse,json,subprocess,tempfile,shutil
from pathlib import Path
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,sha,block,transform_body


def stage(source,output,parents,opaque_clock=False):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied source required')
    path='src/asymmetric_native.cuh';original=(source/path).read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact frozen composition baseline required')
    text=original;inputs=[]
    with tempfile.TemporaryDirectory(prefix='beamz-compose-') as temporary:
        d=Path(temporary);base=d/'base.cuh';base.write_text(original)
        for parent in map(Path,parents):
            parent=parent.resolve()
            for item in source.rglob('*'):
                if item.is_file() and item.relative_to(source).as_posix()!=path:
                    other=parent/item.relative_to(source)
                    if not other.is_file() or sha(item.read_bytes())!=sha(other.read_bytes()):raise ValueError('Composition only permits consumer changes: '+str(other))
            value=(parent/path).read_text();left=d/'left.cuh';right=d/'right.cuh';left.write_text(text);right.write_text(value)
            result=subprocess.run(['git','merge-file','--stdout',str(left),str(base),str(right)],text=True,capture_output=True)
            if result.returncode:raise ValueError('Overlapping composition rejected: '+str(parent))
            text=result.stdout;inputs.append(dict(source=str(parent),consumer_sha256=sha(value.encode())))
    if opaque_clock:
        for signature in ('void UniformBody(','void FixedCubeCachedBody('):
            start,end,body=block(text,signature);text=text[:start]+transform_body(body,1,clock='opaque-mov')+text[end:]
    if text==original:raise ValueError('Empty composition')
    shutil.copytree(source,output);(output/path).write_text(text)
    proof=dict(schema='beamz.explicit-trace-composition.v1',parent_sha256=PARENT_CONSUMER,composed_parents=inputs,
               opaque_phase_clock=opaque_clock,consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
               independent_validation_and_timing_required=True,expected_kernels=66,GPU_executed=False)
    (output.parent/'composition-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);p.add_argument('--parents',type=Path,nargs='+',required=True);p.add_argument('--opaque-clock',action='store_true');a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.parents,a.opaque_clock)))
