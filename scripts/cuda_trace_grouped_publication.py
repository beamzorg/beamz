"""Group disjoint field stores under one surface and one final-version guard.

The native runtime allocates distinct field components and a separate trail
arena. Publication values and byte addresses stay unchanged; only independent
store nodes are reordered. Every existing synchronization boundary is retained.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,sha,block
from scripts.cuda_trace_effect_schedule import prove_native_publication


def transform(text):
    anchor='if(surface>=0){p.trails[';sites=text.count(anchor)
    if sites!=4:raise ValueError('Exact four publication sites required')
    cursor=0
    for _ in range(sites):
        publication=text.index(anchor,cursor);start=text.rfind('for(int c=0;c<3;++c){',0,publication)
        assert start>=0
        _,length,loop=block(text[start:],'for(int c=0;c<3;++c){')
        a,b,surface=block(loop,'if(surface>=0)');c,d,final=block(loop,'if(n==Depth)')
        prefix=loop[loop.index('{')+1:a]
        assert not loop[b:c].strip() and not loop[d:-1].strip()
        assert 'p.trails[' not in prefix and '.data[' not in prefix
        def grouped(condition,body):
            return condition+'{\n    #pragma unroll\n    for(int c=0;c<3;++c){'+prefix+body[body.index('{')+1:-1]+'}\n   }'
        replacement=grouped('if(surface>=0)',surface)+'\n   '+grouped('if(n==Depth)',final)
        pragma=text.rfind('#pragma unroll',0,start)
        assert text[pragma:start].strip()=='#pragma unroll'
        text=text[:pragma]+replacement+text[start+length:];cursor=pragma+len(replacement)
    assert text.count(anchor)==0
    return text,sites


def stage(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    text,sites=transform(original)
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.trace-grouped-trail-publication.v1',publication_sites=sites,effect_schedule=prove_native_publication(),
               expected_component_order=list(range(3)),addresses_and_values_unchanged=True,
               memory_fact='Private native component buffers and trail arena are disjoint',added_barriers=0,
               parent_sha256=PARENT_CONSUMER,consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
               expected_kernels=66,GPU_executed=False)
    (output.parent/'grouped-publication-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output)))
