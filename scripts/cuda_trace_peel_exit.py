"""Peel the terminal bulk phase using finite-window cache liveness.

The main Fast/Layout1/isotropic-H body executes1..Depth exactly. Its final
iteration can use constant rotating heads, skip dead E cache publication and
emit terminal state stores without an inner n==Depth test. Other paths keep
the admitted dynamic-window body. No floating-point expression is changed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

from scripts.cuda_trace_phase_cycle import ROOT,block,sha
from scripts.cuda_trace_cache_exit import prove_exit


def transform(body):
    anchor='for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n)'
    begin,end,loop=block(body,anchor)
    iteration=loop[loop.index('{')+1:-1]
    guard='if(n<(Fast?Depth:window[1]))'
    start,stop,publication=block(iteration,guard)
    interior=iteration[:start]+publication[publication.index('{')+1:-1]+iteration[stop:]
    start,stop,_=block(interior,'if(n==Depth)')
    interior=interior[:start]+interior[stop:]
    start,stop,_=block(iteration,guard)
    terminal=iteration[:start]+iteration[stop:]
    clock='int step;asm volatile("mov.b32 %0,%1;":"=r"(step):"r"(n));'
    if terminal.count(clock)!=2:
        raise ValueError('Both reviewed opaque phase clocks required')
    terminal=terminal.replace(clock,'constexpr int step=Depth;')
    start,stop,final=block(terminal,'if(n==Depth)')
    terminal=terminal[:start]+final[final.index('{')+1:-1]+terminal[stop:]
    replacement='''if constexpr(Fast&&Layout==1&&IsotropicH){
 #pragma unroll 1
 for(int n=1;n<Depth;++n){'''+interior+'''}
 {constexpr int n=Depth;'''+terminal+'''}
 }else{
 #pragma unroll 1
 '''+loop+'\n }'
    pragma=body.rfind('#pragma unroll 1',0,begin)
    if body[pragma:begin].strip()!='#pragma unroll 1':
        raise ValueError('Exact loop pragma required')
    return body[:pragma]+replacement+body[end:]


def stage(source,output,expected):
    source,output=Path(source).resolve(),Path(output).resolve()
    if (output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents
            or source in output.parents or output in source.parents):
        raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=expected:raise ValueError('Explicit immutable parent hash required')
    begin,end,body=block(original,'void UniformBody(')
    text=original[:begin]+transform(body)+original[end:]
    shutil.copytree(source,output)
    (output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.trace-peeled-cache-exit.v1',proof=prove_exit(),
                parent_source=str(source),parent_consumer_sha256=expected,
                consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
                admission='Fast && Layout1 && isotropic-H; complete1..Depth window',
                sequence_checks=[dict(depth=d,old=list(range(1,d+1)),new=list(range(1,d))+[d]) for d in (4,16)],
                final_heads_constant=True,source_paired_pipeline_unchanged=True,expected_kernels=66,GPU_executed=False)
    (output.parent/'peeled-exit-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--expected-consumer-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
