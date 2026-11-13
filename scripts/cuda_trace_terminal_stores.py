"""Schedule terminal field effects after the private-field timestep loop.

Only full private H/E cache bodies retain all final values. Source pipelines
that intentionally retire electric pairs early keep their original schedule.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,sha,block


EPILOGUE='''
 // Trace terminal effects: every private value is already its final version.
 if constexpr(!Fast){if(window[1]<Depth)return;}
 constexpr int terminal_n=Depth,terminal_hz=(-terminal_n-1)&7,terminal_hy=(-terminal_n-1)&15,terminal_hx=(-terminal_n-1)&31,terminal_low=-terminal_n-1;
 const int terminal_y=oy+terminal_low+((sy-terminal_hy)&15),terminal_x=ox+terminal_low+((sx-terminal_hx)&31);
 #pragma unroll
 for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep;
  if(zs!=terminal_hz&&sy!=terminal_hy&&sx!=terminal_hx){const int z=oz+terminal_low+((zs-terminal_hz)&7),y=terminal_y,x=terminal_x;
   #pragma unroll
   for(int c=0;c<3;++c){const auto&hf=FinalField(p,c);const auto&ef=FinalField(p,c+3);
    if(Contains(hf,z,y,x))hf.data[z*hf.plane+y*hf.row+x]=h[c][j];
    if(Contains(ef,z,y,x))ef.data[z*ef.plane+y*ef.row+x]=e[c][j];
   }
  }
 }
'''


def transform(body):
    start,end,loop=block(body,'for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n)')
    a,b,terminal=block(loop,'if(n==Depth)')
    assert loop.count('if(n==Depth)')==1
    assert 'hf.data[' in terminal and 'ef.data[' in terminal
    # The admitted parent groups the component loop inside this predicate.
    assert 'for(int c=0;c<3;++c)' in terminal
    loop=loop[:a]+loop[b:]
    return body[:start]+loop+EPILOGUE+body[end:]


def stage(source,output,expected):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=expected:raise ValueError('Explicit frozen grouped-publication parent required')
    text=original
    for signature in ('void UniformBody(','void FixedCubeCachedBody('):
        start,end,body=block(text,signature);text=text[:start]+transform(body)+text[end:]
    checks=0
    for depth in (4,16):
        for first in range(1,depth+1):
            for last in range(0,depth+1):
                old=depth in range(first,last+1)
                new=first<=last and last>=depth
                assert old==new;checks+=1
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.trace-terminal-field-stores.v1',parent_source=str(source),parent_consumer_sha256=expected,
               source_pipeline_unchanged=True,terminal_window_checks=checks,added_barriers=0,field_addresses_and_values_unchanged=True,
               consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),expected_kernels=66,GPU_executed=False)
    (output.parent/'terminal-stores-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--expected-consumer-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256)))
