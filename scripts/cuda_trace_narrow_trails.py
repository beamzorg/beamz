"""Use a checked 32-bit trail-word index while retaining 64-bit pointers.

Prepare rejects a required trail arena larger than UINT32_MAX words before
launch. Invalid-base sentinels, ownership tags and metadata remain 64 bit.
This is an explicit research admission bound, not truncation of large arenas.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,sha


def stage(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    text=original
    old='if(owners>std::numeric_limits<size_t>::max()/w.depth/6/SurfacePoints||w.trail_floats<owners*w.depth*6*SurfacePoints)return false;'
    assert text.count(old)==1
    text=text.replace(old,old+'\n if(owners*w.depth*6*SurfacePoints>std::numeric_limits<uint32_t>::max())return false;')
    old='p.trails+base+static_cast<size_t>(version-1)*6*SurfacePoints+surface;'
    assert text.count(old)==1
    text=text.replace(old,'p.trails+(uint32_t(base)+uint32_t((version-1)*6*SurfacePoints+surface));')
    changed=0
    for field in ('c','c+3'):
        old=f'p.trails[bases[0]+((n-1)*6+{field})*SurfacePoints+surface]'
        count=text.count(old);assert count==4
        text=text.replace(old,f'p.trails[uint32_t(bases[0])+uint32_t(((n-1)*6+{field})*SurfacePoints+surface)]');changed+=count
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.trace-checked-32bit-trail-indices.v1',publication_addresses_changed=changed,
               maximum_required_arena_words=2**32-1,larger_arenas_rejected_before_launch=True,
               base_sentinel_and_ownership_tags_64bit=True,pointers_64bit=True,FP_and_schedule_unchanged=True,
               parent_sha256=PARENT_CONSUMER,consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
               expected_kernels=66,GPU_executed=False)
    (output.parent/'narrow-trails-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output)))
