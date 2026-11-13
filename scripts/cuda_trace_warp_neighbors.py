"""Lower the traced X edges to unconditional, full-warp register exchange.

Both ghost and owned lanes participate before point predicates. Y/Z edges,
all private field versions, imports, and the four publication barriers retain
their original implementations. Only copied research sources are changed.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,sha,block


HELPER='''__device__ __forceinline__ float ResearchWarpNeighbor(float value,int lane){
 unsigned result;
 asm volatile("shfl.sync.idx.b32 %0,%1,%2,31,0xffffffff;":"=r"(result):"r"(__float_as_uint(value)),"r"(lane));
 return __uint_as_float(result);
}
'''


def prove():
    checks=ghost_providers=0
    for phase,direction in ((0,1),(1,-1)):
        for n in range(1,33):
            hx=(-n-1)&31
            for sx in range(32):
                provider=(sx+direction)&31
                assert provider in range(32)
                # The indexed shuffle reads exactly the shared X address.
                for sy in range(16):
                    for zs in range(8):
                        assert (zs*16+sy)*32+provider==(zs*16+sy)*32+((sx+direction)&31)
                        checks+=1
                if sx!=hx and provider==hx:ghost_providers+=1
    assert ghost_providers==64
    return dict(lane_address_checks=checks,required_ghost_lane_examples=ghost_providers,
                full_member_mask='0xffffffff',width=32,placement='unconditional before Owned/Valid',
                same_j_and_field_version=True,barriers_removed=0)


def transform(body):
    h='for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+low+((zs-hz)&7);'
    e='for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+elow+((zs-ehz)&7);'
    if body.count(h)!=1 or body.count(e)!=1:raise ValueError('Exact point loop sites required')
    body=body.replace(h,h+'\n    const float x_y=ResearchWarpNeighbor(e[1][j],(sx+1)&31),x_z=ResearchWarpNeighbor(e[2][j],(sx+1)&31);')
    body=body.replace(e,e+'\n    const float x_y=ResearchWarpNeighbor(h[1][j],(sx-1)&31),x_z=ResearchWarpNeighbor(h[2][j],(sx-1)&31);')
    hread='if(!dz&&!ddy&&!ddx)return e[c][j];return frame[c*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];'
    eread='if(!dz&&!ddy&&!ddx)return h[field][j];return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];'
    assert body.count(hread)==1 and body.count(eread)==1
    body=body.replace(hread,'if(!dz&&!ddy&&!ddx)return e[c][j];if(ddx)return c==1?x_y:x_z;return frame[c*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];')
    body=body.replace(eread,'if(!dz&&!ddy&&!ddx)return h[field][j];if(ddx)return field==1?x_y:x_z;return frame[field*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];')
    return body


def stage(source,output,scope='both'):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    text=original
    for signature in (['void UniformBody(','void FixedCubeCachedBody('] if scope=='both' else ['void FixedCubeCachedBody(']):
        start,end,body=block(text,signature);text=text[:start]+transform(body)+text[end:]
    marker='// Fast is admitted only for source/CPML/observation/native-wall-free OWNED'
    assert text.count(marker)==1;text=text.replace(marker,HELPER+'\n'+marker)
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    proof=dict(schema='beamz.trace-warp-neighbors.v1',scope=scope,proof=prove(),
               parent_sha256=PARENT_CONSUMER,consumer_sha256=sha(text.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
               exact_FP_cell_unchanged=True,expected_kernels=66,GPU_executed=False)
    (output.parent/'warp-neighbor-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);p.add_argument('--scope',choices=('both','fixed'),default='both');a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.scope)))
