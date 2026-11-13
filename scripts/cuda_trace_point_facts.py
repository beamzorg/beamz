"""Compact invariant facts for one physical point's cache residency.

Facts are refreshed when a point first enters the owned region. They do not
encode evolving field/psi values, source amplitudes or equal profile values.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA

AXES=(1,0,0,2,2,1)


def native_mask(shape,p):
    return sum(int(all(0<=q<n+d for q,n,d in zip(p,shape,delta,strict=True)))<<f for f,delta in enumerate(SUPPORT_DELTA))


def encode(shape,p):
    mask=native_mask(shape,p)
    if not mask:return 0
    tag=mask
    for axis,(n,q) in enumerate(zip(shape,p,strict=True)):
        code=q if q<12 else 24+q-n if q>=n-12 else 31
        if not 0<=code<=31:raise ValueError('Noncanonical point fact')
        tag|=code<<(6+axis*5)
    return tag


def code(tag,axis):return (tag>>(6+axis*5))&31


def active(tag,slot):
    q=code(tag,AXES[slot%6]);return q<12 or (13 if slot//6 else 12)<=q<=24


def packed(tag,slot):
    q=code(tag,AXES[slot%6]);return q-int(slot>=6 and q>=12)


def wall(tag,axis):return code(tag,axis) in (0,24)


CPP='''// Generated native geometry facts, refreshed on physical-point entry.
__device__ __forceinline__ uint32_t EncodeNativePointFacts(const Parameters&p,int z,int y,int x){
 const uint32_t mask=NativeSupportMask(p,z,y,x);if(!mask)return 0;
 const int coordinates[3]={z,y,x};uint32_t tag=mask;
 #pragma unroll
 for(int axis=0;axis<3;++axis){const int q=coordinates[axis],size=p.physical_n[axis];
  const int code=q<12?q:q>=size-12?24+q-size:31;
  tag|=uint32_t(code)<<(6+axis*5);
 }
 return tag;
}
__host__ __device__ __forceinline__ int NativePointAxisCode(uint32_t tag,int axis){return (tag>>(6+5*axis))&31;}
__host__ __device__ __forceinline__ bool NativePointLocalActive(uint32_t tag,int slot){
 const int c=NativePointAxisCode(tag,beamz::cuda::yee::CpmlAxis(slot%6));
 return c<12||(c>=(slot>=6?13:12)&&c<=24);
}
__host__ __device__ __forceinline__ int NativePointPacked(uint32_t tag,int slot){
 const int c=NativePointAxisCode(tag,beamz::cuda::yee::CpmlAxis(slot%6));return c-int(slot>=6&&c>=12);
}
__host__ __device__ __forceinline__ bool NativePointElectricWall(uint32_t tag,int axis){
 const int c=NativePointAxisCode(tag,axis);return c==0||c==24;
}
'''


def verify_predicates(shape):
    values=[sorted(set(range(-2,15))|set(range(n-14,n+3))|{n//2}) for n in shape]
    checks=0
    for p in itertools.product(*values):
        tag=encode(shape,p);mask=native_mask(shape,p)
        assert tag&63==mask
        for f in range(6):
            if not mask&(1<<f):continue
            family,component=divmod(f,3)
            for term in range(component*2,component*2+2):
                slot=family*6+term;axis=AXES[term];coord=p[axis];size=shape[axis]+family
                expected=coord<12 or coord>=size-12
                assert active(tag,slot)==expected,(shape,p,f,slot,tag)
                if expected:
                    expected_packed=coord if coord<12 else 24+coord-size
                    assert packed(tag,slot)==expected_packed
                    assert 0<=expected_packed<24
                if family:assert wall(tag,axis)==(coord==0 or coord==shape[axis])
                checks+=1
    return dict(shape=shape,point_cases=1 if not values else len(values[0])*len(values[1])*len(values[2]),term_checks=checks)


def verify_lifetime(cache=(16,16,32),owned=(15,15,31),shape=(64,96,128),origin=(45,75,93),first=1,steps=35,negative=False):
    points=tuple(itertools.product(*(range(c) for c in cache)))
    state={q:0xffffffff for q in points};checks=refreshes=0
    for n in range(first,steps+1):
        low=-n-1
        for q in points:
            d=tuple((x-low)%c for x,c in zip(q,cache,strict=True))
            if not all(1<=x<=b for x,b in zip(d,owned,strict=True)):continue
            p=tuple(o+low+x for o,x in zip(origin,d,strict=True))
            refresh=n==first or any(x==1 for x in d)
            if refresh and not(negative and n!=first):state[q]=encode(shape,p);refreshes+=1
            if state[q]!=encode(shape,p):raise AssertionError(('stale physical-point fact',n,q,p,state[q],encode(shape,p)))
            # All three H and three E sites share this owned physical point;
            # phase-independent facts can be reused by both arithmetic phases.
            checks+=2
    return dict(cache=cache,owned=owned,shape=shape,origin=origin,first=first,steps=steps,phase_fact_checks=checks,refreshes=refreshes)


def report():
    predicates=[verify_predicates(s) for s in ((27,29,31),(32,35,37),(512,512,512),(564,564,564))]
    lifetimes=[verify_lifetime(first=f,origin=o) for f in (1,4,13) for o in ((0,0,0),(45,75,93),(60,90,122))]
    lifetimes+=[verify_lifetime(owned=(14,15,31),first=f) for f in (1,4,13)]
    try:verify_lifetime(negative=True)
    except AssertionError as e:negative=str(e)
    else:raise AssertionError('Missing refresh was not detected')
    return dict(schema='beamz.physical-point-fact-cache.v1',passed=True,predicates=predicates,lifetimes=lifetimes,negative_control=negative,
                encoding='6support bits,3five-bit H-packed/interior codes; E high-layer correction and wall flags derived from same code',
                limit='Only facts for valid canonical staggered updates/actual tangential CPML terms. H coordinateN code24 is never used as a profile index by a valid H term. No CUDA implementation or speedup claim.',
                producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=report();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,term_checks=sum(x['term_checks']for x in r['predicates']),phase_fact_checks=sum(x['phase_fact_checks']for x in r['lifetimes']))))
