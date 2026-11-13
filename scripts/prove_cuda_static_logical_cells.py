"""Canonical version oracle for fixed logical cells and time-invariant routes.

All previous-version operands shift by(-1,-1,-1) in the logical cache; current
H operands use their ordinary offsets. Each phase computes into one temporary
private bank, joins all old-family readers, then replaces that shared family.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA
from scripts.cuda_space_time_ir import FIELDS,yee_operator


def simulate(minimum=(13,17,21),owned=(7,15,31),depth=4,swizzle=True,negative=None):
    shape=[tuple(n+d for n,d in zip(minimum,delta)) for delta in SUPPORT_DELTA];resident=tuple(b+1 for b in owned);cz,cy,cx=resident;pad=math.prod(resident)
    logical=list(itertools.product(*(range(n) for n in resident)));own=[d for d in logical if all(d)];halo=[d for d in logical if not all(d)]
    def address(d,wrong=False):
        z,y,x=d;assert all(0<=v<n for v,n in zip(d,resident)),('Static route out of cache',d)
        if swizzle and not wrong:x^=(y|((z&1)<<4)) if cx==32 else y^(z&1)
        return (z*cy+y)*cx+x
    assert len({address(d) for d in logical})==pad,'Shared map is not bijective'
    valid=lambda f,p:all(0<=v<n for v,n in zip(p,shape[f]));linear=lambda f,p:(p[0]*shape[f][1]+p[1])*shape[f][2]+p[2]
    token=lambda f,p,n:(n,f,p) if valid(f,p) else None
    output=[[None]*math.prod(s) for s in shape];psi=[[0]*len(a) for a in output];history={};dead=object()
    counts=tuple((n+1+depth+b-1)//b for n,b in zip(minimum,owned));reads=imports=tasks=late=0
    operands={u.output.field:tuple(r for r in sorted(u.reads) if r.field in FIELDS) for u in yee_operator(cpml_axes=(0,1,2)).updates if u.output.field in FIELDS}
    def imported(f,p,n):
        nonlocal imports
        if not valid(f,p):return None
        producer=tuple((v+n)//b for v,b in zip(p,owned));actual=output[f][linear(f,p)] if n==depth else history.get((producer,n,f,p),dead)
        assert actual==token(f,p,n),('Static ghost packet version',producer,n,f,p,actual)
        imports+=1;return actual
    for q in sorted(itertools.product(*(range(c) for c in counts)),key=lambda q:(sum(q),q)):
        origin=tuple(v*b for v,b in zip(q,owned));first=max(1,*(o-n for o,n in zip(origin,minimum)));last=min(depth,*(o+b-1 for o,b in zip(origin,owned)));tasks+=1
        if first>last:continue
        late+=first>1;frame=[[dead]*pad for _ in range(6)];temporary=[[dead]*pad for _ in range(3)]
        for d in logical if first==1 else halo:
            p=tuple(o-first+v for o,v in zip(origin,d))
            for f in range(6):frame[f][address(d)]=token(f,p,0) if first==1 else imported(f,p,first-1)
        for n in range(first,last+1):
            physical=lambda d:tuple(o-n-1+v for o,v in zip(origin,d))
            for phase in (0,1):
                # Private arithmetic completes for ALL owned points before any
                # shared same-family output may overwrite an old center.
                for d in own:
                    p=physical(d)
                    for f in range(phase*3,phase*3+3):
                        if not valid(f,p):temporary[f%3][address(d)]=None;continue
                        for r in operands[FIELDS[f]]:
                            sf=FIELDS.index(r.field);sp=tuple(v+dv for v,dv in zip(p,r.offset))
                            if not valid(sf,sp):continue
                            shift=0 if phase and sf<3 else 1
                            if negative=='missing-previous-logical-shift' and shift:shift=0
                            if negative=='shift-current-H' and phase and sf<3:shift=1
                            source=tuple(v+dv-shift for v,dv in zip(d,r.offset));actual=frame[sf][address(source)];expected=token(sf,sp,n-1+r.version)
                            assert actual==expected,('Static canonical operand',q,n,FIELDS[f],r,d,source,actual,expected)
                            reads+=1
                        assert psi[f][linear(f,p)]==n-1,('Auxiliary version',q,n,f,p);psi[f][linear(f,p)]=n
                        temporary[f%3][address(d)]=token(f,p,n)
                        if negative==('early-H-shared-store' if phase==0 else 'early-E-shared-store'):frame[f][address(d)]=temporary[f%3][address(d)]
                # Reader retirement join, then owned and ghost publication.
                for d in own:
                    for f in range(phase*3,phase*3+3):frame[f][address(d)]=temporary[f%3][address(d)]
                if phase==0 or n<last:
                    for d in halo:
                        p=physical(d)
                        for f in range(phase*3,phase*3+3):frame[f][address(d,negative=='wrong-ghost-map')]=imported(f,p,n)
                # Family-visible join precedes its first consumer phase.
            for d in own:
                if n!=depth and not any(v==b for v,b in zip(d,owned)):continue
                p=physical(d)
                for f in range(6):
                    if not valid(f,p):continue
                    actual=frame[f][address(d)];assert actual==token(f,p,n),('Static export',q,n,f,p,actual)
                    if n==depth:output[f][linear(f,p)]=actual
                    else:history[q,n,f,p]=actual
    assert all(v==token(f,p,depth) for f,s in enumerate(shape) for p in itertools.product(*(range(n) for n in s)) for v in [output[f][linear(f,p)]]),'Incomplete fields'
    assert all(v==depth for a in psi for v in a),'Incomplete auxiliary histories'
    return dict(minimum=minimum,owned=owned,depth=depth,swizzle=swizzle,tasks=tasks,late_tasks=late,canonical_reads=reads,ghost_reads=imports,
                time_invariant_shared_routes=True,persistent_private_field_values=0,private_phase_output_values=3*pad,shared_field_families=6,phase_reader_retirement_joins=2,phase_publication_joins=2)


def report():
    cases=[simulate(owned=owned,depth=depth,swizzle=swizzle) for owned in ((7,15,31),(15,15,15)) for depth in (4,16) for swizzle in (False,True)]
    negatives={}
    for name in ('missing-previous-logical-shift','shift-current-H','early-H-shared-store','early-E-shared-store','wrong-ghost-map'):
        try:simulate(depth=4,negative=name)
        except AssertionError as error:negatives[name]=str(error)
        else:raise AssertionError('Ineffective static route negative:'+name)
    return dict(schema='beamz.static-logical-cell-routes.v1',passed=True,cases=cases,negative_controls=negatives,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Canonical field/auxiliary version operands and actualphysical owner packets under fixed logical coordinates; single temporary output bank, all6shared planes and4WAR/publication joins. No CUDA resource/visibility, numerical or speed claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=report();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(r['cases']),canonical_reads=sum(c['canonical_reads'] for c in r['cases']),ghost_reads=sum(c['ghost_reads'] for c in r['cases']),negative_controls=r['negative_controls'])))
