"""Prove two shared in-place Yee phases from canonical field/version operands."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from scripts.cuda_native_geometry_ir import SUPPORT_DELTA
from scripts.cuda_space_time_ir import FIELDS,yee_operator


def simulate(minimum=(13,17,21),owned=(7,15,31),depth=4,negative=None):
    resident=tuple(b+1 for b in owned);cz,cy,cx=resident;pad=math.prod(resident);assert pad==4096
    logical=list(itertools.product(*(range(n) for n in resident)))
    shape=[tuple(n+d for n,d in zip(minimum,s)) for s in SUPPORT_DELTA];counts=tuple((n+1+depth+b-1)//b for n,b in zip(minimum,owned))
    def slot(s):
        z,y,x=(v%n for v,n in zip(s,resident));x^=(y|((z&1)<<4)) if cx==32 else (y^(z&1));return (z*cy+y)*cx+x
    assert len({slot(s) for s in logical})==pad
    valid=lambda f,p:all(0<=v<n for v,n in zip(p,shape[f]));index=lambda f,p:(p[0]*shape[f][1]+p[1])*shape[f][2]+p[2]
    token=lambda f,p,n:(f,p,n) if valid(f,p) else None
    output=[[None]*math.prod(s) for s in shape];psi=[[0]*len(a) for a in output];history={};reads=0;ghost_reads=0;tasks=0;late=0
    updates=[u for u in yee_operator(cpml_axes=(0,1,2)).updates if u.output.field in FIELDS]
    operands={u.output.field:[r for r in sorted(u.reads) if r.field in FIELDS] for u in updates};same_family=0
    for u in updates:
        f=FIELDS.index(u.output.field)
        for r in operands[u.output.field]:
            sf=FIELDS.index(r.field)
            if sf//3==f//3:
                assert sf==f and r.offset==(0,0,0) and r.version==0,('Unsafe in-place same-family dependency',f,r)
                same_family+=1
    assert same_family==6
    def imported(f,p,n):
        nonlocal ghost_reads
        if not valid(f,p):return None
        q=tuple((v+n)//b for v,b in zip(p,owned));value=output[f][index(f,p)] if n==depth else history.get((q,n,f,p))
        assert value==token(f,p,n),('Global ghost operand',q,n,f,p,value);ghost_reads+=1;return value
    for q in sorted(itertools.product(*(range(n) for n in counts)),key=lambda q:(sum(q),q)):
        origin=tuple(v*b for v,b in zip(q,owned));first=max(1,*(o-n for o,n in zip(origin,minimum)));last=min(depth,*(o+b-1 for o,b in zip(origin,owned)));tasks+=1
        if first>last:continue
        late+=first>1;frame=[[None]*pad for _ in range(6)]
        for s in logical:
            if first==1:
                p=tuple(o+(v if v!=d-1 else -1) for o,v,d in zip(origin,s,resident))
                for f in range(6):frame[f][slot(s)]=token(f,p,0)
            else:
                prev=first-1;g=tuple(-first%n for n in resident);d=tuple((v-a)%n for v,a,n in zip(s,g,resident));p=tuple(o-first+x for o,x in zip(origin,d))
                if any(v==0 for v in d):
                    for f in range(6):frame[f][slot(s)]=imported(f,p,prev)
        for n in range(first,last+1):
            g=tuple((-n-1)%d for d in resident);physical=lambda s:tuple(o-n-1+(v-a)%d for o,v,a,d in zip(origin,s,g,resident))
            own=[s for s in logical if all(v!=a for v,a in zip(s,g))];halo=[s for s in logical if any(v==a for v,a in zip(s,g))]
            for phase in range(2):
                if negative=='E-overwrite-before-H-join' and phase==0:
                    for s in own:
                        for f in range(3,6):frame[f][slot(s)]=token(f,physical(s),n)
                if negative=='next-H-overwrite-before-E-join' and phase==1:
                    for s in own:
                        for f in range(3):frame[f][slot(s)]=token(f,physical(s),n+1)
                for s in own:
                    p=physical(s)
                    for f in range(phase*3,phase*3+3):
                        if not valid(f,p):frame[f][slot(s)]=None;continue
                        for r in operands[FIELDS[f]]:
                            sf=FIELDS.index(r.field);sp=tuple(v+dx for v,dx in zip(p,r.offset))
                            if not valid(sf,sp):continue
                            adjacent=tuple(v+dx for v,dx in zip(s,r.offset));actual=frame[sf][slot(adjacent)];expected=token(sf,sp,n-1+r.version)
                            assert actual==expected,('In-place phase operand',q,n,phase,f,str(r),s,sp,actual,expected);reads+=1
                        assert psi[f][index(f,p)]==n-1,('Auxiliary version order',q,n,f,p);psi[f][index(f,p)]=n;frame[f][slot(s)]=token(f,p,n)
                for s in own:
                    p=physical(s);d=tuple((v-a)%n for v,a,n in zip(s,g,resident));surface=any(v==b for v,b in zip(d,owned))
                    if not surface and n!=depth:continue
                    for f in range(phase*3,phase*3+3):
                        if not valid(f,p):continue
                        value=frame[f][slot(s)];assert value==token(f,p,n)
                        if n==depth:output[f][index(f,p)]=value
                        else:history[q,n,f,p]=value
                if phase==0 or n<last:
                    for s in halo:
                        p=physical(s)
                        for f in range(phase*3,phase*3+3):frame[f][slot(s)]=None if negative==('missing-H-ghost' if phase==0 else 'missing-E-ghost') else imported(f,p,n)
                # H join: publishes H, retires all E-old readers. E join is
                # the existing writer-fence/publication CTA barrier and also
                # retires all H readers before the next H in-place updates.
    assert all(v==token(f,p,depth) for f,s in enumerate(shape) for p in itertools.product(*(range(n) for n in s)) for v in [output[f][index(f,p)]]),'Incomplete final fields'
    assert all(v==depth for a in psi for v in a),'Incomplete auxiliary outputs'
    return dict(minimum=minimum,owned=owned,depth=depth,tasks=tasks,late_tasks=late,canonical_reads=reads,ghost_reads=ghost_reads,same_family_center_only_updates=same_family,shared_family_values=6*pad,persistent_private_field_values=0,step_CTA_barriers=2)


def report():
    cases=[simulate(owned=b,depth=d) for b in ((7,15,31),(15,15,15)) for d in (4,16)];negatives={}
    for name in ('E-overwrite-before-H-join','next-H-overwrite-before-E-join','missing-H-ghost','missing-E-ghost'):
        try:simulate(negative=name)
        except AssertionError as e:negatives[name]=str(e)
        else:raise AssertionError('Ineffective shared-phase negative:'+name)
    return dict(passed=True,schema='beamz.double-shared-inplace-phase-model.v1',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=cases,negative_controls=negatives,
                proof='Canonical same-family operands are only the same field/point old center. Each thread overwrites that owned center only after reading it; ghost writes are disjoint from all owned same-family center reads. The H join publishes H and retires E-old stencil reads; the final release publication join publishes E and retires H stencil reads.',
                scope='Canonical six-field and auxiliary version operands; in-place dependency/ownership proof. No FP32/CUDA-memory-visibility or performance claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=report();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(r['cases']),canonical_reads=sum(c['canonical_reads'] for c in r['cases']),negatives=r['negative_controls'])))
