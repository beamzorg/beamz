"""Prove whole-tile physical-frontier reuse across overlapping temporal macros.

Task(m,q) waits for its three lower axial neighbors in macro m, and one upper
corner task q+ceil(depth/owned) in macro m-1 (clipped to the prepared grid).
This model intentionally lets a later macro overwrite the physical frontier
before unrelated earlier-macro tasks finish. Every canonical read checks its
exact global version. Sources/raw observations remain a separate CUDA binding.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA
from scripts.cuda_space_time_ir import FIELDS,yee_operator


def macro_parent_theorem():
    """One-dimensional bounds compose across all three physical coordinates."""
    cases=[]
    for b in (3,5,7,15,31):
        for depth in (4,16,32):
            shift=(depth+b-1)//b;checks=0
            for c in range(5):
                # Every startup read needs the preceding macro's final owner.
                for d in range(b+1):
                    p=c*b-1+d
                    assert (p+depth)//b<=c+shift;checks+=1
                for n in range(1,depth+1):
                    for d in range(1,b+1):
                        p=c*b-1+d-n
                        # Every preceding-macro writer and stencil reader of
                        # a newly published point must already have retired.
                        # radius1 covers all Yee component offsets; exact
                        # version relationships are checked by token schedules.
                        for k in range(1,depth+1):
                            for offset in (-1,0,1):
                                assert (p-offset+k)//b<=c+shift;checks+=1
            cases.append(dict(owned=b,depth=depth,parent_shift=shift,checks=checks))
    return dict(passed=True,axis_cases=cases,
                statement='Previous-macro final producers and all previous-macro point readers are componentwise dominated by the upper-corner parent q+ceil(depth/owned). Three axial completion edges imply that dominance. Initial same-macro WAR uses the existing physical-frontier theorem.',
                clipped_grid='A valid preceding-macro consumer is inside the prepared grid; clipping the dominating parent to its upper corner preserves domination.')


def token_pipeline(minimum=(9,11,13),owned=(3,5,7),depth=4,macros=3,seed=13,negative=None):
    shapes=[tuple(n+d for n,d in zip(minimum,ds)) for ds in SUPPORT_DELTA]
    maximum=tuple(n+1 for n in minimum)
    counts=tuple((m+depth+b-1)//b for m,b in zip(maximum,owned))
    shift=tuple((depth+b-1)//b for b in owned)
    if negative=='floor-macro-parent':shift=tuple(depth//b for b in owned)
    if negative=='missing-macro-parent':shift=None
    points=tuple(itertools.product(*(range(c) for c in counts)))
    tasks=tuple((m,*q) for m in range(macros) for q in points)
    predecessors={};successors={task:[] for task in tasks};pending={}
    for task in tasks:
        m,*q=task;deps=[]
        for axis in range(3):
            if q[axis]:
                parent=q.copy();parent[axis]-=1;deps.append((m,*parent))
        if m and shift is not None:deps.append((m-1,*(min(x+d,c-1) for x,d,c in zip(q,shift,counts))))
        predecessors[task]=tuple(deps);pending[task]=len(deps)
        for parent in deps:successors[parent].append(task)
    ready=[t for t,n in pending.items() if not n];order=[];rng=random.Random(seed)
    while ready:
        # Bias toward later macros: stress early physical-frontier overwrites.
        highest=max(t[0] for t in ready)
        choices=[i for i,t in enumerate(ready) if t[0]==highest] if rng.random()<.85 else list(range(len(ready)))
        task=ready.pop(rng.choice(choices));order.append(task)
        for child in successors[task]:
            pending[child]-=1
            if not pending[child]:ready.append(child)
    assert len(order)==len(tasks),'Macro DAG progress failed'
    cache=tuple(itertools.product(*(range(b+1) for b in owned)))
    owned_q=tuple(q for q in cache if all(q));ghost_q=tuple(q for q in cache if not all(q))
    reads={u.output.field:tuple(r for r in sorted(u.reads) if r.field in FIELDS) for u in yee_operator(cpml_axes=(0,1,2)).updates if u.output.field in FIELDS}
    memory=[[0]*math.prod(s) for s in shapes]
    stats=dict(bootstrap_reads=0,ghost_reads=0,traced_reads=0,publications=0,early_macro_starts=0)
    finished=[0]*macros;started=set();raw_versions=[set() for _ in FIELDS]
    def valid(f,p):return all(0<=v<n for v,n in zip(p,shapes[f]))
    def index(f,p):return (p[0]*shapes[f][1]+p[1])*shapes[f][2]+p[2]
    def load(f,p,version,why,task,n):
        if not valid(f,p):return
        actual=memory[f][index(f,p)]
        if actual!=version:raise AssertionError(dict(task=task,local_step=n,field=FIELDS[f],point=p,why=why,expected=version,actual=actual,negative=negative))
    for task in order:
        m,*q=task;origin=tuple(t*b for t,b in zip(q,owned));base=m*depth
        if m not in started:
            if m and finished[m-1]<len(points):stats['early_macro_starts']+=1
            started.add(m)
        first=max(1,*(o-n+1 for o,n in zip(origin,maximum)))
        last=min(depth,*(o+b-1 for o,b in zip(origin,owned)))
        if first>last:finished[m]+=1;continue
        local=[{} for _ in FIELDS]
        if first==1:
            for d in cache:
                p=tuple(o-1+x for o,x in zip(origin,d))
                for f in range(6):
                    load(f,p,base,'bootstrap',task,0);local[f][p]=base if valid(f,p) else -1;stats['bootstrap_reads']+=int(valid(f,p))
        else:
            for d in ghost_q:
                p=tuple(o-first+x for o,x in zip(origin,d))
                for f in range(6):
                    load(f,p,base+first-1,'late ghost',task,first-1);local[f][p]=base+first-1 if valid(f,p) else -1
        for n in range(first,last+1):
            for phase in (0,1):
                for d in owned_q:
                    p=tuple(o-1+x-n for o,x in zip(origin,d))
                    for f in range(3*phase,3*phase+3):
                        if not valid(f,p):continue
                        for read in reads[FIELDS[f]]:
                            sf=FIELDS.index(read.field);sp=tuple(x+dx for x,dx in zip(p,read.offset))
                            if not valid(sf,sp):continue
                            expected=base+n-1+read.version
                            actual=local[sf].get(sp,-999)
                            if actual!=expected:raise AssertionError(('local read',task,n,f,p,read,sp,expected,actual))
                            stats['traced_reads']+=1
                        local[f][p]=base+n
                        effect=(base+n,p)
                        if effect in raw_versions[f]:raise AssertionError('Duplicate physical observation ownership')
                        raw_versions[f].add(effect)
                if phase==0:
                    for d in ghost_q:
                        p=tuple(o-1+x-n for o,x in zip(origin,d))
                        for f in range(6):
                            if n==depth and f>=3:continue
                            load(f,p,base+n,'current ghost',task,n);local[f][p]=base+n if valid(f,p) else -1;stats['ghost_reads']+=int(valid(f,p))
            for d in owned_q:
                if n!=depth and not any(x==b for x,b in zip(d,owned)):continue
                p=tuple(o-1+x-n for o,x in zip(origin,d))
                for f in range(6):
                    if valid(f,p):memory[f][index(f,p)]=base+n;stats['publications']+=1
        finished[m]+=1
    assert all(v==macros*depth for a in memory for v in a),'Missing final physical field value'
    assert all(len(a)==math.prod(shape)*depth*macros for a,shape in zip(raw_versions,shapes)),'Missing raw version effect'
    assert stats['early_macro_starts']>0,'No actual macro overlap exercised'
    return dict(minimum=minimum,owned=owned,depth=depth,macros=macros,seed=seed,counts=counts,macro_parent_shift=shift,tasks=len(tasks),**stats)


def report():
    cases=[token_pipeline(minimum=n,owned=b,depth=d,macros=m,seed=s)
           for n,b,d,m in [((9,11,13),(3,5,7),4,3),((9,11,13),(3,5,7),16,3),((13,17,21),(7,15,31),16,4)]
           for s in (13,37)]
    negatives={}
    for negative in ('floor-macro-parent','missing-macro-parent'):
        try:token_pipeline(negative=negative)
        except AssertionError as e:negatives[negative]=str(e)
        else:raise AssertionError('Ineffective macro negative:'+negative)
    return dict(schema='beamz.physical-frontier-macro-pipeline.v1',passed=True,theorem=macro_parent_theorem(),schedules=cases,negative_controls=negatives,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Canonical field version reads, bootstrap WAR and exactly-once per-cell raw-version effects under early later-macro execution; full FP32 numerical rounding, CUDA visibility, CPML/source/DFT lowering and performance remain separate.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=report();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,schedules=len(r['schedules']),traced_reads=sum(c['traced_reads'] for c in r['schedules']),early_macro_starts=sum(c['early_macro_starts'] for c in r['schedules']))))
