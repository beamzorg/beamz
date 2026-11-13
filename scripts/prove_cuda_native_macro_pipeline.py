"""Prove overlapping macros with two native field banks and compact histories.

Global affine levels join, while task order within a level is randomized.
Startup reads use the macro-parity input bank; endpoints use the other bank.
All intermediate ghost reads use a four-level owner-tagged face history ring.
Every canonical read checks its exact global version, including old bootstrap
readers while a later macro is already overwriting the other field bank.
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
    stride=1+sum(shift)
    if negative=='short-macro-stride':stride-=1
    epochs=3 if negative=='short-history-ring' else 4
    points=tuple(itertools.product(*(range(c) for c in counts)))
    tasks=tuple((m,*q) for m in range(macros) for q in points)
    levels={};slots={};order=[]
    for task in tasks:levels.setdefault(sum(task[1:])+task[0]*stride,[]).append(task)
    frontier=max(map(len,levels.values()))
    for level,items in sorted(levels.items()):
        random.Random(seed+level).shuffle(items)
        for slot,task in enumerate(items):slots[task]=slot
        execution=items.copy();random.Random(seed*7+level).shuffle(execution);order.extend(execution)
    def owner(task):return ((sum(task[1:])+task[0]*stride)%epochs)*frontier+slots[task]
    owners={};history={}
    cache=tuple(itertools.product(*(range(b+1) for b in owned)))
    owned_q=tuple(q for q in cache if all(q));ghost_q=tuple(q for q in cache if not all(q))
    reads={u.output.field:tuple(r for r in sorted(u.reads) if r.field in FIELDS) for u in yee_operator(cpml_axes=(0,1,2)).updates if u.output.field in FIELDS}
    banks=[[[0]*math.prod(s) for s in shapes],[[-1]*math.prod(s) for s in shapes]]
    stats=dict(bootstrap_reads=0,ghost_reads=0,traced_reads=0,publications=0,early_macro_starts=0)
    finished=[0]*macros;started=set();raw_versions=[set() for _ in FIELDS]
    def valid(f,p):return all(0<=v<n for v,n in zip(p,shapes[f]))
    def index(f,p):return (p[0]*shapes[f][1]+p[1])*shapes[f][2]+p[2]
    def load(f,p,version,why,task,n):
        if not valid(f,p):return
        m=task[0]
        if why=='bootstrap':
            bank=(m+1)%2 if negative=='wrong-initial-bank' else m%2
            actual=banks[bank][f][index(f,p)]
        else:
            local_version=version-m*depth
            producer=(m,*(int((x+local_version)//b) for x,b in zip(p,owned)))
            slot=owner(producer)
            if owners.get(slot)!=producer:raise AssertionError(dict(reason='History owner clobbered',producer=producer,consumer=task,actual=owners.get(slot),negative=negative))
            actual=history[slot].get((local_version,f,p),-999)
        if actual!=version:raise AssertionError(dict(task=task,local_step=n,field=FIELDS[f],point=p,why=why,expected=version,actual=actual,negative=negative))
    for task in order:
        m,*q=task;origin=tuple(t*b for t,b in zip(q,owned));base=m*depth
        slot=owner(task);owners[slot]=task;history[slot]={}
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
                    if not valid(f,p):continue
                    if any(x==b for x,b in zip(d,owned)) and (n<depth or f<3):history[slot][(n,f,p)]=base+n
                    if n==depth:banks[(m+1)%2][f][index(f,p)]=base+n
                    stats['publications']+=1
        finished[m]+=1
    assert all(v==macros*depth for a in banks[macros%2] for v in a),'Missing final native bank field value'
    assert all(len(a)==math.prod(shape)*depth*macros for a,shape in zip(raw_versions,shapes)),'Missing raw version effect'
    assert stats['early_macro_starts']>0,'No actual macro overlap exercised'
    return dict(minimum=minimum,owned=owned,depth=depth,macros=macros,seed=seed,counts=counts,macro_parent_shift=shift,macro_level_stride=stride,history_epochs=epochs,max_frontier=frontier,tasks=len(tasks),**stats)


def report():
    cases=[token_pipeline(minimum=n,owned=b,depth=d,macros=m,seed=s)
           for n,b,d,m in [((9,11,13),(3,5,7),4,3),((9,11,13),(3,5,7),16,3),((13,17,21),(7,15,31),16,4),((13,17,21),(7,15,31),16,8)]
           for s in (13,37)]
    negatives={}
    for negative in ('short-macro-stride','short-history-ring','wrong-initial-bank'):
        try:token_pipeline(negative=negative)
        except AssertionError as e:negatives[negative]=str(e)
        else:raise AssertionError('Ineffective macro negative:'+negative)
    return dict(schema='beamz.native-banked-macro-pipeline.v1',passed=True,theorem=macro_parent_theorem(),schedules=cases,negative_controls=negatives,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                history_theorem='Same-macro ghost consumers advance1..3globallevels; any next four-level ring writer is strictly after every old consumer. Slot injectivity within each level prevents same-level aliasing.',
                scope='Canonical reads against two native parity banks and actual owner-tagged compact history ring under globallevel joins; exactly-once raw effects, terminalHtrails and earlymacro overwrites. CUDA visibility and source/DFT rounding remain separate.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=report();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,schedules=len(r['schedules']),traced_reads=sum(c['traced_reads'] for c in r['schedules']),early_macro_starts=sum(c['early_macro_starts'] for c in r['schedules']))))
