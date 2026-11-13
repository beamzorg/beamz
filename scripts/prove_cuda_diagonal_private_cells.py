"""Canonical operand/version oracle for diagonal-warp logical register caches.

Threads follow XY diagonals; a previous physical center is the previous Z slot
and previous lane of the SAME warp. Descending Z preserves old registers.
Two contiguous Z groups use shared lower-Z3 snapshots protected by phase joins.
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


def simulate(minimum=(13,17,21),depth=4,threads=1024,seed=13,negative=None):
    dims=(8,16,32);owned=(7,15,31);pad=math.prod(dims);local_z=8 if threads==512 else 4;groups=threads//512
    shape=[tuple(n+d for n,d in zip(minimum,delta)) for delta in SUPPORT_DELTA];counts=tuple((n+1+depth+b-1)//b for n,b in zip(minimum,owned))
    logical=list(itertools.product(*(range(n) for n in dims)));own=[d for d in logical if all(d)];ghost=[d for d in logical if not all(d)]
    def point(t,j):x=t%32;u=(t//32)%16;return (t//512)*local_z+j,(x+u)%16,x
    def inverse(d):z,y,x=d;return (z//local_z)*512+((y-x)%16)*32+x,z%local_z
    assert len({point(t,j) for t in range(threads) for j in range(local_z)})==pad
    assert all(point(*inverse(d))==d for d in logical)
    def slot(d):z,y,x=d;assert 0<=z<8 and 0<=y<16 and 0<=x<32;return (z*16+y)*32+((x+2*y+z)%32)
    def boundary_slot(y,x):return y*32+((x+2*y)%32)
    assert len({slot(d) for d in logical})==pad
    valid=lambda f,p:all(0<=v<n for v,n in zip(p,shape[f]));linear=lambda f,p:(p[0]*shape[f][1]+p[1])*shape[f][2]+p[2]
    token=lambda f,p,n:(n,f,p) if valid(f,p) else None
    output=[[None]*math.prod(s) for s in shape];history={};dead=object();rng=random.Random(seed)
    operands={u.output.field:tuple(r for r in sorted(u.reads) if r.field in FIELDS) for u in yee_operator(cpml_axes=()).updates if u.output.field in FIELDS}
    stats=dict(canonical_reads=0,private_diagonal_reads=0,private_current_reads=0,shared_XY_reads=0,boundary_old_reads=0,boundary_current_reads=0,global_ghost_reads=0,warp_events=0,tasks=0,late_tasks=0)
    def imported(f,p,n):
        if not valid(f,p):return None
        q=tuple((v+n)//b for v,b in zip(p,owned));value=output[f][linear(f,p)] if n==depth else history.get((q,n,f,p),dead)
        assert value==token(f,p,n),('Diagonal global packet version',q,n,f,p,value)
        stats['global_ghost_reads']+=1;return value
    for q in sorted(itertools.product(*(range(c) for c in counts)),key=lambda q:(sum(q),q)):
        origin=tuple(v*b for v,b in zip(q,owned));first=max(1,*(o-n for o,n in zip(origin,minimum)));last=min(depth,*(o+b-1 for o,b in zip(origin,owned)));stats['tasks']+=1
        if first>last:continue
        stats['late_tasks']+=first>1
        cache=[[[None]*local_z for _ in range(threads)] for _ in range(6)];frame=[[None]*pad for _ in range(3)];snap=[[None]*512 for _ in range(6)]
        def get(f,d):t,j=inverse(d);return cache[f][t][j]
        def put(f,d,v):t,j=inverse(d);cache[f][t][j]=v
        # Coalesced initial IO and diagonal private reloads are represented by
        # their shared point identity; both Z snapshots precede first H math.
        for d in logical:
            p=tuple(o-first+v for o,v in zip(origin,d))
            for f in range(6):
                value=token(f,p,0) if first==1 else imported(f,p,first-1) if not all(d) else None
                put(f,d,value)
                if f>=3:frame[f-3][slot(d)]=value
                if groups==2 and d[0]==3:snap[f][boundary_slot(d[1],d[2])]=value
        for n in range(first,last+1):
            physical=lambda d:tuple(o-n-1+v for o,v in zip(origin,d))
            def old_diagonal(f,t,j):
                z,y,x=point(t,j)
                if j>0:
                    lane=((x+1) if negative=='wrong-shuffle-lane' else (x-1))%32;source=(t//32)*32+lane
                    stats['private_diagonal_reads']+=1;return cache[f][source][j-1]
                assert z>=4,'A logical low-Z ghost cannot execute owned math'
                stats['boundary_old_reads']+=1;return snap[f][boundary_slot(y-1,x-1)]
            def read(phase,r,t,j):
                f=FIELDS.index(r.field);d=point(t,j)
                if phase==0:
                    if not any(r.offset):return old_diagonal(f,t,j)
                    if r.offset==(1,0,0):
                        lane=(t%32-1)%32;source=(t//32)*32+lane;stats['private_diagonal_reads']+=1;return cache[f][source][j]
                    stats['shared_XY_reads']+=1;return frame[f-3][slot(tuple(v+dv-1 for v,dv in zip(d,r.offset)))]
                if f>=3:return old_diagonal(f,t,j)
                if not any(r.offset):stats['private_current_reads']+=1;return cache[f][t][j]
                if r.offset==(-1,0,0):
                    if j>0:stats['private_current_reads']+=1;return cache[f][t][j-1]
                    stats['boundary_current_reads']+=1;return snap[f][boundary_slot(d[1],d[2])]
                stats['shared_XY_reads']+=1;return frame[f][slot(tuple(v+dv for v,dv in zip(d,r.offset)))]
            def compute(phase):
                if negative==('early-H-boundary' if phase==0 else 'early-E-boundary') and groups==2:
                    for y in range(16):
                        for x in range(32):
                            d=(3,y,x);p=physical(d)
                            for f in range(phase*3,phase*3+3):snap[f][boundary_slot(y,x)]=token(f,p,n)
                ascending=negative==('ascending-H-Z' if phase==0 else 'ascending-E-Z')
                order=list(range(local_z)) if ascending else list(reversed(range(local_z)))
                positions=[[0,0] for _ in range(threads//32)];active=list(range(threads//32))
                while active:
                    warp=rng.choice(active);ji,c=positions[warp];j=order[ji];f=phase*3+c;updates=[]
                    for lane in range(32):
                        t=warp*32+lane;d=point(t,j)
                        if not all(d):continue
                        p=physical(d)
                        if not valid(f,p):updates.append((t,None));continue
                        for r in operands[FIELDS[f]]:
                            sf=FIELDS.index(r.field);sp=tuple(v+dv for v,dv in zip(p,r.offset))
                            if not valid(sf,sp):continue
                            actual=read(phase,r,t,j);expected=token(sf,sp,n-1+r.version)
                            assert actual==expected,('Diagonal canonical operand',q,n,phase,FIELDS[f],r,d,t,j,actual,expected,negative)
                            stats['canonical_reads']+=1
                        updates.append((t,token(f,p,n)))
                    for t,value in updates:cache[f][t][j]=value
                    stats['warp_events']+=1;c+=1
                    if c==3:ji+=1;c=0
                    if ji==len(order):active.remove(warp)
                    else:positions[warp]=[ji,c]
            def publish(phase):
                # Reader-retirement join precedes shared-family alias reuse.
                for d in own:
                    for f in range(phase*3,phase*3+3):frame[f%3][slot(d)]=get(f,d)
                if phase==0 or n<last:
                    for d in ghost:
                        p=physical(d)
                        for f in range(phase*3,phase*3+3):frame[f%3][slot(d)]=imported(f,p,n)
                # The owned and coalesced-ghost writers jointly fill lowerZ3
                # before publication; private ghost reloads occur AFTER join.
                if groups==2:
                    for y in range(16):
                        for x in range(32):
                            for f in range(phase*3,phase*3+3):snap[f][boundary_slot(y,x)]=frame[f%3][slot((3,y,x))]
                if (phase==0 or n<last) and negative!=('skip-H-ghost-reload' if phase==0 else 'skip-E-ghost-reload'):
                    for d in ghost:
                        for f in range(phase*3,phase*3+3):put(f,d,frame[f%3][slot(d)])
                # Coalesced IO consumes shared owned values while the opposite
                # phase reads this immutable family. The next join retires IO.
                for d in own:
                    if n!=depth and not any(v==b for v,b in zip(d,owned)):continue
                    p=physical(d)
                    for f in range(phase*3,phase*3+3):
                        if not valid(f,p):continue
                        actual=frame[f%3][slot(d)];assert actual==token(f,p,n),('Diagonal export',q,n,f,p,actual)
                        if n==depth:output[f][linear(f,p)]=actual
                        else:history[q,n,f,p]=actual
            compute(0);publish(0);compute(1);publish(1)
    assert all(v==token(f,p,depth) for f,s in enumerate(shape) for p in itertools.product(*(range(n) for n in s)) for v in [output[f][linear(f,p)]]),'Incomplete diagonal native fields'
    if groups==2:assert stats['boundary_old_reads'] and stats['boundary_current_reads'],'No contiguous-Z boundary communication exercised'
    return dict(minimum=minimum,owned=owned,depth=depth,threads=threads,private_Z=local_z,Zgroups=groups,seed=seed,shared_bytes=61568 if groups==2 else 49280,**stats)


def report():
    cases=[simulate(depth=depth,threads=threads,seed=seed) for depth in (4,16) for threads in (512,1024) for seed in (13,37)]
    negatives={}
    for name in ('wrong-shuffle-lane','ascending-H-Z','ascending-E-Z','early-H-boundary','early-E-boundary','skip-H-ghost-reload','skip-E-ghost-reload'):
        try:simulate(negative=name)
        except AssertionError as error:negatives[name]=str(error)
        else:raise AssertionError('Ineffective diagonal cache negative:'+name)
    return dict(schema='beamz.diagonal-private-logical-cells.v1',passed=True,cases=cases,negative_controls=negatives,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Canonical six-field/version reads with actual diagonal thread inverse, warp lane forwarding, independently interleaved warps descending privateZ, stable old/current contiguousZ snapshots, rank-owned ghost packets and coalesced shared IO. No CUDA visibility/resource, FP or speed claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=report();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(r['cases']),canonical_reads=sum(c['canonical_reads'] for c in r['cases']),negative_controls=r['negative_controls'])))
