"""Concurrent canonical field/version oracle for phase-level adjacent-Z handoff.

Supernodes reserve both groups before notifying STARTED. H and E frames have
separate ready/reader-ACK epochs, while external packets retain GPU-style
whole-step readiness and all-seven clipped bootstrap. Integer tokens prove
ownership/liveness only; actual CUDA visibility and FP32 checks are separate.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA
from scripts.cuda_paired_supernodes import compile_pairs,native_plan
from scripts.cuda_space_time_ir import FIELDS,yee_operator
from scripts.prove_cuda_partial_plane_pipeline import Task


def simulate(minimum=(11,13,15),owned=(3,5,7),depth=4,workers=4,seed=13,negative=None,queue_order='fifo',schedule='random'):
    shapes=[tuple(n+d for n,d in zip(minimum,delta)) for delta in SUPPORT_DELTA]
    counts=tuple((n+1+depth+b-1)//b for n,b in zip(minimum,owned));nz,ny,nx=counts
    plan=native_plan(counts,seed);pair=compile_pairs(plan);sn=pair['node_count']
    flat=lambda q:(q[0]*ny+q[1])*nx+q[2]
    node=lambda q:((q[0]//2)*ny+q[1])*nx+q[2]
    tasks={};members=[[] for _ in range(sn)]
    for q in itertools.product(*(range(n) for n in counts)):
        origin=tuple(x*b for x,b in zip(q,owned));first=max(1,*(o-n for o,n in zip(origin,minimum)));last=min(depth,*(o+b-1 for o,b in zip(origin,owned)))
        t=Task(q,origin,first,last,0);t.frame=None;t.h_ready=0;t.e_ready=0;t.h_ack=0;t.e_ack=0;tasks[q]=t;members[node(q)].append(q)
    def low_parent(t):return tasks.get((t.q[0]-1,t.q[1],t.q[2])) if t.q[0]&1 else None
    def upper_child(t):return tasks.get((t.q[0]+1,t.q[1],t.q[2])) if not t.q[0]&1 else None
    def overlap(t,n):
        child=upper_child(t)
        return child is not None and child.first<=n<=child.last and t.first<=n<=t.last
    owner=lambda t:pair['owner_slots'][flat(t.q)]
    cache=tuple(itertools.product(*(range(b+1) for b in owned)));owned_points=tuple(d for d in cache if all(d));ghosts=tuple(d for d in cache if not all(d))
    reads={u.output.field:tuple(r for r in sorted(u.reads) if r.field in FIELDS) for u in yee_operator(cpml_axes=(0,1,2)).updates if u.output.field in FIELDS}
    output=[[-1]*math.prod(shape) for shape in shapes];psi=[[0]*len(output[f//2]) for f in range(12)]
    tags={};history={};node_started=set();node_done=set();active=[];queued=set();queue=deque();rng=random.Random(seed)
    stats=dict(field_reads=0,psi_checks=0,global_ghost_reads=0,local_H_reads=0,local_E_reads=0,elided_packet_values=0,events=0,late_tasks=0,empty_tasks=0,producer_last_E_reads=0)
    valid=lambda f,p:all(0<=x<n for x,n in zip(p,shapes[f]))
    index=lambda f,p:(p[0]*shapes[f][1]+p[1])*shapes[f][2]+p[2]
    def enqueue():
        for s in range(sn):
            if s not in queued and pair['start'][s]<=node_started and pair['finish'][s]<=node_done:queued.add(s);queue.append(s)
    def begin(s):
        active.append(s)
        for q in members[s]:
            t=tasks[q];slot=owner(t);tags[slot]=q;history[slot]={};t.started=True;t.n=t.first;t.ready=t.first-1
            if t.first>t.last:t.state='finish';t.ready=depth;stats['empty_tasks']+=1
            elif t.first>1:stats['late_tasks']+=1
        # Both native owner tags precede supernode START notification.
        node_started.add(s);enqueue()
    def parents_ready(t,n,bootstrap=False):
        if n<1:return True
        for bits in range(1,8) if bootstrap else (1,2,4):
            if not bootstrap and bits==1 and low_parent(t) is not None and negative!='whole-step-local-Z-wait':continue
            q=tuple(x-((bits>>a)&1) for a,x in enumerate(t.q))
            if q in tasks and tasks[q].ready<n:return False
        return True
    def canonical(t,phase):
        n=t.n
        for d in owned_points:
            p=tuple(o-1+x-n for o,x in zip(t.origin,d))
            for f in range(phase*3,phase*3+3):
                if not valid(f,p):continue
                for r in reads[FIELDS[f]]:
                    sf=FIELDS.index(r.field);sp=tuple(x+dx for x,dx in zip(p,r.offset))
                    if valid(sf,sp):
                        expected=n-1+r.version;actual=t.local[sf].get(sp,-999)
                        assert actual==expected,('Canonical field version',t.q,n,FIELDS[f],r,sp,expected,actual)
                        stats['field_reads']+=1
                for term in (2*f,2*f+1):
                    assert psi[term][index(f,p)]==n-1,('CPML recurrence',t.q,n,term,p)
                    psi[term][index(f,p)]=n;stats['psi_checks']+=1
                t.local[f][p]=n
    def global_read(t,f,p,n):
        if not valid(f,p):return
        producer=tuple((x+n)//b for x,b in zip(p,owned));pt=tasks[producer];slot=owner(pt)
        assert tags.get(slot)==producer,('Old packet owner reused',t.q,producer,n,tags.get(slot))
        actual=output[f][index(f,p)] if n==depth else history[slot].get((n,f,p),-999)
        assert actual==n,('Global packet version',t.q,producer,n,p,FIELDS[f],actual)
        stats['global_ghost_reads']+=1
    def imports(t,phase):
        n=t.n;parent=low_parent(t)
        for d in ghosts:
            p=tuple(o-1+x-n for o,x in zip(t.origin,d));local=parent is not None and d[0]==0 and d[1]>0 and d[2]>0
            for f in range(phase*3,phase*3+3):
                if not valid(f,p):continue
                if local:
                    family='H' if phase==0 else 'E';assert parent.first<=n<=parent.last,('Valid local ghost outside parent window',t.q,parent.q,n,p)
                    assert parent.frame==(family,n),('Local shared frame reused before ACK',t.q,parent.q,family,n,parent.frame,negative)
                    assert parent.local[f].get(p)==n,('Local shared value/version',t.q,parent.q,FIELDS[f],n,p)
                    stats['local_'+family+'_reads']+=1
                    if phase and n==parent.last:stats['producer_last_E_reads']+=1
                else:global_read(t,f,p,n)
                t.local[f][p]=n
        if parent is not None and parent.first<=n<=parent.last:
            if phase:parent.e_ack=n
            else:parent.h_ack=n
    def local_ready(t,phase):
        p=low_parent(t);n=t.n
        return p is None or not p.first<=n<=p.last or (p.h_ready if phase==0 else p.e_ready)>=n
    def payload(t):
        n=t.n;slot=owner(t)
        for d in owned_points:
            if n!=depth and not any(x==b for x,b in zip(d,owned)):continue
            p=tuple(o-1+x-n for o,x in zip(t.origin,d))
            internal=upper_child(t) is not None and d[0]==owned[0] and d[1]<owned[1] and d[2]<owned[2]
            elide=internal and n<depth and (overlap(t,n) or negative=='elide-bootstrap-packet')
            for f in range(6):
                if not valid(f,p):continue
                if n==depth:output[f][index(f,p)]=n
                elif elide:stats['elided_packet_values']+=1
                else:history[slot][(n,f,p)]=n
    def enabled(t):
        n=t.n
        if t.state=='init':return t.first==1 or parents_ready(t,t.first-1,True)
        if t.state=='Himports':return parents_ready(t,n) and local_ready(t,0)
        if t.state=='Hvisible':return not overlap(t,n-1) or t.e_ack>=n-1 or negative=='reuse-E-before-ACK'
        if t.state=='Evisible':return not overlap(t,n) or t.h_ack>=n or negative=='reuse-H-before-ACK'
        if t.state=='Eimports':return local_ready(t,1)
        return True
    enqueue()
    while len(node_done)<sn:
        while queue and len(active)<workers:begin(queue.popleft() if queue_order=='fifo' else queue.pop())
        choices=[tasks[q] for s in active for q in members[s] if not tasks[q].done and enabled(tasks[q])]
        assert choices,('Paired phase deadlock',negative,len(node_done),sn,[(q,tasks[q].state,tasks[q].n,tasks[q].frame) for s in active for q in members[s] if not tasks[q].done])
        if schedule=='producer-fast':
            level=min(t.q[0] for t in choices);choices=[t for t in choices if t.q[0]==level]
        elif schedule=='consumer-fast':
            level=max(sum(t.q) for t in choices);choices=[t for t in choices if sum(t.q)==level]
        elif schedule!='random':raise ValueError('Explicit model schedule required')
        t=rng.choice(choices);n=t.n;stats['events']+=1
        if t.state=='init':
            for d in cache if t.first==1 else ghosts:
                p=tuple(o-t.first+x for o,x in zip(t.origin,d))
                for f in range(6):
                    if t.first>1:global_read(t,f,p,t.first-1)
                    t.local[f][p]=t.first-1 if valid(f,p) else -1
            t.state='H'
        elif t.state=='H':canonical(t,0);t.state='Himports'
        elif t.state=='Himports':imports(t,0);t.state='Hvisible'
        elif t.state=='Hvisible':t.frame=('H',n);t.h_ready=n;t.state='E'
        elif t.state=='E':canonical(t,1);t.state='Evisible'
        elif t.state=='Evisible':
            # A producer's last E still belongs to the upper child's next H.
            if negative!='omit-producer-last-E-frame' or n<t.last:t.frame=('E',n)
            payload(t);t.state='Eimports' if n<t.last and n<depth else 'publish'
        elif t.state=='Eimports':imports(t,1);t.state='publish'
        elif t.state=='publish':
            t.e_ready=n;t.ready=n
            if n<t.last:t.n+=1;t.state='H'
            else:t.state='finish'
        elif t.state=='finish':
            t.ready=depth;t.done=True;s=node(t.q)
            if all(tasks[q].done for q in members[s]):active.remove(s);node_done.add(s);enqueue()
        else:raise AssertionError('Unknown paired phase')
    assert all(v==depth for a in output for v in a),'Incomplete native output'
    assert all(v==depth for a in psi for v in a),'Incomplete CPML state'
    assert stats['local_H_reads'] and stats['local_E_reads'],'No phase-level local handoff exercised'
    return dict(minimum=minimum,owned=owned,depth=depth,workers=workers,seed=seed,queue_order=queue_order,schedule=schedule,counts=counts,nodes=sn,packet_epochs=5,**stats)


def report():
    cases=[]
    for minimum,owned,depth in [((11,13,15),(3,5,7),4),((13,17,21),(7,15,31),16),((15,17,47),(7,15,31),16)]:
        for workers,order in ((1,'fifo'),(4,'lifo'),(82,'fifo')):cases.append(simulate(minimum,owned,depth,workers,queue_order=order))
    for schedule in ('producer-fast','consumer-fast'):cases.append(simulate(workers=82,schedule=schedule))
    negatives={}
    for name in ('whole-step-local-Z-wait','reuse-E-before-ACK','reuse-H-before-ACK','elide-bootstrap-packet','omit-producer-last-E-frame'):
        try:simulate(workers=82,negative=name,schedule='producer-fast')
        except AssertionError as error:negatives[name]=str(error)
        else:raise AssertionError('Ineffective paired handoff negative:'+name)
    return dict(schema='beamz.paired-phase-handoff-model.v1',passed=True,cases=cases,negative_controls=negatives,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                supernode_source_sha256=hashlib.sha256(Path(__file__).with_name('cuda_paired_supernodes.py').read_bytes()).hexdigest(),
                canonical_source_sha256=hashlib.sha256(Path(__file__).with_name('cuda_space_time_ir.py').read_bytes()).hexdigest(),
                scope='Concurrent resident supernode START/COMPLETE and5rankpacket reuse; all canonical field/12auxiliary version operands; two shared family lifetimes with independent H/E ACKs, clipped all7global bootstrap, internal pureZ packet elision and producer-last-Echild liveness. No CUDA visibility/FP or speed claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=report();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(r['cases']),field_reads=sum(c['field_reads'] for c in r['cases']),local_H_reads=sum(c['local_H_reads'] for c in r['cases']),local_E_reads=sum(c['local_E_reads'] for c in r['cases']),negative_controls=r['negative_controls'])))
