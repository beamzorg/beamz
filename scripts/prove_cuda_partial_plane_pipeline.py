"""Discrete-event proof of a started-parent, partial-step native tile pipeline.

This models concurrent resident tasks, exact canonical field reads, mutable
point-local auxiliary histories, clipped windows and a three-Z-plane face ring.
It is a scheduling legality/progress model, not a GPU throughput prediction.
"""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import dataclass,field
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA
from scripts.cuda_space_time_ir import FIELDS,yee_operator


@dataclass
class Task:
    q:tuple[int,int,int]
    origin:tuple[int,int,int]
    first:int
    last:int
    pending:int
    queued:bool=False
    started:bool=False
    done:bool=False
    ready:int=0
    n:int=0
    state:str='init'
    local:list=field(default_factory=lambda:[{} for _ in range(6)])


def simulate(minimum=(11,13,15),owned=(3,5,7),depth=4,workers=8,seed=13,negative=None,planes=3,schedule='random',disposal=False,closed_epochs=None,closed_allocation='global',queue_order='fifo',closed_flags='synthetic',ready_mode='startup7'):
    shapes=[tuple(n+d for n,d in zip(minimum,delta)) for delta in SUPPORT_DELTA]
    maximum=tuple(n+1 for n in minimum)
    counts=tuple((n+depth+b-1)//b for n,b in zip(maximum,owned))
    tasks={}
    for q in itertools.product(*(range(c) for c in counts)):
        origin=tuple(x*b for x,b in zip(q,owned));first=max(1,*(o-n+1 for o,n in zip(origin,maximum)));last=min(depth,*(o+b-1 for o,b in zip(origin,owned)))
        tasks[q]=Task(q,origin,first,last,sum(x>0 for x in q))
    closed=None;previous={};cohort_index=0;cohort_of={};cohort_sizes=[]
    if closed_epochs:
        from scripts.cuda_policy_closed_cohorts import compile_cohorts,schedule_first_owner_plan
        ids=sorted(tasks,key=lambda q:(sum(q),(q[0]*counts[1]+q[1])*counts[2]+q[2]));slots={};ranks={}
        for q in ids:slots[q]=ranks.get(sum(q),0);ranks[sum(q)]=slots[q]+1
        flat=lambda q:(q[0]*counts[1]+q[1])*counts[2]+q[2]
        flags=[0]*len(tasks);slot_vector=[0]*len(tasks)
        for q in tasks:flags[flat(q)]=2 if closed_flags=='homogeneous' else (0,2,1,16,17,3)[(flat(q)*1103515245+12345)%6];slot_vector[flat(q)]=slots[q]
        plan=dict(counts=counts,tile_ids=list(map(flat,ids)),slots=slot_vector,tile_flags=flags,max_frontier=max(ranks.values()),depth=depth)
        if closed_allocation=='schedule-first':plan,closed=schedule_first_owner_plan(plan,closed_epochs)
        elif closed_allocation=='global':closed=compile_cohorts(plan,closed_epochs,'earliest')
        elif closed_allocation=='no-reuse':
            plan.update(owner_slots=list(range(len(tasks))),owner_count=len(tasks),owner_partition='unique-allocation-legality-only')
            closed=compile_cohorts(plan,closed_epochs,'earliest')
        else:raise ValueError('Explicit closed packet allocation required')
        cohort_of={q:closed['cohort_of'][flat(q)] for q in tasks};cohort_sizes=[len(c['tiles']) for c in closed['cohorts']]
        owners={}
        for t in plan['tile_ids']:
            q=(t//(counts[1]*counts[2]),(t//counts[2])%counts[1],t%counts[2])
            slot=plan['owner_slots'][t] if 'owner_slots' in plan else (sum(q)%closed_epochs)*plan['max_frontier']+slots[q];previous[q]=owners.get(slot);owners[slot]=q
    plane_size=counts[1]*counts[2];plane_done=[0]*counts[0];epochs=1 if negative=='one-plane-ring' else planes
    def owner(q):
        if closed_epochs:return plan['owner_slots'][flat(q)] if 'owner_slots' in plan else (sum(q)%closed_epochs)*plan['max_frontier']+slots[q]
        return ((q[0]%epochs)*counts[1]+q[1])*counts[2]+q[2]
    tags={};history={};queue=deque();active=[];completed=0
    packet_states={};acknowledged=set();disposal_checks=0;disposal_count=0
    output=[[-1]*math.prod(shape) for shape in shapes];psi=[[0]*len(a) for a in output]
    cache=tuple(itertools.product(*(range(b+1) for b in owned)));owned_q=tuple(q for q in cache if all(q));ghost_q=tuple(q for q in cache if not all(q))
    reads={u.output.field:tuple(r for r in sorted(u.reads) if r.field in FIELDS) for u in yee_operator(cpml_axes=(0,1,2)).updates if u.output.field in FIELDS}
    rng=random.Random(seed);stats=dict(field_reads=0,psi_version_checks=0,ghost_reads=0,partial_producer_reads=0,payload_writes=0,events=0,peak_resident=0,empty_tasks=0,late_tasks=0)
    def parents(q):
        for a in range(3):
            if q[a]:p=list(q);p[a]-=1;yield tasks[tuple(p)]
    def safe(q):
        if closed:
            old=previous[q]
            if old is None:return True
            if not tasks[old].done:return False
            for bits in range(1,8):
                reader=tuple(x+((bits>>a)&1) for a,x in enumerate(old))
                if reader in tasks and not tasks[reader].done:return False
            return True
        z=q[0]
        if z<epochs:return True
        if negative=='ignore-old-writer-plane':return plane_done[z-epochs+1]==plane_size
        if negative=='ignore-old-reader-plane':return plane_done[z-epochs]==plane_size
        return plane_done[z-epochs]==plane_size and plane_done[z-epochs+1]==plane_size
    def enqueue(q):
        t=tasks[q]
        if closed and cohort_of[q]!=cohort_index:return
        if not t.queued and not t.pending and safe(q):t.queued=True;queue.append(q)
    def valid(f,p):return all(0<=x<n for x,n in zip(p,shapes[f]))
    def index(f,p):return (p[0]*shapes[f][1]+p[1])*shapes[f][2]+p[2]
    def parents_ready(t,n,startup=False):
        if ready_mode=='all7' or ready_mode=='startup7' and startup:
            for bits in range(1,8):
                q=tuple(x-((bits>>a)&1) for a,x in enumerate(t.q))
                if q in tasks and tasks[q].ready<n:return False
            return True
        if ready_mode not in ('startup7','nearest3','all7'):raise ValueError('Explicit ready semantics required')
        return all(p.ready>=n for p in parents(t.q))
    def dispose(key):
        nonlocal disposal_count
        state=packet_states[key]
        assert state['ready'] and not state['disposed'] and (not state['remaining'] or negative=='early-packet-disposal'),('Invalid last-use disposal',key,state)
        state['disposed']=True;disposal_count+=1
    def acknowledge(t,n):
        if not disposal or n<1 or n>=depth:return
        assert (t.q,n) not in acknowledged,('Duplicate consumer ACK',t.q,n)
        acknowledged.add((t.q,n))
        for bits in range(1,8):
            q=tuple(x-((bits>>a)&1) for a,x in enumerate(t.q))
            if q not in tasks:continue
            key=(owner(q),n);state=packet_states[key]
            assert state['producer']==q and state['remaining']>0,('ACK wrong packet owner',t.q,q,n,state)
            state['remaining']-=1
            if not state['remaining'] and state['ready']:dispose(key)
    def load_ghost(t,f,p,version):
        nonlocal disposal_checks
        if not valid(f,p):return
        producer=tuple((x+version)//b for x,b in zip(p,owned));slot=owner(producer)
        assert tags.get(slot)==producer,dict(reason='history tag overwritten',consumer=t.q,producer=producer,actual=tags.get(slot),step=version,negative=negative)
        if disposal and version<depth:
            state=packet_states[slot,version]
            assert (t.q,version) not in acknowledged,('Load after own ACK',t.q,version)
            assert state['producer']==producer and state['ready'] and not state['disposed'],('Read poisoned/unpublished packet',t.q,producer,version,state)
            disposal_checks+=1
        actual=output[f][index(f,p)] if version==depth else history[slot].get((version,f,p),-999)
        assert actual==version,dict(reason='ghost version',consumer=t.q,producer=producer,point=p,field=FIELDS[f],expected=version,actual=actual,negative=negative)
        stats['ghost_reads']+=1
        if tasks[producer].ready<depth:stats['partial_producer_reads']+=1
    def begin(t):
        nonlocal completed
        slot=owner(t.q)
        # Starting a new plane owner must not erase a live old writer/reader.
        if slot in tags:
            old=tags[slot];assert tasks[old].done,('old writer still active',old,t.q)
            for bits in range(1,8):
                q=tuple(x+((bits>>a)&1) for a,x in enumerate(old))
                if q in tasks:assert tasks[q].done,('old reader still active',q,t.q)
        if disposal:
            for n in range(1,depth):
                old=packet_states.get((slot,n))
                if old:assert not old['remaining'],('Owner reused before all ACKs',old)
                readers=math.prod(1+(x+1<c) for x,c in zip(t.q,counts))-1
                packet_states[slot,n]=dict(producer=t.q,remaining=readers,ready=False,disposed=False)
        tags[slot]=t.q;history[slot]={};t.started=True;t.n=t.first
        # NotifyStarted occurs after the actual native owner-tag CTA barrier.
        for a in range(3):
            q=list(t.q);q[a]+=1;q=tuple(q)
            if q in tasks:tasks[q].pending-=1;enqueue(q)
        # All seven ancestor tags must be initialized before base-table checks.
        for bits in range(1,8):
            q=tuple(x-((bits>>a)&1) for a,x in enumerate(t.q))
            if q in tasks:assert tags.get(owner(q))==q,('early ancestor tag check',t.q,q)
        if t.first>t.last:t.state='finish';t.ready=depth;stats['empty_tasks']+=1
        else:
            t.ready=t.first-1
            if t.first>1:stats['late_tasks']+=1
    def execute_math(t,phase):
        n=t.n
        for d in owned_q:
            p=tuple(o-1+x-n for o,x in zip(t.origin,d))
            for f in range(3*phase,3*phase+3):
                if not valid(f,p):continue
                for read in reads[FIELDS[f]]:
                    sf=FIELDS.index(read.field);sp=tuple(x+dx for x,dx in zip(p,read.offset))
                    if not valid(sf,sp):continue
                    expected=n-1+read.version;actual=t.local[sf].get(sp,-999)
                    assert actual==expected,dict(reason='canonical read',task=t.q,step=n,field=FIELDS[f],read=str(read),point=sp,expected=expected,actual=actual)
                    stats['field_reads']+=1
                # Two independently mutable point-local auxiliary terms per
                # field. Treating all points active is stronger than CPML-only
                # activity for ownership/version ordering (no FP emulation).
                assert psi[f][index(f,p)]==n-1,('psi version',t.q,n,f,p,psi[f][index(f,p)])
                stats['psi_version_checks']+=2;psi[f][index(f,p)]=n;t.local[f][p]=n
    def payload(t):
        n=t.n;slot=owner(t.q);assert tags[slot]==t.q
        for d in owned_q:
            surface=any(x==b for x,b in zip(d,owned))
            if n!=depth and not surface:continue
            p=tuple(o-1+x-n for o,x in zip(t.origin,d))
            for f in range(6):
                if not valid(f,p):continue
                if n==depth:output[f][index(f,p)]=n
                elif surface:history[slot][(n,f,p)]=n
                stats['payload_writes']+=1
    if closed:
        for q in tasks:enqueue(q)
    else:enqueue((0,0,0))
    while completed<len(tasks):
        if closed and not active and not queue:
            cohort_index+=1
            assert cohort_index<len(cohort_sizes),('Closed queue lost tasks',completed)
            for q in tasks:enqueue(q)
        while queue and len(active)<workers:
            if queue_order=='fifo':q=queue.popleft()
            elif queue_order=='lifo':q=queue.pop()
            else:raise ValueError('Explicit readyqueue order required')
            begin(tasks[q]);active.append(q)
        stats['peak_resident']=max(stats['peak_resident'],len(active))
        enabled=[]
        for q in active:
            t=tasks[q]
            if t.state=='init' and t.first>1 and not parents_ready(t,t.first-1,True):continue
            if t.state=='imports' and negative!='skip-parent-ready' and not parents_ready(t,t.n):continue
            enabled.append(q)
        assert enabled,dict(reason='queue/step deadlock',completed=completed,total=len(tasks),active=[(q,tasks[q].state,tasks[q].n,tasks[q].ready) for q in active],queued=list(queue),plane_done=plane_done)
        if schedule=='producer-fast':
            first_plane=min(q[0] for q in enabled);enabled=[q for q in enabled if q[0]==first_plane]
        elif schedule=='consumer-fast':
            rank=max(map(sum,enabled));enabled=[q for q in enabled if sum(q)==rank]
        elif schedule!='random':raise ValueError('Explicit scheduling policy required')
        q=rng.choice(enabled);t=tasks[q];stats['events']+=1
        if t.state=='init':
            qs=cache if t.first==1 else ghost_q;version=t.first-1
            if disposal and negative=='premature-late-ack' and version:acknowledge(t,version)
            for d in qs:
                p=tuple(o-t.first+x for o,x in zip(t.origin,d))
                for f in range(6):
                    if t.first>1:load_ghost(t,f,p,version)
                    t.local[f][p]=version if valid(f,p) else -1
            if disposal:
                for n in range(1,version+1):
                    if (t.q,n) not in acknowledged:acknowledge(t,n)
            t.state='H'
        elif t.state=='H':execute_math(t,0);t.state='imports'
        elif t.state=='imports':
            for d in ghost_q:
                p=tuple(o-1+x-t.n for o,x in zip(t.origin,d))
                for f in range(6):
                    if t.n==depth and f>=3:continue
                    load_ghost(t,f,p,t.n);t.local[f][p]=t.n if valid(f,p) else -1
            acknowledge(t,t.n)
            t.state='E'
        elif t.state=='E':execute_math(t,1);payload(t);t.state='publish'
        elif t.state=='publish':
            if disposal and t.n<depth:
                key=(owner(t.q),t.n);state=packet_states[key];state['ready']=True
                if not state['remaining'] or negative=='early-packet-disposal':dispose(key)
            t.ready=min(depth,t.n+1) if negative=='future-ready' else t.n
            if t.n<t.last:t.n+=1;t.state='H'
            else:t.state='finish'
        elif t.state=='finish':
            if disposal:
                for n in range(1,depth):
                    if (t.q,n) not in acknowledged:acknowledge(t,n)
            t.ready=depth;t.done=True;completed+=1;active.remove(q);plane_done[q[0]]+=1
            if closed:
                for candidate in tasks:enqueue(candidate)
            elif plane_done[q[0]]==plane_size:
                for z in (q[0]+epochs-1,q[0]+epochs):
                    if z>=counts[0]:continue
                    for y in range(counts[1]):
                        for x in range(counts[2]):enqueue((z,y,x))
        else:raise AssertionError('Unknown phase')
    assert all(v==depth for a in output for v in a),'Incomplete native output'
    assert all(v==depth for a in psi for v in a),'Incomplete auxiliary history'
    if workers>1:assert stats['partial_producer_reads']>0,'No partial producer consumption exercised'
    if disposal:assert all(not s['remaining'] for s in packet_states.values()),'Unretired final packet readers'
    return dict(minimum=minimum,owned=owned,depth=depth,workers=workers,seed=seed,schedule=schedule,counts=counts,tasks=len(tasks),history_planes=epochs if not closed else None,history_slots=epochs*plane_size if not closed else plan.get('owner_count',closed_epochs*plan['max_frontier']),closed_rank_epochs=closed_epochs,closed_cohort_count=len(cohort_sizes),closed_allocation=closed_allocation,closed_flags=closed_flags,queue_order=queue_order,ready_mode=ready_mode,disposal_checks=disposal_checks,disposed_packets=disposal_count,**stats)


def report():
    cases=[simulate(minimum=n,owned=b,depth=d,workers=w,seed=s)
           for n,b,d in [((11,13,15),(3,5,7),4),((13,17,21),(7,15,31),16)]
           for w in (1,4,82) for s in (13,37)]
    cases.append(simulate(minimum=(13,17,21),owned=(7,15,31),depth=16,workers=82,seed=13,planes=2))
    cases.append(simulate(workers=82,schedule='producer-fast'))
    negatives={}
    for name in ('one-plane-ring','ignore-old-reader-plane','skip-parent-ready','future-ready'):
        try:simulate(negative=name,workers=82 if name=='ignore-old-reader-plane' else 8,schedule='producer-fast' if name=='ignore-old-reader-plane' else 'random')
        except AssertionError as e:negatives[name]=str(e)
        else:raise AssertionError('Ineffective partial-step negative:'+name)
    return dict(schema='beamz.partial-step-native-plane-pipeline.v1',passed=True,cases=cases,negative_controls=negatives,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Discrete event canonical field/auxiliary version reads, owner-tagged3Zplane histories, clippedwindowlate/bootstrap/terminal cases, started-parent readyqueue progress and actualpartialconsumption; no CUDA memory visibility, FP rounding or speed claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=report();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(r['cases']),field_reads=sum(c['field_reads'] for c in r['cases']),psi_checks=sum(c['psi_version_checks'] for c in r['cases']),partial_reads=sum(c['partial_producer_reads'] for c in r['cases']))))
