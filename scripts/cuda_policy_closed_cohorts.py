"""Partition native tile hazards into policy-homogeneous closed CUDA cohorts.

All cross-cohort dependencies finish before the next kernel. Within a cohort,
RAW dependencies may hand off individual steps, and owner reuse still waits
for full writer/reader retirement. No cross-kernel spin protocol is assumed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path


def policy(flags):
    flags&=63
    if not flags:return 0
    if not flags&17:return 1
    if flags==1:return 2
    if flags==16:return 3
    if flags==17:return 4
    return 5


def hazards(plan,epochs=4,*,omit_war=False):
    nz,ny,nx=plan['counts'];ids=plan['tile_ids'];slots=plan['slots'];frontier=plan['max_frontier']
    if epochs<4:raise ValueError('Rank ring must retain at least4epochs')
    if len(ids)!=nz*ny*nx or len(set(ids))!=len(ids):raise ValueError('Unique complete prepared tile IDs required')
    position={t:i for i,t in enumerate(ids)};parents=[set() for _ in ids];finish=[set() for _ in ids]
    owners={}
    for t in ids:
        z,y,x=t//(ny*nx),(t//nx)%ny,t%nx
        if z:parents[t].add(t-ny*nx)
        if y:parents[t].add(t-nx)
        if x:parents[t].add(t-1)
        slot=plan['owner_slots'][t] if 'owner_slots' in plan else ((z+y+x)%epochs)*frontier+slots[t];old=owners.get(slot)
        if old is not None and not omit_war:
            finish[t].add(old);oz,oy,ox=old//(ny*nx),(old//nx)%ny,old%nx
            for bits in range(1,8):
                dz,dy,dx=bits&1,(bits>>1)&1,(bits>>2)&1
                if oz+dz<nz and oy+dy<ny and ox+dx<nx:finish[t].add(old+dz*ny*nx+dy*nx+dx)
        owners[slot]=t
        if any(position[p]>=position[t] for p in parents[t]|finish[t]):raise ValueError('Prepared owner/RAW hazards must strictly advance input order')
    return parents,finish,position


def assign_owner_slots(plan,epochs,mode='policy'):
    """Separate packet ownership by physics to avoid artificial mixed-class WARs."""
    if mode not in ('policy','signature'):raise ValueError('Explicit owner partition required')
    _,ny,nx=plan['counts'];counts=Counter();maximum=Counter();classes=[]
    for flags in plan['tile_flags']:
        p=policy(flags);classes.append(p if mode=='policy' else (p*16+((flags&14)>>1)))
    for t in plan['tile_ids']:
        rank=t//(ny*nx)+(t//nx)%ny+t%nx;key=(rank,classes[t]);counts[key]+=1;maximum[classes[t]]=max(maximum[classes[t]],counts[key])
    bases={};owners=0
    for k in sorted(maximum):bases[k]=owners;owners+=epochs*maximum[k]
    counts.clear();slots=[-1]*len(classes)
    for t in plan['tile_ids']:
        rank=t//(ny*nx)+(t//nx)%ny+t%nx;k=classes[t];slots[t]=bases[k]+rank%epochs*maximum[k]+counts[rank,k];counts[rank,k]+=1
    result=dict(plan);result.update(owner_slots=slots,owner_count=owners,owner_partition=mode);return result


def compile_cohorts(plan,epochs=4,heuristic='largest-ready',*,omit_war=False):
    parents,finish,position=hazards(plan,epochs,omit_war=omit_war);n=len(parents);classes=list(map(policy,plan['tile_flags']))
    dependencies=[a|b for a,b in zip(parents,finish)];remaining=list(map(len,dependencies));children=[[] for _ in range(n)]
    for t,deps in enumerate(dependencies):
        for p in deps:children[p].append(t)
    ready=[[] for _ in range(6)];cohort_of=[-1]*n;cohorts=[];total=0
    for t in range(n):
        if not remaining[t]:heapq.heappush(ready[classes[t]],(position[t],t))
    while total<n:
        available=[p for p in range(6) if ready[p]]
        if not available:raise ValueError('Closed cohort compiler deadlocked')
        if heuristic=='largest-ready':selected=max(available,key=lambda p:(len(ready[p]),-ready[p][0][0]))
        elif heuristic=='earliest':selected=min(available,key=lambda p:ready[p][0][0])
        else:raise ValueError('Unknown explicit closure heuristic')
        root_count=len(ready[selected]);tiles=[];level={};internal_start=internal_finish=external=0;cohort=len(cohorts)
        while ready[selected]:
            _,t=heapq.heappop(ready[selected]);tiles.append(t);cohort_of[t]=cohort;total+=1
            internal=[p for p in dependencies[t] if cohort_of[p]==cohort]
            if any(cohort_of[p]<0 for p in dependencies[t]):raise ValueError('Unscheduled external dependency')
            level[t]=1+max((level[p] for p in internal),default=0)
            internal_start+=sum(cohort_of[p]==cohort for p in parents[t]);internal_finish+=sum(cohort_of[p]==cohort for p in finish[t]);external+=sum(cohort_of[p]!=cohort for p in dependencies[t])
            for child in children[t]:
                remaining[child]-=1
                if not remaining[child]:heapq.heappush(ready[classes[child]],(position[child],child))
        cohorts.append(dict(policy=selected,tiles=tiles,initial_roots=root_count,longest_internal_full_task_path=max(level.values()),internal_start_edges=internal_start,internal_finish_edges=internal_finish,external_edges=external))
    # Independent coverage and actual typed-hazard closure audit.
    for t in range(n):
        for p in parents[t]|finish[t]:
            if cohort_of[p]>cohort_of[t]:raise AssertionError('Future kernel dependency')
            if cohort_of[p]==cohort_of[t] and classes[p]!=classes[t]:raise AssertionError('Mixed policy inside kernel')
    size=Counter(c['policy'] for c in cohorts);tile_count=Counter(classes);largest=sorted(cohorts,key=lambda c:len(c['tiles']),reverse=True)
    return dict(schema='beamz.policy-closed-cohort-plan.v1',epochs=epochs,heuristic=heuristic,tile_count=n,cohort_count=len(cohorts),
                counts_by_policy=dict(size),tiles_by_policy=dict(tile_count),
                largest_cohorts=[{k:(len(v) if k=='tiles' else v) for k,v in c.items()} for c in largest[:12]],
                singleton_cohorts=sum(len(c['tiles'])==1 for c in cohorts),small_under82_cohorts=sum(len(c['tiles'])<82 for c in cohorts),
                tiles_in_under82_cohorts=sum(len(c['tiles']) for c in cohorts if len(c['tiles'])<82),
                trail_bytes=plan.get('owner_count',epochs*plan['max_frontier'])*plan['depth']*6*735*4,
                owner_partition=plan.get('owner_partition','global'),
                start_edges=sum(map(len,parents)),finish_edges=sum(map(len,finish)),
                cohort_of=cohort_of,cohorts=cohorts,passed=True,
                scope='Typed RAW-start and WAR/WAW-completion task partition; cross-kernel completion edges only, per-step handoff confined to one policy kernel. Planner estimate, not CUDA memory visibility or speed.')


def schedule_first_owner_plan(plan,epochs=4):
    """Choose physics closures first; allocate internal rings and export intervals.

    Internal packets have no readers outside their writer's kernel. Exported
    packets get separate colors until their final consumer kernel completes.
    This avoids making artificial storage reuse edges split physics regions.
    """
    physics=compile_cohorts(plan,epochs,'earliest',omit_war=True);mapping=physics['cohort_of'];nz,ny,nx=plan['counts']
    last_reader=[];exports=[];rank_counts=Counter();rank_of=[];maximum=0
    for t in range(len(mapping)):
        z,y,x=t//(ny*nx),(t//nx)%ny,t%nx;rank=z+y+x;rank_of.append(rank);last=mapping[t]
        for bits in range(1,8):
            dz,dy,dx=bits&1,(bits>>1)&1,(bits>>2)&1
            if z+dz<nz and y+dy<ny and x+dx<nx:last=max(last,mapping[t+dz*ny*nx+dy*nx+dx])
        last_reader.append(last);exports.append(last>mapping[t])
        if last==mapping[t]:rank_counts[mapping[t],rank]+=1;maximum=max(maximum,rank_counts[mapping[t],rank])
    owners=[-1]*len(mapping);internal_size=epochs*maximum;rank_counts.clear();busy=[];available=[];next_owner=internal_size
    for cohort,c in enumerate(physics['cohorts']):
        while busy and busy[0][0]<cohort:
            _,slot=heapq.heappop(busy);heapq.heappush(available,slot)
        for t in c['tiles']:
            if exports[t]:
                if available:slot=heapq.heappop(available)
                else:slot=next_owner;next_owner+=1
                owners[t]=slot;heapq.heappush(busy,(last_reader[t],slot))
            else:
                key=(cohort,rank_of[t]);owners[t]=rank_of[t]%epochs*maximum+rank_counts[key];rank_counts[key]+=1
    result=dict(plan);result.update(tile_ids=[t for c in physics['cohorts'] for t in c['tiles']],owner_slots=owners,owner_count=next_owner,owner_partition='schedule-first-internal-ring-and-export-intervals')
    audited=compile_cohorts(result,epochs,'earliest')
    if audited['cohort_of']!=mapping:raise AssertionError('Storage allocation fragmented or reordered chosen physics closures')
    start,finish,_=hazards(result,epochs)
    for t,parents in enumerate(finish):
        if any(mapping[p]>mapping[t] for p in parents):raise AssertionError('Export slot released before a future reader')
    result['schedule_first_allocation']=dict(physics_cohort_count=physics['cohort_count'],internal_ring_owners=internal_size,export_owners=next_owner-internal_size,
        exported_packets=sum(exports),internal_packets=len(mapping)-sum(exports),owner_count=next_owner,
        physical_cohorts_preserved=True,scope='Rank rings only for packets with exclusively same-kernel readers; interval coloring reserves exports through final consumer kernel. Full version/FP/CUDA validation separate.')
    return result,audited


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());a.output.mkdir(exist_ok=False);reports=[]
    for epochs in (4,8,16,32):
        for heuristic in ('largest-ready','earliest'):
            r=compile_cohorts(plan,epochs,heuristic);r['input_sha256']=hashlib.sha256(a.plan.read_bytes()).hexdigest();r['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            (a.output/f'epochs{epochs}-{heuristic}.json').write_text(json.dumps(r)+'\n');reports.append({k:v for k,v in r.items() if k not in ('cohorts','cohort_of')})
    (a.output/'summary.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps([{k:r[k] for k in ('epochs','heuristic','cohort_count','tiles_in_under82_cohorts','trail_bytes')} for r in reports]))
