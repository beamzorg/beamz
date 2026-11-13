"""CPU event proof for packed H/E readiness; not a GPU timing predictor."""
from __future__ import annotations
import random

try:
    from scripts.cuda_persistent_tile_pipeline import PipelineHazard
except ModuleNotFoundError:
    from cuda_persistent_tile_pipeline import PipelineHazard


def simulate(dag,depth=16,workers=4,seed=19,tile_windows=None,negative=None,h_publish_before_parent=False,e_publication_steps=1):
    n=len(dag['jobs']);rng=random.Random(seed)
    spans=tile_windows or [(1,depth)]*n
    owners=[None]*(4*dag['frontier']);ready=[0]*len(owners)
    h_written=[0]*n;e_written=[0]*n;e_fenced=[0]*n;completed=set();claimed=set();next_job=0
    h_start=[[None]*depth for _ in range(n)];h_pub=[[None]*depth for _ in range(n)]
    e_start=[[None]*depth for _ in range(n)];e_pub=[[None]*depth for _ in range(n)]
    h_work=[[rng.randrange(1,10) for _ in range(depth)] for _ in range(n)]
    e_work=[[rng.randrange(1,10) for _ in range(depth)] for _ in range(n)]
    states=[{'state':'idle','worker':i} for i in range(workers)]
    clock=0;checks=overlap_e=0

    def slot(t):return (dag['ranks'][t]&3)*dag['frontier']+dag['slots'][t]
    def parents(t,counter):
        nonlocal checks
        for parent in dag['raw_predecessors'][t]:
            checks+=1
            if owners[slot(parent)]!=parent or parent not in claimed or ready[slot(parent)]<counter:return False
        return True
    def verify_payload(t,step,family):
        for parent in dag['raw_predecessors'][t]:
            first,last=spans[parent]
            if first<=step<=last:
                written=h_written[parent] if family==0 else e_fenced[parent]
                if written<step:raise PipelineHazard(('Unready opposite/current family trail',t,parent,step,family))

    while len(completed)!=n:
        progress=False;order=list(range(workers));rng.shuffle(order)
        for wi in order:
            s=states[wi];state=s['state']
            if state=='idle' and next_job<n:
                t=dag['jobs'][next_job];next_job+=1;s.update(state='claim',tile=t)
                progress=True
            elif state=='claim':
                t=s['tile']
                deps=dag['claim_completion_predecessors'][t]
                if negative=='drop-WAR':deps=set()
                if deps<=completed:
                    old=dag['previous_slot_holder'].get(t)
                    if old is not None and not({old}|dag['raw_successors'][old])<=completed:
                        raise PipelineHazard(('Old phase consumer not retired before reuse',old,t))
                    if owners[slot(t)]!=old:raise PipelineHazard('Unexpected phase slot holder')
                    ready[slot(t)]=0;owners[slot(t)]=t;claimed.add(t)
                    first,last=spans[t];s.update(state='startup',step=first,last=last)
                    if first>last:s['state']='complete'
                    progress=True
            elif state=='startup':
                t,step=s['tile'],s['step']
                if parents(t,2*(step-1)):
                    verify_payload(t,step-1,1) if step>1 else None
                    h_start[t][step-1]=clock;s.update(state='H',until=clock+h_work[t][step-1]);progress=True
                    if negative=='omit-H-writer-join':ready[slot(t)]=2*step-1
            elif state=='H' and s['until']<=clock:
                if h_publish_before_parent:
                    t,step=s['tile'],s['step'];h_written[t]=step;h_pub[t][step-1]=clock
                    if negative!='H-fence-does-not-cover-E':
                        e_fenced[t]=max(e_fenced[t],step-1)
                        for old in range(step-1):
                            if e_pub[t][old] is None:e_pub[t][old]=clock
                    ready[slot(t)]=2*step-1
                s['state']='waitH';progress=True
            elif state=='waitH':
                t,step=s['tile'],s['step']
                if parents(t,2*step-1):
                    verify_payload(t,step,0)
                    if negative=='mixed-H-packet':verify_payload(t,step,1)
                    if not h_publish_before_parent:
                        h_written[t]=step;h_pub[t][step-1]=clock;ready[slot(t)]=2*step-1
                        e_fenced[t]=max(e_fenced[t],step-1)
                    if any(e_written[p]<step for p in dag['raw_predecessors'][t]):overlap_e+=1
                    e_start[t][step-1]=clock;s.update(state='E',until=clock+e_work[t][step-1]);progress=True
            elif state=='E' and s['until']<=clock:
                t,step=s['tile'],s['step'];e_written[t]=step
                publish=step%e_publication_steps==0 or step==s['last']
                if negative=='omit-last-E-publication':publish=step%e_publication_steps==0
                if publish:
                    e_fenced[t]=step
                    for old in range(step):
                        if e_pub[t][old] is None:e_pub[t][old]=clock
                    if negative!='wait-before-E-release':ready[slot(t)]=2*step
                s['state']='waitE';progress=True
            elif state=='waitE':
                t,step=s['tile'],s['step']
                if parents(t,2*step):
                    verify_payload(t,step,1)
                    if negative=='wait-before-E-release':ready[slot(t)]=2*step
                    if step==s['last']:s['state']='complete'
                    else:
                        step+=1;h_start[t][step-1]=clock
                        s.update(state='H',step=step,until=clock+h_work[t][step-1])
                        if negative=='omit-H-writer-join':ready[slot(t)]=2*step-1
                    progress=True
            elif state=='complete':
                t=s['tile'];ready[slot(t)]=2*depth;completed.add(t);s['state']='idle';progress=True
        if progress:continue
        future=[s['until'] for s in states if s['state'] in ('H','E') and s['until']>clock]
        if not future:raise PipelineHazard(('Phase DAG deadlock/watchdog',[(s['state'],s.get('tile')) for s in states]))
        clock=min(future)
    early_h=sum(h_pub[t][step]<h_pub[p][step] for t in range(n) for p in dag['raw_predecessors'][t] for step in range(depth) if h_pub[t][step] is not None and h_pub[p][step] is not None)
    return {'tiles':n,'depth':depth,'workers':workers,'simulated_ticks':clock,'RAW_phase_checks':checks,'H_publications_before_current_parent_H':early_h,
        'E_starts_before_upstream_E_same_version':overlap_e,'H_start':h_start,'H_ready':h_pub,'E_start':e_start,'E_ready':e_pub,
        'scope':'CPU dependency/packed-counter proof only; ticks are not CUDA timings'}
