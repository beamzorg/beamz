"""Compile adjacent-Z native tiles into typed START/COMPLETE supernodes.

Fusion changes storage legality: a four-rank packet ring can require a reader
inside a fused node to COMPLETE before that node may START. Five ranks provide
a strict supernode-rank order for every old-writer/reader retirement edge.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path
import random


def native_plan(counts,seed=0):
    nz,ny,nx=counts;rng=random.Random(seed);ranks={}
    for z,y,x in itertools.product(range(nz),range(ny),range(nx)):ranks.setdefault(z+y+x,[]).append((z*ny+y)*nx+x)
    slots=[-1]*(nz*ny*nx);ids=[]
    for rank in sorted(ranks):
        row=ranks[rank]
        if seed:rng.shuffle(row)
        for i,t in enumerate(row):slots[t]=i
        ids.extend(row)
    return dict(counts=counts,tile_ids=ids,slots=slots,max_frontier=max(map(len,ranks.values())))


def compile_pairs(plan,epochs=5,*,omit_reader=False,require_strict_rank=True):
    nz,ny,nx=plan['counts'];n=nz*ny*nx;sn=((nz+1)//2)*ny*nx
    if epochs<4:raise ValueError('At least four original packet rank epochs required')
    pair=lambda t:((t//(ny*nx))//2*ny+(t//nx)%ny)*nx+t%nx
    rank=lambda t:t//(ny*nx)+(t//nx)%ny+t%nx
    srank=lambda t:2*(t//(ny*nx))+(t//nx)%ny+t%nx
    start=[set() for _ in range(sn)];finish=[set() for _ in range(sn)];owners=[-1]*n;previous=[-1]*n;last={};hazards=0
    for t in plan['tile_ids']:
        z,y,x=t//(ny*nx),(t//nx)%ny,t%nx;node=pair(t)
        for parent in ([t-ny*nx] if z else [])+([t-nx] if y else [])+([t-1] if x else []):
            if pair(parent)!=node:start[node].add(pair(parent))
        owner=(rank(t)%epochs)*plan['max_frontier']+plan['slots'][t];owners[t]=owner;old=last.get(owner);previous[t]=-1 if old is None else old
        if old is not None:
            oz,oy,ox=old//(ny*nx),(old//nx)%ny,old%nx
            retire={old}
            if not omit_reader:
                for bits in range(1,8):
                    dz,dy,dx=bits&1,(bits>>1)&1,(bits>>2)&1
                    if oz+dz<nz and oy+dy<ny and ox+dx<nx:retire.add(old+dz*ny*nx+dy*nx+dx)
            for reader in retire:
                assert rank(reader)<rank(t),('Native packet retirement lacks rank order',reader,t)
                parent=pair(reader);assert parent!=node,('Fused node requires its own future reader COMPLETE',reader,t,old,node)
                if require_strict_rank:assert srank(parent)<srank(node),('Fused packet retirement does not strictly advance supernode rank',reader,t,parent,node)
                finish[node].add(parent);hazards+=1
        last[owner]=t
    children=[[] for _ in range(sn)];pending=[]
    for t in range(sn):
        for p in start[t]|finish[t]:children[p].append(t)
        pending.append(len(start[t]|finish[t]))
        for p in start[t]:assert srank(p)<srank(t),'START supernode rank order'
    ready=deque(t for t in range(sn) if not pending[t]);order=[]
    while ready:
        t=ready.popleft();order.append(t)
        for child in children[t]:
            pending[child]-=1
            if not pending[child]:ready.append(child)
    assert len(order)==sn,'Fused START/COMPLETE graph cycle'
    return dict(counts=(int((nz+1)//2),ny,nx),original_counts=(nz,ny,nx),epochs=epochs,node_count=sn,original_count=n,
                owner_count=epochs*plan['max_frontier'],owner_slots=owners,previous_holder=previous,start=start,finish=finish,order=order,hazard_checks=hazards,
                start_edges=sum(map(len,start)),finish_edges=sum(map(len,finish)),odd_upper_dummy=nz%2==1)


def report():
    cases=[]
    for counts in ((3,3,3),(5,4,4),(6,5,7),(8,7,9),(12,11,10),(76,36,18),(84,39,19)):
        for seed in (0,13,37):
            p=native_plan(counts,seed);c=compile_pairs(p)
            cases.append({k:v for k,v in c.items() if k not in ('owner_slots','previous_holder','start','finish','order')})
    witness=None
    for counts in itertools.product(range(3,9),repeat=3):
        for seed in (0,13,37):
            try:compile_pairs(native_plan(counts,seed),4,require_strict_rank=False)
            except AssertionError as error:witness=dict(counts=counts,seed=seed,error=str(error));break
        if witness:break
    assert witness,'Four-epoch fusion negative must have a concrete witness'
    return dict(schema='beamz.paired-supernode-DAG.v1',passed=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=cases,
                negative_controls={'four-rank-ring-contraction':witness},
                proof='Original owner reuse has reader rank <=writer_rank-epochs+3. A Z-pair spans original ranks [S,S+1]. epochs>=5 implies every completion edge S_parent<S_child. External axial START edges also strictly advance S; internal lowZ->upperZ stays inside node.',
                scope='Packet ownership and fused DAG legality; arbitrary per-rank slot permutations, odd Z counts and large geometry. No CUDA visibility, phase handoff, FP or speed claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=report();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(r['cases']),hazard_checks=sum(c['hazard_checks'] for c in r['cases']),negative_controls=r['negative_controls'])))
