"""Prove and simulate physical-field reuse for the owned moving-box schedule.

This is a legality model, not a GPU execution result. Full rank joins remain
required. A point is published when it leaves a tile or at the macro endpoint;
imports read its last published version directly from the working field bank.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from pathlib import Path

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA
from scripts.cuda_space_time_ir import FIELDS, yee_operator


def componentwise_theorem(owned, depth):
    """Check RAW and WAR links without assuming chronological task execution."""
    op = yee_operator(cpml_axes=(0, 1, 2))
    counts = {"field_dependences": 0, "initial_WAR": 0, "cross_task_RAW_WAR": 0}
    for update in op.updates:
        if update.output.field not in FIELDS:
            continue
        for read in update.reads:
            if read.field not in FIELDS:
                continue
            dt = read.version - 1
            delta = tuple(d + dt for d in read.offset)
            assert all(d in (-1, 0) for d in delta), (update, read, delta)
            counts["field_dependences"] += 1
            for b, d in zip(owned, delta, strict=True):
                for source_q in range(-b, 2*b):
                    consumer_q = source_q - d
                    producer = source_q // b
                    consumer = consumer_q // b
                    next_writer = (source_q + 1) // b
                    # A cross-tile source necessarily leaves its producer at
                    # the next time step. Every consumer is coordinatewise no
                    # later than that next writer, so a distinct task is earlier
                    # in rank. Equality can only be the writer's own CTA.
                    assert producer <= consumer <= next_writer
                    if producer != consumer:
                        assert source_q % b == b-1
                        assert producer != next_writer
                    counts["cross_task_RAW_WAR"] += 1
    for b in owned:
        for tile in range(4):
            for p in range(tile*b-1, tile*b+b):
                first_writer = (p+1)//b
                assert tile <= first_writer
                counts["initial_WAR"] += 1
    return dict(owned=owned, depth=depth, checks=counts,
                intra_task="H uses prior E before imports; E uses current H; both complete before outgoing publication. Startup completes before publication.",
                rank_ties="Coordinatewise order plus equal sum implies the same tile. Distinct CTAs of equal rank have disjoint physical writes to each field.")


def token_schedule(minimum=(11, 13, 15), owned=(3, 5, 7), depth=4, seed=13, negative=None, order_kind="rank", heterogeneous_z=False):
    """Execute canonical versioned reads against a single physical field bank."""
    shapes = [tuple(n+d for n, d in zip(minimum, ds, strict=True)) for ds in SUPPORT_DELTA]
    maximum = tuple(n+1 for n in minimum)
    counts = tuple((m+depth+b-1)//b for m,b in zip(maximum,owned,strict=True))
    ranks = {}
    for tile in itertools.product(*(range(c) for c in counts)):
        ranks.setdefault(sum(tile), []).append(tile)
    order = []
    for rank in sorted(ranks, reverse=negative == "reverse-ranks"):
        same = ranks[rank].copy(); random.Random(seed+rank).shuffle(same); order.extend(same)
    if order_kind == "three-parent-DAG":
        # Kahn scheduling with only the three immediate axial predecessors.
        # Choosing a random ready task intentionally allows unequal ranks to
        # overtake each other. No global rank join or trail-reuse edge remains.
        pending={tile:sum(t>0 for t in tile) for tile in order}
        ready=[tile for tile,n in pending.items() if not n];order=[];rng=random.Random(seed)
        while ready:
            tile=ready.pop(rng.randrange(len(ready)));order.append(tile)
            for axis in range(3):
                successor=list(tile);successor[axis]+=1;successor=tuple(successor)
                if successor not in pending:continue
                pending[successor]-=1
                if not pending[successor]:ready.append(successor)
        assert len(order)==math.prod(counts) and not any(pending.values()), "Three-parent DAG failed progress"
    elif order_kind != "rank":
        raise ValueError("Explicit rank or three-parent-DAG order required")
    memory = [[0] * math.prod(s) for s in shapes]
    if negative == "output-cleared-before-bootstrap":
        memory = [[-1] * len(m) for m in memory]
    field_reads = {f: sorted(r for r in u.reads if r.field in FIELDS)
                   for u in yee_operator(cpml_axes=(0,1,2)).updates
                   if (f := u.output.field) in FIELDS}
    stats = dict(bootstrap_reads=0, ghost_reads=0, traced_reads=0, publications=0, late_tiles=0)
    def valid(f,p): return all(0 <= x < n for x,n in zip(p,shapes[f],strict=True))
    def index(f,p): return (p[0]*shapes[f][1]+p[1])*shapes[f][2]+p[2]
    def load(f,p,v,why,tile,n):
        if not valid(f,p): return
        actual = memory[f][index(f,p)]
        if actual != v:
            raise AssertionError(dict(negative=negative,why=why,tile=tile,n=n,field=FIELDS[f],point=p,expected=v,actual=actual))
    parent_owned=owned;execution=[];point_sets={}
    for tile in order:
        origin=tuple(t*b for t,b in zip(tile,parent_owned,strict=True))
        split=heterogeneous_z and (sum((a+1)*v for a,v in enumerate(tile))+seed)%3!=0
        if split:
            if parent_owned[0]%2:raise ValueError("Even parent Z ownership required for two equal children")
            child_owned=(parent_owned[0]//2,*parent_owned[1:]);children=[0,1]
            if negative=="reverse-internal-children":children.reverse()
            execution.extend(((*tile,child),(origin[0]+child*child_owned[0],*origin[1:]),child_owned) for child in children)
        else:execution.append((tile,origin,parent_owned))
    for tile,origin,owned in execution:
        if owned not in point_sets:
            cache_q=tuple(itertools.product(*(range(b+1) for b in owned)))
            point_sets[owned]=(cache_q,tuple(q for q in cache_q if all(q)),tuple(q for q in cache_q if not all(q)))
        cache_q,owned_q,ghost_q=point_sets[owned]
        first = max(1,*(o-m+1 for o,m in zip(origin,maximum,strict=True)))
        last = min(depth,*(o+b-1 for o,b in zip(origin,owned,strict=True)))
        if first > last: continue
        local = [{} for _ in FIELDS]
        if first == 1:
            for q in cache_q:
                p = tuple(o-1+d for o,d in zip(origin,q,strict=True))
                for f in range(6):
                    load(f,p,0,"bootstrap",tile,0)
                    local[f][p] = 0 if valid(f,p) else -1
                    stats["bootstrap_reads"] += int(valid(f,p))
        else:
            stats["late_tiles"] += 1
            for q in ghost_q:
                p = tuple(o-first+d for o,d in zip(origin,q,strict=True))
                for f in range(6):
                    load(f,p,first-1,"late ghost",tile,first-1)
                    local[f][p] = first-1 if valid(f,p) else -1
        for n in range(first,last+1):
            for phase in (0,1):
                for q in owned_q:
                    p = tuple(o-1+d-n for o,d in zip(origin,q,strict=True))
                    for f in range(3*phase,3*phase+3):
                        if not valid(f,p): continue
                        for read in field_reads[FIELDS[f]]:
                            sf = FIELDS.index(read.field)
                            sp = tuple(x+d for x,d in zip(p,read.offset,strict=True))
                            if not valid(sf,sp): continue
                            expected = n-1+read.version
                            actual = local[sf].get(sp,-999)
                            if actual != expected:
                                raise AssertionError(("local read",tile,n,FIELDS[f],p,read,sp,expected,actual))
                            stats["traced_reads"] += 1
                        local[f][p] = n
                if phase == 0:
                    for q in ghost_q:
                        p = tuple(o-1+d-n for o,d in zip(origin,q,strict=True))
                        for f in range(6):
                            if n == depth and f >= 3: continue  # dead terminal E ghosts
                            required = n-1 if negative == "stale-ghost-version" else n
                            load(f,p,required,"current ghost",tile,n)
                            local[f][p] = required if valid(f,p) else -1
                            stats["ghost_reads"] += int(valid(f,p))
            for q in owned_q:
                if n != depth and not any(d == b for d,b in zip(q,owned,strict=True)): continue
                p = tuple(o-1+d-n for o,d in zip(origin,q,strict=True))
                for f in range(6):
                    if valid(f,p):
                        memory[f][index(f,p)] = n+1 if negative == "publish-future-version" else n
                        stats["publications"] += 1
    assert all(v == depth for field in memory for v in field), "Missing final publication"
    overtakes=sum(sum(a)>sum(b) for a,b in zip(order,order[1:]))
    return dict(minimum=minimum,owned=parent_owned,depth=depth,seed=seed,order_kind=order_kind,rank_overtakes=overtakes,
                heterogeneous_z=heterogeneous_z,local_owned_shapes=sorted(point_sets),executed_tasks=len(execution),**stats)


def report():
    geometries = ((3,5,7),(7,15,31),(15,15,15))
    cases = [token_schedule(minimum=n,owned=b,depth=d,seed=s,order_kind=order_kind)
             for n,b in (((11,13,15),geometries[0]),((17,19,23),geometries[1]),((17,19,23),geometries[2]))
             for d in (4,16) for s in (13,37) for order_kind in ("rank","three-parent-DAG")]
    negatives = {}
    for negative in ("reverse-ranks","output-cleared-before-bootstrap","stale-ghost-version","publish-future-version"):
        try: token_schedule(negative=negative)
        except AssertionError as e: negatives[negative] = str(e)
        else: raise AssertionError("Negative passed: "+negative)
    return dict(schema="beamz.physical-frontier-legality.v1",passed=True,
                theorem=[componentwise_theorem(b,d) for b in geometries for d in (4,16)],
                schedules=cases,negative_controls=negatives,
                limits="Box ownership under full rank joins or complete three-parent task dependencies. No per-step producer overlap, CUDA visibility proof, pointer alias admission or allocation claim.",
                proof_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    r=report();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(dict(passed=r["passed"],schedules=len(r["schedules"]),traced_reads=sum(c["traced_reads"] for c in r["schedules"]),negative_controls=len(r["negative_controls"]))))
