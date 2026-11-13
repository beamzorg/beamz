"""Canonical word/version and barrier proof for the512-thread warp frontier.

This independently enumerates actual field-read routes, all compact shared
publishers and stochastic warp orders. Numerical FP32/CPML output parity and
compiled CUDA resources remain separate gates.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import cuda_warp_frontier_layout as geometry
from scripts.cuda_space_time_ir import FIELDS
from scripts.prove_cuda_frontier_schedule import Hazard, actual_reads


POINTS = tuple(itertools.product(range(8), range(16), range(32)))


def prove_layout(warp_y=4):
    point = lambda tid, j: geometry.point(tid, j, warp_y=warp_y)
    owner = lambda p: geometry.owner(p, warp_y=warp_y)
    index = lambda f, p: geometry.index(f, p, warp_y=warp_y)
    WORDS = geometry.descriptor(warp_y)["shared_words"]
    words = {}
    for f in range(6):
        for p in POINTS:
            key = index(f, p)
            if key is None:
                continue
            if key in words:
                raise Hazard(("Shared layout aliases distinct producer fields/points", key, words[key], (f, p)))
            words[key] = f, p
    assert set(words) == set(range(WORDS))
    owner_points = {point(tid, j) for tid in range(512) for j in range(8)}
    if owner_points != set(POINTS):
        raise Hazard("512-thread owner mapping misses or repeats a cache point")
    if any(owner(point(tid, j)) != tid for tid in range(512) for j in range(8)):
        raise Hazard("Inverse owner mapping does not recover the original lane")
    reads, cases = actual_reads(), []
    for n in range(1, 17):
        heads = tuple((-n-1) % s for s in (8, 16, 32))
        writers, phase_reads = {}, [set(), set()]
        for f in range(6):
            for p in POINTS:
                key = index(f, p)
                if key is not None:
                    # Howned and Z/Y ghosts use private point owners; lowX-only
                    # ghosts use coalesced stage producer i. E is point-owned.
                    tid = owner(p)
                    if f < 3 and p[2] == heads[2] and p[0] != heads[0] and p[1] != heads[1]:
                        dz, dy = (p[0]-heads[0]) % 8, (p[1]-heads[1]) % 16
                        tid = (dz-1)*15+dy-1
                    if key in writers:
                        raise Hazard(("Multiple shared producers", key))
                    writers[key] = tid
                if not all(a != b for a, b in zip(p, heads, strict=True)):
                    continue
                for read in reads[f]:
                    src = FIELDS.index(read.field)
                    if src//3 == f//3 or not any(read.offset):
                        continue
                    q = tuple((a+b) % s for a, b, s in zip(p, read.offset, (8, 16, 32), strict=True))
                    if read.offset[0]:
                        if owner(q) != owner(p):
                            raise Hazard("Z neighbor escaped its register owner")
                    elif owner(q)//32 == owner(p)//32:
                        # A full-mask shuffle is materialized before predicates;
                        # no future private ghost write changes that SSA value.
                        pass
                    else:
                        remote = index(src, q)
                        if remote is None or words.get(remote) != (src, q):
                            raise Hazard(("Crosswarp source lacks exact published word", f, p, read, q, remote))
                        phase_reads[f//3].add(remote)
        h_writes = {k for k, (f, _) in words.items() if f < 3}
        e_writes = set(words)-h_writes
        if phase_reads[0] & h_writes or phase_reads[1] & e_writes:
            raise Hazard("Same-phase shared producer overwrites an active shared reader")
        cases.append({"step": n, "H_remote_words_read": len(phase_reads[0]),
                      "E_remote_words_read": len(phase_reads[1]),
                      "unique_shared_producers": len(writers)})
    return {"layout": geometry.descriptor(warp_y), "bijection_words": len(words), "actual_route_cases": cases}


def simulate(*, first=1, steps=4, seed=19, negative=None, warp_y=4):
    point = lambda tid, j: geometry.point(tid, j, warp_y=warp_y)
    owner = lambda p: geometry.owner(p, warp_y=warp_y)
    index = lambda f, p: geometry.index(f, p, warp_y=warp_y)
    WORDS = geometry.descriptor(warp_y)["shared_words"]
    reads, rng = actual_reads(), random.Random(seed)
    h, e, shared, stage = {}, {}, [None]*WORDS, {}
    clocks, pending = {"H": first, "E": first}, {}
    checks = 0
    routes = {r: 0 for r in ("private-center", "register-Z", "warp-X", "warp-Y", "shared-face")}

    def physical(p, n):
        return tuple(64-n-1+((a+n+1) % s) for a, s in zip(p, (8, 16, 32), strict=True))

    def token(f, n, p):
        return f, n, physical(p, n)

    def require(value, f, n, world, reason):
        nonlocal checks
        checks += 1
        if value != (f, n, world):
            raise Hazard((negative, reason, value, (f, n, world)))

    def store(f, p, value):
        key = index(f, p)
        if key is not None:
            if negative == "alias-family-faces" and f < 3:
                key += WORDS//2
            shared[key] = value

    def clock(family, n):
        nonlocal checks
        checks += 1
        if clocks[family] != n:
            raise Hazard((negative, family, clocks[family], n))

    def read_value(f, read, p):
        src = FIELDS.index(read.field)
        if src//3 == f//3 or not any(read.offset):
            routes["private-center"] += 1
            return h[src, p] if src < 3 else e[src-3, p]
        private = h if src < 3 else e
        q = tuple((a+b) % s for a, b, s in zip(p, read.offset, (8, 16, 32), strict=True))
        if read.offset[0]:
            routes["register-Z"] += 1
            if owner(q) != owner(p):
                raise Hazard("Z register owner changed")
            return private[src % 3, q]
        if owner(q)//32 == owner(p)//32:
            routes["warp-X" if read.offset[2] else "warp-Y"] += 1
            if negative == "wrong-Y-shuffle" and read.offset[1]:
                tid = owner(p)
                # Incorrectly carry the old width32-X lane interpretation.
                wrong_tid = (tid//32)*32+((tid+read.offset[1]) % 32)
                q = point(wrong_tid, p[0])
            return private[src % 3, q]
        routes["shared-face"] += 1
        key = index(src, q)
        if key is None:
            raise Hazard(("Unpublished source", src, q))
        return shared[key]

    for p in POINTS:
        for c in range(3):
            h[c, p], e[c, p] = token(c, first-1, p), token(c+3, first-1, p)
            store(c+3, p, e[c, p])

    def run_h(warp, n, heads):
        clock("H", n)
        # Full-mask shuffles are captured before each j's active predicate.
        # Complete all H arithmetic in this warp before importing private E.
        for tid in range(warp*32, warp*32+32):
            for j in range(8):
                p = point(tid, j)
                if not all(a != b for a, b in zip(p, heads, strict=True)):
                    continue
                for f in range(3):
                    for read in reads[f]:
                        src = FIELDS.index(read.field)
                        require(read_value(f, read, p), src, n-1,
                                tuple(a+b for a, b in zip(physical(p, n), read.offset, strict=True)), "H exact canonical source")
                    h[f, p] = token(f, n, p)
                    store(f, p, h[f, p])
        for tid in range(warp*32, warp*32+32):
            for j in range(8):
                p = point(tid, j)
                if p[0] == heads[0] or p[1] == heads[1]:
                    for c in range(3):
                        h[c, p], e[c, p] = token(c, n, p), token(c+3, n, p)
                        store(c, p, h[c, p])
                        if negative == "early-E-shared-import":
                            store(c+3, p, e[c, p])
            if tid < 105:
                dz, dy = 1+tid//15, 1+tid % 15
                p = ((heads[0]+dz) % 8, (heads[1]+dy) % 16, heads[2])
                for f in range(6):
                    stage[f, tid] = token(f, n, p)
                    if f < 3:
                        store(f, p, stage[f, tid])
                    elif negative == "early-E-shared-import":
                        store(f, p, stage[f, tid])

    def run_e(warp, n, heads):
        clock("E", n)
        # All lanes retire private lowX packets before the first H shuffle.
        for tid in range(warp*32, warp*32+32):
            for j in range(8):
                p = point(tid, j)
                if p[2] == heads[2] and p[1] != heads[1] and p[0] != heads[0]:
                    key = (((p[0]-heads[0]) % 8)-1)*15+((p[1]-heads[1]) % 16)-1
                    for c in range(3):
                        if (c, key) not in stage or (c+3, key) not in stage:
                            raise Hazard((negative, "Unpublished lowX stage", c, key))
                        h[c, p], e[c, p] = stage[c, key], stage[c+3, key]
        for tid in range(warp*32, warp*32+32):
            for j in range(8):
                p = point(tid, j)
                if all(a != b for a, b in zip(p, heads, strict=True)):
                    for f in range(3, 6):
                        for read in reads[f]:
                            src = FIELDS.index(read.field)
                            require(read_value(f, read, p), src, n-1 if src >= 3 else n,
                                    tuple(a+b for a, b in zip(physical(p, n), read.offset, strict=True)), "E exact canonical source")
                        e[f-3, p] = token(f, n, p)
                    for c in range(3):
                        require(h[c, p], c, n, physical(p, n), "H trail/final value")
                        require(e[c, p], c+3, n, physical(p, n), "E trail/final value")
                for c in range(3):
                    store(c+3, p, e[c, p])

    for n in range(first, first+steps):
        heads = tuple((-n-1) % s for s in (8, 16, 32))
        hs, es = list(range(16)), list(range(16))
        rng.shuffle(hs)
        rng.shuffle(es)
        for i, warp in enumerate(hs):
            run_h(warp, n, heads)
            if negative == "missing-PF1" and i == 0:
                run_e(es[-1], n, heads)
        clocks.update(pending)
        pending.clear()
        pending["H"] = n+1
        if negative == "single-shared-clock":
            clocks["E"] = n+1
        for i, warp in enumerate(es):
            run_e(warp, n, heads)
            if negative == "missing-PF2" and i == 0:
                clocks["H"] = n+1
                run_h(hs[-1], n+1, tuple((-n-2) % s for s in (8, 16, 32)))
        clocks.update(pending)
        pending.clear()
        pending["E"] = n+1
    return {"warp_y": warp_y, "first": first, "steps": steps, "seed": seed, "actual_checks": checks,
            "read_routes": routes, "loop_barriers": 2*steps, "result": "pass"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--warp-y", type=int, choices=(2, 4, 8), default=4)
    a = p.parse_args()
    cases = [simulate(first=f, steps=n, seed=s, warp_y=a.warp_y) for f, n, s in itertools.product(
        (1, 4, 13), (1, 2) if a.quick else (1, 2, 8, 16), (7, 29))]
    negatives = []
    for name in ("alias-family-faces", "early-E-shared-import", "wrong-Y-shuffle",
                 "missing-PF1", "missing-PF2", "single-shared-clock"):
        try:
            # n3's low Y head aligns to an actual inter-warp publication face.
            # n1/n2 ghost imports can be disjoint from every remote consumer.
            simulate(first=1, steps=4, seed=7, negative=name, warp_y=a.warp_y)
        except Hazard as exc:
            negatives.append({"mutation": name, "rejected": True, "witness": str(exc)})
        else:
            raise AssertionError(f"Unsafe mutation escaped: {name}")
    proof = {"schema": "beamz.warp-frontier-512-oracle.v1", "layout_proof": prove_layout(a.warp_y),
             "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "positive_cases": cases, "negative_controls": negatives,
             "actual_checks": sum(c["actual_checks"] for c in cases),
             "scope": "actual field addresses, versions, compact publication words, phase clocks; no FP/CPML-state alias/resource/performance claim"}
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(proof, indent=2)+"\n")
    print(json.dumps({"positive_cases": len(cases), "actual_checks": proof["actual_checks"],
                      "negative_rejections": len(negatives), "layout_words": geometry.descriptor(a.warp_y)["shared_words"]}))


if __name__ == "__main__":
    main()
