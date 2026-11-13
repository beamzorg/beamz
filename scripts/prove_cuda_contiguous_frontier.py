"""Actual-read oracle for four permanent field banks and Ey/Hy snapshots.

1024 threads own contiguous four-Z chunks. Canonical field reads determine
same-thread, warp-X, snapshot and shared-bank routes. Random warp permutations
exercise every phase without assuming lockstep between warps. There is no
numerical or compiled-CUDA performance claim.
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

from scripts.prove_cuda_frontier_schedule import Hazard, actual_reads
from scripts.cuda_space_time_ir import FIELDS


def simulate(*, first=1, steps=4, seed=19, early_private_zy_e=True, negative=None):
    reads, rng, shape, origin = actual_reads(), random.Random(seed), (8, 16, 32), (64, 64, 64)
    h, e, shared, snapshots, stage = {}, {}, {}, {}, {}
    clocks, pending_clocks = {"H": first, "E": first}, {}
    checks = 0
    route_counts = {r: 0 for r in ("private", "warp-X", "chunk-snapshot", "shared-bank")}

    def point(tid, j):
        return (tid//512)*4+j, (tid//32) % 16, tid % 32

    def owner(p):
        z, y, x = p
        return (z//4)*512+y*32+x

    def physical(p, n):
        return tuple(o-n-1+((a+n+1) % s) for o, a, s in zip(origin, p, shape, strict=True))

    def token(f, n, p):
        return f, n, physical(p, n)

    def require(value, f, version, p, reason):
        nonlocal checks
        checks += 1
        if value != (f, version, p):
            raise Hazard((negative, reason, value, (f, version, p)))

    def check_clock(family, n):
        nonlocal checks
        checks += 1
        if clocks[family] != n:
            raise Hazard((negative, family, clocks[family], n))

    points = tuple(itertools.product(*(range(s) for s in shape)))
    for p in points:
        for c in range(3):
            h[c, p], e[c, p] = token(c, first-1, p), token(c+3, first-1, p)
            if c != 1:
                shared[c+3, p] = e[c, p]
        if p[0] % 4 == 0:
            snapshots["E", p[0]//4, p[1], p[2]] = e[1, p]

    def canonical_value(f, r, p):
        src = FIELDS.index(r.field)
        q = tuple((a+b) % c for a, b, c in zip(p, r.offset, shape, strict=True))
        if src//3 == f//3 or not any(r.offset):
            route_counts["private"] += 1
            return h[src, p] if src < 3 else e[src-3, p]
        if src % 3 != 1:
            route_counts["shared-bank"] += 1
            return shared.get((src, q))
        family = "H" if src < 3 else "E"
        private = h if src < 3 else e
        if r.offset[2]:
            if owner(p)//32 != owner(q)//32 or p[:2] != q[:2]:
                raise Hazard("X shuffle escaped its warp/plane")
            route_counts["warp-X"] += 1
            return private[1, q]
        if r.offset[0]:
            j = p[0] % 4
            boundary = j == 0 if src < 3 else j == 3
            if boundary:
                route_counts["chunk-snapshot"] += 1
                return snapshots.get((family, (p[0]//4)^1, p[1], p[2]))
            if owner(p) != owner(q):
                raise Hazard("Same-thread Z route changed owner")
            route_counts["private"] += 1
            return private[1, q]
        raise Hazard(("Unexpected Ey/Hy canonical offset", r))

    def run_h(warp, n, heads):
        check_clock("H", n)
        # Complete this warp's entire H loop before its Z/Y private E imports.
        # Other warps cannot read its private Ey except through the immutable
        # first-plane snapshots; X shuffles stay within this warp.
        for tid in range(warp*32, warp*32+32):
            for j in range(4):
                p = point(tid, j)
                if not all(a != b for a, b in zip(p, heads, strict=True)):
                    continue
                for f in range(3):
                    for r in reads[f]:
                        src = FIELDS.index(r.field)
                        require(canonical_value(f, r, p), src, n-1,
                                tuple(a+b for a, b in zip(physical(p, n), r.offset, strict=True)), "H ordered source")
                    h[f, p] = token(f, n, p)
                    if f != 1:
                        shared[f if negative != "alias-H-E-banks" else f+3, p] = h[f, p]
        for tid in range(warp*32, warp*32+32):
            for j in range(4):
                p = point(tid, j)
                if p[0] == heads[0] or p[1] == heads[1]:
                    for c in range(3):
                        h[c, p] = token(c, n, p)
                        if c != 1:
                            shared[c, p] = h[c, p]
                        if early_private_zy_e:
                            e[c, p] = token(c+3, n, p)
            last = point(tid, 3)
            if not (last[2] == heads[2] and last[1] != heads[1] and last[0] != heads[0]):
                snapshots["H", last[0]//4, last[1], last[2]] = h[1, last]
            if negative == "early-E-snapshot":
                q = point(tid, 0)
                snapshots["E", q[0]//4, q[1], q[2]] = token(4, n, q)
            if tid < 105:
                dz, dy = 1+tid//15, 1+tid % 15
                p = ((heads[0]+dz) % 8, (heads[1]+dy) % 16, heads[2])
                for c in range(6):
                    stage[c, tid] = token(c, n, p)
                for c in (0, 2):
                    shared[c, p] = stage[c, tid]
                if p[0] % 4 == 3 and negative != "omit-coalesced-Hy-snapshot":
                    snapshots["H", p[0]//4, p[1], p[2]] = stage[1, tid]

    def run_e(warp, n, heads):
        check_clock("E", n)
        # Entire warp first retires lowX/imports before any Hy shuffle reads it.
        for tid in range(warp*32, warp*32+32):
            for j in range(4):
                p = point(tid, j)
                if p[2] == heads[2] and p[1] != heads[1] and p[0] != heads[0]:
                    key = (((p[0]-heads[0]) % 8)-1)*15+((p[1]-heads[1]) % 16)-1
                    for c in range(3):
                        if (c, key) not in stage or (c+3, key) not in stage:
                            raise Hazard((negative, "Unpublished lowX stage", c, key))
                        h[c, p], e[c, p] = stage[c, key], stage[c+3, key]
                elif not early_private_zy_e and (p[0] == heads[0] or p[1] == heads[1]):
                    for c in range(3):
                        e[c, p] = token(c+3, n, p)
        for tid in range(warp*32, warp*32+32):
            for j in range(4):
                p = point(tid, j)
                if all(a != b for a, b in zip(p, heads, strict=True)):
                    for f in range(3, 6):
                        for r in reads[f]:
                            src = FIELDS.index(r.field)
                            require(canonical_value(f, r, p), src, n-1 if src >= 3 else n,
                                    tuple(a+b for a, b in zip(physical(p, n), r.offset, strict=True)), "E ordered source")
                        e[f-3, p] = token(f, n, p)
                    for c in range(3):
                        require(h[c, p], c, n, physical(p, n), "H publication")
                        require(e[c, p], c+3, n, physical(p, n), "E publication")
                for c in (0, 2):
                    shared[c+3, p] = e[c, p]
            first_point = point(tid, 0)
            snapshots["E", first_point[0]//4, first_point[1], first_point[2]] = e[1, first_point]

    for n in range(first, first+steps):
        heads = tuple((-n-1) % s for s in shape)
        hs, es = list(range(32)), list(range(32))
        rng.shuffle(hs)
        rng.shuffle(es)
        for i, warp in enumerate(hs):
            run_h(warp, n, heads)
            if negative == "missing-PF1" and i == 0:
                run_e(es[-1], n, heads)
        clocks.update(pending_clocks)
        pending_clocks.clear()
        pending_clocks["H"] = n+1
        if negative == "single-shared-clock":
            clocks["E"] = n+1
        for i, warp in enumerate(es):
            run_e(warp, n, heads)
            if negative == "missing-PF2" and i == 0:
                clocks["H"] = n+1
                run_h(hs[-1], n+1, tuple((-n-2) % s for s in shape))
        clocks.update(pending_clocks)
        pending_clocks.clear()
        pending_clocks["E"] = n+1
    return {"first": first, "steps": steps, "seed": seed, "early_private_zy_e": early_private_zy_e,
            "actual_checks": checks, "route_counts": route_counts, "result": "pass"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    cases = [simulate(first=f, steps=n, seed=s, early_private_zy_e=e)
             for f, n, s, e in itertools.product((1, 4, 13), (1, 2) if a.quick else (1, 2, 8, 16), (7, 29), (False, True))]
    negatives = []
    for name in ("alias-H-E-banks", "early-E-snapshot", "missing-PF1", "missing-PF2", "single-shared-clock"):
        try:
            simulate(first=1, steps=2, seed=7, negative=name)
        except Hazard as exc:
            negatives.append({"mutation": name, "rejected": True, "witness": str(exc)})
        else:
            raise AssertionError(f"Unsafe mutation escaped: {name}")
    proof = {"schema": "beamz.contiguous-family-frontier.v1",
             "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "positive_cases": cases, "negative_controls": negatives,
             "actual_checks": sum(c["actual_checks"] for c in cases),
             "scope": "Canonical field address/version routes; no FP numerical, GPU resource/performance or mutable local-state alias claim"}
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(proof, indent=2)+"\n")
    print(json.dumps({"positive_cases": len(cases), "actual_checks": proof["actual_checks"], "negative_rejections": len(negatives)}))


if __name__ == "__main__":
    main()
