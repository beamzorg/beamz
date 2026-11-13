"""Actual-read and visibility oracle for a two-barrier permanent-family cache.

No CUDA performance claim. The canonical ordered Yee/CPML graph supplies all
field reads. Tokens retain tensor, physical point and logical version. Warps
execute in adversarial orders between publication/retirement barriers. Shared
H/E banks and private ghost staging are modeled separately, including coalesced
low-X producers, arbitrary first phase indices and two opaque family clocks.
Startup data must be supplied consistently; native ragged/late-start support
admission remains a separate CUDA validation obligation.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_space_time_ir import FIELDS, yee_operator


class Hazard(AssertionError):
    pass


@dataclass(frozen=True)
class Owner:
    shape: tuple[int, int, int] = (8, 16, 32)
    threads: int = 1024

    def __post_init__(self):
        if self.shape != (8, 16, 32) or self.threads != 1024:
            raise ValueError("Exact selected1024-thread interleaved owner mapping required")

    def point(self, tid, j):
        return tid // 512 + 2*j, (tid // 32) % 16, tid % 32

    def tid(self, p):
        z, y, x = p
        return (z % 2)*512 + y*32 + x


def actual_reads():
    operator = yee_operator(cpml_axes=(0, 1, 2))
    return {FIELDS.index(u.output.field): tuple(sorted(r for r in u.reads if r.field in FIELDS))
            for u in operator.updates if u.output.field in FIELDS}


def prove_footprints():
    """Enumerate actual read locations; prove same-phase banks never conflict.

    Program-order accesses to a private register do not require a CTA barrier.
    A separate writer owns each shared component/point; ghostH is produced by
    either the z/y owner or a coalesced low-X writer. No whole-family guesses
    about the read footprint are used to construct actual shared consumers.
    """
    g, reads = Owner(), actual_reads()
    summaries = []
    for n in range(1, 17):
        heads = tuple((-n-1) % s for s in g.shape)
        phase_reads = [set(), set()]
        phase_writes = [set(), set()]
        writers = {}
        for p in itertools.product(*(range(s) for s in g.shape)):
            owned = all(a != b for a, b in zip(p, heads, strict=True))
            owner = g.tid(p)
            for f in range(6):
                if owned:
                    phase = int(f >= 3)
                    for r in reads[f]:
                        src = FIELDS.index(r.field)
                        if src//3 != f//3 and any(r.offset):
                            q = tuple((a+b) % c for a, b, c in zip(p, r.offset, g.shape, strict=True))
                            phase_reads[phase].add((src, q))
                # Howned and z/yghost are their point owner's publication.
                # LowX excluding z/y is coalesced dz-major/dy-minor (i<T).
                if f < 3 and not owned and p[2] == heads[2] and p[0] != heads[0] and p[1] != heads[1]:
                    dz, dy = (p[0]-heads[0]) % 8, (p[1]-heads[1]) % 16
                    writer = (dz-1)*15 + dy-1
                else:
                    writer = owner
                key = (f, p)
                if key in writers:
                    raise Hazard("Shared producer is not unique")
                writers[key] = writer
                phase_writes[int(f >= 3)].add(key)
        assert not (phase_reads[0] & phase_writes[0])
        assert not (phase_reads[1] & phase_writes[1])
        assert phase_reads[0] <= phase_writes[1]
        assert phase_reads[1] <= phase_writes[0]
        summaries.append({"step": n, "H_actual_shared_reads": len(phase_reads[0]),
                          "E_actual_shared_reads": len(phase_reads[1]),
                          "H_shared_writers": len(phase_writes[0]),
                          "E_shared_writers": len(phase_writes[1])})
    return summaries


def simulate(*, first=1, steps=4, seed=19, origin=(64, 64, 64), negative=None):
    g, reads, rng = Owner(), actual_reads(), random.Random(seed)
    points = tuple(itertools.product(*(range(s) for s in g.shape)))
    h, e, shared = {}, {}, {}
    low_stage = {}
    clocks = {"H": first, "E": first}
    clock_pending = {}
    checks = imports = publications = 0

    def physical(p, n):
        return tuple(o-n-1+((a+n+1) % s) for o, a, s in zip(origin, p, g.shape, strict=True))

    def token(f, n, p):
        return f, n, physical(p, n)

    def require(value, f, version, point, reason):
        nonlocal checks
        checks += 1
        expected = f, version, point
        if value != expected:
            raise Hazard((negative, reason, value, expected))

    def check_clock(family, n):
        nonlocal checks
        checks += 1
        if clocks[family] != n:
            raise Hazard((negative, family, clocks[family], n))

    for p in points:
        for c in range(3):
            h[c, p] = token(c, first-1, p)
            e[c, p] = token(c+3, first-1, p)
            shared[c+3, p] = e[c, p]

    def run_h(warp, n, heads):
        nonlocal imports
        check_clock("H", n)
        for tid in range(32*warp, 32*warp+32):
            for j in range(4):
                p = g.point(tid, j)
                if all(a != b for a, b in zip(p, heads, strict=True)):
                    for f in range(3):
                        for r in reads[f]:
                            src = FIELDS.index(r.field)
                            q = tuple((a+b) % c for a, b, c in zip(p, r.offset, g.shape, strict=True))
                            value = h[src, p] if src < 3 else e[src-3, p] if not any(r.offset) else shared[src, q]
                            require(value, src, n-1,
                                    tuple(a+b for a, b in zip(physical(p, n), r.offset, strict=True)), "H canonical ordered read")
                        h[f, p] = token(f, n, p)
                        shared[f if negative != "alias-H-E-banks" else f+3, p] = h[f, p]
                if p[0] == heads[0] or p[1] == heads[1]:
                    imports += 1
                    for c in range(3):
                        h[c, p] = token(c, n, p)
                        e[c, p] = token(c+3, n, p)
                        if negative != "late-H-publication":
                            shared[c, p] = h[c, p]
                        if negative == "early-E-ghost-publication":
                            shared[c+3, p] = e[c, p]
            if tid < 105:
                dz, dy = 1+tid//15, 1+tid % 15
                p = ((heads[0]+dz) % 8, (heads[1]+dy) % 16, heads[2])
                imports += 1
                for c in range(6):
                    low_stage[c, tid] = token(c, n, p)
                    if c < 3 and negative != "late-H-publication":
                        shared[c, p] = low_stage[c, tid]

    def run_e(warp, n, heads):
        nonlocal publications
        check_clock("E", n)
        for tid in range(32*warp, 32*warp+32):
            for j in range(4):
                p = g.point(tid, j)
                if p[2] == heads[2] and p[1] != heads[1] and p[0] != heads[0]:
                    dz, dy = (p[0]-heads[0]) % 8, (p[1]-heads[1]) % 16
                    key = (dz-1)*15+dy-1
                    for c in range(3):
                        h[c, p], e[c, p] = low_stage[c, key], low_stage[c+3, key]
                if all(a != b for a, b in zip(p, heads, strict=True)):
                    for f in range(3, 6):
                        for r in reads[f]:
                            src = FIELDS.index(r.field)
                            q = tuple((a+b) % c for a, b, c in zip(p, r.offset, g.shape, strict=True))
                            value = e[src-3, p] if src >= 3 else h[src, p] if not any(r.offset) else shared.get((src, q))
                            require(value, src, n-1 if src >= 3 else n,
                                    tuple(a+b for a, b in zip(physical(p, n), r.offset, strict=True)), "E canonical ordered read")
                        e[f-3, p] = token(f, n, p)
                for c in range(3):
                    shared[c+3, p] = e[c, p]
                if all(a != b for a, b in zip(p, heads, strict=True)):
                    for c in range(3):
                        require(h[c, p], c, n, physical(p, n), "H trail/final publication")
                        require(e[c, p], c+3, n, physical(p, n), "E trail/final publication")
                        publications += 2

    for n in range(first, first+steps):
        heads = tuple((-n-1) % s for s in g.shape)
        h_warps, e_warps = list(range(32)), list(range(32))
        rng.shuffle(h_warps)
        rng.shuffle(e_warps)
        if negative == "early-H-clock":
            clocks["H"] = n+1
        for i, warp in enumerate(h_warps):
            run_h(warp, n, heads)
            if negative == "missing-PF1" and i == 0:
                run_e(e_warps[0], n, heads)
        # PF1 completes all H writes, stages and old E readers. It publishes
        # the E clock written after previous PF2 and retires H clock readers.
        clocks.update(clock_pending)
        clock_pending.clear()
        clock_pending["H"] = n+1
        if negative == "single-shared-clock":
            clocks["E"] = n+1
        if negative == "early-E-clock":
            clocks["E"] = n+1
        for i, warp in enumerate(e_warps):
            run_e(warp, n, heads)
            if negative == "missing-PF2" and i == 0:
                # H word is written after PF1, even before it is visible at
                # PF2; the failure being isolated is field visibility.
                clocks["H"] = n+1
                next_heads = tuple((-n-2) % s for s in g.shape)
                run_h(h_warps[-1], n+1, next_heads)
        # PF2 publishes E and H's next clock, retires every H shared reader
        # and every E clock reader. E's next clock needs only next PF1.
        clocks.update(clock_pending)
        clock_pending.clear()
        clock_pending["E"] = n+1
    return {"first": first, "steps": steps, "seed": seed, "origin": origin,
            "actual_field_and_clock_checks": checks, "ghost_import_points": imports,
            "published_components": publications, "loop_barriers": 2*steps,
            "result": "pass"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    cases = []
    for first, steps, seed in itertools.product((1, 4, 13), (1, 2) if a.quick else (1, 2, 8, 16), (7, 29)):
        cases.append(simulate(first=first, steps=steps, seed=seed))
    negatives = []
    for name in ("alias-H-E-banks", "early-E-ghost-publication", "late-H-publication",
                 "missing-PF1", "missing-PF2", "single-shared-clock", "early-H-clock", "early-E-clock"):
        try:
            simulate(first=1, steps=2, seed=7, negative=name)
        except Hazard as exc:
            negatives.append({"mutation": name, "rejected": True, "witness": str(exc)})
        else:
            raise AssertionError(f"Unsafe mutation escaped: {name}")
    proof = {"schema": "beamz.permanent-family-frontier-oracle.v1",
             "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "canonical_reads": {FIELDS[f]: [str(r) for r in rds] for f, rds in actual_reads().items()},
             "same_phase_footprints": prove_footprints(), "positive_cases": cases,
             "negative_controls": negatives,
             "total_actual_checks": sum(c["actual_field_and_clock_checks"] for c in cases),
             "scope": "ordered field/version/address dataflow; no FP numerical, CPML-state alias, compiled CUDA resource or performance claim"}
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(proof, indent=2)+"\n")
    print(json.dumps({"positive_cases": len(cases), "actual_checks": proof["total_actual_checks"],
                      "negative_rejections": len(negatives), "output": str(a.output)}))


if __name__ == "__main__":
    main()
