"""CPU topology/read audit for NoIO cache8x8x32/T512 versus8x16x32/T1024.

This models the existing four-barrier family exchange, not the paired source
schedule. Same four interleaved-Z private slots/thread; no CUDA emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_space_time_ir import FIELDS, yee_operator
from scripts.model_cuda_systolic_traffic import geometry, grid_for, rank_histogram


class Hazard(AssertionError):
    pass


@dataclass(frozen=True)
class Topology:
    y: int
    threads: int

    def __post_init__(self):
        if (self.y, self.threads) not in ((8, 512), (16, 1024)):
            raise ValueError("Exact proposed and current NoIO mappings only")

    @property
    def cache(self):
        return 8, self.y, 32

    @property
    def xy(self):
        return self.y * 32

    @property
    def pad(self):
        return 8 * self.xy

    def point(self, tid, j):
        return tid // self.xy + 2 * j, (tid // 32) % self.y, tid % 32

    def owner(self, p):
        z, y, x = p
        return (z & 1) * self.xy + 32 * y + x, z // 2


def simulate(t, version=1, seed=13, negative=None):
    rng = random.Random(seed)
    op = yee_operator(cpml_axes=(0, 1, 2))
    reads = {FIELDS.index(u.output.field): u.reads for u in op.updates if u.output.field in FIELDS}
    points = {t.point(tid, j) for tid in range(t.threads) for j in range(4)}
    assert len(points) == t.pad
    heads = tuple((-version - 1) % d for d in t.cache)
    frame, private = {}, {}
    checks = 0

    def ghost(p):
        return any(c == h for c, h in zip(p, heads, strict=True))

    def physical(p, n):
        return tuple(64 - n - 1 + ((c + n + 1) % d) for c, d in zip(p, t.cache, strict=True))

    def token(f, n, p):
        return f, n, physical(p, n)

    def need(value, f, n, p, delta, context):
        nonlocal checks
        checks += 1
        expected = f, n, tuple(c + d for c, d in zip(physical(p, version), delta, strict=True))
        if value != expected:
            raise Hazard((negative, context, value, expected))

    for p in points:
        for f in range(6):
            private[f, p] = token(f, version - 1, p)
        frame[p] = [private[f, p] for f in (3, 4, 5)]

    def compute_warp(warp, phase):
        for tid in range(32 * warp, 32 * warp + 32):
            for j in range(4):
                p = t.point(tid, j)
                if ghost(p):
                    continue
                for f in range(3 * phase, 3 * phase + 3):
                    for r in reads[f]:
                        if r.field not in FIELDS:
                            continue
                        source = FIELDS.index(r.field)
                        q = tuple((c + d) % size for c, d, size in zip(p, r.offset, t.cache, strict=True))
                        own_family = source // 3 == phase
                        value = private[source, p] if own_family or not any(r.offset) else frame[q][source % 3]
                        need(value, source, version - 1 if source >= 3 or own_family else version, p, r.offset, "actual H/E read")
                    private[f, p] = token(f, version, p)

    order = list(range(t.threads // 32))
    rng.shuffle(order)
    for warp in order:
        compute_warp(warp, 0)
        if negative == "publish-H-before-old-E-retirement":
            for tid in range(warp * 32, warp * 32 + 32):
                for j in range(4):
                    p = t.point(tid, j)
                    frame[p] = [private[f, p] for f in (0, 1, 2)]
    # Imports provide all six current ghost values; B1 retires all old E reads.
    for p in points:
        if ghost(p):
            for f in range(6):
                private[f, p] = token(f, version, p)
        if negative != "omit-H-publication":
            frame[p] = [private[f, p] for f in (0, 1, 2)]
    # B2 publishes complete current H before any E neighbor reads.
    rng.shuffle(order)
    for warp in order:
        compute_warp(warp, 1)
        if negative == "publish-E-before-H-reader-retirement":
            for tid in range(warp * 32, warp * 32 + 32):
                for j in range(4):
                    p = t.point(tid, j)
                    frame[p] = [private[f, p] for f in (3, 4, 5)]
    # B3 retires current H readers; B4 publishes all E for the next H phase.
    for p in points:
        frame[p] = [private[f, p] for f in (3, 4, 5)]
        for f in range(6):
            if private[f, p] != token(f, version, p):
                raise Hazard(("wrong final field", f, p, private[f, p]))
    return {"cache": t.cache, "threads": t.threads, "slots": 4,
            "private_field_scalars": 24, "barriers": 4, "checks": checks,
            "shared_bytes": 3 * t.pad * 4 + 76 + 6 * 7 * (t.y - 1) * 4}


def report():
    cases = [simulate(t, n, seed) for t in (Topology(8, 512), Topology(16, 1024))
             for n in range(1, 33) for seed in (13, 37)]
    negatives = {}
    for name in ("publish-H-before-old-E-retirement", "omit-H-publication", "publish-E-before-H-reader-retirement"):
        try:
            simulate(Topology(8, 512), version=2, negative=name)
        except Hazard as e:
            negatives[name] = repr(e)
        else:
            raise AssertionError((name, "unsafe protocol passed"))
    models = []
    for t in (Topology(16, 1024), Topology(8, 512)):
        tile = tuple(c - 1 for c in t.cache)
        item = geometry(tile)
        item["threads"] = t.threads
        item["shared_bytes"] = 3 * t.pad * 4 + 76 + 6 * 7 * (t.y - 1) * 4
        item["finite_interior_tile_field_bytes_depth16"] = item["circular_field_payload_bytes_per_update"] + 24 * t.pad / math.prod(tile) / 16 + 24 / 16
        item["domain_envelopes"] = []
        for n in (136, 272, 512, 564, 640):
            grid = grid_for(n, 16, tile)
            hist = rank_histogram(grid)
            trail = 4 * max(hist) * 16 * 6 * item["outgoing_high_union"] * 4
            item["domain_envelopes"].append({"n": n, "depth": 16, "grid": grid,
                                            "ctas": math.prod(grid), "ranks": len(hist),
                                            "max_frontier": max(hist), "four_epoch_trail_bytes": trail})
        models.append(item)
    return {"scope": __doc__, "cases": cases, "reads": sum(c["checks"] for c in cases),
            "negatives": negatives, "models": models,
            "limits": "No CUDA emitted or register/occupancy/carveout/timing prediction. Host plan, source policies, all trail addressing and shape-derived admissions must be rebuilt for the new owned geometry."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = report()
    r["proof_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({"cases": len(r["cases"]), "reads": r["reads"], "negative_controls": len(r["negatives"])}))


if __name__ == "__main__":
    main()
