"""Independent actual-Yee-read oracle for paired cache16x16x16 ownership.

No CUDA generation/resource/performance claim. The existing8x16x32 geometry is
a control. Both use1024 threads, contiguous4-Z ownership, immutable Hx/Hy chunk
snapshots, shared Ey, private Ex/Ez pending pairs and five CTA barriers.
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
class Geometry:
    z: int = 16
    y: int = 16
    x: int = 16
    threads: int = 1024

    def __post_init__(self):
        if (self.z, self.y, self.x, self.threads) not in ((16, 16, 16, 1024), (8, 16, 32, 1024)):
            raise ValueError("Only proposed cube and existing paired control mappings")

    @property
    def shape(self):
        return self.z, self.y, self.x

    @property
    def owned(self):
        return tuple(c - 1 for c in self.shape)

    @property
    def xy(self):
        return self.y * self.x

    @property
    def groups(self):
        return self.threads // self.xy

    @property
    def local_z(self):
        return self.z // self.groups

    def point(self, tid, j):
        return (tid // self.xy) * self.local_z + j, (tid // self.x) % self.y, tid % self.x

    def owner(self, p):
        z, y, x = p
        return (z // self.local_z) * self.xy + y * self.x + x, z % self.local_z

    def linear(self, p):
        z, y, x = p
        return (z * self.y + y) * self.x + x

    def snapshot(self, group, y, x):
        return group * self.xy + y * self.x + x


def actual_reads():
    op = yee_operator(cpml_axes=(0, 1, 2))
    return {FIELDS.index(u.output.field): tuple(sorted(r for r in u.reads if r.field in FIELDS))
            for u in op.updates if u.output.field in FIELDS}


def simulate(g=None, version=1, seed=13, negative=None):  # noqa: C901
    g = Geometry() if g is None else g
    rng, reads = random.Random(seed), actual_reads()
    frame = {p: [None, None, None] for tid in range(g.threads) for j in range(g.local_z) for p in [g.point(tid, j)]}
    h, snapshots, old, pending = {}, {}, {}, {}
    origin = (64, 64, 64)
    heads = tuple((-version - 1) % d for d in g.shape)
    checks, cross_chunk, warp_ops = 0, 0, 0

    def physical(p, n):
        return tuple(o - n - 1 + ((c + n + 1) % d) for o, c, d in zip(origin, p, g.shape, strict=True))

    def token(f, n, p):
        return f, n, physical(p, n)

    def ghost(p):
        return any(c == a for c, a in zip(p, heads, strict=True))

    def other(p, delta):
        return tuple((a + b) % c for a, b, c in zip(p, delta, g.shape, strict=True))

    def require(value, f, n, physical_point, where):
        nonlocal checks
        checks += 1
        if value != (f, n, physical_point):
            raise Hazard((negative, where, value, (f, n, physical_point)))

    for p in frame:
        for f in range(3):
            h[f, p] = token(f, version - 1, p)
            frame[p][f] = token(f + 3, version - 1, p)
    # All H readers see old E until R0. Owned-point physical neighbors are
    # checked against actual affine trace offsets, independently of wrap tags.
    for p in frame:
        if ghost(p):
            continue
        out_p = physical(p, version)
        for f in range(3):
            for read in reads[f]:
                source = FIELDS.index(read.field)
                q = other(p, read.offset)
                value = h[source, p] if source < 3 else frame[q][source - 3]
                required = tuple(a + b for a, b in zip(out_p, read.offset, strict=True))
                require(value, source, version - 1, required, "H actual affine source")
            h[f, p] = token(f, version, p)
    # Existing import/R0 protocol supplies current low H AND current low E.
    for p in frame:
        if ghost(p):
            for f in range(3):
                h[f, p] = token(f, version, p)
                frame[p][f] = token(f + 3, version, p)
        if p[0] % g.local_z == g.local_z - 1:
            group = p[0] // g.local_z
            key = g.snapshot(group, p[1], p[2])
            for f in (0, 1):
                snapshots[f, key] = token(f, version - 1, p) if negative == "stale-snapshot" else h[f, p]

    def prepare_warp(warp, pair):
        for tid in range(warp * 32, warp * 32 + 32):
            for j in (2 * pair, 2 * pair + 1):
                p = g.point(tid, j)
                old[tid, j] = frame[p][0], frame[p][2]
                if negative == "save-old-after-H":
                    old[tid, j] = h[0, p], h[2, p]
                frame[p][0], frame[p][2] = h[0, p], h[2, p]

    def compute_warp(warp, pair):
        nonlocal cross_chunk, warp_ops
        if pair:
            for tid in range(warp * 32, warp * 32 + 32):
                for j, values in zip((0, 1), pending[tid], strict=True):
                    p = g.point(tid, j)
                    frame[p][0], frame[p][2] = values
        new = {tid: [] for tid in range(warp * 32, warp * 32 + 32)}
        for j in (2 * pair, 2 * pair + 1):
            # ALL32 lanes participate, including both16-lane rows and ghosts.
            hy = {}
            for tid in range(warp * 32, warp * 32 + 32):
                p, lane = g.point(tid, j), tid % 32
                segment = (lane // g.x) * g.x
                if negative == "wrong-shuffle-width":
                    segment = 0
                peer = warp * 32 + segment + ((p[2] - 1) % g.x)
                q, peer_j = g.point(peer, j), j
                expected_q = other(p, (0, 0, -1))
                if q != expected_q:
                    raise Hazard((negative, "shuffle changed Y row", p, q, expected_q))
                if negative == "shuffle-only-owned" and not ghost(p) and ghost(q):
                    raise Hazard((negative, "excluded ghost provider", p, q))
                assert peer_j == j
                hy[tid] = h[1, q]
            warp_ops += 1
            for tid in range(warp * 32, warp * 32 + 32):
                p = g.point(tid, j)
                ex, ez = old[tid, j]
                if not ghost(p):
                    out_p = physical(p, version)
                    for c in range(3):
                        for read in reads[c + 3]:
                            f, q = FIELDS.index(read.field), other(p, read.offset)
                            if f >= 3:
                                value = ex if f == 3 else ez if f == 5 else frame[p][1]
                            elif not any(read.offset):
                                value = h[f, p]
                            elif read.offset[0]:
                                if j:
                                    value = h[f, g.point(tid, j - 1)]
                                else:
                                    group = p[0] // g.local_z
                                    previous = (group - 1) % g.groups
                                    if negative == "old-xor1-snapshot":
                                        previous = group ^ 1
                                    key = g.snapshot(previous, p[1], p[2])
                                    value = snapshots[f, key]
                                    cross_chunk += 1
                            elif f == 1:
                                value = hy[tid]
                            else:
                                value = frame[q][0 if f == 0 else 2]
                            required = tuple(a + b for a, b in zip(out_p, read.offset, strict=True))
                            require(value, f, version - 1 if f >= 3 else version, required, "E actual affine source")
                        result = token(c + 3, version, p)
                        if c == 0:
                            ex = result
                        elif c == 1:
                            frame[p][1] = result
                        else:
                            ez = result
                new[tid].append((ex, ez))
        for tid in new:
            pending[tid] = new[tid]

    warps = list(range(g.threads // 32))
    for warp in warps:
        prepare_warp(warp, 0)
    # P01 publishes both first planes and immutable snapshots.
    order = warps.copy()
    rng.shuffle(order)
    for warp in order:
        compute_warp(warp, 0)
        # Legal early preparation while other warps still consume first pair.
        prepare_warp(warp, 1)
        if negative == "restore-before-P23":
            for tid in range(warp * 32, warp * 32 + 32):
                for j, values in zip((0, 1), pending[tid], strict=True):
                    p = g.point(tid, j)
                    frame[p][0], frame[p][2] = values
    # P23 retires first pair readers before previous-pair restoration.
    rng.shuffle(order)
    for warp in order:
        compute_warp(warp, 1)
        if negative == "restore-before-RL":
            for tid in range(warp * 32, warp * 32 + 32):
                for j, values in zip((2, 3), pending[tid], strict=True):
                    p = g.point(tid, j)
                    frame[p][0], frame[p][2] = values
    # RL retires all final-pair Hxy readers; RE publishes all final E.
    for tid in range(g.threads):
        for j, values in zip((2, 3), pending[tid], strict=True):
            p = g.point(tid, j)
            frame[p][0], frame[p][2] = values
    for p in frame:
        for c in range(3):
            require(frame[p][c], c + 3, version, physical(p, version), "RE full E including ghosts")
    return {"cache": g.shape, "owned": g.owned, "threads": g.threads, "version": version,
            "zgroups": g.groups, "local_z": g.local_z, "snapshot_floats": 2 * g.threads,
            "field_register_lower_bound": 20, "barriers": 5,
            "checked_reads": checks, "cross_chunk_snapshot_reads": cross_chunk,
            "full_warp_shuffle_ops": warp_ops, "shuffle_segment_width": g.x}


def report():
    cases = [simulate(g, n, seed) for g in (Geometry(), Geometry(8, 16, 32))
             for n in range(1, 33) for seed in (13, 37)]
    negatives = {}
    for name in ("old-xor1-snapshot", "stale-snapshot", "wrong-shuffle-width", "shuffle-only-owned", "save-old-after-H", "restore-before-P23", "restore-before-RL"):
        try:
            simulate(negative=name, version=2)
        except Hazard as e:
            negatives[name] = repr(e)
        else:
            raise AssertionError((name, "unsafe schedule passed"))
    models = []
    for g in (Geometry(8, 16, 32), Geometry()):
        item = geometry(g.owned)
        item["source_pending_shared_bytes"] = 3 * 4096 * 4 + 2 * 1024 * 4 + 76 + 3 * (g.z - 1) * (g.y - 1) * 4
        item["raw_cta_grids"] = []
        for n in (136, 272, 512, 564, 640):
            grid = grid_for(n, 16, g.owned)
            hist = rank_histogram(grid)
            item["raw_cta_grids"].append({"n": n, "depth": 16, "grid": grid, "ctas": math.prod(grid), "ranks": len(hist), "max_frontier": max(hist), "scope": "rectangular transformed envelope before active-window clipping"})
        models.append(item)
    return {"scope": __doc__, "cases": cases, "checked_reads": sum(c["checked_reads"] for c in cases),
            "negatives": negatives, "analytic_geometry": models,
            "limits": "No emitted CUDA/source binding, physical packed/source/CPML execution, compiler register fit, coalescing or throughput claim. The exact-read proof assumes correct current ghost imports and old-cache bootstrap."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = report()
    r["proof_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({"cases": len(r["cases"]), "reads": r["checked_reads"], "negatives": len(r["negatives"])}))


if __name__ == "__main__":
    main()
