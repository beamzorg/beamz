"""Independent Yee reader/retirement oracle for partial shared Ey schedules.

Integer tags distinguish field, version and cache coordinate. Warp groups may
advance to next-plane publication while other warps still read the current
plane. This proves cache routing/lifetimes, not floating arithmetic, native
packed CPML, source/DFT execution or CUDA compiler convergence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA
from scripts.cuda_space_time_ir import FIELDS, yee_operator
from scripts.cuda_stencil_trace import serialize_updates

REVIEWED_BODY_SHA256 = "d7d93cce8f5472c2f540a2730258322080da2a98258c1139f306478f0d0eb4cc"


def bind_reviewed_body(body):
    """Keep a later source edit from silently inheriting this manual audit."""
    digest = hashlib.sha256(body).hexdigest()
    if digest != REVIEWED_BODY_SHA256:
        raise ValueError("Partial-Ey CUDA source changed; independent source review must be repeated")
    return {"body_sha256": digest, "binding_kind": "exact manually reviewed source; independent logical oracle",
            "scope": "The oracle proves the modeled schedule; this hash records the separately reviewed lowering, not automated CUDA semantic extraction"}


class Hazard(AssertionError):
    pass


@dataclass(frozen=True)
class Layout:
    z: int
    y: int
    threads: int
    x: int = 32

    @property
    def plane(self):
        return self.y * self.x

    @property
    def zgroups(self):
        return max(1, self.threads // self.plane)

    @property
    def ygroups(self):
        return max(1, self.plane // self.threads)

    @property
    def local_z(self):
        return self.z // self.zgroups

    @property
    def slots(self):
        return self.local_z * self.ygroups

    @property
    def volume(self):
        return self.z * self.plane

    def point(self, tid, j, gy=0):
        return tid // self.plane * self.local_z + j, (tid // 32) % self.y + gy * (self.threads // 32), tid % 32

    def owner(self, point):
        z, y, x = point
        if self.zgroups == 2:
            return (z // self.local_z) * self.plane + y * 32 + x, z % self.local_z
        gy, sy = divmod(y, self.threads // 32)
        return sy * 32 + x, gy * self.local_z + z

    def scratch(self, tid, gy=0):
        z, y, x = self.point(tid, 0, gy)
        return (((z // self.local_z) ^ 1) * self.plane if self.zgroups == 2 else 0) + y * 32 + x


def traced_reads():
    operator = yee_operator(cpml_axes=(0, 1, 2))
    return {FIELDS.index(u.output.field): tuple(sorted(r for r in u.reads if r.field in FIELDS))
            for u in operator.updates if u.output.field in FIELDS}


def simulate(layout, *, version=1, seed=0, negative=None, tail_pair=False, shared_ey_private_pending=False, paired_planes=False):  # noqa: C901
    if tail_pair and (layout.z != 8 or layout.zgroups != 1):
        raise ValueError("Tail-pair experiment is scoped to whole-Z cache8 mappings")
    if paired_planes and (not shared_ey_private_pending or layout != Layout(8, 16, 1024)):
        raise ValueError("Full-pair experiment requires shared-Ey cache8x16x32/T1024")
    reads = traced_reads()
    pad, slots, threads = layout.volume, layout.slots, layout.threads
    frame = [[0] * pad for _ in range(2)]
    ey_shared, ey_private = [0] * (pad if shared_ey_private_pending else 3 * pad // 4), [[0] * (slots // 4) for _ in range(threads)]
    h = [[[0] * slots for _ in range(3)] for _ in range(threads)]
    pending = [None] * 2048
    snapshots, private_pending = [None] * 2048, {}
    checked = 0
    heads = ((-version - 1) % layout.z, (-version - 1) % layout.y, (-version - 1) % 32)

    def linear(point):
        z, y, x = point
        return (z * layout.y + y) * 32 + x

    def token(field, n, point):
        return (6 * n + field) * pad + linear(point)

    def other(point, offset):
        return tuple((p + d) % n for p, d, n in zip(point, offset, (layout.z, layout.y, 32), strict=True))

    def ghost(point):
        return any(p == head for p, head in zip(point, heads, strict=True))

    def eye_location(point):
        z, y, x = point
        if shared_ey_private_pending:
            return ey_shared, linear(point)
        if z % 8 < 6:
            return ey_shared, ((z // 8 * 6 + z % 8) * layout.y + y) * 32 + x
        tid, si = layout.owner(point)
        return ey_private[tid], (si // 8) * 2 + si % 2

    def get_eye(point):
        values, index = eye_location(point)
        return values[index]

    def set_eye(point, value):
        values, index = eye_location(point)
        values[index] = value

    def need(value, field, n, point, context):
        nonlocal checked
        expected = token(field, n, point)
        checked += 1
        if value != expected:
            raise Hazard((negative, context, point, FIELDS[field], n, value, expected))

    points = [(tid, gy, j, layout.point(tid, j, gy)) for tid in range(threads)
              for gy in range(layout.ygroups) for j in range(layout.local_z)]
    for tid, gy, j, point in points:
        si = gy * layout.local_z + j
        for c in range(3):
            h[tid][c][si] = token(c, version - 1, point)
        frame[0][linear(point)], frame[1][linear(point)] = token(3, version - 1, point), token(5, version - 1, point)
        set_eye(point, token(4, version - 1, point))
    if negative in ("missing-old-H-retirement", "early-Ey-ghost-import"):
        for _, _, _, point in points:
            if ghost(point):
                if negative == "missing-old-H-retirement":
                    frame[0][linear(point)] = token(3, version, point)
                    frame[1][linear(point)] = token(5, version, point)
                set_eye(point, token(4, version, point))
    for tid, gy, j, point in points:
        if ghost(point):
            continue
        si = gy * layout.local_z + j
        for c in range(3):
            for read in reads[c]:
                f = FIELDS.index(read.field)
                q = other(point, read.offset)
                value = h[tid][f][si] if f < 3 else get_eye(q) if f == 4 else frame[int(f == 5)][linear(q)]
                need(value, f, version - 1, q, "H-old-field")
            h[tid][c][si] = token(c, version, point)
    # R0 has retired every old shared/private Ey read and published H low-X.
    for tid, gy, j, point in points:
        if ghost(point):
            si = gy * layout.local_z + j
            for c in range(3):
                h[tid][c][si] = token(c, version, point)
            frame[0][linear(point)], frame[1][linear(point)] = token(3, version, point), token(5, version, point)
            set_eye(point, token(4, version, point))
        if layout.zgroups == 2 and j == layout.local_z - 1:
            key = (point[0] // layout.local_z) * layout.plane + point[1] * 32 + point[2]
            si = gy * layout.local_z + j
            target = snapshots if shared_ey_private_pending else pending
            if negative != "missing-snapshot-publication":
                target[key], target[1024 + key] = h[tid][0][si], h[tid][1][si]
                if negative == "previous-version-snapshot":
                    target[key], target[1024 + key] = token(0, version - 1, point), token(1, version - 1, point)
    old = {}

    def prepare(tid, j):
        for gy in range(layout.ygroups):
            point = layout.point(tid, j, gy)
            si = gy * layout.local_z + j
            saved = frame[0][linear(point)], frame[1][linear(point)]
            if negative == "save-old-after-H-publish" or (negative == "tail-save-after-H-publish" and j >= layout.local_z - 2):
                saved = h[tid][0][si], h[tid][2][si]
            old[tid, gy, j] = saved
            frame[0][linear(point)], frame[1][linear(point)] = h[tid][0][si], h[tid][2][si]

    def restore(tid, j):
        for gy in range(layout.ygroups):
            point, key = layout.point(tid, j, gy), layout.scratch(tid, gy)
            values = private_pending[tid, gy] if shared_ey_private_pending else (pending[key], pending[1024 + key])
            frame[0][linear(point)], frame[1][linear(point)] = values

    tail_values = {}

    def compute(tid, j, *, tail=False, paired=False):
        for gy in range(layout.ygroups):
            point, si, key = layout.point(tid, j, gy), gy * layout.local_z + j, layout.scratch(tid, gy)
            ex, ez = old[tid, gy, j]
            if j and not paired and (not tail or j == layout.local_z - 2):
                previous = layout.point(tid, j - 1, gy)
                values = private_pending[tid, gy] if shared_ey_private_pending else (pending[key], pending[1024 + key])
                frame[0][linear(previous)], frame[1][linear(previous)] = values
            if not ghost(point):
                for c in range(3):
                    for read in reads[c + 3]:
                        f, q = FIELDS.index(read.field), other(point, read.offset)
                        if f >= 3:
                            value = ex if f == 3 else get_eye(point) if f == 4 else ez
                        elif not any(read.offset):
                            value = h[tid][f][si]
                        elif read.offset[0]:
                            if negative == "shared-previous-Z" and f == 0:
                                value = frame[0][linear(q)]
                            elif j:
                                value = h[tid][f][si - 1]
                            elif layout.zgroups == 2:
                                source_key = key ^ layout.plane if negative == "wrong-snapshot-slot" else key
                                value = (snapshots if shared_ey_private_pending else pending)[f * 1024 + source_key]
                            else:
                                value = h[tid][f][si + layout.local_z - 1]
                        elif f == 1:
                            peer, peer_si = layout.owner(q)
                            assert peer // 32 == tid // 32
                            value = h[peer][1][peer_si]
                        else:
                            value = frame[int(f == 2)][linear(q)]
                        need(value, f, version - 1 if f >= 3 else version, q, "E-read")
                    value = token(c + 3, version, point)
                    if c == 0:
                        ex = value
                        if j == 0 and negative == "early-snapshot-overwrite":
                            pending[key] = ex
                    elif c == 1:
                        set_eye(point, value)
                    else:
                        ez = value
            if tail or paired:
                tail_values[tid, gy, j] = ex, ez
            elif shared_ey_private_pending:
                private_pending[tid, gy] = ex, ez
            else:
                pending[key], pending[1024 + key] = ex, ez

    if not paired_planes:
        for tid in range(threads):
            prepare(tid, 0)
    rng = random.Random(seed)
    warps = list(range(threads // 32))
    ordinary_planes = 0 if paired_planes else layout.local_z - 2 if tail_pair else layout.local_z
    for j in range(ordinary_planes):
        # Pj: all publications complete. Whole warps may now run ahead and
        # publish H(j+1) while another warp is still reading current H(j).
        if seed:
            rng.shuffle(warps)
        for warp in warps:
            for tid in range(32 * warp, 32 * warp + 32):
                compute(tid, j)
            if negative == "missing-plane-retirement" or (negative == "missing-last-retirement" and j == layout.local_z - 1):
                for tid in range(32 * warp, 32 * warp + 32):
                    restore(tid, j)
            if j + 1 < ordinary_planes:
                for tid in range(32 * warp, 32 * warp + 32):
                    prepare(tid, j + 1)
            elif tail_pair:
                # A fast warp can publish BOTH tail planes while a laggard
                # still reads plane5. Those shared planes are disjoint.
                for tid in range(32 * warp, 32 * warp + 32):
                    for tail_j in (layout.local_z - 2, layout.local_z - 1):
                        if negative == "tail-no-second-publication" and tail_j == layout.local_z - 1:
                            for gy in range(layout.ygroups):
                                point = layout.point(tid, tail_j, gy)
                                old[tid, gy, tail_j] = frame[0][linear(point)], frame[1][linear(point)]
                        else:
                            prepare(tid, tail_j)
    if paired_planes:
        for tid in range(threads):
            prepare(tid, 0)
            prepare(tid, 1)
        for base in (0, 2):
            # Ppair publishes both planes and retires ALL prior-pair readers.
            if base:
                for tid in range(threads):
                    for gy in range(layout.ygroups):
                        for j in (base - 2, base - 1):
                            point = layout.point(tid, j, gy)
                            source_j = base - 1 if negative == "paired-forget-first-pending" else j
                            frame[0][linear(point)], frame[1][linear(point)] = tail_values[tid, gy, source_j]
            rng.shuffle(warps)
            for warp in warps:
                for j in (base, base + 1):
                    for tid in range(32 * warp, 32 * warp + 32):
                        if negative == "paired-overwrite-second-old" and j == base + 1:
                            for gy in range(layout.ygroups):
                                old[tid, gy, j] = tail_values[tid, gy, base]
                        compute(tid, j, paired=True)
                if not base:
                    for tid in range(32 * warp, 32 * warp + 32):
                        for j in (2, 3):
                            if negative == "paired-missing-H3" and j == 3:
                                for gy in range(layout.ygroups):
                                    point = layout.point(tid, j, gy)
                                    old[tid, gy, j] = frame[0][linear(point)], frame[1][linear(point)]
                            else:
                                prepare(tid, j)
                        if negative == "paired-early-restore":
                            for gy in range(layout.ygroups):
                                for j in (0, 1):
                                    point = layout.point(tid, j, gy)
                                    frame[0][linear(point)], frame[1][linear(point)] = tail_values[tid, gy, j]
        # RL retires all final-pair readers before either final E plane restores.
        for tid in range(threads):
            for gy in range(layout.ygroups):
                for j in (2, 3):
                    point = layout.point(tid, j, gy)
                    frame[0][linear(point)], frame[1][linear(point)] = tail_values[tid, gy, j]
    if tail_pair:
        # One P67 publishes both planes and retires all plane5 readers.
        rng.shuffle(warps)
        for warp in warps:
            for j in (layout.local_z - 2, layout.local_z - 1):
                for tid in range(32 * warp, 32 * warp + 32):
                    compute(tid, j, tail=True)
            if negative == "tail-restore-before-RL":
                for tid in range(32 * warp, 32 * warp + 32):
                    for gy in range(layout.ygroups):
                        for j in (layout.local_z - 2, layout.local_z - 1):
                            point = layout.point(tid, j, gy)
                            ex, ez = tail_values[tid, gy, j]
                            frame[0][linear(point)], frame[1][linear(point)] = ex, ez
        # RL retires readers of BOTH tail planes; old-save slots now hold results.
        for tid in range(threads):
            for gy in range(layout.ygroups):
                for j in (layout.local_z - 2, layout.local_z - 1):
                    point = layout.point(tid, j, gy)
                    source_j = layout.local_z - 1 if negative == "tail-overwrite-first-result" else j
                    ex, ez = tail_values[tid, gy, source_j]
                    frame[0][linear(point)], frame[1][linear(point)] = ex, ez
    # RL retires all last-plane H readers; RE publishes full current E.
    if not tail_pair and not paired_planes:
        for tid in range(threads):
            restore(tid, layout.local_z - 1)
    for _, _, _, point in points:
        need(frame[0][linear(point)], 3, version, point, "final-Ex")
        need(get_eye(point), 4, version, point, "final-Ey")
        need(frame[1][linear(point)], 5, version, point, "final-Ez")
    return {"checked_field_reads": checked, "layout": layout.__dict__, "version": version,
            "barriers_per_step": (layout.local_z // 2 + 3) if paired_planes else layout.local_z + 3 - int(tail_pair), "cross_Z_snapshots": layout.zgroups == 2,
            "persistent_field_floats_per_thread": 3 * slots + ((4 if paired_planes else 2) * layout.ygroups if shared_ey_private_pending else slots // 4),
            "saved_current_ExEz_floats": (4 if tail_pair or paired_planes else 2) * layout.ygroups,
            "shared_bytes": ((3 * pad + 2048) if shared_ey_private_pending else (2 * pad + 3 * pad // 4 + 2048)) * 4 + 76 + 3 * (layout.z - 1) * (layout.y - 1) * 4}


def report():
    layouts = (Layout(16, 16, 1024), Layout(16, 16, 512), Layout(8, 32, 1024), Layout(8, 32, 512), Layout(8, 32, 256))
    cases = [simulate(layout, version=n, seed=0) for layout in layouts for n in range(1, 17)]
    cases += [simulate(layout, version=n, seed=seed) for layout in layouts for n in (7, 15, 16, 31, 64) for seed in (13, 37)]
    negatives = {}
    for name in ("missing-old-H-retirement", "early-Ey-ghost-import", "save-old-after-H-publish", "early-snapshot-overwrite",
                 "wrong-snapshot-slot", "shared-previous-Z", "missing-plane-retirement", "missing-last-retirement"):
        try:
            simulate(layouts[0], negative=name)
        except Hazard as error:
            negatives[name] = repr(error)
        else:
            raise AssertionError(f"Negative control unexpectedly passed: {name}")
    return {"cases": cases, "valid_cases": len(cases), "negative_controls": negatives,
            "direct_negative_witnesses": direct_witnesses(), "collectives": collective_proof(),
            "scope": __doc__, "source_read_matrix": {FIELDS[f]: [(r.field, r.version, r.offset) for r in reads] for f, reads in traced_reads().items()}}


def direct_witnesses():
    reads = traced_reads()
    assert any(r.field == "Hx" and r.offset == (0, -1, 0) for r in reads[5])
    assert any(r.field == "Ez" and r.offset == (0, 1, 0) for r in reads[0])
    return {
        "missing-current-H-publication": {"reader": "Ez at(z0,y1,x2) needs Hx_n at(z0,y0,x2)",
                                           "unordered_writer": "neighbor still holds Ex_(n-1) until Pj"},
        "missing-final-E-publication": {"reader": "next Hx at last-local-Z,y0,x2 needs Ez_n at y1",
                                         "unordered_writer": "lagging warp still holds Hz_n before last restore; RE is required"},
        "missing-snapshot-publication": {"reader": "chunk0 Ey j0 needs Hx_n at chunk1 lastZ,sameXY",
                                          "unordered_writer": "snapshot producer has not completed until P0"},
        "publish-only-first-Y-half": {"reader": "Ez at y0 needs Hx_n at wrapped y31 in the fourth256-thread Y group",
                                       "unordered_writer": "second/third/fourth Y publications must all precede Pj"},
    }


def collective_proof():
    failures = []
    for head in range(32):
        owned = set(range(32)) - {head}
        h_target, e_target = (head - 1) % 32, (head + 1) % 32
        assert h_target in owned and (h_target + 1) % 32 == head
        assert e_target in owned and (e_target - 1) % 32 == head
        failures.append({"ghost_lane": head, "Hz_Ey_plusX_target": h_target, "Ez_Hy_minusX_target": e_target})
    return {"full_warp_required": True, "inside-owned-branch_negative_witnesses": failures,
            "scope": "Source ghost lane is required by an owned neighbor; both private-Ey(+X) and Hy(-X) shuffles must execute for all32 lanes before predicates"}


def prove_late_bootstrap(layout, minimum=(64, 64, 64), origin=(70, 40, 40), depth=16):
    dimensions = tuple(tuple(n + d for n, d in zip(minimum, delta, strict=True)) for delta in SUPPORT_DELTA)
    maximum = tuple(max(s[a] for s in dimensions) for a in range(3))
    first = max(1, *(o - n + 1 for o, n in zip(origin, maximum, strict=True)))
    if first > depth:
        raise ValueError("Requested late case does not execute")
    previous = first - 1
    cache = layout.z, layout.y, 32
    reads, checked = traced_reads(), 0

    def physical(point, version):
        return tuple(o - version - 1 + ((s + version + 1) % c)
                     for s, o, c in zip(point, origin, cache, strict=True))

    def valid(field, point):
        return all(0 <= p < n for p, n in zip(point, dimensions[field], strict=True))

    for tid in range(layout.threads):
        for gy in range(layout.ygroups):
            for j in range(layout.local_z):
                point = layout.point(tid, j, gy)
                old = physical(point, previous)
                previous_ghost = any((s + previous + 1) % c == 0 for s, c in zip(point, cache, strict=True))
                if first > 1 and not previous_ghost:
                    assert not any(valid(field, old) for field in range(6))
                    checked += 6
                current = physical(point, first)
                current_ghost = any((s + first + 1) % c == 0 for s, c in zip(point, cache, strict=True))
                if current_ghost:
                    continue
                for phase in (0, 1):
                    for field in range(3 * phase, 3 * phase + 3):
                        if not valid(field, current):
                            continue
                        for read in reads[field]:
                            source = FIELDS.index(read.field)
                            q = tuple(p + d for p, d in zip(current, read.offset, strict=True))
                            source_slot = tuple((s + d) % c for s, d, c in zip(point, read.offset, cache, strict=True))
                            source_version = previous if source >= 3 or phase == 0 else first
                            assert physical(source_slot, source_version) == q
                            checked += 1
    return {"first": first, "checked": checked, "origin": origin,
            "scope": "Zero initialized previous-owned cells are physically invalid; all actual valid target reads route to the exact physical previous/current field coordinate"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = report()
    path = Path(__file__).with_name("cuda_native_partial_ey_frame.cuh")
    result["source_binding"] = bind_reviewed_body(path.read_bytes())
    result["reviewed_body_sha256"] = result["source_binding"]["body_sha256"]
    graph = serialize_updates(yee_operator(cpml_axes=(0, 1, 2)).updates)
    result["operator_sha256"] = hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result["proof_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"valid_cases": result["valid_cases"], "checked_reads": sum(c["checked_field_reads"] for c in result["cases"]),
                      "negative_controls": len(result["negative_controls"]), "body_sha256": result["reviewed_body_sha256"]}))


if __name__ == "__main__":
    main()
