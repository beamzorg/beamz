"""Actual-field/version oracle for H-private/E-shared compact Y exchange.

The field trace supplies read identities and offsets. Values retain tensor,
physical coordinate and version. Private H owners, current low-X stage, dense E
shared bank, full-warp X exchanges and compact Y bank are distinct storage.
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

from scripts.cuda_space_time_ir import FIELDS
from scripts.prove_cuda_frontier_schedule import actual_reads, Hazard
from scripts.prove_cuda_traced_lifetime_private_psi512 import Owner512

VOLUME = 7 * 15 * 31


def compact_key(field, point, heads):
    dz, dy, dx = ((p-h) % width for p, h, width in zip(point, heads, (8, 16, 32), strict=True))
    if field not in (0, 2) or not (1 <= dz <= 7 and 0 <= dy <= 14 and 1 <= dx <= 31):
        raise Hazard((field, point, heads, "Compact H Y footprint out of bounds"))
    return (VOLUME if field == 2 else 0) + ((dz-1)*15+dy)*31+dx-1


def prove_footprint():
    graph, owner = actual_reads(), Owner512()
    for field, reads in graph.items():
        for read in reads:
            source = FIELDS.index(read.field)
            if source//3 == field//3 and (source != field or any(read.offset)):
                raise Hazard("Same-family shared/private centers are not same-point same-component")
    cases = []
    for step in range(1, 17):
        heads = tuple((-step-1) % width for width in owner.shape)
        producers, consumers = {}, set()
        for point in itertools.product(range(8), range(16), range(32)):
            delta = tuple((p-h) % width for p, h, width in zip(point, heads, owner.shape, strict=True))
            owned = all(delta)
            if delta[0] and delta[2] and delta[1] < 15:
                for field in (0, 2):
                    key = compact_key(field, point, heads)
                    if key in producers:
                        raise Hazard("Compact Y shared producer is not unique")
                    producers[key] = field, point, owner.tid(point)
            if owned:
                for field in range(3, 6):
                    for read in graph[field]:
                        source = FIELDS.index(read.field)
                        if source >= 3 or not read.offset[1]:
                            continue
                        if read.offset != (0, -1, 0) or source not in (0, 2):
                            raise Hazard("Actual Y consumer does not match Hx/Hz negative-Y class")
                        q = point[0], (point[1]-1) % 16, point[2]
                        consumers.add(compact_key(source, q, heads))
        if consumers != set(producers) or consumers != set(range(2*VOLUME)):
            raise Hazard("Compact Y exact producer/consumer footprint differs")
        cases.append({"step": step, "compact_H_words": len(producers), "same_phase_write_read_conflicts": 0,
                      "lowY_ghost_words": 2*7*31, "highY_owned_words_retained": 0,
                      "E_shared_same_phase_centers": "samepointowner; ghostEwriters disjoint from owned E reads"})
    return cases


def simulate(*, first=1, steps=4, seed=19, origin=(64, 64, 64), negative=None):
    owner, graph, rng = Owner512(), actual_reads(), random.Random(seed)
    points = tuple(itertools.product(range(8), range(16), range(32)))
    h, e, ybank, stage = {}, {}, {}, {}
    clocks, pending = {"H": first, "E": first}, {}
    checks = publications = cross_z = warp_x = cross_y = 0

    def physical(point, version):
        return tuple(o-version-1+((p+version+1) % width) for o, p, width in zip(origin, point, owner.shape, strict=True))

    def token(field, version, point):
        return field, version, physical(point, version)

    def require(value, field, version, point, why):
        nonlocal checks
        checks += 1
        expected = field, version, point
        if value != expected:
            raise Hazard((negative, why, value, expected))

    def is_owned(point, heads):
        return all(p != head for p, head in zip(point, heads, strict=True))

    def check_clock(family, version):
        nonlocal checks
        checks += 1
        if clocks[family] != version:
            raise Hazard((negative, "Wrong family clock", family, clocks[family], version))

    def low_point(tid, heads):
        return (heads[0]+1+tid//15) % 8, (heads[1]+1+tid % 15) % 16, heads[2]

    def publish_y(field, point, value, heads):
        dz, dy, dx = ((p-head) % width for p, head, width in zip(point, heads, owner.shape, strict=True))
        if field in (0, 2) and dz and dx and dy < 15:
            if negative == "omit-lowY-ghost-H" and dy == 0:
                return
            key = compact_key(field, point, heads)
            if negative == "wrong-Y-index":
                key = (key+31) % (2*VOLUME)
            ybank[key] = value

    for point in points:
        for field in range(3):
            h[field, point] = token(field, first-1, point)
            e[field+3, point] = token(field+3, first-1, point)

    def run_h(warp, version, heads):
        check_clock("H", version)
        tids = range(32*warp, 32*warp+32)
        for tid in tids:
            for j in range(8):
                point = owner.point(tid, j)
                if is_owned(point, heads):
                    for field in range(3):
                        for read in graph[field]:
                            source = FIELDS.index(read.field)
                            q = tuple((p+d) % width for p, d, width in zip(point, read.offset, owner.shape, strict=True))
                            value = h[source, point] if source < 3 else e.get((source, q))
                            require(value, source, version-1,
                                    tuple(p+d for p, d in zip(physical(point, version), read.offset, strict=True)), "H actual field read")
                        h[field, point] = token(field, version, point)
                        publish_y(field, point, h[field, point], heads)
        for tid in tids:
            for j in range(8):
                point = owner.point(tid, j)
                if point[0] == heads[0] or point[1] == heads[1]:
                    for field in range(3):
                        h[field, point] = token(field, version, point)
                        publish_y(field, point, h[field, point], heads)
                        if negative == "early-E-ghost":
                            e[field+3, point] = token(field+3, version, point)
        for tid in tids:
            if tid < 105:
                point = low_point(tid, heads)
                for field in range(6):
                    stage[field, tid] = token(field, version, point)

    def run_e(warp, version, heads):
        nonlocal publications, cross_z, warp_x, cross_y
        check_clock("E", version)
        tids = range(32*warp, 32*warp+32)
        # Every private low-X provider is repaired before unconditional shuffles.
        for tid in tids:
            for j in range(8):
                point = owner.point(tid, j)
                if point[2] == heads[2] and point[1] != heads[1] and point[0] != heads[0]:
                    dz, dy = (point[0]-heads[0]) % 8, (point[1]-heads[1]) % 16
                    key = (dz-1)*15+dy-1
                    if negative != "omit-lowX-private-H":
                        for field in range(3):
                            h[field, point] = stage.get((field, key))
                if point[0] == heads[0] or point[1] == heads[1]:
                    for field in range(3, 6):
                        e[field, point] = token(field, version, point)
        for tid in tids:
            if tid < 105:
                point = low_point(tid, heads)
                for field in range(3, 6):
                    e[field, point] = stage.get((field, tid))
        for j in range(8):
            x_values = {}
            for tid in tids:
                point = owner.point(tid, j)
                lane = ((tid % 32) + (1 if negative == "wrong-X-shuffle" else -1)) % 32
                peer = owner.point(32*warp+lane, j)
                if negative == "exclude-ghost-shuffle-lane" and is_owned(point, heads) and not is_owned(peer, heads):
                    raise Hazard("Ghost lane omitted from fullwarp H exchange")
                for field in (1, 2):
                    x_values[field, tid] = h[field, peer]
            for tid in tids:
                point = owner.point(tid, j)
                if not is_owned(point, heads):
                    continue
                for field in range(3, 6):
                    for read in graph[field]:
                        source = FIELDS.index(read.field)
                        q = tuple((p+d) % width for p, d, width in zip(point, read.offset, owner.shape, strict=True))
                        if source >= 3:
                            value = e.get((source, point))
                        elif not any(read.offset):
                            value = h[source, point]
                        elif read.offset[0]:
                            q = owner.point(tid, (j+(1 if negative == "wrong-Z-slot" else 7)) & 7)
                            value = h[source, q]
                            cross_z += 1
                        elif read.offset[2]:
                            value = x_values[source, tid]
                            warp_x += 1
                        else:
                            value = ybank.get(compact_key(source, q, heads))
                            cross_y += 1
                        require(value, source, version-1 if source >= 3 else version,
                                tuple(p+d for p, d in zip(physical(point, version), read.offset, strict=True)), "E actual Hprivate/Eshared/Ybank read")
                    e[field, point] = token(field, version, point)
        # Same thread's E writes and private H supply trails/final before PF2.
        for tid in tids:
            for j in range(8):
                point = owner.point(tid, j)
                if is_owned(point, heads):
                    for field in range(6):
                        value = h[field, point] if field < 3 else e.get((field, point))
                        require(value, field, version, physical(point, version), "prePF2 trail/final publication")
                        publications += 1

    for version in range(first, first+steps):
        heads = tuple((-version-1) % width for width in owner.shape)
        h_warps, e_warps = list(range(16)), list(range(16))
        rng.shuffle(h_warps)
        rng.shuffle(e_warps)
        for i, warp in enumerate(h_warps):
            run_h(warp, version, heads)
            if negative == "missing-PF1" and i == 0:
                run_e(e_warps[0], version, heads)
        clocks.update(pending)
        pending.clear()
        pending["H"] = version+1
        if negative == "single-clock":
            clocks["E"] = version+1
        for i, warp in enumerate(e_warps):
            run_e(warp, version, heads)
            if negative == "missing-PF2" and i == 0:
                clocks["H"] = version+1
                future_heads = tuple((head-1) % width for head, width in zip(heads, owner.shape, strict=True))
                run_h(h_warps[-1], version+1, future_heads)
        clocks.update(pending)
        pending.clear()
        pending["E"] = version+1
    return {"first": first, "steps": steps, "seed": seed,
            "actual_field_clock_checks": checks, "H_Z_private_neighbor_reads": cross_z,
            "H_X_warp_neighbor_reads": warp_x, "H_Y_compact_neighbor_reads": cross_y,
            "published_components": publications, "loop_barriers": 2*steps, "result": "pass"}


NEGATIVES = ("omit-lowY-ghost-H", "wrong-Y-index", "early-E-ghost", "omit-lowX-private-H",
             "wrong-X-shuffle", "wrong-Z-slot", "exclude-ghost-shuffle-lane", "missing-PF1", "missing-PF2", "single-clock")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cases = [simulate(first=first, steps=steps, seed=seed)
             for first, steps, seed in itertools.product((1, 4, 13), (1, 2) if args.quick else (1, 2, 8, 16), (7, 29))]
    negatives = []
    for name in NEGATIVES:
        try:
            simulate(first=1, steps=2, seed=7, negative=name)
        except Hazard as exc:
            negatives.append({"mutation": name, "rejected": True, "witness": str(exc)})
        else:
            raise AssertionError(f"Unsafe hybrid schedule escaped: {name}")
    report = {"schema": "beamz.hybridY-privateH-sharedE.v1", "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "exact_reader_producer_footprints": prove_footprint(), "positive_cases": cases, "negative_controls": negatives,
              "actual_field_clock_checks": sum(case["actual_field_clock_checks"] for case in cases),
              "limits": "CPU token/read/owner/barrier evidence; no numerical CUDA or compilerresource/performance claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"positive_cases": len(cases), "negative_rejections": len(negatives),
                      "actual_field_clock_checks": report["actual_field_clock_checks"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
