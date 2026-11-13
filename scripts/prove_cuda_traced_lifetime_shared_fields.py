"""Per-word/version and interwarp hazard oracle for no-private-field frontier.

Uses actual ordered Yee/CPML field dependencies. Shared field tokens retain
component, physical position and full Yee version; there are no private field
arrays. Shared center read/write is legal only for the same thread owner.
Ghosts use the actual interleaved Z mapping and coalesced low-X writer IDs.
This proves dataflow/visibility, not floating-point arithmetic or CUDA speed.
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
from scripts.prove_cuda_frontier_schedule import Hazard, Owner, actual_reads


def prove_hazards():
    g, graph = Owner(), actual_reads()
    summaries = []
    for step in range(1, 17):
        heads = tuple((-step - 1) % width for width in g.shape)
        accesses = [[], []]
        old_publications, future_ghosts = set(), set()
        for point in itertools.product(*(range(width) for width in g.shape)):
            owned = all(p != h for p, h in zip(point, heads, strict=True))
            owner = g.tid(point)
            if owned:
                for field in range(6):
                    phase = int(field >= 3)
                    for read in graph[field]:
                        source = FIELDS.index(read.field)
                        q = tuple((p + d) % width for p, d, width in zip(point, read.offset, g.shape, strict=True))
                        accesses[phase].append(("read", (source, q), owner))
                    accesses[phase].append(("write", (field, point), owner))
                # Both field families' trail/final readers are before PF2.
                for field in range(6):
                    accesses[1].append(("read", (field, point), owner))
                    old_publications.add((field, point))
            else:
                zghost, yghost, xghost = (p == h for p, h in zip(point, heads, strict=True))
                if xghost and not zghost and not yghost:
                    dz, dy = (point[0] - heads[0]) % 8, (point[1] - heads[1]) % 16
                    writer = (dz - 1) * 15 + dy - 1
                else:
                    writer = owner
                for field in range(6):
                    accesses[int(field >= 3)].append(("write", (field, point), writer))
            next_heads = tuple((h - 1) % width for h, width in zip(heads, g.shape, strict=True))
            if point[2] == next_heads[2] and point[0] != next_heads[0] and point[1] != next_heads[1]:
                future_ghosts.update((field, point) for field in range(3))
        conflicts_checked = 0
        for phase in range(2):
            writers, readers = {}, {}
            for kind, key, tid in accesses[phase]:
                if kind == "write":
                    if key in writers:
                        raise Hazard("Same-phase shared word has multiple writers")
                    writers[key] = tid
                else:
                    readers.setdefault(key, set()).add(tid)
            for key, writer in writers.items():
                # Every same-phase read/write overlap is a center or own-E
                # publication, carried out by the same program-ordered thread.
                if readers.get(key, set()) - {writer}:
                    raise Hazard((phase, key, "Cross-thread same-phase RAW/WAR conflict"))
                conflicts_checked += len(readers.get(key, ()))
        overlap = old_publications & future_ghosts
        if not overlap:
            raise Hazard("Publication-after-PF2 negative witness became vacuous")
        summaries.append({"step": step, "same_owner_ordered_conflicts": conflicts_checked,
                          "old_publication_future_lowX_H_writer_overlap": len(overlap),
                          "result": "pass"})
    return summaries


def simulate(*, first=1, steps=4, seed=19, origin=(64, 64, 64), negative=None):
    g, graph, rng = Owner(), actual_reads(), random.Random(seed)
    points = tuple(itertools.product(*(range(width) for width in g.shape)))
    shared, low_stage = {}, {}
    clocks, pending = {"H": first, "E": first}, {}
    checks = imports = publications = 0

    def physical(point, version):
        return tuple(o - version - 1 + ((p + version + 1) % width)
                     for o, p, width in zip(origin, point, g.shape, strict=True))

    def token(field, version, point):
        return field, version, physical(point, version)

    def require(value, field, version, physical_point, reason):
        nonlocal checks
        checks += 1
        expected = field, version, physical_point
        if value != expected:
            raise Hazard((negative, reason, value, expected))

    def check_clock(family, version):
        nonlocal checks
        checks += 1
        if clocks[family] != version:
            raise Hazard((negative, "opaque phase clock", family, clocks[family], version))

    def is_owned(point, heads):
        return all(p != h for p, h in zip(point, heads, strict=True))

    def low_x_point(tid, heads):
        dz, dy = 1 + tid // 15, 1 + tid % 15
        return (heads[0] + dz) % 8, (heads[1] + dy) % 16, heads[2]

    for point in points:
        for field in range(6):
            shared[field, point] = token(field, first - 1, point)
            if negative == "uninitialized-H-bootstrap" and field < 3:
                shared[field, point] = None

    def compute(phase, tid, point, version):
        family = range(3) if phase == 0 else range(3, 6)
        for field in family:
            for read in graph[field]:
                source = FIELDS.index(read.field)
                q = tuple((p + d) % width for p, d, width in zip(point, read.offset, g.shape, strict=True))
                value = shared.get((source, q))
                source_version = version - 1 if phase == 0 or source >= 3 else version
                require(value, source, source_version,
                        tuple(p + d for p, d in zip(physical(point, version), read.offset, strict=True)),
                        "canonical all-shared field read")
            target = field + 3 if phase == 0 and negative == "alias-H-E-banks" else field
            shared[target, point] = token(field, version, point)

    def run_h(warp, version, heads):
        nonlocal imports
        check_clock("H", version)
        tids = range(32 * warp, 32 * warp + 32)
        for tid in tids:
            for j in range(4):
                point = g.point(tid, j)
                if is_owned(point, heads):
                    compute(0, tid, point, version)
        for tid in tids:
            for j in range(4):
                point = g.point(tid, j)
                if point[0] == heads[0] or point[1] == heads[1]:
                    imports += 1
                    for field in range(3):
                        if negative != "late-H-ghost-publication":
                            shared[field, point] = token(field, version, point)
                        if negative == "early-E-ghost-publication":
                            shared[field + 3, point] = token(field + 3, version, point)
        for tid in tids:
            if tid < 105:
                point = low_x_point(tid, heads)
                imports += 1
                for field in range(6):
                    low_stage[field, tid] = token(field, version, point)
                    if field < 3 and negative != "late-H-ghost-publication":
                        shared[field, point] = low_stage[field, tid]

    def run_e(warp, version, heads):
        nonlocal imports, publications
        check_clock("E", version)
        tids = range(32 * warp, 32 * warp + 32)
        for tid in tids:
            for j in range(4):
                point = g.point(tid, j)
                if point[0] == heads[0] or point[1] == heads[1]:
                    imports += 1
                    for field in range(3, 6):
                        shared[field, point] = token(field, version, point)
        for tid in tids:
            if tid < 105 and negative != "missing-lowX-E-scatter":
                point = low_x_point(tid, heads)
                for field in range(3, 6):
                    shared[field, point] = low_stage.get((field, tid))
        for tid in tids:
            for j in range(4):
                point = g.point(tid, j)
                if is_owned(point, heads):
                    compute(1, tid, point, version)
        for tid in tids:
            for j in range(4):
                point = g.point(tid, j)
                if is_owned(point, heads):
                    for field in range(6):
                        require(shared.get((field, point)), field, version, physical(point, version),
                                "pre-PF2 shared trail/final publication")
                        publications += 1

    for version in range(first, first + steps):
        heads = tuple((-version - 1) % width for width in g.shape)
        h_warps, e_warps = list(range(32)), list(range(32))
        rng.shuffle(h_warps)
        rng.shuffle(e_warps)
        for i, warp in enumerate(h_warps):
            run_h(warp, version, heads)
            if negative == "missing-PF1" and i == 0:
                run_e(e_warps[0], version, heads)
        # PF1 retires old E readers and publishes current H/low-stage.
        clocks.update(pending)
        pending.clear()
        pending["H"] = version + 1
        if negative == "single-shared-clock":
            clocks["E"] = version + 1
        for i, warp in enumerate(e_warps):
            run_e(warp, version, heads)
            if negative == "missing-PF2" and i == 0:
                clocks["H"] = version + 1
                future_heads = tuple((h - 1) % width for h, width in zip(heads, g.shape, strict=True))
                run_h(h_warps[-1], version + 1, future_heads)
        # PF2 retires all H readers, trail/final readers and low-stage readers.
        clocks.update(pending)
        pending.clear()
        pending["E"] = version + 1
        if negative == "trail-publication-after-PF2":
            future_heads = tuple((h - 1) % width for h, width in zip(heads, g.shape, strict=True))
            point = ((heads[0] + 1) % 8, (heads[1] + 1) % 16, future_heads[2])
            assert is_owned(point, heads)
            dz, dy = (point[0] - future_heads[0]) % 8, (point[1] - future_heads[1]) % 16
            future_writer = (dz - 1) * 15 + dy - 1
            assert future_writer != g.tid(point)
            shared[0, point] = token(0, version + 1, point)
            require(shared[0, point], 0, version, physical(point, version),
                    "late H publication races future coalesced ghost writer")
    return {"first": first, "steps": steps, "seed": seed, "origin": origin,
            "actual_field_and_clock_checks": checks, "ghost_import_points": imports,
            "published_components": publications, "loop_barriers": 2 * steps,
            "private_field_arrays": 0, "result": "pass"}


NEGATIVES = ("alias-H-E-banks", "uninitialized-H-bootstrap", "early-E-ghost-publication",
             "late-H-ghost-publication", "missing-lowX-E-scatter", "missing-PF1", "missing-PF2",
             "single-shared-clock", "trail-publication-after-PF2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
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
            raise AssertionError(f"Unsafe schedule escaped: {name}")
    proof = {"schema": "beamz.all-shared-family-frontier.v1",
             "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "actual_reads": {FIELDS[field]: [str(read) for read in rs] for field, rs in actual_reads().items()},
             "same_phase_perword_hazards": prove_hazards(), "positive_cases": cases,
             "negative_controls": negatives,
             "total_actual_checks": sum(case["actual_field_and_clock_checks"] for case in cases),
             "scope": "component/point/version visibility plus same-thread center hazards; no numerical CUDA or performance claim"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"positive_cases": len(cases), "negative_rejections": len(negatives),
                      "total_actual_checks": proof["total_actual_checks"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
