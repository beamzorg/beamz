"""Field-only alias proof for owned moving-cache bootstrap/final scatter.

No backend admission is changed. Corresponding input/final fields must have the
same dense layout; different components and psi/profiles/sources/trails/raw DFT
buffers remain disjoint. Current full rank joins retire earlier bootstrap reads.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA
from scripts.cuda_space_time_ir import FIELDS, yee_operator

REVIEWED_CONSUMER_SHA256 = "f082980ad0687ef165c7807b415e87cbc1003edfdbd2acf7a4ee018dae4d4547"


def axis_links(extent, owned, depth):
    if min(extent, owned, depth) < 1:
        raise ValueError("Positive field extent/ownership/depth required")
    links = same = 0
    for p in range(extent):
        writer = (p + depth) // owned
        readers = {p // owned, (p + 1) // owned}
        # These are exactly the integer tiles with p in[C*B-1,C*B+B-1].
        for reader in readers:
            assert reader * owned - 1 <= p <= reader * owned + owned - 1
            assert reader <= writer
        assert writer * owned - depth <= p < writer * owned + owned - depth
        links += len(readers)
        same += writer in readers
    return links, same


def factorized_case(minimum, owned, depth):
    fields = []
    for field, delta in zip(FIELDS, SUPPORT_DELTA, strict=True):
        shape = tuple(n + d for n, d in zip(minimum, delta, strict=True))
        axes = [axis_links(n, b, depth) for n, b in zip(shape, owned, strict=True)]
        links = math.prod(a[0] for a in axes)
        same = math.prod(a[1] for a in axes)
        fields.append({"field": field, "shape": shape, "words": math.prod(shape),
                       "initial_reader_to_final_writer_links": links,
                       "same_tile_links": same, "strictly_earlier_rank_readers": links - same})
    return {"minimum_zyx": minimum, "owned_zyx": owned, "depth": depth,
            "field_cases": fields, "removed_bank_bytes": 4 * sum(f["words"] for f in fields)}


def trace_initial_coverage(owned):
    checks = 0
    op = yee_operator(cpml_axes=(0, 1, 2))
    for update in op.updates:
        for r in update.reads:
            if r.field not in FIELDS or r.version != 0:
                continue
            # At first output version1, physical owned points span[-1,B-2].
            # Every version0 field operand fits the actual bootstrap[-1,B-1].
            for corner in itertools.product((0, 1), repeat=3):
                p = tuple((-1 if not side else b - 2) + d for side, b, d in zip(corner, owned, r.offset, strict=True))
                assert all(-1 <= x <= b - 1 for x, b in zip(p, owned, strict=True))
                checks += 1
    return checks


def token_schedule(minimum=(27, 29, 31), owned=(7, 15, 31), depth=16, seed=13, negative=None):
    maximum = tuple(n + 1 for n in minimum)
    counts = tuple((m + depth + b - 1) // b for m, b in zip(maximum, owned, strict=True))
    ranks = {}
    for tile in itertools.product(*(range(c) for c in counts)):
        ranks.setdefault(sum(tile), []).append(tile)
    ordered = []
    for rank in sorted(ranks, reverse=negative == "reverse-rank-order"):
        cells = ranks[rank].copy()
        random.Random(seed + rank).shuffle(cells)
        ordered.extend(cells)
    reads = writes = late = 0
    for field, delta in zip(FIELDS, SUPPORT_DELTA, strict=True):
        shape = tuple(n + d for n, d in zip(minimum, delta, strict=True))
        size = math.prod(shape)
        memory = list(range(1, size + 1))  # Nonzero immutable bootstrap witness.
        if negative == "output-upload-or-clear-before-bootstrap":
            memory[:] = [0] * size

        def linear(p, shape=shape):
            return (p[0] * shape[1] + p[1]) * shape[2] + p[2]

        for tile in ordered:
            origin = tuple(t * b for t, b in zip(tile, owned, strict=True))
            first = max(1, *(o - m + 1 for o, m in zip(origin, maximum, strict=True)))
            last = min(depth, *(o + b - 1 for o, b in zip(origin, owned, strict=True)))
            if first > last:
                continue
            if first == 1:
                bounds = [range(max(0, o - 1), min(n, o + b)) for o, b, n in zip(origin, owned, shape, strict=True)]
                for p in itertools.product(*bounds):
                    i = linear(p)
                    if memory[i] != i + 1:
                        raise AssertionError((negative, "initial word overwritten", field, tile, p, memory[i], i + 1))
                    reads += 1
            else:
                # Actual late branch has no Initial calls; previous owned
                # envelope is globally invalid and low ghosts use trails.
                assert any(o - (first - 1) >= m for o, m in zip(origin, maximum, strict=True))
                late += 1
            shift = depth if negative == "reverse-final-shift" else 0 if negative == "unshifted-final-n0" else -depth
            bounds = [range(max(0, o + shift), min(n, o + b + shift)) for o, b, n in zip(origin, owned, shape, strict=True)]
            # An intentionally strong same-rank interleaving: this tile's
            # entire final scatter completes before the next tile bootstraps.
            for p in itertools.product(*bounds):
                i = linear(p)
                memory[i] = -(i + 1)
                writes += 1
        if negative is None and memory != [-(i + 1) for i in range(size)]:
            raise AssertionError("Final owned points were not written exactly once")
    return {"minimum": minimum, "owned": owned, "depth": depth, "seed": seed,
            "initial_reads": reads, "final_writes": writes, "late_field_tile_initializations_skipped": late}


def source_binding(source):
    text = (source / "src/asymmetric_native.cuh").read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != REVIEWED_CONSUMER_SHA256:
        raise ValueError("Consumer changed: repeat the startup/final/rank-join source review")
    start = text.index("FieldPacket ImportPacket(")
    end = text.index("__global__ void BeginEpoch", start)
    assert "Initial(" not in text[start:end]
    assert "if(first==1)" in text
    assert "if(n==Depth)" in text
    assert "if(parallel)for(int policy=0;policy<6;++policy)" in text
    assert "cudaStreamWaitEvent(stream,static_cast<cudaEvent_t>(w.policy_done_events[policy]),0)" in text
    assert "alias mode is not admitted yet" in text
    return {"consumer_sha256": digest,
            "scope": "Manual reviewed source binding: Initial calls are only startup branches; ImportPacket reads trails only; nDepth final stores; per-rank event joins; current Prepare rejects aliases. Not automatic control-flow extraction."}


def report(source):
    cases = [factorized_case(n, b, d) for n in ((27, 29, 31), (35, 49, 65), (64, 96, 128), (136, 136, 136), (512, 512, 512), (564, 564, 564), (640, 640, 640))
             for b in ((7, 15, 31), (15, 15, 15)) for d in (4, 16)]
    schedules = [token_schedule(owned=b, depth=d, seed=s) for b in ((7, 15, 31), (15, 15, 15)) for d in (4, 16) for s in (13, 37)]
    negatives = {}
    for negative in ("reverse-final-shift", "unshifted-final-n0", "output-upload-or-clear-before-bootstrap", "reverse-rank-order"):
        try:
            token_schedule(depth=4, negative=negative)
        except AssertionError as e:
            negatives[negative] = repr(e)
        else:
            raise AssertionError((negative, "unsafe alias schedule passed"))
    return {"scope": __doc__, "factorized_cases": cases, "all_three_dimensional_reader_writer_links": sum(f["initial_reader_to_final_writer_links"] for c in cases for f in c["field_cases"]),
            "trace_initial_read_coverage_checks": sum(trace_initial_coverage(b) for b in ((7, 15, 31), (15, 15, 15))),
            "token_schedules": schedules, "negative_controls": negatives, "source_binding": source_binding(source),
            "theorem": "For each physical word p, initial readers C_a are floor(p_a/B_a) or floor((p_a+1)/B_a). Final owner W_a=floor((p_a+T)/B_a) is >=everyC_a forT>=1. Distinct reader has strictlylower rank; equalrank impliessameCTA. Currentrankjoins/startupprogramorder retire all suchreads before finalscatter.",
            "requires": ["correspondingfieldaliases only, identicaldenseindexlayouts", "fullrankretirement or separatelyprovedequivalentfieldWARpaths", "bootstraploadscompletebeforeownfinalstores", "no outputinitialization/upload/clear aliases input", "all positive-version fieldproviders use trails/register/shared, never reread initialfields", "psi/profiles/material/source/clock/observation/trailbuffers remain disjoint from fields"],
            "limits": "Fieldmemory proof only; doesnotenablecurrentrejectedaliasABI, changeallocations/APIpreservation, prove ordinaryMstepD2samebuffer, or derivepsi/source/DFT/backendnumericalcorrectness."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = report(a.source)
    r["proof_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({"factorized_cases": len(r["factorized_cases"]), "reader_writer_links": r["all_three_dimensional_reader_writer_links"], "token_schedules": len(r["token_schedules"]), "negative_controls": len(r["negative_controls"])}))


if __name__ == "__main__":
    main()
