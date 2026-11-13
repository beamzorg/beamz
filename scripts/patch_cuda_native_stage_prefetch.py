"""Prefetch immutable low-X trails into disjoint shared stage before H work.

Only a source-order experiment: exact ImportPacket values, GPR ghost retirement,
all four phase barriers, fields, CPML and observation arithmetic stay intact.
Later async-copy lowering can use the same lifetime/publication contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_plane_retirement_binding import braced_body

CANONICAL_CELL = "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5"
LOOP = " for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n){\n"
CALL = "  StageRegisterLow<Depth,SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);\n"
B1 = "  __syncthreads(); // B1 retires all old-E frame readers and publishes low imports."
OLD = CALL + B1
SUPPORT = ((0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1))
HELPER_HASHES = {
    "FieldPacket ImportPacket(": "d535f67116a40ec7b96bb97847d891ec7791b73f6afd1172a685b4b3dc2b0e70",
    "void StageRegisterLow(": "0708d6adaa3c66dfeb46c36cfe76c2621db3cc708238f34bc1ceaea824349ea4",
    "void RetireRegisterLow(": "32fc22f678c3745f0cfe504a9d72e788d15b2bd4c64b3335606ff5bbb13b2f34",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def token(minimum, origin, n, dz, dy, field):
    point = (origin[0] - 1 + dz - n, origin[1] - 1 + dy - n, origin[2] - 1 - n)
    shape = tuple(m + d for m, d in zip(minimum, SUPPORT[field], strict=True))
    if all(0 <= q < s for q, s in zip(point, shape, strict=True)):
        return field, n, point
    return "positive-zero"


def prove_stage():
    cases = checks = cross_warp = 0
    for threads in (512, 1024):
        slots, zstep = 4096 // threads, threads // 512
        for minimum in [(5, 7, 9), (24, 25, 31), (64, 96, 128)]:
            for origin in [(0, 0, 0), (7, 15, 31), (21, 30, 62), (84, 105, 155)]:
                rng = random.Random(threads + sum(origin))
                stage = {}
                for n in range(1, 34):
                    # Previous consumers have retired by B2(n-1); the source
                    # places this overwrite after the stronger B4(n-1).
                    publishers = list(range(105))
                    rng.shuffle(publishers)
                    for i in publishers:
                        dz, dy = 1 + i // 15, 1 + i % 15
                        for field in range(6):
                            stage[field * 105 + i] = token(
                                minimum, origin, n, dz, dy, field
                            )
                    # H reads private own fields or frame[0:3*4096], whereas
                    # stage starts after that frame and the metadata. They
                    # have no overlapping shared-memory addresses.
                    frame_end = 3 * 4096
                    stage_begin = frame_end + 8 * 2 + 3
                    assert set(range(frame_end)).isdisjoint(
                        range(stage_begin, stage_begin + 630)
                    )
                    assert (stage_begin + 630) * 4 == 51748
                    # Existing B1 publishes every staged writer before any
                    # arbitrary cross-warp consumer begins.
                    readers = list(range(threads))
                    rng.shuffle(readers)
                    hz, hy, hx = (-n - 1) & 7, (-n - 1) & 15, (-n - 1) & 31
                    read_keys = set()
                    for tid in readers:
                        sy, sx = (tid // 32) & 15, tid & 31
                        if sx != hx or sy == hy:
                            continue
                        for j in range(slots):
                            zs = tid // 512 + j * zstep
                            if zs == hz:
                                continue
                            dz, dy = (zs - hz) & 7, (sy - hy) & 15
                            key = (dz - 1) * 15 + dy - 1
                            for field in range(6):
                                index = field * 105 + key
                                assert stage[index] == token(
                                    minimum, origin, n, dz, dy, field
                                )
                                assert index not in read_keys
                                read_keys.add(index)
                                cross_warp += key // 32 != tid // 32
                                checks += 1
                    assert read_keys == set(range(630))
                cases += 1
    # Counterexamples to moving staging without the publication/retirement
    # edges. Each is an allowed fast/slow-worker ordering in the broken model.
    counterexamples = []
    for threads in (512, 1024):
        n, key, field = 17, 0, 0
        hz, hy, hx = (-n - 1) & 7, (-n - 1) & 15, (-n - 1) & 31
        zs, sy = (hz + 1) & 7, (hy + 1) & 15
        reader = sy * 32 + hx + (512 * (zs % 2) if threads == 1024 else 0)
        assert reader // 32 != key // 32
        expected = token((64, 96, 128), (21, 30, 62), n, 1, 1, field)
        assert expected != "positive-zero"
        for kind, wrong_version in [
            ("missing_current_publication", n - 1),
            ("overwrite_before_previous_readers_retire", n + 1),
        ]:
            observed = token((64, 96, 128), (21, 30, 62), wrong_version, 1, 1, field)
            assert observed != expected
            counterexamples.append(
                {
                    "kind": kind,
                    "threads": threads,
                    "writer_thread": key,
                    "reader_thread": reader,
                    "stage_key": key,
                    "expected": expected,
                    "observed": observed,
                }
            )
    return {
        "mapping_cases": cases,
        "stage_token_checks": checks,
        "cross_warp_reads": cross_warp,
        "versions_per_case": 33,
        "negative_counterexamples": counterexamples,
        "scope": "Shared stage/version/point ownership only; unchanged H/E arithmetic and lower-rank readiness require actual source binding/GPU gates",
    }


def transform(consumer, cell):
    if digest(cell) != CANONICAL_CELL:
        raise ValueError("Canonical twelve-term cell source changed")
    text = consumer.decode()
    for name, expected in HELPER_HASHES.items():
        if digest(braced_body(text, name).encode()) != expected:
            raise ValueError(f"Unreviewed stage/import helper: {name}")
    if "BZ=7,BY=15,BX=31,CZ=8,CY=16,CX=32,Threads=512" not in text:
        raise ValueError("Stage proof requires the reviewed8x16x32 cache")
    for anchor in [
        LOOP,
        OLD,
        "float*low_stage=frame+3*PAD+8*sizeof(size_t)/sizeof(float)+3;",
        "RetireRegisterLow<T,Slots>(low_stage,h,e,n,sy,sx);",
        "__syncthreads(); // B2 publishes all current H values.",
        "__syncthreads(); // B4 publishes E and the next phase counter.",
    ]:
        if text.count(anchor) != 1:
            raise ValueError(f"Changed or ambiguous stage schedule anchor: {anchor!r}")
    required = [
        "if(zs!=hz&&sy!=hy&&sx!=hx)",
        "return frame[c*PAD+Slot(zs+dz,sy+ddy,sx+ddx)]",
        "cudaStreamWaitEvent(stream,static_cast<cudaEvent_t>(w.policy_done_events[policy]),0)",
    ]
    if any(anchor not in text for anchor in required):
        raise ValueError("Field/frame isolation or lower-rank join contract changed")
    changed = text.replace(OLD, B1, 1).replace(LOOP, LOOP + CALL, 1)
    inverse = changed.replace(LOOP + CALL, LOOP, 1).replace(B1, OLD, 1)
    if inverse.encode() != consumer:
        raise ValueError("Exact schedule inverse failed")
    if changed.count("__syncthreads()") != text.count("__syncthreads()"):
        raise ValueError("Prefetch may not change CTA barrier count")
    return changed.encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source, output = a.source.resolve(), a.output.resolve()
    canonical = Path(__file__).resolve().parents[1] / "cuda"
    if (
        output.exists()
        or output == canonical
        or source == output
        or source in output.parents
        or output in source.parents
    ):
        p.error("Output must be new, disjoint and noncanonical")
    consumer, cell = (
        source / "src/asymmetric_native.cuh",
        source / "src/asymmetric_native_cell_generated.cuh",
    )
    changed = transform(consumer.read_bytes(), cell.read_bytes())
    proof = {
        "parent_consumer_sha256": digest(consumer.read_bytes()),
        "consumer_sha256": digest(changed),
        "cell_sha256": digest(cell.read_bytes()),
        "inverse_source_binding": True,
        "stage_lifetime": prove_stage(),
    }
    shutil.copytree(source, output)
    (output / "src/asymmetric_native.cuh").write_bytes(changed)
    (output.parent / "stage-prefetch-proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
