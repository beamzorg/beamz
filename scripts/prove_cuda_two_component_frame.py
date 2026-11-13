"""Actual-trace token oracle for the two-component CUDA frame experiment.

This proves addresses, field versions and thread ownership, not FP arithmetic.
Each shared phase is separated at precisely the corresponding CTA barrier;
negative witnesses demonstrate why the retirement/snapshot edges are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prove_cuda_plane_retirement import dependencies

FIELDS = ("Hx", "Hy", "Hz", "Ex", "Ey", "Ez")
CACHE = (16, 16, 32)


def point(slot, n):
    low = -n - 1
    ghost = tuple(low & (d - 1) for d in CACHE)
    return tuple(100 + low + ((s - g) & (d - 1)) for s, g, d in zip(slot, ghost, CACHE, strict=True))


def token(field, n, slot):
    return (field, n, point(slot, n))


def run(threads, steps, negative=None):
    deps = dependencies()
    slots = [(z, y, x) for z in range(16) for y in range(16) for x in range(32)]
    h = {s: {f: token(f, 0, s) for f in FIELDS[:3]} for s in slots}
    ey = {s: token("Ey", 0, s) for s in slots}
    frame = {s: {0: token("Ex", 0, s), 1: token("Ez", 0, s)} for s in slots}
    eycross = {
        (z // 8, y, x): ey[z, y, x]
        for z in (0, 8)
        for y in range(16)
        for x in range(32)
    }
    checks = shuffles = crossz = crossy = 0
    for n in range(1, steps + 1):
        ghost = ((-n - 1) & 15, (-n - 1) & 15, (-n - 1) & 31)
        owned = [s for s in slots if all(a != g for a, g in zip(s, ghost, strict=True))]
        for s in owned:
            z, y, x = s
            for f in FIELDS[:3]:
                for read in deps[f]:
                    source = tuple(
                        (a + d) & (size - 1)
                        for a, d, size in zip(s, read.offset, CACHE, strict=True)
                    )
                    if read.field.startswith("H"):
                        got = h[s][read.field]
                    elif read.field == "Ey":
                        if read.offset[0] and threads == 1024 and z % 8 == 7:
                            got = eycross[(z // 8) ^ 1, y, x]
                            crossz += 1
                        else:
                            got = ey[source]
                        if read.offset[2]:
                            assert source[0:2] == s[0:2]
                            shuffles += 1
                    else:
                        got = frame[source][0 if read.field == "Ex" else 1]
                    wanted = (
                        read.field,
                        n - 1,
                        tuple(p + d for p, d in zip(point(s, n), read.offset, strict=True)),
                    )
                    assert got == wanted, ("H", threads, n, s, f, read, got, wanted)
                    checks += 1
                h[s][f] = token(f, n, s)
        # Current ghost imports. No shared old-E overwrite before T1.
        for s in slots:
            if s not in h:
                raise AssertionError("Missing fixed slot")
            if any(a == g for a, g in zip(s, ghost, strict=True)):
                h[s] = {f: token(f, n, s) for f in FIELDS[:3]}
                ey[s] = token("Ey", n, s)
        # T1 publishes low-X packets and retires all H readers; T2 then publishes
        # the final Hx/Hy chunk-boundary planes, including low-X imports.
        hcross = {
            (f, z // 8, y, x): h[z, y, x][f]
            for f in ("Hx", "Hy")
            for z in (7, 15)
            for y in range(16)
            for x in range(32)
        }
        count = 8 if threads == 1024 else 16
        for j in range(count):
            zs = (j, j + 8) if threads == 1024 else (j,)
            group = [(z, y, x) for z in zs for y in range(16) for x in range(32)]
            saved = {s: frame[s].copy() for s in group}
            for s in group:
                if negative != "missing_Pj_publication":
                    frame[s] = {0: h[s]["Hx"], 1: h[s]["Hz"]}
            # Pj publishes every current Hx/Hz plane before any XY read.
            next_e = {}
            for s in group:
                z, y, x = s
                if any(a == g for a, g in zip(s, ghost, strict=True)):
                    next_e[s] = {0: token("Ex", n, s), 1: token("Ez", n, s)}
                    continue
                for f in FIELDS[3:]:
                    for read in deps[f]:
                        source = tuple(
                            (a + d) & (size - 1)
                            for a, d, size in zip(s, read.offset, CACHE, strict=True)
                        )
                        if read.field.startswith("E"):
                            got = (
                                ey[s]
                                if read.field == "Ey"
                                else saved[s][0 if read.field == "Ex" else 1]
                            )
                        elif not any(read.offset):
                            got = h[s][read.field]
                        elif read.offset[0]:
                            if threads == 1024 and z % 8 == 0:
                                got = hcross[
                                    read.field,
                                    ((z // 8) ^ 1)
                                    if negative != "wrong_H_chunk"
                                    else z // 8,
                                    y,
                                    x,
                                ]
                                crossz += 1
                            else:
                                got = h[source][read.field]
                        elif read.field == "Hy":
                            got = h[source]["Hy"]
                            assert source[0:2] == s[0:2]
                            shuffles += 1
                        else:
                            if (
                                negative == "missing_Qj_retirement"
                                and read.offset[1]
                                and all(a != g for a, g in zip(source, ghost, strict=True))
                            ):
                                # The producer warp has completed E and overwrites its shared plane
                                # while this slower neighboring warp still reads current H.
                                frame[source][0 if read.field == "Hx" else 1] = token(
                                    "Ex" if read.field == "Hx" else "Ez", n, source
                                )
                            got = frame[source][0 if read.field == "Hx" else 1]
                            if read.offset[1]:
                                crossy += 1
                        wanted = (
                            read.field,
                            n if read.field.startswith("H") else n - 1,
                            tuple(p + d for p, d in zip(point(s, n), read.offset, strict=True)),
                        )
                        assert got == wanted, ("E", threads, n, s, f, read, got, wanted)
                        checks += 1
                next_e[s] = {0: token("Ex", n, s), 1: token("Ez", n, s)}
                ey[s] = token("Ey", n, s)
            # Qj retires all current H XY readers. Both Z-chunks progress together;
            # no other j's E read uses this shared plane (Z reads are private/snapshot).
            for s in group:
                frame[s] = next_e[s]
        if negative != "stale_Ey_chunk_snapshot":
            eycross = {
                (z // 8, y, x): ey[z, y, x]
                for z in (0, 8)
                for y in range(16)
                for x in range(32)
            }
        # T3 publishes full E frame and Ey boundary snapshots for next H.
        for s in slots:
            assert ey[s] == token("Ey", n, s)
            for k, f in enumerate(("Ex", "Ez")):
                assert frame[s][k] == token(f, n, s)
    return {
        "threads": threads,
        "steps": steps,
        "operator_reads": checks,
        "warp_shuffle_reads": shuffles,
        "chunk_boundary_reads": crossz,
        "shared_Y_reads": crossy,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cases = [run(t, 33) for t in (512, 1024)]
    negative = []
    # Replay the same actual-trace checker with deliberately broken storage
    # operations. Each must fail at a real field read; numerical zero does not
    # conceal the wrong version or physical coordinate in this token oracle.
    for kind in (
        "stale_Ey_chunk_snapshot",
        "wrong_H_chunk",
        "missing_Qj_retirement",
        "missing_Pj_publication",
    ):
        try:
            run(1024, 2, negative=kind)
        except AssertionError as exc:
            negative.append(
                {"kind": kind, "actual_trace_counterexample": repr(exc.args)}
            )
        else:
            raise AssertionError(f"Broken schedule was not rejected:{kind}")
    body = Path(__file__).with_name("cuda_native_two_component_frame.cuh")
    j = {
        "pass": True,
        "cases": cases,
        "body_sha256": hashlib.sha256(body.read_bytes()).hexdigest(),
        "negative_witnesses": negative,
        "scope": "All actual canonical Yee field reads and CPML-local dependency invariants; storage/ownership only, no floating-point or native bounds equivalence claim",
    }
    a.output.write_text(json.dumps(j, indent=2) + "\n")
    print(json.dumps(j))


if __name__ == "__main__":
    main()
