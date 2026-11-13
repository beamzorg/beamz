"""Actual-Yee four-barrier cube routing for T256/512/1024, independent of CUDA.

Each thread's fixed interleaved Z slots hold own old H/E values; all spatial
opposite-family neighbors are shared. This deliberately does not reuse or alter
the paired source/half-Y schedule oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_space_time_ir import FIELDS, yee_operator


def point(threads, tid, j, negative=None):
    if threads not in (256, 512, 1024) or not 0 <= tid < threads or not 0 <= j < 4096 // threads:
        raise ValueError("Exact cube fixed owner domain required")
    step = 1 if negative == "old-ZStep" else threads // 256
    base = tid // (512 if negative == "old-ZBase" else 256)
    return base + j * step, (tid // 16) % 16, tid % 16


def simulate(threads, version=1, negative=None):
    entries = [(tid, j, point(threads, tid, j, negative)) for tid in range(threads) for j in range(4096 // threads)]
    points = {p for _, _, p in entries}
    if len(points) != 4096 or any(not all(0 <= c < 16 for c in p) for p in points):
        raise AssertionError((negative, "incomplete/aliased cache owner map"))
    reads = {FIELDS.index(u.output.field): tuple(r for r in u.reads if r.field in FIELDS)
             for u in yee_operator(cpml_axes=(0, 1, 2)).updates if u.output.field in FIELDS}
    heads = (-version - 1) % 16

    def physical(p, n):
        return tuple(64 - n - 1 + (c + n + 1) % 16 for c in p)

    def token(f, n, p):
        return f, n, physical(p, n)

    def ghost(p):
        return heads in p

    private = {(f, p): token(f, version - 1, p) for p in points for f in range(6)}
    frame = {p: [private[f, p] for f in (3, 4, 5)] for p in points}
    checks = 0

    def phase(family):
        nonlocal checks
        order = entries.copy()
        random.Random(version + family).shuffle(order)
        for _, _, p in order:
            if ghost(p):
                continue
            for f in range(3 * family, 3 * family + 3):
                for r in reads[f]:
                    source = FIELDS.index(r.field)
                    q = tuple((a + b) % 16 for a, b in zip(p, r.offset, strict=True))
                    own = source // 3 == family
                    if own and (source != f or any(r.offset)):
                        raise AssertionError("Same-phase inputs are not own old component")
                    value = private[source, p] if own or not any(r.offset) else frame[q][source % 3]
                    n = version - 1 if own or source >= 3 else version
                    required = source, n, tuple(a + b for a, b in zip(physical(p, version), r.offset, strict=True))
                    checks += 1
                    if value != required:
                        raise AssertionError((negative, "wrong affine field read", value, required))
                private[f, p] = token(f, version, p)

    if negative == "early-H-publication":
        # A fast neighboring owner has overwritten old Ez before this pending
        # Hx(+Y) reader; omitted old-E retirement allows this interleaving.
        frame[0, 1, 0][2] = token(2, version, (0, 1, 0))
    phase(0)
    # Imports/R1 followed by complete H publication/R2.
    for p in points:
        if ghost(p):
            for f in range(6):
                private[f, p] = token(f, version, p)
        frame[p] = [private[f, p] for f in (0, 1, 2)]
    if negative == "early-E-publication":
        # Ex(0,0,0) requires shared Hz at(0,15,0); an early E frame writer
        # in that distinct XY owner must not run before current H retirement.
        frame[0, 15, 0][2] = token(5, version, (0, 15, 0))
    phase(1)
    # R3 followed by complete E publication/R4.
    for p in points:
        frame[p] = [private[f, p] for f in (3, 4, 5)]
        for f in range(6):
            if private[f, p] != token(f, version, p):
                raise AssertionError((negative, "wrong final field", f, p))
    return {"threads": threads, "slots": 4096 // threads, "ZStep": threads // 256,
            "field_scalar_lower_bound": 6 * (4096 // threads), "barriers": 4,
            "checked_affine_reads": checks}


def report():
    cases = [simulate(t, n) for t in (256, 512, 1024) for n in range(1, 33)]
    negatives = {}
    for name in ("old-ZStep", "old-ZBase", "early-H-publication", "early-E-publication"):
        try:
            simulate(512, 2, name)
        except AssertionError as e:
            negatives[name] = repr(e)
        else:
            raise AssertionError((name, "unsafe mapping passed"))
    return {"cases": cases, "reads": sum(c["checked_affine_reads"] for c in cases),
            "negative_controls": negatives, "scope": __doc__,
            "limits": "Actual canonical affine reader/owner proof under correct bootstrap/current ghost providers. No packed CPML/source/DFT/physical support arithmetic, CUDA convergence, register-fit or performance result."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = report()
    r["proof_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({"cases": len(r["cases"]), "reads": r["reads"], "negatives": len(r["negative_controls"])}))


if __name__ == "__main__":
    main()
