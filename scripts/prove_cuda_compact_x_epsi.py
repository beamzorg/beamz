"""Guarded lifetime/address proof for a compact X-only shared E-psi bank.

Logical state tokens, not a CUDA/resource/performance claim. This retains the
old cache8x16x32 even/odd Z mapping, existing E phase retirement barriers and
first-owned/global-bootstrap/high-face publication protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_space_time_ir import yee_operator
from scripts.cuda_trace_guarded_storage import (
    FIXED_E_COLORS,
    OwnedRegion,
    terms_from_ir,
)

OLD_BYTES, WORDS, TOTAL_BYTES = 51748, 3072, 64036
CACHE, OWNED = (8, 16, 32), (7, 15, 31)


def index(color, zs, ys, packed, negative=None):
    if not (color in (0, 1) and zs in range(8) and ys in range(16) and packed in range(24)):
        raise ValueError("Active canonical X packed coordinate and cache slots required")
    if negative == "wrong-color":
        color = 0
    ystride = 8 if negative == "wrong-y-stride" else 16
    columns = 11 if negative == "wrong-packed-modulus" else 12
    return ((color * 8 + zs) * ystride + ys) * 12 + packed % columns


def admission(nx, depth):
    if type(nx) is not int or nx < 64 or depth not in (4, 16):
        raise ValueError("Native compact class requires physical Nx>=64 and depth4/16")
    separation = nx - 11  # E target extent Nx+1, high packed12 starts Nx-11.
    visited_span = OWNED[2] + depth - 2
    assert separation > visited_span
    return {"nx": nx, "depth": depth, "equal_color_column_cross_side_separation": separation,
            "maximum_trajectory_coordinate_span": visited_span,
            "cross_side_same_word_reuse_within_macro": False}


def lifecycle(nx=64, depth=16, origin=(48, 48, 31), *, negative=None, admit=True):
    if admit:
        admission(nx, depth)
    minimum = (96, 96, nx)
    region = OwnedRegion(minimum, origin, OWNED, depth)
    terms, _ = terms_from_ir(yee_operator(cpml_axes=(0, 1, 2)), region)
    terms = [term for term in terms if term["axis"] == 2]
    assert {term["term"] for term in terms} == {3, 4}
    first = max(1, *(o - n for o, n in zip(origin, minimum, strict=True)))
    last = min(depth, *(o + b - 1 for o, b in zip(origin, OWNED, strict=True)))
    bank, final, published = {}, {}, {}
    accesses, warm, same_word_cross_thread_reuses = 0, 0, 0
    cross_side_reuses, reuse_witnesses, cross_side_witnesses = 0, [], []
    cross_thread_reuses = 0
    prior_owner = {}
    for n in range(first, last + 1):
        owners = {}
        for term in terms:
            color = FIXED_E_COLORS[term["term"]]
            for dz in range(1, 8):
                for dy in range(1, 16):
                    z, y = origin[0] - n + dz - 1, origin[1] - n + dy - 1
                    if not (0 <= z < term["shape"][0] and 0 <= y < term["shape"][1]):
                        continue
                    for dx in range(1, 32):
                        x = origin[2] - n + dx - 1
                        if not 0 <= x < term["shape"][2] or not (x < 12 or x >= nx - 11):
                            continue
                        packed = x if x < 12 else 23 + x - nx
                        zs, ys, xs = (z - origin[0]) % 8, (y - origin[1]) % 16, (x - origin[2]) % 32
                        tid, slot = (zs & 1) * 512 + ys * 32 + xs, zs >> 1
                        address = index(color, zs, ys, packed, negative)
                        logical = term["term"], z, y, x
                        if address in owners and owners[address][0] != logical:
                            raise AssertionError((negative, "co-owned alias", n, address, owners[address], logical))
                        owners[address] = logical, tid, slot
                        if address in prior_owner and prior_owner[address] != (logical, tid, slot):
                            same_word_cross_thread_reuses += 1
                            cross_thread_reuses += prior_owner[address][1] != tid
                            old = prior_owner[address][0]
                            cross_side = (old[3] < 12) != (x < 12)
                            cross_side_reuses += cross_side
                            if cross_side and len(cross_side_witnesses) < 2:
                                cross_side_witnesses.append({"address": address, "previous": bank[address], "current": [logical, n], "previous_owner": prior_owner[address][1:], "current_owner": [tid, slot]})
                            if len(reuse_witnesses) < 2:
                                reuse_witnesses.append({"address": address, "previous": bank[address], "current": [logical, n], "previous_owner": prior_owner[address][1:], "current_owner": [tid, slot]})
                        fresh = n == first or dz == 1 or dy == 1 or dx == 1
                        if negative == "omit-first-owned":
                            fresh = False
                        elif negative == "omit-turnover-reload":
                            fresh = n == first
                        expected = logical, n - 1
                        actual = expected if fresh else bank.get(address)
                        if actual != expected:
                            raise AssertionError((negative, "stale bank", n, address, actual, expected))
                        bank[address] = logical, n
                        final[logical] = logical, n
                        if negative != "omit-flush" and (n == depth or dz == 7 or dy == 15 or dx == 31):
                            published[logical] = logical, n
                        accesses += 1
                        warm += not fresh
        # Existing B3/B4 retires EVERY E load/store before the next version;
        # this is the proof edge permitting a different TID to reuse a word.
        prior_owner.update(owners)
    if any(published.get(p) != value for p, value in final.items()):
        raise AssertionError((negative, "missing outgoing/final publication"))
    return {"nx": nx, "depth": depth, "origin": origin, "window": [first, last],
            "active_reads": accesses, "warm_reads": warm,
            "word_owner_turnovers_between_retired_versions": same_word_cross_thread_reuses,
            "cross_side_word_reuses": cross_side_reuses, "cross_thread_word_reuses": cross_thread_reuses, "reuse_witnesses": reuse_witnesses,
            "cross_side_witnesses": cross_side_witnesses,
            "visited_words": len(bank), "required_new_barriers": 0}


def report():
    words = {index(c, z, y, k) for c in range(2) for z in range(8) for y in range(16) for k in range(12)}
    assert words == set(range(WORDS)) and OLD_BYTES + WORDS * 4 == TOTAL_BYTES < 65536
    cases = [lifecycle(nx, d, (48, 48, ox))
             for nx in (64, 65, 96, 136) for d in (4, 16)
             for ox in sorted({0, 31, ((nx + d) // 31) * 31})]
    negatives = {}
    for name in ("wrong-color", "wrong-y-stride", "wrong-packed-modulus", "omit-first-owned", "omit-turnover-reload", "omit-flush"):
        try:
            lifecycle(origin=(48, 48, 0) if name == "wrong-packed-modulus" else (48, 48, 31), negative=name)
        except AssertionError as e:
            negatives[name] = repr(e)
        else:
            raise AssertionError((name, "negative unexpectedly passed"))
    try:
        lifecycle(32, 4, (48, 48, 0), admit=False)
    except AssertionError as e:
        negatives["insufficient-slab-separation"] = repr(e)
    else:
        raise AssertionError("Nx32 must expose same-version low/high column alias")
    # An extended-depth diagnostic allows cross-side reuse only after many
    # retired versions; it is outside the proposed4/16 native admission.
    extended = lifecycle(64, 64, (80, 80, 62), admit=False)
    return {"bank_words": WORDS, "bytes_halfopen": [OLD_BYTES, TOTAL_BYTES], "native_admission": "flags&14==8, Nx>=64, depth4/16; existing NoIO/valid/LocalActive/disjoint guards",
            "cases": cases, "negatives": negatives, "extended_depth_diagnostic": extended,
            "cross_side_argument": "equal packed%12 points differ by Nx-11; owned trajectory span31+Depth-2 is smaller for admitted depths",
            "synchronization": "Within-version map injective; inter-version owner turnover follows existing E B3/B4 retirement/publication. Preserve an equivalent inter-version retirement edge if extending the admitted depth.",
            "limits": "Predicted capacity only; no CUDA/resource/carveout/performance result. Global bootstrap and inter-CTA outgoing-psi ordering inherited from original schedule proof."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = report()
    r["proof_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({"cases": len(r["cases"]), "reads": sum(c["active_reads"] for c in r["cases"]), "warm": sum(c["warm_reads"] for c in r["cases"]), "words": WORDS, "bytes": TOTAL_BYTES}))


if __name__ == "__main__":
    main()
