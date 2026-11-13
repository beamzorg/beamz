"""Exclusive-owner SoA shared E-psi bank: address and lifecycle proof.

Logical 32-bit float words, not emitted instruction widths/performance. The old
NoIO1024 point mapping owns even/odd Z planes, four slots/thread. Existing
one-axis guard coloring, global boot/flush and H-global fallback remain required.
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

THREADS, SLOTS, COLORS = 1024, 4, 2
OLD_BYTES, BANK_WORDS = 51748, 8192
TOTAL_BYTES = OLD_BYTES + 4 * BANK_WORDS
CACHE, OWNED = (8, 16, 32), (7, 15, 31)


def index(color, slot, tid, negative=None):
    if not (color in range(2) and slot in range(4) and tid in range(1024)):
        raise ValueError("Canonical color/slot/thread domain required")
    if negative == "wrong-tid":
        tid &= 511
    if negative == "wrong-color":
        color = 0
    stride = 512 if negative == "wrong-stride" else 1024
    return (color * SLOTS + slot) * stride + tid


def point(tid, slot):
    return tid // 512 + 2 * slot, (tid // 32) & 15, tid & 31


def owner(cache_point):
    z, y, x = cache_point
    return (z & 1) * 512 + y * 32 + x, z >> 1


def collision(negative):
    seen = {}
    for c in range(2):
        for slot in range(4):
            for tid in range(1024):
                address = index(c, slot, tid, negative)
                current = c, slot, tid
                if address in seen:
                    a = seen[address]
                    return {
                        "negative": negative,
                        "word": address,
                        "first": a,
                        "second": current,
                        "cache_points": [point(a[2], a[1]), point(tid, slot)],
                    }
                seen[address] = current
    raise AssertionError("Negative index unexpectedly injective")


def guarded_collision(negative):
    """Both aliased words can carry warm active states in a one-axis region."""
    witness = collision(negative)
    origin, n = (17, 48, 48), 8 if negative == "wrong-stride" else 9
    region = OwnedRegion((96, 96, 96), origin, OWNED, 16)
    terms, _ = terms_from_ir(yee_operator(cpml_axes=(0, 1, 2)), region)
    logical = []
    for color, slot, tid in (witness["first"], witness["second"]):
        cache = point(tid, slot)
        rel = tuple((p + n + 1) % c for p, c in zip(cache, CACHE, strict=True))
        physical = tuple(o - n - 1 + r for o, r in zip(origin, rel, strict=True))
        # Z-only active terms1/2 have colors0/1; all relative coordinates>1
        # means warm values genuinely need the previous shared state.
        term = terms[1 if color == 0 else 2]
        assert FIXED_E_COLORS[term["term"]] == color and term["axis"] == 0
        assert all(1 < r <= b for r, b in zip(rel, OWNED, strict=True))
        assert all(0 <= p < size for p, size in zip(physical, term["shape"], strict=True))
        assert physical[0] < 12
        logical.append({"term": term["term"], "point": physical, "version": n, "warm": True})
    return {**witness, "guarded_states": logical, "region": {"origin": origin, "depth": 16, "axis": 0}}


def prove_mapping():
    seen = set()
    for c in range(2):
        for slot in range(4):
            for tid in range(1024):
                i = index(c, slot, tid)
                assert i not in seen and 0 <= i < 8192
                seen.add(i)
                assert i % 1024 == tid and (i // 1024) % 4 == slot and i // 4096 == c
                assert owner(point(tid, slot)) == (tid, slot)
                assert OLD_BYTES <= OLD_BYTES + 4 * i <= TOTAL_BYTES - 4
    warps = 0
    for c in range(2):
        for slot in range(4):
            for warp in range(32):
                indices = [index(c, slot, 32 * warp + lane) for lane in range(32)]
                assert indices == list(range(indices[0], indices[0] + 32))
                # The bank begins at word offset9mod32; shared-bank permutation
                # still has one distinct bank per32-bit lane access.
                assert len({(OLD_BYTES // 4 + i) % 32 for i in indices}) == 32
                warps += 1
    regions = {
        "field_frame": [0, 49152],
        "bases": [49152, 49216],
        "phase_window": [49216, 49228],
        "low_stage": [49228, 51748],
        "Epsi_bank": [51748, 84516],
    }
    assert all(
        a[1] <= b[0] for a, b in zip(regions.values(), list(regions.values())[1:], strict=False)
    )
    return {
        "distinct_words": len(seen),
        "warp_access_groups": warps,
        "regions_bytes_halfopen": regions,
        "bank_base_mod128": OLD_BYTES % 128,
        "total_dynamic_shared_bytes": TOTAL_BYTES,
        "negative_indices": {
            name: guarded_collision(name)
            for name in ("wrong-tid", "wrong-stride", "wrong-color")
        },
        "wrong_base_witness": {
            "incorrect_base": OLD_BYTES - 4,
            "overwrites": "last old low-stage float",
        },
    }


def lifecycle(axis, depth=16, *, high=False, clipped=False, negative=None):
    minimum = (96, 96, 96)
    origin = [48, 48, 48]
    origin[axis] = 98 if high else 0 if clipped else 12 if depth == 4 else 18
    region = OwnedRegion(minimum, tuple(origin), OWNED, depth)
    terms, _ = terms_from_ir(yee_operator(cpml_axes=(0, 1, 2)), region)
    terms = [term for term in terms if term["axis"] == axis]
    first = max(1, *(o - n for o, n in zip(origin, minimum, strict=True)))
    last = min(depth, *(o + b - 1 for o, b in zip(origin, OWNED, strict=True)))
    bank, private, published, final = {}, {}, {}, {}
    reads, warm, boots, stores, point_identity = 0, 0, 0, 0, {}
    for n in range(first, last + 1):
        for dz in range(1, 8):
            for dy in range(1, 16):
                for dx in range(1, 32):
                    rel = dz, dy, dx
                    physical = tuple(
                        o - n + r - 1 for o, r in zip(origin, rel, strict=True)
                    )
                    cache = tuple(
                        (p - o) % c
                        for p, o, c in zip(physical, origin, CACHE, strict=True)
                    )
                    tid, slot = owner(cache)
                    for term in terms:
                        if not all(
                            0 <= p < limit
                            for p, limit in zip(physical, term["shape"], strict=True)
                        ):
                            continue
                        coordinate = physical[axis]
                        if not (
                            coordinate < 12 or coordinate >= term["shape"][axis] - 12
                        ):
                            continue
                        logical = term["term"], physical
                        identity = tid, slot, FIXED_E_COLORS[term["term"]]
                        if logical in point_identity:
                            assert point_identity[logical] == identity
                        point_identity[logical] = identity
                        c = identity[2]
                        address = index(c, slot, tid)
                        fresh = n == first or any(r == 1 for r in rel)
                        if negative == "omit-reload":
                            fresh = False
                        expected = term["term"], physical, n - 1
                        got_shared = expected if fresh else bank.get(address)
                        got_private = expected if fresh else private.get(identity)
                        if got_shared != expected or got_private != expected:
                            raise AssertionError(
                                (
                                    negative,
                                    "wrong prior state",
                                    n,
                                    logical,
                                    identity,
                                    got_shared,
                                    expected,
                                )
                            )
                        reads += 1
                        warm += not fresh
                        boots += fresh
                        value = term["term"], physical, n
                        bank[address] = value
                        private[identity] = value
                        flush = n == depth or any(
                            r == b for r, b in zip(rel, OWNED, strict=True)
                        )
                        if flush and negative != "omit-flush":
                            published[logical] = value
                            stores += 1
                        final[logical] = value
    if any(published.get(logical) != value for logical, value in final.items()):
        raise AssertionError((negative, "last owner did not publish recurrence"))
    return {
        "axis": axis,
        "depth": depth,
        "origin": origin,
        "window": [first, last],
        "active_reads": reads,
        "warm_reads": warm,
        "global_boots": boots,
        "global_flushes": stores,
        "fixed_owner_points": len(point_identity),
        "new_barriers": 0,
    }


def report():
    result = {
        "scope": __doc__,
        "mapping": prove_mapping(),
        "lifecycle_cases": [
            lifecycle(a, d, high=h, clipped=c)
            for a in range(3)
            for d in (4, 16)
            for h, c in ((False, False), (True, False), (False, True))
        ],
        "negative_lifetimes": {},
        "requires": [
            "one-hot axis mask only; multi-axis keeps global fallback",
            "per-component Valid and LocalActive before any bank access",
            "first-owned reload/global-high-face-or-final flush unchanged",
            "H psi remains global",
            "no cross-thread shared bank access; fixed tid/si/c index only",
        ],
        "limits": "No performance/register-fit claim. Volatile is not synchronization and does not guarantee instruction/access count; inspect emitted SASS.",
    }
    for name in ("omit-reload", "omit-flush"):
        try:
            lifecycle(0, negative=name)
        except AssertionError as error:
            result["negative_lifetimes"][name] = repr(error)
        else:
            raise AssertionError("Unsafe lifecycle passed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = report()
    result["oracle_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "mapping_words": result["mapping"]["distinct_words"],
                "lifecycle_cases": len(result["lifecycle_cases"]),
                "active_reads": sum(
                    case["active_reads"] for case in result["lifecycle_cases"]
                ),
                "warm_reads": sum(
                    case["warm_reads"] for case in result["lifecycle_cases"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
