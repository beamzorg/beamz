"""Stationary E-psi state isomorphism from interleaved to contiguous Z owners.

The physical activation/first-owned/high-face proof is geometry dependent, not
dependent on which fixed thread owns the cache coordinate. This pass proves the
full address relabeling and retains the original independently tested lifecycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prove_cuda_shared_epsi_bank import index, lifecycle, owner


def contiguous_owner(p):
    z, y, x = p
    if not (z in range(8) and y in range(16) and x in range(32)):
        raise ValueError("Canonical cache8x16x32 coordinate required")
    return (z // 4) * 512 + y * 32 + x, z % 4


def point(tid, slot):
    if tid not in range(1024) or slot not in range(4):
        raise ValueError("Contiguous1024-thread/four-slot class required")
    return (tid // 512) * 4 + slot, (tid // 32) % 16, tid % 32


def prove_permutation(negative=None):
    addresses = {}
    relabel = {}
    for color in range(2):
        for z in range(8):
            for y in range(16):
                for x in range(32):
                    p = z, y, x
                    old_tid, old_slot = owner(p)
                    tid, slot = contiguous_owner(p)
                    assert point(tid, slot) == p
                    if negative == "old-interleaved-slot":
                        slot = z // 2
                    if negative == "old-interleaved-thread":
                        tid = old_tid
                    before, after = index(color, old_slot, old_tid), index(color, slot, tid)
                    if after in addresses:
                        raise AssertionError((negative, "aliased new storage", addresses[after], p, after))
                    addresses[after] = p
                    relabel[before] = after
    assert set(relabel) == set(relabel.values()) == set(range(8192))
    return {"mapped_words": len(relabel), "new_indices_minmax": [min(addresses), max(addresses)],
            "old_owner": "((zs&1)*512+ys*32+xs,zs//2)",
            "new_owner": "((zs//4)*512+ys*32+xs,zs%4)",
            "stationarity": "For physical p while owned, cache coordinates=(p-origin)%cache stay fixed; therefore both ownership labels stay fixed. First-owned and high/final global events are unchanged."}


def report():
    negatives = {}
    for name in ("old-interleaved-slot", "old-interleaved-thread"):
        try:
            prove_permutation(name)
        except AssertionError as e:
            negatives[name] = repr(e)
        else:
            raise AssertionError("Wrong mixed ownership passed")
    cases = [lifecycle(a, d, high=h, clipped=c) for a in range(3) for d in (4, 16)
             for h, c in ((False, False), (True, False), (False, True))]
    return {"mapping": prove_permutation(), "negative_controls": negatives,
            "inherited_geometry_lifecycle_cases": cases,
            "inherited_oracle_sha256": hashlib.sha256(Path(__file__).with_name("prove_cuda_shared_epsi_bank.py").read_bytes()).hexdigest(),
            "scope": __doc__, "requires": ["same actual one-axis colors/Valid/LocalActive", "global boot/windowfirst/relative1 reload", "highunion or depth final flush", "separate mutable psi arrays and profile ranges", "correct paired field ghost/provider schedule"],
            "limits": "Ownership/address isomorphism plus inherited logical lifecycle, not an extracted CUDA execution proof or resource/performance claim."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = report()
    r["proof_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({"words": r["mapping"]["mapped_words"], "lifecycle_cases": len(r["inherited_geometry_lifecycle_cases"]), "negatives": len(r["negative_controls"])}))


if __name__ == "__main__":
    main()
