"""Packed CPML local-recurrence ordering proof for moving-box ownership.

No GPU/float arithmetic: each packed word carries its physical point and latest
global timestep. H/E and derivative-term banks are separate. Owners are ordered
by rank; each owner processes its point's versions in increasing time.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict

try:
    from scripts.prove_cuda_systolic_schedule import component_shapes
except ModuleNotFoundError:
    from prove_cuda_systolic_schedule import component_shapes


AXES = (1, 0, 0, 2, 2, 1)


def packed_records(shape, thickness, *, padded=False, alias_faces=False, alias_phases=False):
    shapes = component_shapes(shape)
    records, physical_by_word = [], {}
    for family in "HE":
        for term, axis in enumerate(AXES):
            dims = shapes[family + "xyz"[term // 2]]
            assert dims[axis] >= 2 * thickness
            ranges = [range(n) for n in dims]
            ranges[axis] = tuple(range(thickness)) + tuple(range(dims[axis] - thickness, dims[axis]))
            packed_dims = list(dims)
            packed_dims[axis] = 2 * thickness
            row = packed_dims[2] + (3 if padded else 0)
            plane = row * (packed_dims[1] + (2 if padded else 0))
            for p in itertools.product(*ranges):
                q = list(p)
                if p[axis] >= dims[axis] - thickness:
                    q[axis] = (0 if alias_faces else thickness) + p[axis] - (dims[axis] - thickness)
                index = q[0] * plane + q[1] * row + q[2]
                word = ("H" if alias_phases else family), term, index
                physical = family, term, p
                assert word not in physical_by_word, ("Packed physical alias", word,
                                                      physical_by_word.get(word), physical)
                physical_by_word[word] = physical
                records.append((family, term, p, word))
    expected = sum(2 * thickness * math.prod(shapes[f + "xyz"[t // 2]][a]
                                            for a in range(3) if a != axis)
                   for f in "HE" for t, axis in enumerate(AXES))
    assert len(records) == expected
    return records


def prove(shape=(7, 9, 11), tile=(3, 4, 5), thickness=2, chunks=(2, 3), *,
          padded=False, reverse_ranks=False, reverse_time=False,
          alias_faces=False, alias_phases=False):
    records = packed_records(shape, thickness, padded=padded,
                             alias_faces=alias_faces, alias_phases=alias_phases)
    bank = [0] * len(records)
    previous_owner = [None] * len(records)
    reads = writes = transitions = 0
    base = 0
    for steps in chunks:
        events = defaultdict(list)
        for i, (family, _, p, _) in enumerate(records):
            previous_owner[i] = None
            rank_owner = {}
            for n in range(1, steps + 1):
                owner = tuple((a + n) // b for a, b in zip(p, tile, strict=True))
                rank = sum(owner)
                if rank in rank_owner:
                    assert rank_owner[rank] == owner, "Distinct same-rank owners share one psi word"
                rank_owner[rank] = owner
                events[owner, n, family].append(i)
        order = sorted(events, key=lambda e: ((-1 if reverse_ranks else 1) * sum(e[0]),
                                              e[0], -e[1] if reverse_time else e[1], e[2] == "E"))
        for owner, n, family in order:
            for i in events[owner, n, family]:
                assert bank[i] == base + n - 1, ("Wrong old psi clock", records[i], owner, n, bank[i], base)
                old_owner = previous_owner[i]
                if old_owner is not None and old_owner != owner:
                    assert all(a <= b for a, b in zip(old_owner, owner, strict=True))
                    assert sum(old_owner) < sum(owner)
                    transitions += 1
                # New psi and the corrected derivative/field consume this
                # register value now; no later observer rereads an old psi.
                next_clock = base + n
                bank[i] = next_clock
                assert bank[i] == next_clock
                previous_owner[i] = owner
                reads += 1
                writes += 1
        base += steps
        assert all(clock == base for clock in bank)
    assert reads == writes == len(records) * sum(chunks)
    return {"shape": shape, "tile": tile, "cpml_thickness": thickness,
            "chunks": chunks, "padded_pitches": padded,
            "distinct_packed_words": len(records), "read_then_write_events": writes,
            "owner_rank_transitions": transitions, "same_rank_conflicts": 0,
            "field_aux_payload_bytes_fp32": 8 * writes, "result": "pass"}


def main():
    cases = [prove(padded=p) for p in (False, True)]
    cases += [prove((25, 27, 29), (7, 7, 15), thickness=12, chunks=chunks)
              for chunks in ((2,), (2, 3))]
    negatives = []
    for label, kwargs in (("reverse box ranks", {"reverse_ranks": True}),
                          ("reverse local versions", {"reverse_time": True}),
                          ("overlapping packed low/high faces", {"alias_faces": True}),
                          ("aliased H/E packed banks", {"alias_phases": True})):
        try:
            prove(**kwargs)
        except AssertionError:
            negatives.append(label)
        else:
            raise AssertionError(f"Negative control escaped: {label}")
    print(json.dumps({"scope": __doc__, "cases": cases,
                      "negative_controls_rejected": negatives,
                      "conditions": ["mutable ordinary psi loads/stores, not read-only __ldg",
                                     "CTA phase barriers order global psi between threads",
                                     "rank kernel boundaries order owners",
                                     "new psi is consumed immediately by its own field update",
                                     "no historical or spatial-neighbor psi observers"]}, indent=2))


if __name__ == "__main__":
    main()
