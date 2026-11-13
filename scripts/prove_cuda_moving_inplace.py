"""Initial/final field-bank alias proof for owned moving-box execution.

Assumes injective physical field storage, rank-ordered launches, all version-zero
reads retired before a box's final scatter, and no external/preserved observers.
It proves initial-state anti-dependencies, not floating arithmetic/CUDA ordering.
"""

from __future__ import annotations

import itertools
import json

try:
    from scripts.prove_cuda_systolic_schedule import component_shapes
except ModuleNotFoundError:
    from prove_cuda_systolic_schedule import component_shapes


def box_positions(low, widths):
    return itertools.product(*(range(a, a + b) for a, b in zip(low, widths, strict=True)))


def prove(shape=(5, 6, 7), tile=(2, 3, 4), steps=4, *, initial_low_halo=1, wrong_final_shift=False):
    if steps < 1:
        raise ValueError("At least one timestep is required")
    fields = tuple(f + c for f in "HE" for c in "xyz")
    shapes = component_shapes(shape)
    grid = tuple((n + steps + b) // b for n, b in zip(shape, tile, strict=True))
    boxes = list(itertools.product(*(range(n) for n in grid)))
    initial, final = {}, {}
    shift = steps - int(wrong_final_shift)

    def valid(field, p):
        return all(0 <= a < b for a, b in zip(p, shapes[field], strict=True))

    for box in boxes:
        origin = tuple(a * b for a, b in zip(box, tile, strict=True))
        low = tuple(a - initial_low_halo for a in origin)
        width = tuple(a + initial_low_halo for a in tile)
        initial[box] = {(field, p) for p in box_positions(low, width) for field in fields if valid(field, p)}
        low_final = tuple(a - shift for a in origin)
        final[box] = {(field, p) for p in box_positions(low_final, tile) for field in fields if valid(field, p)}
    pairs = 0
    for producer in boxes:
        for consumer in boxes:
            if producer == consumer or sum(consumer) < sum(producer):
                continue
            assert any(c > p for p, c in zip(producer, consumer, strict=True))
            collision = final[producer] & initial[consumer]
            assert not collision, ("Future/same-rank version-zero read clobbered", producer, consumer, next(iter(collision)))
            pairs += 1
    # Challenge within-rank order. Each box completes all its v0 reads before
    # its own scatter; other same-rank boxes may already have scattered.
    all_fields = {(field, p) for field in fields for p in box_positions((0, 0, 0), shapes[field])}
    bank = dict.fromkeys(all_fields, 0)
    writes = set()
    for box in sorted(boxes, key=lambda b: (sum(b), tuple(-a for a in b))):
        for key in initial[box]:
            assert bank[key] == 0, ("Aliased initial cache sees final value", box, key)
        for key in final[box]:
            assert key not in writes
            writes.add(key)
            bank[key] = steps
    assert writes == all_fields
    assert all(v == steps for v in bank.values())
    return {"shape": shape, "tile": tile, "steps": steps, "initial_low_halo": initial_low_halo,
            "boxes": len(boxes), "future_consumer_pairs_checked": pairs,
            "unique_final_field_values": len(writes), "result": "pass"}


def main():
    cases = [prove(shape, tile, steps) for shape in ((3, 4, 5), (5, 6, 7))
             for tile in ((2, 3, 4), (3, 2, 3)) for steps in (1, 2, 4)]
    negatives = []
    for label, kw in (("T1 with an unadmitted two-cell initial halo", {"initial_low_halo": 2}),
                      ("T1 wrong final-coordinate shift", {"wrong_final_shift": True})):
        try:
            prove(steps=1, **kw)
        except AssertionError:
            negatives.append(label)
        else:
            raise AssertionError(f"Negative control escaped: {label}")
    print(json.dumps({"scope": __doc__, "cases": cases,
                      "negative_controls_rejected": negatives,
                      "exclusions": ["immutable caller input without a separate snapshot",
                                     "external/deferred initial-state readers",
                                     "overlapping component/material/monitor buffers",
                                     "late version-zero reads after own final scatter"]}, indent=2))


if __name__ == "__main__":
    main()
