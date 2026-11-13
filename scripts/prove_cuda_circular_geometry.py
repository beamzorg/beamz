"""Independent padded-face/circular-index proof, including partial CTA tails."""

from __future__ import annotations

import itertools
import json
import math


def prove(tile=(7, 7, 15), steps=16, *, old_owned_mask=False, omit_tail_guard=False,
          wrong_backward_sign=False, missing_head_roll=False):
    bz, by, bx = tile
    cache = tuple(b + 1 for b in tile)
    pz, py, px = cache
    volume, padded = math.prod(tile), math.prod(cache)
    lanes = (volume + 255) // 256
    owned = []
    skipped = 0
    for thread in range(256):
        for j in range(lanes):
            q = thread + j * 256
            if q >= volume and not omit_tail_guard:
                skipped += 1
                continue
            point = q // (by * bx), (q // bx) % by, q % bx
            assert all(0 <= a < b for a, b in zip(point, tile, strict=True)), (thread, j, q, point)
            owned.append(point)
    assert len(owned) == len(set(owned)) == volume

    def linear(point):
        z, y, x = point
        return (z * py + y) * px + x

    def wrapped(point):
        return tuple(a % b for a, b in zip(point, cache, strict=True))

    def fast(point, head):
        result = []
        for a, h, size in zip(point, head, cache, strict=True):
            value = a + h
            result.append(value - size if value >= size else value)
        return tuple(result)

    # Global physical modulo is the oracle, independently of CUDA's head API.
    wrap_cases = 0
    for size in cache:
        offset = size * ((steps + 1 + size - 1) // size)
        for coordinate in range(-steps - 1, size + 1):
            actual = ((coordinate % (1 << 32)) & (size - 1)) if size & (size - 1) == 0 else (coordinate + offset) % size
            assert actual == coordinate % size
            wrap_cases += 1
    for n in range(1, steps + 1):
        head = tuple((-1 if missing_head_roll else -n) % size for size in cache)
        for q in owned:
            position = fast(q, head)
            expected = wrapped(tuple(a - n for a in q))
            assert position == expected
            for axis in range(3):
                for direction in (-1, 1):
                    target = list(expected)
                    coordinate = expected[axis]
                    stride = (py * px, px, 1)[axis]
                    if direction > 0:
                        delta = stride if coordinate + 1 < cache[axis] else -(cache[axis] - 1) * stride
                    else:
                        delta = -stride if coordinate else (cache[axis] - 1) * stride
                    target[axis] = (target[axis] + direction) % cache[axis]
                    actual_delta = -delta if wrong_backward_sign and direction < 0 else delta
                    assert linear(expected) + actual_delta == linear(target)
                    wrap_cases += 1

    low = []
    for i in range(padded - volume):
        if i < py * px:
            point = 0, i // px, i % px
        elif i - py * px < bz * px:
            j = i - py * px
            point = 1 + j // px, 0, j % px
        else:
            j = i - py * px - bz * px
            point = 1 + j // by, 1 + j % by, 0
        low.append(point)
        global_u = tuple(3 * b + p - 1 for b, p in zip(tile, point, strict=True))
        expected_local = tuple(u % b for u, b in zip(global_u, tile, strict=True))
        actual_local = tuple(((p - 1) & (b - 1)) if old_owned_mask else (p - 1 if p else b - 1)
                             for p, b in zip(point, tile, strict=True))
        assert actual_local == expected_local, (point, actual_local, expected_local)
        assert any(a == b - 1 for a, b in zip(actual_local, tile, strict=True))
        for n in range(1, steps + 1):
            next_head = tuple((-n - 1) % size for size in cache)
            assert fast(point, next_head) == wrapped(tuple(p - n - 1 for p in point))
    expected_low = {p for p in itertools.product(*(range(v) for v in cache)) if 0 in p}
    assert len(low) == len(set(low)) == padded - volume
    assert set(low) == expected_low
    return {"owned_zyx": tile, "cache_zyx": cache, "steps": steps,
            "owned_cells": volume, "thread_cell_slots": lanes * 256,
            "skipped_tail_slots": skipped, "low_halo_cells": len(low),
            "wrap_neighbor_cases": wrap_cases, "result": "pass"}


def main():
    cases = [prove(tile, steps=n) for tile in ((4, 8, 32), (8, 8, 16), (7, 7, 15))
             for n in (4, 8, 16, 64)]
    negatives = []
    for label, kwargs in (("old owned-dimension bitmask", {"old_owned_mask": True}),
                          ("unguarded partial CTA tail", {"omit_tail_guard": True}),
                          ("wrong E backward-neighbor sign", {"wrong_backward_sign": True}),
                          ("missing rolling-head advance", {"missing_head_roll": True})):
        try:
            prove(**kwargs)
        except AssertionError:
            negatives.append(label)
        else:
            raise AssertionError(f"Negative control escaped: {label}")
    print(json.dumps({"scope": __doc__, "cases": cases,
                      "negative_controls_rejected": negatives}, indent=2))


if __name__ == "__main__":
    main()
