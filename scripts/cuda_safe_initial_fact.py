"""Exact initial-cache support fact, independent of owned-trajectory support."""

from __future__ import annotations

import itertools


def certify(shapes, tile, owned=(7, 15, 31), *, policy=0):
    if (len(shapes) != 6 or any(len(shape) != 3 or min(shape) < 1 for shape in shapes)
            or len(tile) != 3 or min(tile) < 0 or len(owned) != 3 or min(owned) < 1):
        raise ValueError("Six positive component supports and a nonnegative3D tile required")
    if policy not in (0, 2, 3, 4):
        return False
    origin = tuple(t * b for t, b in zip(tile, owned, strict=True))
    return all(o - 1 >= 0 and o + b - 1 < size
               for shape in shapes for o, b, size in zip(origin, owned, shape, strict=True))


def prove(shapes, tile, owned=(7, 15, 31), *, policy=0):
    admitted = certify(shapes, tile, owned, policy=policy)
    origin = tuple(t * b for t, b in zip(tile, owned, strict=True))
    checked = 0
    if admitted:
        for slot in itertools.product(*(range(b + 1) for b in owned)):
            point = tuple(o + (s if s < b else -1) for o, s, b in zip(origin, slot, owned, strict=True))
            for shape in shapes:
                assert all(0 <= p < size for p, size in zip(point, shape, strict=True))
                checked += 1
    return {"admitted": admitted, "cache_zyx": [b + 1 for b in owned],
            "all_component_cache_points_checked": checked,
            "scope": "Inclusive startup box only; no owned trajectory or physics substitution"}
