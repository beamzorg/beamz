"""Independent physical-coordinate token proof for a circular padded cache.

The owned u-box has shape B; a (B+1) physical cache slides upwind. Persistent
physical slots remove own-field gathers. This proves dependency/version/slot
and publication-disjointness properties, not CUDA memory ordering or arithmetic.
"""

from __future__ import annotations

import itertools
import json
import math

try:
    from scripts.cuda_stencil_trace import trace_yee_step
    from scripts.prove_cuda_systolic_schedule import component_shapes
except ModuleNotFoundError:
    from cuda_stencil_trace import trace_yee_step
    from prove_cuda_systolic_schedule import component_shapes


def positions(low, shape):
    return itertools.product(*(range(a, a + b) for a, b in zip(low, shape, strict=True)))


def prove(tile=(3, 4, 5), shape=(12, 15, 19), steps=8, *, owner_box=(1, 1, 1),
          early_import=False, stale_h=False, publish_all_owned=False):
    cache_shape = tuple(b + 1 for b in tile)
    origin = tuple(a * b for a, b in zip(tile, owner_box, strict=True))
    fields = tuple("H" + c for c in "xyz") + tuple("E" + c for c in "xyz")
    shapes = component_shapes(shape)
    initial_low = tuple(a - 1 for a in origin)
    cache = {}
    reads = imports = 0

    def slot(p):
        # Reference mapping only. CUDA should use a nonnegative rolling head,
        # avoiding signed remainder for physical coordinates below zero.
        return tuple((a - b) % c for a, b, c in zip(p, initial_low, cache_shape, strict=True))

    def valid(field, p):
        return all(0 <= a < b for a, b in zip(p, shapes[field], strict=True))

    for p in positions(initial_low, cache_shape):
        for field in fields:
            cache[field, slot(p)] = field, 0, p
    previous_publication = set()
    for n in range(1, steps + 1):
        low = tuple(a - n for a in origin)
        owned = set(positions(low, tile))
        cache_low = tuple(a - 1 for a in low)
        incoming = set(positions(cache_low, cache_shape)) - owned
        assert len(incoming) == math.prod(cache_shape) - math.prod(tile)
        assert {slot(p) for p in owned}.isdisjoint({slot(p) for p in incoming})
        # H writes cannot overwrite any previous outgoing publication read.
        assert {slot(p) for p in owned}.isdisjoint(previous_publication)

        def import_faces(n=n, incoming=incoming):
            nonlocal imports
            for p in incoming:
                owner = tuple((a + n) // b for a, b in zip(p, tile, strict=True))
                if any(valid(f, p) for f in fields):
                    assert all(a <= b for a, b in zip(owner, owner_box, strict=True))
                    assert sum(owner) < sum(owner_box)
                for field in fields:
                    version = n - 1 if stale_h and field.startswith("H") else n
                    cache[field, slot(p)] = field, version, p
                    imports += 1

        if early_import:
            import_faces()  # A deliberate overwrite before H has read old E.
        equations = trace_yee_step(n - 1)
        for phase, family in enumerate("HE"):
            if phase and not early_import:
                import_faces()  # H barrier, then imports, then publication barrier.
            for update in equations:
                if not update.output.field.startswith(family):
                    continue
                for p in owned:
                    if valid(update.output.field, p):
                        for read in update.reads:
                            q = tuple(a + b for a, b in zip(p, read.offset, strict=True))
                            if valid(read.field, q):
                                actual = cache.get((read.field, slot(q)))
                                expected = read.field, read.version, q
                                assert actual == expected, (n, phase, actual, expected)
                                reads += 1
                    cache[update.output.field, slot(p)] = update.output.field, n, p
        # E barrier precedes outgoing reads; all cached values now have clock n.
        for p in positions(cache_low, cache_shape):
            for field in fields:
                assert cache[field, slot(p)] == (field, n, p)
        high = tuple(a + b - 1 for a, b in zip(low, tile, strict=True))
        outgoing = {p for p in owned if any(a == b for a, b in zip(p, high, strict=True))}
        if publish_all_owned:
            outgoing = owned
        previous_publication = {slot(p) for p in outgoing}
    return {"tile_zyx": tile, "cache_zyx": cache_shape, "steps": steps,
            "valid_versioned_reads": reads, "imported_field_slots": imports,
            "result": "pass"}


def main():
    cases = [prove(tile, steps=n, owner_box=owner) for tile in ((3, 4, 5), (4, 3, 5))
             for n in (2, 4, 8) for owner in ((1, 1, 1), (3, 3, 3))]
    cases += [prove((7, 7, 15), (23, 25, 47), steps=n, owner_box=owner)
              for n in (4, 8, 16) for owner in ((1, 1, 1), (3, 3, 3))]
    negatives = []
    for label, kw in (("import before H completes", {"early_import": True}),
                      ("H halo at n-1", {"stale_h": True}),
                      ("unconditional all-owned publication reads", {"publish_all_owned": True})):
        try:
            prove(steps=4, **kw)
        except AssertionError:
            negatives.append(label)
        else:
            raise AssertionError(f"Negative control escaped: {label}")
    print(json.dumps({"scope": __doc__, "cases": cases,
                      "negative_controls_rejected": negatives}, indent=2))


if __name__ == "__main__":
    main()
