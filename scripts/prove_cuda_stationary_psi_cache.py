"""Unique-owner lazy publication proof for cached physical CPML recurrences.

CPML12 and actual staggered packed slabs are modeled. Integer recurrences test
storage/version routing independently of floating-point arithmetic. This is not
a kernel, a global-memory-order sanitizer, or a register/performance prediction.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA, derive_geometry
from scripts.cuda_space_time_ir import FIELDS


def owner(point, version, owned):
    return tuple((p + version) // b for p, b in zip(point, owned, strict=True))


def cache_tiles(point, version, owned, counts):
    choices = []
    for p, b, limit in zip(point, owned, counts, strict=True):
        q, remainder = divmod(p + version + 1, b)
        options = (q - 1, q) if not remainder else (q,)
        choices.append(tuple(q for q in options if 0 <= q < limit))
    return set(itertools.product(*choices))


def pml_axes(tile, owned, depth, minimum):
    return tuple(a for a, (t, b, size) in enumerate(zip(tile, owned, minimum, strict=True))
                 if t * b - depth < 12 or t * b + b - 2 >= size - 12)


def packed_key(point, shape, axis, *, wrong_e_extent=False, phase=0):
    size = shape[axis] - int(wrong_e_extent and phase == 1)
    p = point[axis]
    assert p < 12 or p >= size - 12
    packed = p if p < 12 else p + 24 - size
    assert 0 <= packed < 24, "Packed CPML coordinate outside the actual slab"
    coordinate = list(point)
    coordinate[axis] = packed
    slab = list(shape)
    slab[axis] = 24
    return (coordinate[0] * slab[1] + coordinate[1]) * slab[2] + coordinate[2]


def sample_points(shape, owned, axis):
    choices = []
    for a, (size, b) in enumerate(zip(shape, owned, strict=True)):
        values = (0, 1, 11, size - 12, size - 2, size - 1) if a == axis else (0, 1, b - 1, b, size // 2, size - 2, size - 1)
        choices.append(sorted({x for x in values if 0 <= x < size}))
    return [p for p in itertools.product(*choices) if p[axis] < 12 or p[axis] >= shape[axis] - 12]


def prove_owner_boundary_classes(owned=(5, 15, 31)):
    """Exhaust every relative cache coordinate; translation/version cancel."""
    tile = (3, 4, 5)
    origin = tuple(t * b for t, b in zip(tile, owned, strict=True))
    starts = incoming = 0
    for delta in itertools.product(*(range(b + 1) for b in owned)):
        p0 = tuple(o - 1 + d for o, d in zip(origin, delta, strict=True))
        first_owner = owner(p0, 1, owned)
        assert all(t >= c for t, c in zip(first_owner, tile, strict=True))
        assert sum(first_owner) != sum(tile) or first_owner == tile
        starts += 1
        if not any(d == 0 for d in delta):
            continue
        n = 7  # Arbitrary n; its cancellation is checked in the equalities.
        point = tuple(o - n - 1 + d for o, d in zip(origin, delta, strict=True))
        producer, following = owner(point, n, owned), owner(point, n + 1, owned)
        assert producer == tuple(t - int(d == 0) for t, d in zip(tile, delta, strict=True))
        assert following == tuple(t + int(d == b) for t, d, b in zip(tile, delta, owned, strict=True))
        local = tuple(p + n - t * b for p, t, b in zip(point, producer, owned, strict=True))
        assert any(r == b - 1 for r, b in zip(local, owned, strict=True))
        assert all(t >= c for t, c in zip(following, tile, strict=True))
        assert sum(following) != sum(tile) or following == tile
        incoming += 1
    return {"owned_zyx": owned, "startup_classes": starts, "incoming_classes": incoming,
            "future_overwrite_before_consumer": False, "same_rank_other_owner": False}


def prove(minimum=(32, 35, 37), depth=16, *, owned=(5, 15, 31), mode="single-axis",
          omit_h_publication=False, omit_e_publication=False, omit_low_refresh=False,
          wrong_e_extent=False, publish_halo=False, future_publication=False,
          load_policy="first-owned"):
    assert mode in ("single-axis", "all-cache")
    assert load_policy in ("first-owned", "startup-and-low")
    assert len(minimum) == len(owned) == 3 and min(minimum) >= 24 and min(owned) >= 2
    assert depth >= 1
    counts = tuple(math.ceil((n + 1 + depth) / b) for n, b in zip(minimum, owned, strict=True))
    stats = {"sampled_physical_recurrences": 0, "owned_updates": 0, "cached_updates": 0,
             "direct_updates": 0, "startup_loads": 0, "low_imports": 0, "first_owned_loads": 0,
             "high_publications": 0, "final_publications": 0}
    packed_owners = {}
    for term in derive_geometry()["cpml_terms"]:
        phase, axis = term["phase"], term["axis_zyx"]
        term_id = 6 * phase + int(term["term"][-1])
        delta = SUPPORT_DELTA[FIELDS.index(term["output"])]
        shape = tuple(n + d for n, d in zip(minimum, delta, strict=True))
        for point in sample_points(shape, owned, axis):
            packed = packed_key(point, shape, axis, wrong_e_extent=wrong_e_extent, phase=phase)
            key = (term_id, packed)
            assert key not in packed_owners or packed_owners[key] == point, "Packed slab aliases physical points"
            packed_owners[key] = point
            expected = [17 * packed + term_id]
            for n in range(1, depth + 1):
                expected.append(expected[-1] * 3 + 7 * n + 11 * term_id)
            global_version, global_value = 0, expected[0]
            tiles = set().union(*(cache_tiles(point, n, owned, counts) for n in range(depth + 1)))
            updates = 0
            for tile in sorted(tiles, key=lambda t: (sum(t), t)):
                axes = pml_axes(tile, owned, depth, minimum)
                cached = mode == "all-cache" or axes == (axis,)
                private = None
                if cached and load_policy == "startup-and-low" and tile in cache_tiles(point, 0, owned, counts):
                    # For EVERY startup point owner(p,1)>=tile. No previous
                    # rank can have overwritten global psi0 before this read.
                    assert all(p >= t for p, t in zip(owner(point, 1, owned), tile, strict=True))
                    assert global_version == 0 and global_value == expected[0], "Startup saw overwritten psi0"
                    private = (0, global_value)
                    stats["startup_loads"] += 1
                for n in range(1, depth + 1):
                    physical_owner = owner(point, n, owned)
                    if physical_owner == tile:
                        assert axis in axes, "A supposedly CPML-free tile owns an active point"
                        if cached:
                            if load_policy == "first-owned" and (n == 1 or owner(point, n - 1, owned) != tile):
                                assert (global_version, global_value) == (n - 1, expected[n - 1]), "First owned load needs exact psi_(n-1)"
                                if n == 1 or not omit_low_refresh:
                                    private = (n - 1, global_value)
                                stats["first_owned_loads"] += 1
                            assert private == (n - 1, expected[n - 1]), "Cached recurrence missed previous physical value"
                            value = private[1] * 3 + 7 * n + 11 * term_id
                            private = (n, value)
                            stats["cached_updates"] += 1
                        else:
                            assert (global_version, global_value) == (n - 1, expected[n - 1]), "Direct recurrence saw stale global state"
                            value = global_value * 3 + 7 * n + 11 * term_id
                            global_version, global_value = n, value
                            stats["direct_updates"] += 1
                        assert value == expected[n]
                        updates += 1
                        stats["owned_updates"] += 1
                        local = tuple(p + n - t * b for p, t, b in zip(point, tile, owned, strict=True))
                        high = any(x == b - 1 for x, b in zip(local, owned, strict=True))
                        if cached and (high or n == depth):
                            omit = omit_h_publication if phase == 0 else omit_e_publication
                            if not omit:
                                global_version, global_value = n, value
                                if high:
                                    stats["high_publications"] += 1
                                else:
                                    stats["final_publications"] += 1
                                if future_publication and high and n < depth:
                                    global_version, global_value = n + 1, expected[n + 1]
                    in_cache = tile in cache_tiles(point, n, owned, counts)
                    low = in_cache and physical_owner != tile
                    if cached and low and load_policy == "startup-and-low":
                        # Low faces are the producer's high faces. The NEXT
                        # physical owner is >=this consumer componentwise.
                        # Extra high axes can send it to a still later rank.
                        assert all(p >= t for p, t in zip(owner(point, n + 1, owned), tile, strict=True))
                        assert (global_version, global_value) == (n, expected[n]), "Low halo needs exact psi_n, not stale/future state"
                        if publish_halo:
                            assert physical_owner == tile, "Only the unique owned point may publish psi"
                        if not omit_low_refresh:
                            private = (n, global_value)
                        stats["low_imports"] += 1
                    if cached and not in_cache:
                        private = None
            assert updates == depth, "Physical recurrence skipped or duplicated a version"
            assert (global_version, global_value) == (depth, expected[depth]), "Final packed slab not complete"
            stats["sampled_physical_recurrences"] += 1
    if mode == "all-cache":
        assert stats["cached_updates"] > 0
    volume = math.prod(owned)
    cache_volume = math.prod(b + 1 for b in owned)
    high_surface = volume - math.prod(b - 1 for b in owned)
    low_union = cache_volume - volume
    if load_policy == "first-owned":
        ideal_bytes = 4 * 4 * (2 * volume + 2 * (depth - 1) * high_surface) / (depth * volume)
    else:
        ideal_bytes = 4 * 4 * (cache_volume + depth * low_union + depth * high_surface + volume - high_surface) / (depth * volume)
    return {"minimum_support_zyx": minimum, "owned_zyx": owned, "depth": depth, "mode": mode,
            "load_policy": load_policy, "statistics": stats, "private_state_per_single_axis_point": 4,
            "private_state_per_thread_cache6x16x32": 24,
            "additional_low_x_staging_bytes": 0 if load_policy == "first-owned" else 4 * owned[0] * owned[1] * 4,
            "ideal_all_active_psi_bytes_per_update": ideal_bytes,
            "direct_four_term_psi_bytes_per_update": 32,
            "traffic_limits": "Payload estimate only; assumes all points active and counts padded low imports/high publications, not measured DRAM",
            "result": "pass", "scope": "Sampled packed-slab integer oracle + exact owner inequalities; not GPU/arithmetic proof"}


def main():
    cases = [prove(minimum, depth, mode=mode, load_policy=policy)
             for minimum, depth in (((32, 35, 37), 4), ((64, 96, 128), 16))
             for mode in ("single-axis", "all-cache") for policy in ("startup-and-low", "first-owned")]
    rejected = []
    for option in ("omit_h_publication", "omit_e_publication", "omit_low_refresh",
                   "wrong_e_extent", "publish_halo", "future_publication"):
        try:
            prove(mode="all-cache", load_policy="startup-and-low" if option == "publish_halo" else "first-owned", **{option: True})
        except AssertionError:
            rejected.append(option)
        else:
            raise AssertionError(f"Negative control escaped: {option}")
    boundary = [prove_owner_boundary_classes(b) for b in ((5, 15, 31), (3, 3, 7), (7, 7, 15))]
    print(json.dumps({"owner_boundary_classes": boundary, "cases": cases,
                      "negative_controls_rejected": rejected}, indent=2))


if __name__ == "__main__":
    main()
