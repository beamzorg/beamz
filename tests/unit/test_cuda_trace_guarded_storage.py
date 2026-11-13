"""Independent enumerated states/versions challenge affine storage admission."""

import itertools
import random

import pytest

from scripts.cuda_space_time_ir import IllegalSchedule, yee_operator
from scripts.cuda_trace_guarded_storage import (
    FIXED_E_COLORS,
    OwnedRegion,
    certify,
    terms_from_ir,
)


def active(term, point):
    return (all(0 <= p < n for p, n in zip(point, term["shape"], strict=True))
            and (point[term["axis"]] < 12 or point[term["axis"]] >= term["shape"][term["axis"]] - 12))


def points(region, n):
    return itertools.product(*(range(o - n, o + b - n) for o, b in zip(region.origin, region.owned, strict=True)))


def enumerated_edges(region, terms):
    result = set()
    for n in range(1, region.depth + 1):
        for point in points(region, n):
            enabled = [term["term"] for term in terms if active(term, point)]
            result.update(itertools.combinations(enabled, 2))
    return result


def test_integer_guard_intersections_match_independent_enumeration():
    rng = random.Random(1632)
    for _ in range(35):
        minimum = tuple(rng.randrange(24, 39) for _ in range(3))
        owned = (3, 4, 5)
        origin = tuple(rng.randrange(0, n + 12) for n in minimum)
        region = OwnedRegion(minimum, origin, owned, rng.randrange(2, 10))
        terms, _ = terms_from_ir(yee_operator(cpml_axes=(0, 1, 2)), region)
        result = certify(region, threads=30)
        expected = enumerated_edges(region, terms)
        assert set(map(tuple, result["edges"])) == expected
        for witness in result["edge_witnesses"]:
            a, b = witness["states"]
            assert active(terms[a], witness["point_zyx"]) and active(terms[b], witness["point_zyx"])
            assert tuple(witness["point_zyx"]) in set(points(region, witness["version"]))
        assert result["fixed_two_safe"] == all(FIXED_E_COLORS[a] != FIXED_E_COLORS[b] for a, b in expected)


@pytest.mark.parametrize("origin", [(497, 15, 248), (21, 495, 248)])
def test_time_separated_two_faces_allow_two_colors_despite_multiple_axis_flags(origin):
    # First case: high-Z at1..2 then low-Y from4. Second: high-Y through7
    # then low-Z from10, with the complete trajectory away from pinned E nodes.
    region = OwnedRegion((512, 512, 512), origin, (7, 15, 31), 16)
    result = certify(region)
    assert result["fixed_two_safe"] and result["optimal_color_count"] == 2
    assert result["warm_private_use_witness"] is not None
    assert result["mapping"]["slots"] == 4
    assert result["mapping"]["fixed_two_private_floats_per_thread"] == 8
    assert {tuple(edge) for edge in result["edges"]} == {(0, 5), (1, 2)}


def test_corner_requires_six_states_and_supplies_exact_negative_witness():
    region = OwnedRegion((32, 32, 32), (0, 0, 0), (3, 4, 5), 8)
    result = certify(region, threads=30)
    assert not result["fixed_two_safe"]
    assert result["optimal_color_count"] == 6
    witness = result["fixed_two_counterexample"]
    assert FIXED_E_COLORS[witness["states"][0]] == FIXED_E_COLORS[witness["states"][1]]


def simulate_cache(region, *, omit_reload=False, omit_flush=False, force=False):
    terms, _ = terms_from_ir(yee_operator(cpml_axes=(0, 1, 2)), region)
    if not force:
        assert certify(region, threads=(region.owned[1] + 1) * (region.owned[2] + 1))["fixed_two_safe"]
    cache_dims = tuple(b + 1 for b in region.owned)
    cache, published, last = {}, {}, {}
    for n in range(1, region.depth + 1):
        for point in points(region, n):
            relative = tuple(p - o + n + 1 for p, o in zip(point, region.origin, strict=True))
            slot = tuple((p - o) % d for p, o, d in zip(point, region.origin, cache_dims, strict=True))
            fresh = n == 1 or 1 in relative
            flush = n == region.depth or any(d == b for d, b in zip(relative, region.owned, strict=True))
            for term in terms:
                if not active(term, point):
                    continue
                state = term["term"]
                color = FIXED_E_COLORS[state]
                expected = (state, point, n - 1)
                loaded = expected if fresh and (not omit_reload or n == 1) else cache.get((slot, color))
                if loaded != expected:
                    return False
                updated = (state, point, n)
                cache[slot, color] = updated
                last[state, point] = updated
                if flush and (not omit_flush or n == region.depth):
                    published[state, point] = updated
    return bool(last) and published == last


@pytest.mark.parametrize("origin", [(0, 48, 48), (128, 48, 48), (24, 48, 48)])
def test_private_slot_tokens_survive_wrap_and_exact_first_last_owned_rules(origin):
    region = OwnedRegion((128, 128, 128), origin, (3, 4, 5), 16)
    assert simulate_cache(region)
    assert not simulate_cache(region, omit_reload=True)
    assert not simulate_cache(region, omit_flush=True)


def test_forced_two_color_corner_corrupts_retained_previous_state():
    region = OwnedRegion((32, 32, 32), (0, 0, 0), (3, 4, 5), 8)
    assert not simulate_cache(region, force=True)


def test_actual_fixed_xy_mapping_has_four_slots_at1024_and_eight_at512():
    region = OwnedRegion((128, 128, 128), (7, 45, 62), (7, 15, 31), 16)
    for threads, slots in [(512, 8), (1024, 4)]:
        result = certify(region, threads=threads)
        step = result["mapping"]["z_stride"]
        locations = set()
        for tid in range(threads):
            for j in range(slots):
                locations.add((tid // 512 + j * step, tid % 512 // 32, tid % 32))
        assert locations == set(itertools.product(range(8), range(16), range(32)))


def test_noncanonical_precision_or_partial_thread_mapping_rejects():
    region = OwnedRegion((128, 128, 128), (7, 45, 62), (7, 15, 31), 16)
    with pytest.raises(IllegalSchedule, match="FP32"):
        certify(region, yee_operator(cpml_axes=(0, 1, 2), precision="bfloat16"))
    with pytest.raises(IllegalSchedule, match="Z stride"):
        certify(region, threads=768)
