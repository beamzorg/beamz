"""Owned trajectory support does not prove the full six-field startup box."""

import itertools

import pytest

from scripts.cuda_safe_initial_fact import certify, prove


def shapes(n):
    return tuple(tuple(size + d for size, d in zip(n, delta, strict=True))
                 for delta in ((0, 0, 1), (0, 1, 0), (1, 0, 0),
                               (1, 1, 0), (1, 0, 1), (0, 1, 1)))


@pytest.mark.parametrize("policy", [0, 2, 3, 4])
def test_exact_full_cache_fact_covers_all_six_staggered_supports(policy):
    result = prove(shapes((64, 96, 128)), (2, 2, 2), policy=policy)
    assert result["admitted"]
    assert result["all_component_cache_points_checked"] == 6 * 8 * 16 * 32


def test_owned_valid_trajectory_can_have_an_invalid_positive_initial_ghost():
    supports, origin, owned = shapes((62, 64, 96)), (56, 15, 31), (7, 15, 31)
    for n in range(1, 5):
        for q in itertools.product(*(range(b) for b in owned)):
            point = tuple(o + v - n for o, v in zip(origin, q, strict=True))
            assert all(0 <= p < size for shape in supports
                       for p, size in zip(point, shape, strict=True))
    assert not certify(supports, (8, 1, 1))
    assert origin[0] + owned[0] - 1 == supports[0][0]  # Hx positive-Z initial ghost.


def test_noio_context_and_zero_origin_are_never_annotated():
    supports = shapes((64, 96, 128))
    assert not certify(supports, (2, 2, 2), policy=1)
    assert not certify(supports, (2, 2, 2), policy=5)
    assert not certify(supports, (0, 2, 2))


@pytest.mark.parametrize("threads", [512, 1024])
def test_both_actual_private_slot_mappings_cover_the_full_cache_once(threads):
    slots = 8 * 16 * 32 // threads
    actual = []
    for tid in range(threads):
        for j in range(slots):
            z, y, x = tid // 512 + j * (threads // 512), (tid // 32) % 16, tid % 32
            actual.append((z, y, x))
    assert len(actual) == len(set(actual)) == 8 * 16 * 32
    assert set(actual) == set(itertools.product(range(8), range(16), range(32)))
