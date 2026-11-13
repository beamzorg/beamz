"""Shared-bank mapping is exclusive, SoA and lifecycle-equivalent to private."""

import random

import pytest

from scripts.prove_cuda_shared_epsi_bank import (
    OLD_BYTES,
    TOTAL_BYTES,
    collision,
    guarded_collision,
    index,
    lifecycle,
    owner,
    point,
    prove_mapping,
)


def test_all8192_words_are_injective_bounded_and_warp_bank_conflict_free():
    proof = prove_mapping()
    assert proof["distinct_words"] == 8192 and proof["warp_access_groups"] == 256
    assert proof["total_dynamic_shared_bytes"] == 84516
    assert proof["bank_base_mod128"] == 36
    assert OLD_BYTES == 51748 and TOTAL_BYTES == 84516


@pytest.mark.parametrize("negative", ["wrong-tid", "wrong-stride", "wrong-color"])
def test_bad_indexing_has_distinct_owner_collision_witness(negative):
    witness = collision(negative)
    assert witness["first"] != witness["second"]
    assert index(*witness["first"], negative=negative) == index(
        *witness["second"], negative=negative
    )
    assert all(state["warm"] for state in guarded_collision(negative)["guarded_states"])


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("depth", [4, 16])
@pytest.mark.parametrize("high,clipped", [(False, False), (True, False), (False, True)])
def test_shared_and_private_banks_have_identical_warm_and_global_flush_events(
    axis, depth, high, clipped
):
    result = lifecycle(axis, depth, high=high, clipped=clipped)
    assert result["active_reads"] > 0 and result["global_boots"] > 0
    assert result["global_flushes"] > 0 and result["new_barriers"] == 0


@pytest.mark.parametrize("negative", ["omit-reload", "omit-flush"])
def test_stale_bank_or_missing_global_publication_is_detected(negative):
    with pytest.raises(AssertionError):
        lifecycle(0, negative=negative)


def test_old_even_odd_Z_map_not_new_contiguous_source_map():
    assert point(0, 1) == (2, 0, 0)
    assert point(512, 1) == (3, 0, 0)
    assert owner((7, 15, 31)) == (1023, 3)


def test_exclusive_thread_accesses_commute_under_arbitrary_legal_interleavings():
    # Every per-thread read-modify-write sequence is ordered. Cross-thread
    # schedules can interleave freely because no addresses are shared.
    bank = {index(c, s, t): 0 for c in range(2) for s in range(4) for t in range(1024)}
    rounds = [0] * 1024
    ready = list(range(1024))
    rng = random.Random(13)
    while ready:
        t = rng.choice(ready)
        n = rounds[t]
        for c in range(2):
            for s in range(4):
                address = index(c, s, t)
                assert bank[address] == n
                bank[address] = n + 1
        rounds[t] += 1
        if rounds[t] == 4:
            ready.remove(t)
    assert set(bank.values()) == {4}
