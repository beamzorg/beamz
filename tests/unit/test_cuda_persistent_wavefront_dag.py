"""A ready queue needs field dependencies AND trail-slot reader retirement."""
import random

import pytest

from scripts.cuda_persistent_wavefront_dag import build, csr, rank_slots, simulate


@pytest.mark.parametrize("counts", [(1, 1, 1), (2, 3, 4), (8, 4, 3), (12, 9, 6)])
@pytest.mark.parametrize("workers", [1, 3, 16])
def test_arbitrary_ready_completion_keeps_all_raw_and_slot_lifetimes(counts, workers):
    slots, frontier = rank_slots(counts)
    dag = build(counts, slots, frontier)
    rng = random.Random(817)
    times = [rng.randrange(1, 100) for _ in slots]
    r = simulate(dag, times, workers)
    assert r["tiles"] == len(slots)
    assert r["makespan"] >= r["work_lower_bound"]
    assert r["makespan"] >= r["critical_path_lower_bound"]
    c = csr(dag)
    assert c["offsets"][-1] == len(c["edges"]) == dag["all_edges"]
    assert c["initial_ready"] == [0]


def test_raw_only_wavefront_can_destroy_an_active_readers_trail():
    counts = (8, 4, 3)
    slots, frontier = rank_slots(counts)
    dag = build(counts, slots, frontier, retain_reuse_edges=False)
    durations = [1] * len(slots)
    durations[12] = 10000  # First +Z consumer of the initial tile.
    with pytest.raises(AssertionError, match="old readers"):
        simulate(dag, durations, 16)
    safe = build(counts, slots, frontier)
    simulate(safe, durations, 16)


def test_same_rank_slot_alias_and_insufficient_epoch_lifetimes_reject():
    counts = (4, 4, 4)
    slots, frontier = rank_slots(counts)
    with pytest.raises(ValueError, match="unique"):
        build(counts, [0] * len(slots), frontier)
    with pytest.raises(ValueError, match="at least4"):
        build(counts, slots, frontier, epochs=3)
