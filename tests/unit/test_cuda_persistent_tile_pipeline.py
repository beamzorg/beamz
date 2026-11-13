"""Step-readiness differs from whole-producer completion and slot retirement."""

import random

import pytest

from scripts.cuda_persistent_wavefront_dag import rank_slots
from scripts.cuda_persistent_tile_pipeline import PipelineHazard, prepare, simulate, windows


def permuted(counts, seed):
    slots, frontier = rank_slots(counts)
    rng = random.Random(seed)
    ny, nx = counts[1:]
    ranks = {}
    for t in range(len(slots)):
        ranks.setdefault(t//(ny*nx)+(t//nx) % ny+t % nx, []).append(t)
    for group in ranks.values():
        values = list(range(len(group)))
        rng.shuffle(values)
        for t, value in zip(group, values, strict=True):
            slots[t] = value
    return slots, frontier


@pytest.mark.parametrize("counts", [(1, 1, 1), (2, 3, 4), (8, 4, 3), (12, 9, 6)])
@pytest.mark.parametrize("workers", [1, 3, 16])
@pytest.mark.parametrize("locality", [False, True])
def test_fifo_executing_workers_preserve_RAW_and_full_WAR(counts, workers, locality):
    slots, frontier = permuted(counts, 317)
    dag = prepare(counts, slots, frontier, locality=locality)
    r = simulate(dag, workers=workers, seed=29)
    assert r["tiles"] == len(slots)
    assert r["ready_resets_before_claim"] == len(slots)


def test_RAW_waits_are_steps_not_whole_producer_completion():
    slots, frontier = rank_slots((8, 4, 3))
    dag = prepare((8, 4, 3), slots, frontier)
    r = simulate(dag, workers=16)
    assert r["steps_started_before_RAW_parent_completion"] > 100


@pytest.mark.parametrize("extent,depth", [((25, 25, 25), 4), ((25, 31, 47), 16), ((64, 96, 128), 16)])
def test_empty_and_early_late_windows_publish_only_supported_versions(extent, depth):
    counts = tuple((n+1+depth+b-1)//b for n, b in zip(extent, (7, 15, 31), strict=True))
    slots, frontier = rank_slots(counts)
    dag = prepare(counts, slots, frontier)
    spans = windows(counts, extent, depth)
    assert any(first > 1 or last < depth for first, last in spans)
    if depth == 16:
        assert any(first > last for first, last in spans)
    simulate(dag, workers=16, depth=depth, tile_windows=spans)


def test_unstarted_worker_cannot_claim_a_future_fifo_job():
    slots, frontier = rank_slots((8, 4, 3))
    dag = prepare((8, 4, 3), slots, frontier)
    r = simulate(dag, workers=4, worker_start_delays=[0, 1000, 10000, 1000000])
    assert r["tiles"] == len(slots)


def test_future_step_wait_is_rejected_instead_of_hanging():
    slots, frontier = rank_slots((2, 3, 4))
    dag = prepare((2, 3, 4), slots, frontier)
    with pytest.raises(PipelineHazard, match="Deadlock|watchdog"):
        simulate(dag, workers=16, negative="future-step")


def test_missing_slot_reader_retirement_poison_is_detected():
    slots, frontier = rank_slots((8, 4, 3))
    dag = prepare((8, 4, 3), slots, frontier)
    times = [[1]*16 for _ in slots]
    times[12] = [10000]*16
    with pytest.raises(PipelineHazard, match="Old consumer"):
        simulate(dag, workers=16, step_times=times, negative="drop-WAR")


def test_future_job_predecessor_fails_admission():
    slots, frontier = rank_slots((2, 3, 4))
    with pytest.raises(ValueError, match="lower job"):
        prepare((2, 3, 4), slots, frontier, jobs=list(reversed(range(len(slots)))))


def test_WAR_edges_are_not_subtracted_when_also_RAW():
    slots, frontier = permuted((8, 4, 3), 917)
    dag = prepare((8, 4, 3), slots, frontier)
    intersections = [dag["raw_predecessors"][t]&dag["claim_completion_predecessors"][t] for t in dag["jobs"]]
    assert any(intersections)


def test_treating_overlapping_WAR_as_only_RAW_step_readiness_is_unsafe():
    slots, frontier = rank_slots((8, 4, 3))
    dag = prepare((8, 4, 3), slots, frontier)
    with pytest.raises(PipelineHazard, match="Old consumer"):
        simulate(dag, workers=16, seed=0, negative="subtract-RAW-from-WAR")


def test_claim_tag_before_ready_reset_exposes_old_holders_counter():
    slots, frontier = rank_slots((8, 4, 3))
    dag = prepare((8, 4, 3), slots, frontier)
    with pytest.raises(PipelineHazard, match="stale data"):
        simulate(dag, workers=16, seed=0, negative="reset-after-tag")
