"""Bucketed Morton changes host ready priority while retaining actual slots."""

from pathlib import Path

import pytest

from scripts.cuda_persistent_wavefront_dag import build, rank_slots
from scripts.patch_cuda_persistent_rank_bucket import transformed
from scripts.prove_cuda_persistent_rank_bucket import oracle, verify_order


SOURCE = Path(".cache/cuda-five-hour-2026-11-13-1059/persistent-p5-g4-floor-global-v1/source/src/native_persistent.h")


def parent():
    if not SOURCE.exists():
        pytest.skip("Frozen floor/global P5 header required")
    return SOURCE.read_text()


@pytest.mark.parametrize("window", (2, 4))
def test_only_ready_priority_changes_and_rank_slot_validation_stays_identical(window):
    old = parent()
    new = transformed(old, window)
    assert "using Ready=std::tuple<int,uint64_t,int32_t>" in new
    assert f"BeamzPersistentTileRank(t,p)/{window},BeamzPersistentMorton(t,p)" in new
    assert f"BeamzPersistentTileRank(child,p)/{window},BeamzPersistentMorton(child,p)" in new
    before = old[:old.index(" if(locality){")]
    after = new[:new.index(" if(locality){")].replace("\n#include <tuple>", "", 1)
    assert before == after
    validation = " for(size_t i=0;i<count;++i)out.job_index[out.job_ids[i]]=i;"
    assert old[old.index(validation):] == new[new.index(validation):]


def test_unknown_source_or_window_cannot_inherit_proof():
    with pytest.raises(ValueError, match="Exact reviewed"):
        transformed("wrong", 2)
    with pytest.raises(ValueError, match="windows2/4"):
        transformed(parent(), 3)


@pytest.mark.parametrize("window", (2, 4))
def test_independent_available_priority_oracle_keeps_all_RAW_and_WAR_edges(window):
    counts = (12, 9, 6)
    slots, frontier = rank_slots(counts)
    slots = [frontier - 1 - s for s in slots]
    dag = build(counts, slots, frontier)
    jobs = oracle(dag, window)
    verify_order(dag, jobs)
    assert len(jobs) == 648
    bad = jobs.copy()
    bad[0], bad[-1] = bad[-1], bad[0]
    with pytest.raises(AssertionError, match="dependency"):
        verify_order(dag, bad)
