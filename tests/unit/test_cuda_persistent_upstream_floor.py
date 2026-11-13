"""Ready floors remain acquired facts until the current consumer completes."""
from pathlib import Path

import pytest

from scripts.cuda_persistent_tile_pipeline import prepare,simulate,windows,PipelineHazard
from scripts.cuda_persistent_wavefront_dag import rank_slots
from scripts.patch_cuda_native_persistent_relaxed_poll import relaxed_poll
from scripts.patch_cuda_native_persistent_upstream_floor import floor_cache


@pytest.mark.parametrize('group',[1,2,4,16])
@pytest.mark.parametrize('locality',[False,True])
@pytest.mark.parametrize('workers',[1,3,16])
def test_floor_preserves_actual_payload_reads_with_slot_reuse(group,locality,workers):
    counts=(8,4,3);slots,frontier=rank_slots(counts);dag=prepare(counts,slots,frontier,locality=locality)
    plain=simulate(dag,workers=workers,publication_steps=group)
    cached=simulate(dag,workers=workers,publication_steps=group,cache_upstream_floor=True)
    assert cached['step_starts']==plain['step_starts']
    assert cached['RAW_step_checks']<=plain['RAW_step_checks']
    assert cached['upstream_floor_hits']>0


def test_floor_with_empty_late_and_early_windows():
    extent=(25,31,47);depth=16;counts=tuple((n+1+depth+b-1)//b for n,b in zip(extent,(7,15,31),strict=True))
    slots,frontier=rank_slots(counts);dag=prepare(counts,slots,frontier)
    result=simulate(dag,workers=16,publication_steps=4,tile_windows=windows(counts,extent,depth),cache_upstream_floor=True)
    assert result['tiles']==len(slots)


def test_floor_does_not_replace_full_reader_retirement():
    counts=(8,4,3);slots,frontier=rank_slots(counts);dag=prepare(counts,slots,frontier)
    times=[[1]*16 for _ in slots];times[12]=[10000]*16
    with pytest.raises(PipelineHazard,match='Old consumer|Cached acquired'):
        simulate(dag,workers=16,publication_steps=4,cache_upstream_floor=True,step_times=times,negative='drop-WAR')


def test_source_floor_reset_acquired_values_and_cold_storage():
    old=relaxed_poll(Path('scripts/cuda_native_persistent.cuh').read_text());new=floor_cache(old)
    assert 'uint32_t epoch,upstream_floor;' in new
    assert 'C().upstream_floor=0;' in new
    assert 'ready=Acquire(C().ready+owner);if(ready>=uint32_t(step))return ready;' in new
    assert 'const uint32_t seen=WaitProducer(producer,owner,step);if(C().abort_local)return;floor=seen<floor?seen:floor;' in new
    assert 'C().upstream_floor=floor;' in new
    assert old[old.index('struct PersistentEnvironment:'):].replace('C().epoch=Acquire(p.epoch);C().tile=-1;', 'C().epoch=Acquire(p.epoch);C().depth=Depth;C().upstream_floor=0;C().tile=-1;')==new[new.index('struct PersistentEnvironment:'):]
