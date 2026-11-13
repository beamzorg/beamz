"""Explicit-fence recipe: odd H-ready certifies a prior E prefix.

Its omitted-fence negatives are not used to reject the separately reviewed
cumulative-join release alternative, which has additional HB paths.
"""
from pathlib import Path

import pytest

from scripts.cuda_persistent_wavefront_dag import rank_slots
from scripts.cuda_persistent_tile_pipeline import prepare,windows,PipelineHazard
from scripts.cuda_persistent_phase_pipeline import simulate
from scripts.patch_cuda_native_persistent_e_batch import e_batch


@pytest.mark.parametrize('group',[2,4])
@pytest.mark.parametrize('depth',[4,16])
@pytest.mark.parametrize('workers',[1,3,16])
@pytest.mark.parametrize('locality',[False,True])
def test_every_counter_exposed_E_version_is_actually_fenced(group,depth,workers,locality):
    counts=(8,4,3);slots,f=rank_slots(counts);d=prepare(counts,slots,f,locality=locality)
    r=simulate(d,depth,workers,h_publish_before_parent=True,e_publication_steps=group)
    for tile in range(len(slots)):
        for n in range(2,depth+1):
            for parent in d['raw_predecessors'][tile]:
                assert r['E_ready'][parent][n-2]<=r['H_start'][tile][n-1]


@pytest.mark.parametrize('group',[2,4])
def test_mandatory_unaligned_window_end_and_empty_tiles(group):
    extent=(25,31,47);depth=16;counts=tuple((n+17+b-1)//b for n,b in zip(extent,(7,15,31),strict=True))
    slots,f=rank_slots(counts);d=prepare(counts,slots,f);spans=windows(counts,extent,depth)
    r=simulate(d,workers=16,h_publish_before_parent=True,e_publication_steps=group,tile_windows=spans)
    assert r['tiles']==len(slots)


def test_H_join_must_fence_every_previous_E_writer():
    slots,f=rank_slots((8,4,3));d=prepare((8,4,3),slots,f)
    with pytest.raises(PipelineHazard,match='Unready opposite/current'):
        simulate(d,workers=16,h_publish_before_parent=True,e_publication_steps=2,negative='H-fence-does-not-cover-E')


def test_skipping_unaligned_last_E_publication_is_rejected():
    extent=(25,31,47);counts=tuple((n+17+b-1)//b for n,b in zip(extent,(7,15,31),strict=True))
    slots,f=rank_slots(counts);d=prepare(counts,slots,f)
    with pytest.raises(PipelineHazard,match='Unready opposite/current'):
        simulate(d,workers=16,h_publish_before_parent=True,e_publication_steps=4,tile_windows=windows(counts,extent,16),negative='omit-last-E-publication')


@pytest.mark.parametrize('group',[2,4])
def test_source_H_fence_unconditional_all_writers_and_last_E_mandatory(group):
    p=Path('.cache/cuda-five-hour-2026-11-13-1059/persistent-paired-special-phase-v5/source/src/native_persistent.cuh')
    if not p.exists():pytest.skip('Private true-phase source unavailable')
    old=p.read_text();new=e_batch(old,group)
    assert '  __threadfence(); // Every H/psiH/observationH/trailH writer publishes before R0.' in new
    assert 'if(ShouldPublishE(n,Fast?Depth:window[1]))__threadfence();' in new
    assert 'if(!threadIdx.x&&ShouldPublishE(n,Fast?Depth:window[1]))PublishAndWait(2*n,2*(Fast?Depth:window[1]));' in new
    assert 'return step%EPublicationSteps==0||step==last;' in new
    assert old.count('__syncthreads()')==new.count('__syncthreads()')
    assert [line for line in old.splitlines() if 'AdvanceNativeCell<' in line]==[line for line in new.splitlines() if 'AdvanceNativeCell<' in line]
