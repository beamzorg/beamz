"""Grouped release events retain all versions until their consumers retire."""
from pathlib import Path

import pytest

from scripts.cuda_persistent_wavefront_dag import rank_slots
from scripts.cuda_persistent_tile_pipeline import prepare,simulate,windows
from scripts.patch_cuda_native_persistent import clone_body
from scripts.patch_cuda_native_persistent_batch import batch
from scripts.patch_cuda_native_persistent_specialize import specialize


@pytest.mark.parametrize('group',[1,2,4,16])
@pytest.mark.parametrize('locality',[False,True])
@pytest.mark.parametrize('workers',[1,3,16])
def test_grouped_publication_handles_RAW_WAR_delayed_workers(group,locality,workers):
    counts=(8,4,3);slots,frontier=rank_slots(counts)
    dag=prepare(counts,slots,frontier,locality=locality)
    result=simulate(dag,workers=workers,publication_steps=group)
    assert result['tiles']==len(slots)
    assert result['in_body_publications']==len(slots)*max(1,16//group)


@pytest.mark.parametrize('group',[1,2,4,16])
def test_unaligned_last_window_and_empty_tiles(group):
    extent=(25,31,47);depth=16
    counts=tuple((n+1+depth+b-1)//b for n,b in zip(extent,(7,15,31),strict=True))
    slots,frontier=rank_slots(counts);dag=prepare(counts,slots,frontier)
    spans=windows(counts,extent,depth)
    result=simulate(dag,workers=16,publication_steps=group,tile_windows=spans)
    assert result['tiles']==len(slots)
    for tile,(first,last) in enumerate(spans):
        if first<=last:assert result['step_ready'][tile][last-1] is not None


@pytest.mark.parametrize('group',[1,2,4,16])
def test_actual_generated_both_bodies_fence_and_release_same_predicate(group):
    parent=Path('.cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src/asymmetric_native.cuh')
    if not parent.exists():pytest.skip('Frozen parent unavailable')
    original=parent.read_text()
    template=Path('scripts/cuda_native_persistent.cuh').read_text().replace('// BEAMZ_PERSISTENT_PAIRED_BODY_INSERT',clone_body(original))
    specialized,_=specialize(template,original);generated=batch(specialized,group)
    assert specialized.count('__syncthreads()')==generated.count('__syncthreads()')
    assert f'constexpr int PublicationSteps={group};' in generated
    assert 'return step%PublicationSteps==0||step==last;' in generated
    assert 'if(ShouldPublish(step,last))Release(C().ready+C().owner,uint32_t(step));' in generated
    assert 'if(step<last)WaitUpstream(step+1);' in generated
    assert 'if(ShouldPublish(n,window[1]))__threadfence();' in generated
    assert 'if(ShouldPublish(n,Fast?Depth:window[1]))__threadfence();' in generated


def test_publication_groups_cannot_expose_future_or_unfenced_versions():
    # The release at endpoint r orders every store in [previous+1,r]. A
    # consumer acquiring ready>=s can use every requested version <=s.
    for group in [1,2,4,16]:
        for first in range(1,17):
            for last in range(first,17):
                releases=[n for n in range(first,last+1) if n%group==0 or n==last]
                assert releases[-1]==last
                for requested in range(first,last+1):
                    endpoint=next(r for r in releases if r>=requested)
                    assert requested<=endpoint<=last
