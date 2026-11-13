"""Joined cumulative GPU releases replace individual writer fences.

These tests use the reviewed synchronization path. Older missing-writer-fence
negatives describe a different explicit-fence recipe and do not test this one.
"""
from pathlib import Path
import random

import pytest

from scripts.patch_cuda_native_persistent_cumulative_release import cumulative_release,OLD,NEW,ALLOWED
from scripts.patch_cuda_native_persistent_global_atomics import GLOBAL_HELPERS


@pytest.mark.parametrize('variant',[
    'persistent-specialized-g4-floor-global-v1',
    'persistent-paired-special-phase-v5',
    'persistent-paired-special-phase-v5-E2',
    'persistent-paired-special-phase-v5-E4'])
def test_exact_fence_only_inverse_and_mandatory_GPU_releases(variant):
    p=Path('.cache/cuda-five-hour-2026-11-13-1059')/variant/'source/src/native_persistent.cuh'
    if not p.exists():pytest.skip('Private reviewed source unavailable')
    old=p.read_text();new=cumulative_release(old)
    assert new.replace(NEW,OLD)==old
    assert new.count(OLD)==0 and new.count(NEW)==2
    assert GLOBAL_HELPERS in new
    assert old.count('__syncthreads()')==new.count('__syncthreads()')
    assert old.count('__syncwarp(')==new.count('__syncwarp(')
    assert [line for line in old.splitlines() if 'AdvanceNativeCell<' in line]==[line for line in new.splitlines() if 'AdvanceNativeCell<' in line]
    with pytest.raises(ValueError,match='Exact reviewed'):
        cumulative_release(old.replace('st.release.gpu.global.u32','st.relaxed.gpu.global.u32',1))


def reaches(edges,start,target):
    todo=[start];seen=set()
    while todo:
        node=todo.pop()
        if node==target:return True
        if node not in seen:seen.add(node);todo.extend(edges.get(node,()))
    return False


def test_CTA_barrier_and_GPU_release_carries_ALL1024_generic_writes():
    # GPU release is cumulative over the cross-thread CTA happens-before
    # relation. There is intentionally no per-writer GPU fence node.
    edges={('store',i):{'CTA_join'} for i in range(1024)}
    edges.update({'CTA_join':{'GPU_release'},'GPU_release':{'GPU_acquire'},'GPU_acquire':{'cg_read'}})
    assert all(reaches(edges,('store',i),'cg_read') for i in range(1024))
    edges['GPU_release'].clear()
    assert not reaches(edges,('store',777),'cg_read')


@pytest.mark.parametrize('seed',[1,7,317])
def test_warp_RMW_join_cumulative_GPU_release_all_arrival_orders(seed):
    order=list(range(32));random.Random(seed).shuffle(order)
    edges={('store',i):{('warp_sync',i//32)} for i in range(1024)}
    for warp in range(32):edges[('warp_sync',warp)]={('acq_rel_RMW',warp)}
    for a,b in zip(order,order[1:]):edges[('acq_rel_RMW',a)]={('acq_rel_RMW',b)}
    edges[('acq_rel_RMW',order[-1])]={'acquire32'}
    edges.update({'acquire32':{'GPU_release'},'GPU_release':{'GPU_acquire'},'GPU_acquire':{'cg_read'}})
    assert all(reaches(edges,('store',i),'cg_read') for i in range(1024))
    # A missing writer-to-warp join is still a real negative, unlike omitting
    # a redundant individual fence from this intact cumulative chain.
    edges[('store',777)].clear()
    assert not reaches(edges,('store',777),'cg_read')


def test_prior_E_store_joins_next_H_prefix_without_an_individual_fence():
    edges={'E_n_store':{'next_H_warp_sync'},'next_H_warp_sync':{'CTA_RMW_chain'},
        'CTA_RMW_chain':{'leader_acquire32'},'leader_acquire32':{'GPU_release_odd2n1'},
        'GPU_release_odd2n1':{'GPU_acquire_ge_even2n'},'GPU_acquire_ge_even2n':{'read_E_n'}}
    assert reaches(edges,'E_n_store','read_E_n')
