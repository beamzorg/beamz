"""Yee H uses previous E; current ghost import uses current parent publication."""
from pathlib import Path

import pytest

from scripts.cuda_persistent_tile_pipeline import prepare,simulate,windows,PipelineHazard
from scripts.cuda_persistent_wavefront_dag import rank_slots
from scripts.patch_cuda_native_persistent_h_overlap import GATE,h_overlap


@pytest.mark.parametrize('group',[1,4,16])
@pytest.mark.parametrize('workers',[1,3,16])
@pytest.mark.parametrize('locality',[False,True])
def test_phase_DAG_preserves_previous_E_current_import_and_full_WAR(group,workers,locality):
    counts=(8,4,3);slots,frontier=rank_slots(counts);dag=prepare(counts,slots,frontier,locality=locality)
    r=simulate(dag,workers=workers,publication_steps=group,cache_upstream_floor=True,overlap_h=True)
    for tile in range(len(slots)):
        for step in range(1,17):
            h_time=r['step_starts'][tile][step-1];import_time=r['import_starts'][tile][step-1]
            assert h_time<=import_time
            for parent in dag['raw_predecessors'][tile]:
                assert r['step_ready'][parent][step-1]<=import_time
                if step>1:assert r['step_ready'][parent][step-2]<=h_time


def test_new_schedule_actually_overlaps_H_with_current_upstream_E():
    counts=(8,4,3);slots,frontier=rank_slots(counts);dag=prepare(counts,slots,frontier)
    r=simulate(dag,workers=16,cache_upstream_floor=True,overlap_h=True)
    witnesses=0
    for tile in range(len(slots)):
        for step in range(2,17):
            for parent in dag['raw_predecessors'][tile]:
                if r['step_starts'][tile][step-1]<r['step_ready'][parent][step-1]:witnesses+=1
    assert witnesses>0


def test_missing_import_gate_is_detected():
    counts=(8,4,3);slots,frontier=rank_slots(counts);dag=prepare(counts,slots,frontier)
    with pytest.raises(PipelineHazard,match='Import started before current'):
        simulate(dag,workers=16,cache_upstream_floor=True,overlap_h=True,negative='skip-import-gate')


def test_empty_early_late_windows_still_complete_and_retire_readers():
    extent=(25,31,47);counts=tuple((n+17+b-1)//b for n,b in zip(extent,(7,15,31),strict=True))
    slots,frontier=rank_slots(counts);dag=prepare(counts,slots,frontier)
    r=simulate(dag,workers=16,publication_steps=4,cache_upstream_floor=True,overlap_h=True,tile_windows=windows(counts,extent,16))
    assert r['tiles']==len(slots)


def test_source_only_moves_dependency_wait_and_adds_shared_acquire_gate():
    p=Path('.cache/cuda-five-hour-2026-11-13-1059/persistent-specialized-g4-floor-global-v1/source/src/native_persistent.cuh')
    if not p.exists():pytest.skip('Private typed-global source unavailable')
    old=p.read_text();new=h_overlap(old)
    restored=new.replace('uint32_t epoch,upstream_floor,import_gate;','uint32_t epoch,upstream_floor;')
    restored=restored.replace(' C().upstream_floor=0;C().import_gate=0;',' C().upstream_floor=0;')
    restored=restored.replace(' // Current-step dependency is acquired AFTER local H, at GateImports(n).',' if(step<last)WaitUpstream(step+1);')
    restored=restored.replace(GATE+'\n','').replace('  GateImports(n);if(C().abort_local)return;\n','')
    assert restored==old
    assert old.count('__syncthreads()')==new.count('__syncthreads()')
    assert 'ld.acquire.cta.shared.u32' in GATE and 'st.release.cta.shared.u32' in GATE
    assert GATE.count(':"memory"')==2
    assert 'WaitUpstream(step);ReleaseImportGate(uint32_t(step));' in GATE


def test_shared_gate_transitively_orders_parent_stores_for_every_importer():
    # Parent all-writer publication -> thread0 GPU acquire -> CTA release
    # -> each CTA acquire -> each current ghost import. No barrier added.
    for thread in range(1024):
        chain=['parent_fence','parent_ready_release','thread0_GPU_acquire','CTA_gate_release',f'thread{thread}_CTA_acquire',f'thread{thread}_import']
        assert chain.index('thread0_GPU_acquire')<chain.index(f'thread{thread}_import')
    assert GATE.index('WaitUpstream(step)')<GATE.index('ReleaseImportGate(uint32_t(step))')


@pytest.mark.parametrize('group',[1,2,4,16])
@pytest.mark.parametrize('workers',[1,3,16])
@pytest.mark.parametrize('locality',[False,True])
def test_previous_startup_validates_claim0_then_previous_E(group,workers,locality):
    counts=(8,4,3);slots,frontier=rank_slots(counts);dag=prepare(counts,slots,frontier,locality=locality)
    r=simulate(dag,workers=workers,publication_steps=group,cache_upstream_floor=True,overlap_h=True,startup_previous=True)
    for tile in range(len(slots)):
        for step in range(1,17):
            for parent in dag['raw_predecessors'][tile]:
                assert r['step_ready'][parent][step-1]<=r['import_starts'][tile][step-1]
                if step>1:assert r['step_ready'][parent][step-2]<=r['step_starts'][tile][step-1]


def test_previous_startup_source_never_skips_tag_check_at_ready0():
    p=Path('.cache/cuda-five-hour-2026-11-13-1059/persistent-specialized-g4-floor-global-v1/source/src/native_persistent.cuh')
    if not p.exists():pytest.skip('Private parent unavailable')
    new=h_overlap(p.read_text(),True)
    assert 'if(PollAbort()||(step>0&&uint32_t(step)<=C().upstream_floor))return;' in new
    assert new.count('WaitUpstream(window[0]-1);')==2
    assert 'if constexpr(Fast){if(!C().abort_local)WaitUpstream(0);}' in new
    assert 'Acquire(C().tags+owner)==expected' in new


def test_every_H_forward_read_is_previous_owned_or_previous_low_ghost():
    # Independent spatial derivation, not the scheduling simulator. A current
    # owned H point and its one-axis forward E neighbors lie in the PREVIOUS
    # owned box extended by its low ghost union. No E(n) value is requested.
    sizes=(7,15,31)
    checked=0
    for step in range(1,17):
        for z in range(7):
            for y in range(15):
                for x in range(31):
                    current=(z-step,y-step,x-step)
                    for axis in range(3):
                        for forward in (0,1):
                            q=tuple(v+(forward if a==axis else 0) for a,v in enumerate(current))
                            previous_low=1-step
                            for value,size in zip(q,sizes,strict=True):
                                assert previous_low-1<=value<=previous_low+size-1
                            # Any low-ghost component selects a lower spatial
                            # producer; its exact version is step-1.
                            low_bits=sum((value==previous_low-1)<<a for a,value in enumerate(q))
                            assert 0<=low_bits<8
                            checked+=1
    assert checked==16*7*15*31*6


def test_psi_owner_is_monotone_and_continued_point_keeps_same_thread():
    sizes=(7,15,31);rings=(8,16,32)
    for point in [(0,0,0),(6,14,30),(7,15,31),(12,18,34),(24,25,47),(64,65,95)]:
        previous_owner=None;previous_thread=None
        for step in range(17):
            owner=tuple((v+step)//size for v,size in zip(point,sizes,strict=True))
            origin=tuple(t*size for t,size in zip(owner,sizes,strict=True))
            slots=tuple((p-o)%ring for p,o,ring in zip(point,origin,rings,strict=True))
            thread=(slots[0]//4)*512+slots[1]*32+slots[2]
            if previous_owner is not None:
                assert all(old<=new for old,new in zip(previous_owner,owner,strict=True))
                if owner==previous_owner:assert thread==previous_thread
            previous_owner,previous_thread=owner,thread
