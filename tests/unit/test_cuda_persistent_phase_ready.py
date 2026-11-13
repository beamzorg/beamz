"""Packed phase counters cannot authorize unready opposite-family imports."""
from pathlib import Path

import pytest

from scripts.cuda_persistent_phase_pipeline import simulate
from scripts.cuda_persistent_tile_pipeline import prepare,windows,PipelineHazard
from scripts.cuda_persistent_wavefront_dag import rank_slots
from scripts.patch_cuda_native_persistent_phase_ready import phase_ready,H_PUBLISH,E_IMPORT


@pytest.mark.parametrize('depth',[4,16])
@pytest.mark.parametrize('workers',[1,3,16])
@pytest.mark.parametrize('locality',[False,True])
def test_actual_phase_DAG_RAW_WAR_and_packed_counters(depth,workers,locality):
    counts=(8,4,3);slots,f=rank_slots(counts);d=prepare(counts,slots,f,locality=locality)
    r=simulate(d,depth,workers)
    for tile in range(len(slots)):
        for n in range(1,depth+1):
            for parent in d['raw_predecessors'][tile]:
                assert r['H_ready'][parent][n-1]<=r['E_start'][tile][n-1]
                if n>1:assert r['E_ready'][parent][n-2]<=r['H_start'][tile][n-1]


def test_E_really_starts_before_current_parent_E_is_ready():
    slots,f=rank_slots((8,4,3));d=prepare((8,4,3),slots,f)
    assert simulate(d,workers=16)['E_starts_before_upstream_E_same_version']>100


def test_mixed_H_packet_would_read_unready_E_and_is_rejected():
    slots,f=rank_slots((8,4,3));d=prepare((8,4,3),slots,f)
    with pytest.raises(PipelineHazard,match='Unready opposite'):
        simulate(d,workers=16,negative='mixed-H-packet')


def test_phase_support_windows_and_empty_tiles_complete():
    extent=(25,31,47);depth=16;counts=tuple((n+17+b-1)//b for n,b in zip(extent,(7,15,31),strict=True))
    slots,f=rank_slots(counts);d=prepare(counts,slots,f)
    r=simulate(d,workers=16,tile_windows=windows(counts,extent,depth))
    assert r['tiles']==len(slots)


def test_source_orders_family_publications_and_late_E_import_correctly():
    base=Path('.cache/cuda-five-hour-2026-11-13-1059/persistent-p5-g4-H-overlap-v2/source')
    if not base.exists():pytest.skip('Private parent unavailable')
    old=(base/'src/native_persistent.cuh').read_text();new,packet=phase_ready(old,(base/'src/asymmetric_native.cuh').read_text())
    assert old.count('__syncthreads()')==new.count('__syncthreads()')
    assert 'mask&=Family==0?7u:56u;' in packet
    for phase in (0,1):
        for c in range(3):assert old.count(f'AdvanceNativeCell<{phase},{c},false>')==new.count(f'AdvanceNativeCell<{phase},{c},false>')
    assert new.index(H_PUBLISH)<new.index('// R0:')<new.index('PublishAndWait(2*n-1,2*window[1])')
    assert new.index('// RLpair')<new.index('PublishAndWait(2*n,2*window[1])')<new.index(E_IMPORT)<new.index('// RE:')
    assert 'WaitUpstream(2*(window[0]-1));' in new
    assert 'Release(C().ready+C().owner,uint32_t(2*Depth));' in new
    assert new.count('ImportPhasePacket<Depth,SupportSafe,1>')==1
    assert 'if(zs==hz||ys==hy||sx==hx){const auto v=ImportPacket<Depth,SupportSafe>' not in new


def test_owned_H_values_are_identical_when_published_before_E():
    # H arithmetic completes before imports. Those imports write ONLY ghost
    # slots. New H publications are restricted to the complement, so each
    # private published word equals the original later per-pair H publication.
    checked=0
    for n in range(1,17):
        hz,hy,hx=(-n-1)&7,(-n-1)&15,(-n-1)&31
        for z in range(8):
            for y in range(16):
                for x in range(32):
                    ghost=z==hz or y==hy or x==hx
                    published=z!=hz and y!=hy and x!=hx
                    assert ghost!=published
                    if published:checked+=3
    assert checked==16*7*15*31*3


def test_exact_ordered_E_graph_and_H_graph_unchanged():
    base=Path('.cache/cuda-five-hour-2026-11-13-1059/persistent-p5-g4-H-overlap-v2/source')
    if not base.exists():pytest.skip('Private parent unavailable')
    old=(base/'src/native_persistent.cuh').read_text();new,_=phase_ready(old,(base/'src/asymmetric_native.cuh').read_text())
    assert [line for line in old.splitlines() if 'AdvanceNativeCell<' in line]==[line for line in new.splitlines() if 'AdvanceNativeCell<' in line]
    start=old.index('  {const int step=*phase_step,');end=old.index('  GateImports(n);',start)
    assert old[start:end] in new
    for plane in ('j0','j1'):
        start=old.index('{const int j='+plane+',zs=zbase+j;')
        end=old.index('\n    if(zs!=hz&&ys!=hy&&sx!=hx){const int surface=',start)
        assert old[start:end] in new
    assert 'if(!threadIdx.x)*phase_step=n+1;' in new
    assert 'ImportPhasePacket<Depth,SupportSafe,1>(p,bases,n,' in new
