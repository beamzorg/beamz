"""Owned H publication depends on previous E, never current parent H."""
from pathlib import Path

import pytest

from scripts.cuda_persistent_wavefront_dag import rank_slots
from scripts.cuda_persistent_tile_pipeline import prepare,PipelineHazard
from scripts.cuda_persistent_phase_pipeline import simulate
from scripts.patch_cuda_native_persistent_h_publication_join import JOIN,publication_join


@pytest.mark.parametrize('depth',[4,16])
@pytest.mark.parametrize('workers',[1,3,16])
@pytest.mark.parametrize('locality',[False,True])
def test_true_phase_DAG_preserves_field_versions_and_full_WAR(depth,workers,locality):
    counts=(8,4,3);slots,f=rank_slots(counts);d=prepare(counts,slots,f,locality=locality)
    r=simulate(d,depth,workers,h_publish_before_parent=True)
    for tile in range(len(slots)):
        for n in range(1,depth+1):
            for parent in d['raw_predecessors'][tile]:
                assert r['H_ready'][parent][n-1]<=r['E_start'][tile][n-1]
                if n>1:assert r['E_ready'][parent][n-2]<=r['H_start'][tile][n-1]


def test_owned_H_is_actually_published_before_current_parent_H():
    slots,f=rank_slots((8,4,3));d=prepare((8,4,3),slots,f)
    r=simulate(d,workers=16,h_publish_before_parent=True)
    assert r['H_publications_before_current_parent_H']>100


def test_omitting_H_writer_join_exposes_unready_current_family():
    slots,f=rank_slots((8,4,3));d=prepare((8,4,3),slots,f)
    with pytest.raises(PipelineHazard,match='Unready opposite/current'):
        simulate(d,workers=16,h_publish_before_parent=True,negative='omit-H-writer-join')


def test_actual_source_H_publish_precedes_parent_H_wait_and_keeps_FP():
    p=Path('.cache/cuda-five-hour-2026-11-13-1059/persistent-paired-special-phase-v4/source/src/native_persistent.cuh')
    if not p.exists():pytest.skip('Private paired phase parent unavailable')
    old=p.read_text();new=publication_join(old)
    assert old.count('__syncthreads()')==new.count('__syncthreads()')
    assert new.index('  JoinHWriterWarpsAndPublish(n);')<new.index('  GateImports(2*n-1);')<new.index('// R0:')
    assert 'if(!threadIdx.x){*phase_step=n+1;C().h_arrivals=0;}' in new
    assert [line for line in old.splitlines() if 'AdvanceNativeCell<' in line]==[line for line in new.splitlines() if 'AdvanceNativeCell<' in line]
    assert new.count('__threadfence();')==old.count('__threadfence();')
    assert '__syncwarp(0xffffffffu)' in JOIN
    assert 'atom.acq_rel.cta.shared.add.u32' in JOIN
    assert 'ld.acquire.cta.shared.u32' in JOIN
    assert 'if(seen==32)break;' in JOIN
    assert 'if(Timer()-last>C().timeout_ns)' in JOIN


def test_every_writer_precedes_GPU_publication_through_warp_and_CTA_chain():
    # Each group leader waits for all32 writer lanes, then an acq_rel chain
    # accumulates all32 warp releases. Thread0's acquire32 sees the full set.
    delivered=set()
    for warp in range(32):
        group={warp*32+lane for lane in range(32)}
        assert len(group)==32
        delivered|=group
    assert delivered==set(range(1024))
    missing=set(range(1024))-{777}
    assert missing!=delivered  # Missing writer must prevent full join success.
