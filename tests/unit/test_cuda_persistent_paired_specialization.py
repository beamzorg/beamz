"""Policy specialization keeps paired field ownership/arithmetic unchanged."""
from pathlib import Path

import pytest

from scripts.patch_cuda_native_persistent_paired_specialize import paired_specialize
from scripts.patch_cuda_native_persistent import clone_body
from scripts.patch_cuda_native_persistent_batch import batch
from scripts.patch_cuda_native_persistent_relaxed_poll import relaxed_poll
from scripts.patch_cuda_native_persistent_upstream_floor import floor_cache


def baseline():
    p=Path('.cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src/asymmetric_native.cuh')
    if not p.exists():pytest.skip('Frozen parent unavailable')
    template=Path('scripts/cuda_native_persistent.cuh').read_text()
    return floor_cache(relaxed_poll(batch(template.replace('// BEAMZ_PERSISTENT_PAIRED_BODY_INSERT',clone_body(p.read_text())),4)))


@pytest.mark.parametrize('noio',[False,True])
def test_paired_specialization_retains_all_cell_nodes_barriers_and_storage(noio):
    old=baseline();new=paired_specialize(old,noio)
    for phase in (0,1):
        for c in range(3):
            token=f'AdvanceNativeCell<{phase},{c},false>'
            assert new.count(token)==old.count(token)
    assert new.count('__syncthreads()')==old.count('__syncthreads()')
    for token in ['static_assert(CZ==8&&CY==16&&CX==32&&T==1024&&LocalZ==4&&YGroups==1',
                  'float*ey_shared=frame+2*PAD;', 'float*snapshots=frame+3*PAD;',
                  'StageSourcePendingH<Depth,SupportSafe,T>',
                  'if(ShouldPublish(n,Fast?Depth:window[1]))__threadfence();']:
        assert token in new
    # Full inner operation schedule is copied bytewise; only its env template
    # and outer startup/publication last-step selection differ.
    a=old.index('  {const int step=*phase_step,');b=old.index('  if(ShouldPublish(n,window[1]))',a)
    assert old[a:b] in new


def test_paired_bulk_uses_fast_startup_wait_and_never_reads_uninitialized_window():
    new=paired_specialize(baseline())
    assert 'if constexpr(Fast){if(!C().abort_local)WaitUpstream(1);}' in new
    assert 'PublishAndWait(n,Fast?Depth:window[1]);' in new
    assert 'const uint8_t flags=p.flags[tile];' in new
    assert 'PersistentPairedBody<Depth,true,0,false>(p);' in new
    assert 'PersistentPairedBody<Depth,false,5,false>(p);' in new
    assert 'PersistentUniformBody' not in new


def test_phase_specialization_keeps_fixed_controller_and_paired_FP_graph():
    p=Path('.cache/cuda-five-hour-2026-11-13-1059/persistent-p5-phase-ready-v4/source/src/native_persistent.cuh')
    if not p.exists():pytest.skip('Fixed private phase source unavailable')
    old=p.read_text();new=paired_specialize(old,True)
    assert 'if(C().abort_local)return;floor=seen<floor?seen:floor;' in new
    assert 'PublishAndWait(2*n-1,2*(Fast?Depth:window[1]));' in new
    assert 'PublishAndWait(2*n,2*(Fast?Depth:window[1]));' in new
    assert 'if constexpr(Fast){if(!C().abort_local)WaitUpstream(0);}' in new
    assert 'WaitUpstream(2*(window[0]-1));' in new
    for specialization in ('true,0,false','false,1,true','false,5,false'):
        assert f'PersistentPairedBody<Depth,{specialization}>(p);' in new
    assert [line for line in old.splitlines() if 'AdvanceNativeCell<' in line]==[line for line in new.splitlines() if 'AdvanceNativeCell<' in line]
    assert old.count('__syncthreads()')==new.count('__syncthreads()')
    # All explicit owned H and paired E publication statements stay exact.
    for line in old.splitlines():
        if 'p.trails[bases[0]' in line or 'hf.data[' in line or 'ef.data[' in line:assert line in new
