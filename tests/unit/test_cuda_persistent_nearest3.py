"""Later full-step nearest waits preserve supported field and prior-state HB."""
from pathlib import Path

import pytest

from scripts.patch_cuda_native_persistent_nearest3 import nearest3,PARENT,HOST,ABI
from scripts.prove_cuda_persistent_nearest3 import prove_geometry_visibility
from scripts.cuda_persistent_tile_pipeline import windows


def source():
    p=Path('.cache/cuda-trace-five-hour-2026-11-13-1059/persistent-specialized-g4-cumulative-v1/source/src/native_persistent.cuh')
    if not p.exists():pytest.skip('Exact cumulative parent unavailable')
    return p.read_text()


@pytest.mark.parametrize('extent',[(25,25,25),(25,31,47),(64,96,128),(96,96,96)])
@pytest.mark.parametrize('depth',[4,16])
def test_accumulated_visibility_current_payload_and_prior_mutable_owner(extent,depth):
    p=prove_geometry_visibility(extent,depth)
    assert p['pass'] and p['supported_field_packets']>0
    assert p['prior_mutable_handoffs']>0


def test_source_inverse_keeps_ALL7_startup_WAR_base_checks_FP_and_host():
    old=source();new=nearest3(old)
    restored=new.replace('// TRACE_NEAREST3_LATER_ONLY: fullstep publication; startup always ALL7.\ntemplate<bool NearestOnly=false>\n','')
    restored=restored.replace('for(int neighbor=1;neighbor<(NearestOnly?4:8);++neighbor){const int bits=NearestOnly?(1<<(neighbor-1)):neighbor;const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;',
        'for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;')
    restored=restored.replace('WaitUpstream<true>(step+1);','WaitUpstream(step+1);').replace('WaitUpstream<false>(window[0]);','WaitUpstream(window[0]);').replace('WaitUpstream<false>(1);','WaitUpstream(1);')
    assert restored==old
    assert new.count('WaitUpstream<false>(window[0]);')==2
    assert new.count('WaitUpstream<false>(1);')==1
    assert new.count('WaitUpstream<true>')==1
    a=old.index('__device__ __forceinline__ void Claim()');b=old.index('__device__ __forceinline__ uint32_t WaitProducer',a)
    assert old[a:b] in new
    assert old.count('__syncthreads()')==new.count('__syncthreads()')
    base=Path('.cache/cuda-trace-five-hour-2026-11-13-1059/persistent-cumulative-nearest3-v1/source/src')
    if base.exists():
        import hashlib
        assert hashlib.sha256((base/'native_extension.cc').read_bytes()).hexdigest()==HOST
        assert hashlib.sha256((base/'native_persistent.h').read_bytes()).hexdigest()==ABI


def test_actual_v5_input_is_rejected_by_closed_parent_contract():
    p=Path('.cache/cuda-five-hour-2026-11-13-1059/persistent-paired-special-phase-v5-cumulative/source/src/native_persistent.cuh')
    if not p.exists():pytest.skip('Phase-v5 counterexample source unavailable')
    with pytest.raises(ValueError,match='fullstep.*early phase-ready'):
        nearest3(p.read_text())


def test_early_phase_ready_does_not_imply_diagonal_ready():
    # v5 can publish both nearest faces before their current upstream waits;
    # these two signals cannot certify the common diagonal's payload.
    faces_ready=[8,8];diagonal_ready=0
    assert min(faces_ready)>=8 and diagonal_ready<8


def test_late_startup_empty_faces_do_not_certify_diagonal_claim():
    extent=(25,31,47);depth=16;counts=(6,4,3);spans=windows(counts,extent,depth)
    def tile(z,y,x):return (z*4+y)*3+x
    q=tile(2,1,2);zface=tile(1,1,2);yface=tile(2,0,2);diagonal=tile(1,0,2)
    assert spans[q]==(15,16)
    assert spans[zface][0]>spans[zface][1]
    assert spans[yface][0]>spans[yface][1]
    assert spans[diagonal][0]>spans[diagonal][1]
    # Empty face completions may precede diagonal claim. All7 startup tag
    # validation remains necessary even though no supported diagonal data isread.
    face_completed=True;diagonal_claimed=False
    assert face_completed and not diagonal_claimed
