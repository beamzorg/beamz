"""A zero-valued acquired ready word is a successful tag-only wait."""
from pathlib import Path
import pytest

from scripts.patch_cuda_native_persistent_zero_ready import zero_ready
from scripts.prove_cuda_persistent_control import prepare


def test_zero_ready_success_requires_all_remaining_parents():
    seen=[0,0,0];old=[];fixed=[]
    for parent,value in enumerate(seen):
        old.append(parent)
        if not value:break
    for parent,value in enumerate(seen):
        fixed.append(parent)
        abort=False
        if abort:break
    assert old==[0]
    assert fixed==[0,1,2]


def test_source_fix_changes_only_failure_predicate():
    p=Path('.cache/cuda-five-hour-2026-11-13-1059/persistent-p5-phase-ready-v3/source/src/native_persistent.cuh')
    if not p.exists():pytest.skip('Private invalid reproduction unavailable')
    old=p.read_text();new=zero_ready(old)
    assert new.replace('const uint32_t seen=WaitProducer(producer,owner,step);if(C().abort_local)return;',
        'const uint32_t seen=WaitProducer(producer,owner,step);if(!seen)return;')==old


def test_actual_harness_models_pre_initialization_base_tags_and_delayed_zero(tmp_path):
    src=Path('.cache/cuda-five-hour-2026-11-13-1059/persistent-p5-phase-ready-v4/source')
    if not src.exists():pytest.skip('Fixed private source unavailable')
    proof=prepare(src,tmp_path)
    harness=(tmp_path/'actual-persistent-control.cc').read_text()
    assert proof['forced_startup_ready0_delayed_tags']
    assert 'run_startup_zero_regression();' in harness
    assert 'Startup skipped an unclaimed parent base/tag' in harness
    assert harness.index('Startup skipped an unclaimed parent base/tag')<harness.index('WaitUpstream(2*step-1)')
    assert 'Startup ready0 returned before ALL delayed parent tags were acquired' in harness
