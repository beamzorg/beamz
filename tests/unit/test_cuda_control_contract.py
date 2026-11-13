import hashlib

import pytest

from scripts.run_cuda_plane_experiment import verify


def make_plan(path, intended_sha):
    return {"baseline": str(path),
            "baseline_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "expected_control_sha256": intended_sha,
            "dependencies": {}, "fixture_hashes": {}, "workers": []}


def test_wrong_parent_rejected_even_when_path_and_advertised_hash_agree(tmp_path):
    weaker = tmp_path / "compact.so"
    weaker.write_bytes(b"weaker compact control")
    intended = hashlib.sha256(b"strong Observer control").hexdigest()
    with pytest.raises(ValueError, match="control contract"):
        verify(make_plan(weaker, intended))


def test_control_binary_replacement_rejected_before_next_process(tmp_path):
    control = tmp_path / "Observer.so"
    control.write_bytes(b"pinned Observer control")
    plan = make_plan(control, hashlib.sha256(control.read_bytes()).hexdigest())
    verify(plan)
    control.write_bytes(b"different bytes at the same path")
    with pytest.raises(ValueError, match="control contract"):
        verify(plan)


def test_unbound_historical_plan_requires_explicit_contract_before_rerun(tmp_path):
    control = tmp_path / "control.so"
    control.write_bytes(b"control")
    plan = make_plan(control, hashlib.sha256(control.read_bytes()).hexdigest())
    del plan["expected_control_sha256"]
    with pytest.raises(ValueError, match="explicit control contract"):
        verify(plan)
