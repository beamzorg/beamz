"""Safe cached-read speculation is source-bound and cannot speculate psi."""

import re
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.cuda_space_time_ir import IllegalSchedule
from scripts.cuda_trace_cached_read_facts import EXPECTED, certify, prove_addresses
from scripts.generate_cuda_native_phase import generate
from scripts.validate_cuda_native_phase_host import validate


def test_all_actual_cached_read_addresses_are_bounded_with_or_without_swizzle():
    proof = prove_addresses()
    assert proof["checked_cached_reads"] == 147456 and proof["both_swizzles"]


def test_only_raw_derivative_guards_change_and_mutable_state_guards_remain():
    old, new = generate(), generate(cached_read_fact=EXPECTED)
    assert old["source"].count("float derivative_") == 12
    assert new["source"].count("const float derivative_") == 12
    assert "if(valid_0){derivative_" not in new["source"]
    for pattern in (r"env.LoadLocal\((\d+),", r"env.StoreLocal\((\d+),"):
        assert re.findall(pattern, old["source"]) == re.findall(pattern, new["source"])
    assert new["source"].count("env.LocalActive(") == old["source"].count("env.LocalActive(") == 6
    assert new["source"].count("if(valid_") == old["source"].count("if(valid_") - 12
    assert new["proof"]["cached_read_speculation"]["fact"]["consumer_sha256"] == EXPECTED.consumer_sha256


def test_forged_or_changed_callback_certificate_rejects():
    with pytest.raises(IllegalSchedule):
        generate(cached_read_fact=replace(EXPECTED, difference_sha256="0" * 64))
    with pytest.raises(IllegalSchedule):
        certify("auto read=[](){return global_phi[0];};")


def test_actual_frozen_consumer_callbacks_are_the_reviewed_source():
    path = Path(__file__).resolve().parents[2] / ".cache/cuda-five-hour-2026-11-13/runtime-joint-phase-v1/source/src/asymmetric_native.cuh"
    if not path.exists():
        pytest.skip("Immutable local research consumer not distributed")
    source = path.read_text()
    assert certify(source) == EXPECTED
    with pytest.raises(IllegalSchedule):
        certify(source.replace("return frame[c*PAD+Slot", "return global_profile[c*PAD+Slot", 1))


@pytest.mark.parametrize("inactive,zero,metallic", [(False, False, 0), (False, False, 63), (True, True, 63)])
def test_actual_speculated_header_matches_independent_scalar_oracle(tmp_path, inactive, zero, metallic):
    if not shutil.which("c++"):
        pytest.skip("C++20 compiler required")
    result = validate(tmp_path, inactive=inactive, zero=zero, metallic=metallic, depth=5, cached_read_fact=EXPECTED)
    assert result["run_exit_code"] == 0 and '"errors":0' in result["stdout"]
