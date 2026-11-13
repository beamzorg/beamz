"""Four-scalar NoIO mux preserves point ownership, versions and word bits."""

import shutil
from pathlib import Path

import pytest

from scripts.patch_cuda_noio_mux import HELPERS, PARENT, mux_body, original_body, point_blocks, transformed
from scripts.prove_cuda_noio_mux import token_case, validate
from scripts.prove_cuda_half_y_noio import Hazard


SOURCE = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source")


def parent():
    if not SOURCE.exists():
        pytest.skip("Frozen selected52865 source required")
    return {n: (SOURCE / "src" / n).read_bytes() for n in PARENT}


def test_unbound_source_cannot_inherit_mux_proof():
    with pytest.raises(ValueError, match="Exact selected52865"):
        transformed({n: b"wrong" for n in PARENT})


def test_only_main_point_loops_roll_and_other_original_bodies_stay_identical():
    before = parent()
    after, proof = transformed(before)
    old, new = before["asymmetric_native.cuh"].decode(), after["asymmetric_native.cuh"].decode()
    assert original_body(new) == original_body(old)
    body = mux_body(old)
    for block in point_blocks(body):
        assert block.count("#pragma unroll 1") == 1
        assert "#pragma unroll\n     for(int c=0;c<3;++c)" in block
        assert "NoIOSelect4(" in block and "NoIOUpdate4(" in block
        assert "[j]" not in block
    assert body.count("__syncthreads()") == original_body(old).count("__syncthreads()")
    assert body.count("for(int j=0;j<Slots;++j)") == original_body(old).count("for(int j=0;j<Slots;++j)")
    assert new.count(HELPERS) == 1
    assert proof["expected_moving_kernels"] == 62
    for name in PARENT:
        if name != "asymmetric_native.cuh":
            assert after[name] == before[name]


@pytest.mark.parametrize("version", (1, 4, 16, 32))
def test_trace_tokens_keep_same_four_bank_owners_across_ring_heads(version):
    result = token_case(version)
    assert result["result"] == "pass"
    assert result["ownership_bijection_checks"] == 4096
    assert result["actual_traced_field_read_checks"] > 50000


@pytest.mark.parametrize("negative", ("wrong-select-slot", "wrong-update-slot"))
def test_incorrect_mux_indices_fail_trace_tokens(negative):
    with pytest.raises(Hazard):
        token_case(1, negative=negative)


def test_actual_CPP_helpers_preserve_signedzero_NaN_and_all_unselected_words(tmp_path):
    if not shutil.which("c++"):
        pytest.skip("Host C++ compiler required")
    before = parent()
    after, _ = transformed(before)
    source = tmp_path / "source" / "src"
    source.mkdir(parents=True)
    for name, value in after.items():
        (source / name).write_bytes(value)
    result = validate(source.parent, tmp_path / "oracle", versions=(1, 16, 32), seeds=(13,))
    bits = result["actual_CXX_bits"]
    assert bits["errors"] == 0
    assert bits["select_bit_checks"] == 16384
    assert bits["update_and_unselected_word_bit_checks"] == 65536
    assert bits["signed_zero_checks"] > 0 and bits["NaN_payload_checks"] > 0
    assert len(result["negative_controls_rejected"]) == 2
