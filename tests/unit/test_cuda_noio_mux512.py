"""Eight scalar words retain the existing512-thread NoIO point ownership."""

import shutil
from pathlib import Path

import pytest

from scripts.patch_cuda_noio_mux import original_body, point_blocks
from scripts.patch_cuda_noio_mux512 import HELPERS, PARENT, body512, transformed
from scripts.prove_cuda_noio_mux512 import Topology, token_case, validate
from scripts.prove_cuda_half_y_noio import Hazard


SOURCE = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source")


def parent():
    if not SOURCE.exists():
        pytest.skip("Frozen selected52865 required")
    return {n: (SOURCE / "src" / n).read_bytes() for n in PARENT}


def test_exact_existing512_mapping_covers_all_cache_points_and_eight_slots():
    t = Topology(16, 512)
    assert t.point(0, 7) == (7, 0, 0)
    assert t.point(511, 7) == (7, 15, 31)
    points = {t.point(tid, j) for tid in range(512) for j in range(8)}
    assert len(points) == 4096
    assert all(t.owner(t.point(tid, j)) == (tid, j) for tid in range(512) for j in range(8))


def test_only_selected_wrapper_launch_and_cloned_body_change():
    before = parent()
    after, proof = transformed(before)
    old, new = before["asymmetric_native.cuh"].decode(), after["asymmetric_native.cuh"].decode()
    assert original_body(new) == original_body(old)
    assert "__launch_bounds__(512,1) void AsymmetricNoIOUniform1024(" in new
    assert "AsymmetricNoIOUniform1024<Depth><<<uniform_count,512,SharedBytes,stream>>>" in new
    clone = body512(old)
    assert "T==512&&Slots==8&&ZStep==1" in clone
    assert clone.count("__syncthreads()") == original_body(old).count("__syncthreads()")
    for block in point_blocks(clone):
        assert block.count("#pragma unroll 1") == 1
        assert "NoIOSelect8(" in block and "NoIOUpdate8(" in block
        assert "[j]" not in block
    assert new.count(HELPERS) == 1
    assert proof["expected_moving_kernels"] == 62
    for name in PARENT:
        if name != "asymmetric_native.cuh":
            assert after[name] == before[name]


def test_wrong_parent_is_rejected():
    with pytest.raises(ValueError, match="Exact selected52865"):
        transformed({n: b"wrong" for n in PARENT})


@pytest.mark.parametrize("version", (1, 4, 16, 32))
def test_trace_tokens_preserve_both_phases_all_eight_banks_and_Z_neighbors(version):
    result = token_case(version)
    assert result["result"] == "pass"
    assert result["actual_traced_field_read_checks"] == 97650


@pytest.mark.parametrize("negative", ("wrong-select-slot", "wrong-update-slot", "wrong-Z-neighbor"))
def test_incorrect_private_and_shared_Z_routes_fail(negative):
    with pytest.raises(Hazard):
        token_case(1, negative=negative)


def test_actual_CPP_helpers_preserve_all_eight_words_including_nan_and_signedzero(tmp_path):
    if not shutil.which("c++"):
        pytest.skip("Host C++ required")
    after, _ = transformed(parent())
    source = tmp_path / "source" / "src"
    source.mkdir(parents=True)
    for name, value in after.items():
        (source / name).write_bytes(value)
    result = validate(source.parent, tmp_path / "oracle", versions=(1, 16, 32), seeds=(13,))
    bits = result["actual_CXX_bits"]
    assert bits["errors"] == 0
    assert bits["select_bit_checks"] == 32768
    assert bits["update_and_unselected_word_bit_checks"] == 262144
    assert bits["signed_zero_checks"] > 0 and bits["NaN_payload_checks"] > 0
    assert len(result["negative_controls_rejected"]) == 3
