"""Rolled ghost imports preserve each of the841 ghost owners and owned words."""

import shutil
from pathlib import Path

import pytest

from scripts.patch_cuda_noio_mux512_ghost import PARENT_CONSUMER, retire_helper, transformed
from scripts.prove_cuda_noio_mux512_ghost import validate


PARENT = Path(".cache/cuda-cyclic-axis-2026-11-13/noio-mux512-v1/source")


def original():
    if not PARENT.exists():
        pytest.skip("Frozen main-only512 mux required")
    return (PARENT / "src/asymmetric_native.cuh").read_text()


def test_exact_parent_binding_rejects_changed_source():
    with pytest.raises(ValueError, match="Exact reviewed"):
        transformed("wrong")


def test_only_ghost_loops_change_and_retirement_keeps_component_unrolling():
    before = original()
    after, proof = transformed(before)
    helper = retire_helper(before)
    assert "#pragma unroll 1\n  for(int j=0;j<Slots;++j)" in helper
    assert "#pragma unroll\n   for(int c=0;c<3;++c)" in helper
    assert "NoIOUpdate8(h[c],j,stage[c*LowXPoints+key])" in helper
    assert "NoIOMux512RetireLow<T,Slots>(low_stage,h,e,n,sy,sx)" in after
    assert after.count("__syncthreads()") == before.count("__syncthreads()")
    assert proof["parent_consumer_sha256"] == PARENT_CONSUMER
    assert proof["expected_moving_kernels"] == 62
    assert proof["exact_source_inverse"]


def test_actual_CPP_ghost_partition_updates_exactly_736_plus105_points(tmp_path):
    if not shutil.which("c++"):
        pytest.skip("Host C++ required")
    text, _ = transformed(original())
    source = tmp_path / "source" / "src"
    source.mkdir(parents=True)
    (source / "asymmetric_native.cuh").write_text(text)
    report = validate(PARENT, source.parent, tmp_path / "oracle", versions=(1, 16, 32))
    bits = report["actual_CXX_ghost_bits"]
    assert bits["errors"] == 0
    assert bits["actual_ghost_and_unselected_word_bit_checks"] == 786432
    assert bits["YZ_points"] == 32 * 736
    assert bits["low_X_points"] == 32 * 105
    assert bits["wrong_head_counterexamples"] > 0
