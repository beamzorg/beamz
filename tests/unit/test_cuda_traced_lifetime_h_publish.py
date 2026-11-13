"""Actual graph-footprint and moving ghost counterexamples for H publication."""

from pathlib import Path

import pytest

from scripts.patch_cuda_traced_lifetime_h_publish import (
    CONSUMERS,
    PUBLICATION,
    generated_publication,
    prove_publication,
    shared_reads,
    transform,
)
from scripts.patch_cuda_traced_lifetime_warp_x import PARENTS, transform as warp_transform


def parent_files(mode):
    directory = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src")
    if not directory.is_dir():
        pytest.skip("Exact frozen research source required")
    parent = {name: (directory / name).read_bytes() for name in PARENTS}
    return parent if mode == "none" else warp_transform(parent, mode)


@pytest.mark.parametrize("mode", list(CONSUMERS))
def test_full_ring_thread_owner_proof_and_exact_live_footprint(mode):
    result = prove_publication(mode)
    expected = 10416 if mode in ("e", "both") else 11286
    assert result["total_published_h_words_per_step"] == expected
    assert result["checked_single_writer_payloads"] == 32 * expected
    assert result["dense_private_ghost_state_and_dense_E_shared_publication_retained"]


@pytest.mark.parametrize("mode", list(CONSUMERS))
def test_missing_hy_zghost_store_is_detected(mode):
    with pytest.raises(ValueError, match="misses a consumed Hy"):
        prove_publication(mode, omit_z0_hy=True)


def test_sparse_E_ghost_publication_would_break_the_next_moving_window():
    with pytest.raises(ValueError, match="next-window ghost"):
        prove_publication(sparse_e_ghosts=True)


@pytest.mark.parametrize("mode", list(CONSUMERS))
def test_exact_source_inverse_preserves_all_other_schedule_and_physics(mode):
    parent = parent_files(mode)
    changed = transform(parent, mode)
    original = parent["asymmetric_native.cuh"].decode()
    result = changed["asymmetric_native.cuh"].decode()
    assert result.replace(generated_publication(mode), PUBLICATION, 1) == original
    assert result.count("__syncthreads()") == original.count("__syncthreads()")
    for name in parent:
        if name != "asymmetric_native.cuh":
            assert changed[name] == parent[name]


def test_warp_mode_must_match_actual_input_lowering():
    with pytest.raises(ValueError, match="Exact selected52865/warp"):
        transform(parent_files("none"), "both")


def test_X_neighbors_only_disappear_after_E_warp_lowering():
    assert (0, 0, -1) in shared_reads("h")["Hy"]
    assert (0, 0, -1) not in shared_reads("e")["Hy"]
