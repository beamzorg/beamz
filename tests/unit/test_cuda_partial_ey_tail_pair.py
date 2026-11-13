"""Independent reader tags for last-two-plane publication batching."""

import pytest

from scripts.prove_cuda_partial_ey_lifetimes import (
    Hazard,
    Layout,
    collective_proof,
    simulate,
)


@pytest.mark.parametrize("threads", [1024, 512, 256])
@pytest.mark.parametrize("version", [2, 7, 16, 31])
def test_tail_pair_preserves_versions_under_cross_warp_run_ahead(threads, version):
    result = simulate(Layout(8, 32, threads), version=version, seed=37, tail_pair=True)
    assert result["barriers_per_step"] == 10
    assert result["saved_current_ExEz_floats"] == 4 * (1024 // threads)
    assert result["checked_field_reads"] > 100000


@pytest.mark.parametrize("negative", ["tail-no-second-publication", "tail-save-after-H-publish",
                                      "tail-restore-before-RL", "tail-overwrite-first-result"])
def test_missing_last_uses_and_publications_have_actual_reader_failures(negative):
    with pytest.raises(Hazard):
        simulate(Layout(8, 32, 1024), version=2, seed=13, tail_pair=True, negative=negative)


def test_fusion_scope_rejects_cross_chunk_Z_snapshot_geometry():
    with pytest.raises(ValueError):
        simulate(Layout(16, 16, 1024), tail_pair=True)


def test_required_ghost_warp_participation_remains_fullmask():
    assert len(collective_proof()["inside-owned-branch_negative_witnesses"]) == 32
