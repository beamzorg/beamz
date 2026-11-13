"""Generic cube fallback mappings retain actual opposite-family neighbor reads."""

import pytest

from scripts.prove_cuda_cube_uniform_mapping import point, simulate


@pytest.mark.parametrize("threads", [256, 512, 1024])
def test_every4096_cache_cell_has_exactly_one_fixed_owner(threads):
    positions = {point(threads, t, j) for t in range(threads) for j in range(4096 // threads)}
    assert len(positions) == 4096
    assert min(p[0] for p in positions) == 0 and max(p[0] for p in positions) == 15


@pytest.mark.parametrize("threads", [256, 512, 1024])
@pytest.mark.parametrize("version", [1, 2, 15, 16])
def test_actual_affine_field_versions_across_interleaved_slots(threads, version):
    proof = simulate(threads, version)
    assert proof["slots"] == 4096 // threads
    assert proof["field_scalar_lower_bound"] == 6 * (4096 // threads)
    assert proof["checked_affine_reads"] == 101250 and proof["barriers"] == 4


@pytest.mark.parametrize("negative", ["old-ZStep", "old-ZBase", "early-H-publication", "early-E-publication"])
def test_wrong_mapping_or_missing_family_retirement_exposes_actual_failure(negative):
    with pytest.raises(AssertionError):
        simulate(512, 2, negative)
