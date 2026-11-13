"""X-only state placement is guarded by actual CPML shape/activation geometry."""

import pytest

from scripts.prove_cuda_compact_x_epsi import (
    OLD_BYTES,
    TOTAL_BYTES,
    admission,
    index,
    lifecycle,
)


def test_complete_compact_domain_is_injective_bounded_and_fits_predicted64k():
    words = {index(c, z, y, k) for c in range(2) for z in range(8) for y in range(16) for k in range(12)}
    assert words == set(range(3072))
    assert OLD_BYTES + 4 * len(words) == TOTAL_BYTES == 64036 < 65536
    for c in range(2):
        for z in range(8):
            for y in range(16):
                # At most12 active columns; shared bank addresses distinct.
                assert len({(OLD_BYTES // 4 + index(c, z, y, k)) % 32 for k in range(12)}) == 12


@pytest.mark.parametrize("nx", [64, 65, 96, 136, 512, 640])
@pytest.mark.parametrize("depth", [4, 16])
def test_same_packed_column_cross_side_cannot_be_visited_in_one_admitted_macro(nx, depth):
    proof = admission(nx, depth)
    assert proof["equal_color_column_cross_side_separation"] > proof["maximum_trajectory_coordinate_span"]


@pytest.mark.parametrize("nx,depth", [(32, 16), (63, 16), (64, 32), (64, 64)])
def test_stronger_admission_is_explicit_and_depth_does_not_silently_expand(nx, depth):
    with pytest.raises(ValueError):
        admission(nx, depth)


@pytest.mark.parametrize("nx,origin", [(64, 0), (64, 31), (64, 62), (65, 62), (136, 124)])
def test_full_shape_validity_and_actual_first_owned_flush_protocol(nx, origin):
    proof = lifecycle(nx, 16, (48, 48, origin))
    assert proof["active_reads"] > 0 and proof["warm_reads"] > 0
    assert proof["cross_side_word_reuses"] == 0
    assert proof["cross_thread_word_reuses"] == 0
    assert proof["required_new_barriers"] == 0


@pytest.mark.parametrize("negative", ["wrong-color", "wrong-y-stride", "wrong-packed-modulus", "omit-first-owned", "omit-turnover-reload", "omit-flush"])
def test_real_active_recurrences_reject_bad_address_and_lifetime_rules(negative):
    origin = (48, 48, 0) if negative == "wrong-packed-modulus" else (48, 48, 31)
    with pytest.raises(AssertionError):
        lifecycle(origin=origin, negative=negative)


def test_small_domain_exposes_simultaneous_low_high_column_alias():
    with pytest.raises(AssertionError, match="co-owned alias"):
        lifecycle(32, 4, (48, 48, 0), admit=False)


def test_outside_native_depth_cross_thread_reuse_requires_existing_version_retirement():
    proof = lifecycle(64, 64, (80, 80, 62), admit=False)
    assert proof["cross_side_word_reuses"] > 0
    for witness in proof["cross_side_witnesses"]:
        assert witness["current"][1] > witness["previous"][1]
        assert witness["current_owner"] != witness["previous_owner"]
