"""Changing fixed Z owners must not mix the old slot and thread formulas."""

import pytest

from scripts.prove_cuda_contiguous_epsi_map import (
    contiguous_owner,
    point,
    prove_permutation,
)


def test_all8192_colored_states_have_a_complete_owner_address_relabeling():
    proof = prove_permutation()
    assert proof["mapped_words"] == 8192
    assert proof["new_indices_minmax"] == [0, 8191]


@pytest.mark.parametrize("negative", ["old-interleaved-slot", "old-interleaved-thread"])
def test_mixing_old_and_new_mapping_has_concrete_state_collision(negative):
    with pytest.raises(AssertionError, match="aliased new storage"):
        prove_permutation(negative)


def test_actual_contiguous_si_is_j_and_thread_group_owns_four_adjacent_planes():
    assert point(0, 1) == (1, 0, 0)
    assert point(512, 0) == (4, 0, 0)
    assert contiguous_owner((3, 15, 31)) == (511, 3)
    assert contiguous_owner((7, 15, 31)) == (1023, 3)


@pytest.mark.parametrize("p", [(-1, 0, 0), (8, 0, 0), (0, 16, 0), (0, 0, 32)])
def test_cache_envelope_is_not_silently_expanded(p):
    with pytest.raises(ValueError):
        contiguous_owner(p)
