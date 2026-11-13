"""Uniform plane iteration and fixed-register selector admission negatives."""

import pytest

from scripts.cuda_plane_retirement_rolled_binding import (
    END,
    NEW_PREFIX,
    reconstruct_parent,
    selector_certificate,
)


def test_all_components_and_ghost_lanes_keep_uniform_plane_iteration():
    result = selector_certificate()
    assert result["selected_component_tokens"] == 16 * 512 * 3
    assert result["uniform_barrier_participations"] == 17 * 512 * 15
    assert result["selector_matches_per_thread_plane"] == 1


def test_wrong_slot_selector_is_rejected():
    with pytest.raises(ValueError, match="component/slot identity"):
        selector_certificate(wrong_selector=True)


def test_barrier_inside_owned_branch_rejects_ghost_lane_divergence():
    with pytest.raises(ValueError, match="divergent ghost participation"):
        selector_certificate(barrier_inside_owned=True)


def test_changed_constant_k_mapping_cannot_reconstruct_admitted_parent():
    source = NEW_PREFIX.replace("if(j==k)", "if(j==k+1)") + END
    with pytest.raises(ValueError, match="exact constant-k selector changed"):
        reconstruct_parent(source)


def test_dynamic_private_array_index_is_not_admitted_by_rolled_lowering():
    source = NEW_PREFIX + "\nfloat value=h[0][j];\n" + END
    with pytest.raises(ValueError, match="Dynamic private H indexing"):
        reconstruct_parent(source)


def test_partial_hp_substitution_cannot_change_final_or_center_h_values():
    source = NEW_PREFIX + "\nhp[c];hp[field];\n" + END
    with pytest.raises(ValueError, match="selector use changed"):
        reconstruct_parent(source)
