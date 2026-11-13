"""Final E is dead only inside a complete macro, never across its field output."""
import pytest

from scripts.patch_cuda_traced_exit_elision import prove


@pytest.mark.parametrize("depth", [4, 16])
def test_all_cpml_classes_have_no_final_e_trail_reader(depth):
    p = prove(depth)
    assert p["cpml_axis_classes"] == 8
    assert p["electric_reads_checked"] > 0
    assert p["latest_consumed_electric_version"] == depth - 1


@pytest.mark.parametrize("depth", [4, 16])
def test_extending_consumers_without_final_field_reload_invalidates_elision(depth):
    with pytest.raises(ValueError, match="trail is live"):
        prove(depth, include_next_macro=True)


@pytest.mark.parametrize("depth", [4, 16])
def test_penultimate_e_history_cannot_be_eliminated(depth):
    with pytest.raises(ValueError, match="trail is live"):
        prove(depth, erase_previous=True)
