"""Independent word/version and hazard evidence for no-private field lowering."""

from pathlib import Path

import pytest

from scripts.patch_cuda_traced_lifetime_shared_fields import transform
from scripts.prove_cuda_traced_lifetime_shared_fields import NEGATIVES, Hazard, prove_hazards, simulate


@pytest.mark.parametrize("first", [1, 4, 13])
@pytest.mark.parametrize("seed", [7, 29])
def test_adversarial_warp_orders_retain_all_canonical_field_versions(first, seed):
    result = simulate(first=first, steps=2, seed=seed)
    assert result["private_field_arrays"] == 0
    assert result["actual_field_and_clock_checks"] > 200000
    assert result["loop_barriers"] == 4


@pytest.mark.parametrize("negative", NEGATIVES)
def test_unsafe_schedule_mutation_is_detected(negative):
    with pytest.raises(Hazard):
        simulate(first=1, steps=2, seed=7, negative=negative)


def test_perword_conflicts_are_same_owner_and_trail_retirement_is_nonvacuous():
    result = prove_hazards()
    assert len(result) == 16
    assert all(case["same_owner_ordered_conflicts"] > 0 for case in result)
    assert all(case["old_publication_future_lowX_H_writer_overlap"] > 0 for case in result)


@pytest.mark.parametrize("scope", [1, 2, 3])
def test_lowered_source_has_no_private_fields_and_retires_trails_before_PF2(scope):
    source = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src/asymmetric_native.cuh")
    if not source.is_file():
        pytest.skip("Exact frozen research source required")
    parent = source.read_text()
    changed, proof = transform(parent, scope=scope)
    begin = changed.index("void PermanentFamilyBody(")
    end = changed.index("void UniformBody(", begin)
    body = changed[begin:end]
    assert "float h[3]" not in body and "float e[3]" not in body
    assert "h[c][j]" not in body and "e[c][j]" not in body
    assert body.index("PF1 publishes") < body.index("// PF1 retired all old-E neighbors")
    assert body.index("p.trails[bases[0]") < body.index("// PF2 publishes")
    assert proof["source_bodies_and_canonical_cell_header_unchanged"]
    assert proof["permanent_private_field_floats_per_thread"] == 0
    marker = "// Private source-only storage schedule; source-bound"
    assert parent[parent.index(marker):parent.index("inline bool MakeSource")] == changed[changed.index(marker):changed.index("inline bool MakeSource")]


@pytest.mark.parametrize("scope", [0, 4, 7])
def test_initial_scope_does_not_admit_observers_or_unknown_classes(scope):
    with pytest.raises(ValueError, match="bulk/NoIO"):
        transform("", scope=scope)
