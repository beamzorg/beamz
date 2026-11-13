"""Fail-closed source/order admission, separate from the lifetime interpreter."""

import pytest

from scripts.cuda_plane_retirement_binding import (
    ORDER_ANCHORS,
    braced_body,
    inspect_schedule,
    validate,
)
from scripts.generate_cuda_native_cell import generate_native_cell


def lexical_step():
    # A compact lexical fixture deliberately tests source-order admission only;
    # actual frozen consumer admission is separately run against its full SHA.
    return "\n".join(ORDER_ANCHORS)


def test_actual_site_multiplicity_is_eighteen_not_four():
    result = inspect_schedule(lexical_step())
    assert result["static_barrier_sites"] == 4
    assert result["barriers_per_step"] == 18
    assert [site["executions_per_step"] for site in result["barrier_sites"]] == [1, 15, 1, 1]


def test_brace_extraction_ignores_comments_and_strings():
    source = 'void fun(){ // {\n const char*s="}"; if(true){ f(); } /* } */ } trailing'
    assert braced_body(source, "void fun()") == source[source.index("{") + 1:source.rindex("}")]
    with pytest.raises(ValueError, match="exactly one"):
        braced_body(source + source, "void fun()")
    with pytest.raises(ValueError, match="Unbalanced"):
        braced_body("void fun(){", "void fun()")


@pytest.mark.parametrize("mutation", ["missing_R3", "extra_barrier", "early_final_store", "early_ghost", "unconditional_Plast"])
def test_unsafe_or_changed_emitted_site_orders_are_rejected(mutation):
    step = lexical_step()
    if mutation == "missing_R3":
        step = step.replace(ORDER_ANCHORS[9], "// omitted")
    elif mutation == "extra_barrier":
        step += "\n__syncthreads(); // new site"
    elif mutation == "early_final_store":
        step = step.replace(ORDER_ANCHORS[10], "")
        step = ORDER_ANCHORS[10] + "\n" + step
    elif mutation == "early_ghost":
        step = step.replace(ORDER_ANCHORS[11], "")
        step = ORDER_ANCHORS[11] + "\n" + step
    else:
        step = step.replace("if(j!=CZ-1)__syncthreads();", "__syncthreads();")
    with pytest.raises(ValueError):
        inspect_schedule(step)


def test_changed_cell_or_stale_proof_cannot_admit_fusion():
    cell = generate_native_cell()["source"]
    with pytest.raises(ValueError, match="Canonical native cell changed"):
        validate("", cell + "\n", {})
    with pytest.raises(ValueError, match="Stale plane-retirement proof source"):
        validate("", cell, {"proof_source_sha256": "0" * 64})
