"""CPU-only checks for field-bank bootstrap/final alias legality."""

import itertools

import pytest

from scripts.prove_cuda_field_bank_alias import (
    axis_links,
    factorized_case,
    source_binding,
    token_schedule,
    trace_initial_coverage,
)


@pytest.mark.parametrize("owned", [(7, 15, 31), (15, 15, 15)])
@pytest.mark.parametrize("depth", [4, 16])
def test_actual_staggered_fields_and_rank_interleavings(owned, depth):
    case = factorized_case((19, 23, 29), owned, depth)
    assert len(case["field_cases"]) == 6
    assert trace_initial_coverage(owned) > 0
    result = token_schedule((19, 23, 29), owned, depth, seed=812)
    assert result["final_writes"] == sum(f["words"] for f in case["field_cases"])


@pytest.mark.parametrize("depth", [1, 4, 16])
def test_reader_set_and_rank_inequality_exhaustive(depth):
    for extent, owned in itertools.product(range(1, 18), range(1, 9)):
        links, same = axis_links(extent, owned, depth)
        direct = direct_same = 0
        for p in range(extent):
            readers = [c for c in range(extent + 2) if c * owned - 1 <= p <= c * owned + owned - 1]
            writer = (p + depth) // owned
            assert all(c <= writer for c in readers)
            direct += len(readers)
            direct_same += writer in readers
        assert (links, same) == (direct, direct_same)


@pytest.mark.parametrize("negative", [
    "reverse-final-shift", "unshifted-final-n0",
    "output-upload-or-clear-before-bootstrap", "reverse-rank-order",
])
def test_unsafe_schedules_clobber_seeded_input(negative):
    with pytest.raises(AssertionError, match="initial word overwritten"):
        token_schedule(depth=4, negative=negative)


def test_positive_depth_required():
    with pytest.raises(ValueError):
        axis_links(9, 7, 0)


def test_source_binding_requires_reviewed_consumer(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/asymmetric_native.cuh").write_text("// changed schedule\n")
    with pytest.raises(ValueError, match="repeat the startup/final/rank-join source review"):
        source_binding(tmp_path)
