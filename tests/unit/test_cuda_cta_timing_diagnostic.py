"""Exact body/source bindings and honest, storage-safe interval diagnostics."""

from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_cuda_cta_intervals import ring_dependencies, summarize, union_length, valid_component_work
from scripts.cuda_persistent_wavefront_dag import rank_slots
from scripts.patch_cuda_cta_timing_diagnostic import PARENT, transformed
from scripts.prove_cuda_cta_timing_diagnostic import crosscheck_DAG, protocol, prove


def sample():
    arrays = {"start_ns": np.asarray([10, 30], np.uint64), "end_ns": np.asarray([20, 40], np.uint64),
              "smid": np.asarray([0, 5], np.int32), "tile_flags": np.asarray([2, 17], np.uint8),
              "tile_ids": np.asarray([0, 1], np.int32), "slots": np.asarray([0, 0], np.int32),
              "rank_offsets": np.asarray([0, 1, 2], np.int32)}
    meta = {"counts_zyx": (1, 1, 2), "physical_SM_count": 3, "max_frontier": 1, "epochs": 4, "depth": 16}
    return arrays, meta


def test_interval_union_does_not_double_count_overlapping_resident_CTAs():
    assert union_length([0, 5, 20], [10, 15, 25], 0, 30) == 20
    assert union_length([0, 5, 20], [10, 15, 25], 7, 22) == 10


def test_SM_presence_uses_physical_count_and_sparse_start_SMIDs_without_utilization_claim():
    arrays, meta = sample()
    result = summarize(arrays, meta)
    assert result["observed_start_SMIDs"] == [0, 5]
    assert result["SM_count_without_recorded_start"] == 1
    assert result["last_macro_CTA_envelope_ns"] == 30
    assert result["recorded_CTA_presence_SM_ns"] == 20
    assert result["fraction_SM_envelope_with_recorded_CTA_present"] == pytest.approx(2 / 9)
    assert result["rank_order_violations"] == result["conservative_dependency_timestamp_violations"] == 0
    assert result["fixed_duration_unlimited_workers_DAG_critical_path_ns"] == 20
    assert result["diagnostic_only"] and result["DAG_duration_model_is_not_a_hardware_bound"]


@pytest.mark.parametrize("mutation", ["missing-start", "backwards-end", "missing-SMID", "wrong-shape", "duplicate-tile", "wrong-rank-range"])
def test_incomplete_or_misbound_interval_records_fail_closed(mutation):
    arrays, meta = sample()
    if mutation == "missing-start":
        arrays["start_ns"][0] = 0
    elif mutation == "backwards-end":
        arrays["end_ns"][1] = 29
    elif mutation == "missing-SMID":
        arrays["smid"][1] = -1
    elif mutation == "wrong-shape":
        arrays["start_ns"] = arrays["start_ns"][:1]
    elif mutation == "duplicate-tile":
        arrays["tile_ids"][1] = 0
    else:
        arrays["rank_offsets"][1] = 2
    with pytest.raises(ValueError):
        summarize(arrays, meta)


@pytest.mark.parametrize("mutation", ["block-index-instead-of-tile", "nonuniform-terminal-return", "omitted-reset"])
def test_unsafe_timing_protocol_mutations_have_counterexamples(mutation):
    with pytest.raises(AssertionError):
        protocol(negative=mutation)


def test_parent_and_independent_ready_DAG_include_identical_ring_reader_retirement_edges():
    rows = crosscheck_DAG()
    assert len(rows) == 8 and all(r["exact_predecessor_sets_agree"] for r in rows)
    counts = (4, 4, 4)
    slots, frontier = rank_slots(counts)
    _, summary = ring_dependencies(counts, np.asarray(slots, np.int32), frontier)
    assert summary["new_direct_WAR_edges"] > 0
    assert summary["acyclic_by_strict_rank"]


def test_exact_component_work_counts_include_staggering_and_clip_outside_domain():
    # An interior tile owns7*15*31 points for each of6 components, every step.
    values = valid_component_work((12, 12, 12), (512, 512, 512), 16)
    tile = (10 * 12 + 10) * 12 + 10
    assert values[tile] == 6 * 7 * 15 * 31 * 16
    assert valid_component_work((1, 1, 1), (25, 25, 25), 16)[0] < values[tile]


def test_actual_instrumented_source_retains_entire_iteration_bodies_and_bound_protocol(tmp_path):
    parent = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source")
    source = Path(".cache/cuda-cyclic-axis-2026-11-13/cta-timing-v3/source")
    if not source.exists() or not parent.exists():
        pytest.skip("Frozen parent and private diagnostic source required")
    result = prove(parent, source, tmp_path / "proof.json")
    assert len(result["original_iteration_bindings"]) == 2
    assert result["source_structure"]["diagnostic_only"]
    assert len(result["negative_endpoint_cases"]) == 3


def test_wrong_source_cannot_inherit_diagnostic_pointer_or_reset_proof():
    with pytest.raises(ValueError, match="Exact selected52865"):
        transformed({n: b"wrong" for n in PARENT})
