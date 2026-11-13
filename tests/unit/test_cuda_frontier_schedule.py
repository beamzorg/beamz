"""Adversarial storage/visibility checks for the new permanent-family pipeline."""

from pathlib import Path

import pytest

from scripts.prove_cuda_frontier_schedule import Hazard, prove_footprints, simulate
from scripts.patch_cuda_permanent_family_frontier import transformed


def test_actual_read_footprints_have_no_same_phase_conflicts():
    cases = prove_footprints()
    assert len(cases) == 16
    assert all(c["H_actual_shared_reads"] > 0 and c["E_actual_shared_reads"] > 0 for c in cases)


@pytest.mark.parametrize("first,steps,seed", [(1, 2, 7), (4, 2, 29), (13, 2, 19)])
def test_arbitrary_warp_order_reads_exact_physical_versions(first, steps, seed):
    report = simulate(first=first, steps=steps, seed=seed)
    assert report["actual_field_and_clock_checks"] > 100000
    assert report["loop_barriers"] == 2*steps


@pytest.mark.parametrize("mutation", ["alias-H-E-banks", "early-E-ghost-publication",
    "late-H-publication", "missing-PF1", "missing-PF2", "single-shared-clock",
    "early-H-clock", "early-E-clock"])
def test_unsafe_clock_and_memory_schedules_are_rejected(mutation):
    with pytest.raises(Hazard):
        simulate(first=1, steps=2, seed=7, negative=mutation)


def test_lowering_rejects_an_unbound_parent():
    with pytest.raises(ValueError, match="Exact selected52865"):
        transformed("not the frozen source")


def test_scope_is_bound_to_existing_classes():
    parent = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src/asymmetric_native.cuh")
    if not parent.exists():
        pytest.skip("Frozen private research parent unavailable")
    for scope in (1, 2, 4, 7):
        code, report = transformed(parent.read_text(), scope=scope)
        assert report["shared_bytes"] < 99*1024
        assert report["paired_source_bodies_byte_identical"]
        assert "volatile int*h_step" in code and "volatile int*e_step" in code
    with pytest.raises(ValueError, match="Scope bits"):
        transformed(parent.read_text(), scope=8)
