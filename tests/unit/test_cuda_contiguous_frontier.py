"""Canonical-read checks for component-specific four-bank field ownership."""

from pathlib import Path

import pytest

from scripts.prove_cuda_contiguous_frontier import Hazard, simulate
from scripts.patch_cuda_contiguous_family_frontier import transformed


@pytest.mark.parametrize("early_private_zy_e", [False, True])
@pytest.mark.parametrize("first,steps,seed", [(1, 2, 7), (4, 2, 29), (13, 2, 19)])
def test_component_routes_read_exact_physical_versions(early_private_zy_e, first, steps, seed):
    report = simulate(first=first, steps=steps, seed=seed, early_private_zy_e=early_private_zy_e)
    assert report["actual_checks"] > 100000
    assert all(n > 0 for n in report["route_counts"].values())


@pytest.mark.parametrize("mutation", ["alias-H-E-banks", "early-E-snapshot",
    "missing-PF1", "missing-PF2", "single-shared-clock"])
def test_unsafe_component_and_clock_schedules_are_rejected(mutation):
    with pytest.raises(Hazard):
        simulate(first=1, steps=2, seed=7, negative=mutation)


def test_lowering_keeps_source_bodies_and_binds_owner_mapping():
    parent = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src/asymmetric_native.cuh")
    if not parent.exists():
        pytest.skip("Private frozen research parent unavailable")
    for defer in (False, True):
        code, proof = transformed(parent.read_text(), scope=7, defer_zy_e=defer)
        assert proof["shared_bytes"] == 76328
        assert proof["paired_source_bodies_byte_identical"]
        assert proof["defer_private_zy_e_until_PF1"] == defer
        assert "zbase=(threadIdx.x/512)*4;" in code
        assert "__shfl_sync(0xffffffffu,e[1][j],(sx+1)&31)" in code
        assert "__shfl_sync(0xffffffffu,h[1][j],(sx+31)&31)" in code
