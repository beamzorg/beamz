"""Immutable code-family and causal-selector contracts for concurrent NoIO."""

from pathlib import Path

import pytest

from scripts.patch_cuda_concurrent_noio_axes import PARENT_HASHES, transformed


def parent_files():
    root = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src")
    if not root.exists():
        pytest.skip("Private frozen research parent unavailable")
    return {n: (root/n).read_bytes() for n in PARENT_HASHES}


def test_parallel_selector_changes_only_one_host_constant():
    parent = parent_files()
    parallel, serial = transformed(parent, concurrent=True), transformed(parent, concurrent=False)
    assert parallel.keys() == serial.keys()
    for name in parallel:
        if name != "native_extension.cc":
            assert parallel[name] == serial[name]
    assert serial["native_extension.cc"].replace(
        b"const bool BeamzResearchConcurrentNoIOAxesEnabled=false;",
        b"const bool BeamzResearchConcurrentNoIOAxesEnabled=true;") == parallel["native_extension.cc"]


def test_ordered_cell_header_and_uniform_body_are_frozen():
    parent = parent_files()
    child = transformed(parent)
    assert child["asymmetric_native_cell_generated.cuh"] == parent["asymmetric_native_cell_generated.cuh"]
    before, after = parent["asymmetric_native.cuh"], child["asymmetric_native.cuh"]
    start = b"template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody"
    end = b"template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__"
    assert before[before.index(start):before.index(end)] == after[after.index(start):after.index(end)]


@pytest.mark.parametrize("name", list(PARENT_HASHES))
def test_unbound_source_or_planner_fails_closed(name):
    parent = parent_files()
    parent[name] += b"\n// unreviewed mutation\n"
    with pytest.raises(ValueError, match="Exact selected52865"):
        transformed(parent)
