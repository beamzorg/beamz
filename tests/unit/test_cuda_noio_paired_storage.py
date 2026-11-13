"""Resource experiment changes only the existing certified NoIO consumer."""

import importlib.util
import sys
from pathlib import Path

import pytest


def producer():
    scripts = str(Path(__file__).resolve().parents[2] / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("noio_paired_producer", Path(scripts) / "patch_cuda_v12_noio_paired_storage.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)


def test_original_and_paired_classes_use_same_admission_and_correct_callback_facts():
    p = producer()
    assert "UniformBody<Depth,1,false,true,1,true>" in p.ORIGINAL
    assert "SourcePendingPairedBody<Depth,1,false,true,1,true,1024,-1>" in p.REPLACEMENT
    assert "__launch_bounds__(1024,1)" in p.ORIGINAL and "__launch_bounds__(1024,1)" in p.REPLACEMENT


def test_unverified_parent_fails_closed():
    p = producer()
    with pytest.raises(ValueError, match="Exact a939"):
        p.transformed({name: p.ORIGINAL.encode() for name in p.PARENT_HASHES})


def test_actual_source_retains_paired_math_and_updates_only_noio_allocation():
    p = producer()
    source = Path(".cache/cuda-five-hour-2026-11-13/source-pending-paired-on-v12/source/src")
    if not source.exists():
        pytest.skip("Frozen v12 source required")
    parent = {name: (source / name).read_bytes() for name in p.PARENT_HASHES}
    result = p.transformed(parent)
    old, new = parent["asymmetric_native.cuh"].decode(), result["asymmetric_native.cuh"].decode()
    begin = "// Private source-only storage schedule; source-bound"
    end = "inline bool MakeSource("
    assert new[new.index(begin):new.index(p.REPLACEMENT)] == old[old.index(begin):old.index(end)]
    assert "SourcePendingSharedBytes,stream>>>(p,ids)" in new
    for name in p.PARENT_HASHES:
        if name != "asymmetric_native.cuh":
            assert result[name] == parent[name]
