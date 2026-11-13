"""Architectural guard: benchmark values must not select temporal kernels."""
from pathlib import Path
import re


def test_shared_kernel_has_no_benchmark_selectors() -> None:
    source = (Path(__file__).resolve().parents[2] / "cuda/src/temporal/kernel.cuh").read_text()
    assert not re.search(r"physical_n\[[^\]]+\]\s*==\s*\d+", source)
    assert not re.search(r"(?:presence|SourceMask)\s*==\s*\d+", source)
    assert not re.search(r"s\.count\s*==\s*(?:2|6)\b", source)
    assert "FixedCube" not in source and "ModeSource54" not in source
    assert source.count("void TemporalEffects(") == 1
    assert "source<s.count" in source
    # Single-source FMA and pre-rounded coincident products are distinct
    # ordinary CUDA arithmetic contracts, rather than workload selectors.
    assert "__fmaf_rn" in source and "__fmul_rn" in source
