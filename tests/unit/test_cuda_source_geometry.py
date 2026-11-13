"""Coincident geometry lowering keeps the admitted source FP graph intact."""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

from scripts.patch_cuda_v12_coincident_source_geometry import (
    COINCIDENT,
    ORIGINAL,
    PARENT_HASHES,
    REPLACEMENT,
    compose_text,
    transformed,
)


def test_exact_source_inverse_and_general_callback_retained():
    text = "prefix\n" + ORIGINAL + "\nsuffix"
    result = compose_text(text)
    assert result.replace(REPLACEMENT, ORIGINAL) == text
    assert "if constexpr(SourceMask==20)" in result
    assert "}else{\n" + ORIGINAL in result


def test_hoist_keeps_exact_product_add_and_integer_offset_expression():
    for expression in (
        "((source*s.nz+sz)*s.ny+sy)*s.nx+sx",
        "__fadd_rn(value,__fmul_rn(__ldg(s.coefficients+offset),__ldg(s.waveforms+source*s.ntime+time)))",
        "*s.current_step+p.macro_step+version-1",
    ):
        assert ORIGINAL.count(expression) == COINCIDENT.count(expression) == 1
    assert COINCIDENT.index("if(s.count>0)") < COINCIDENT.index("s.starts[0]")
    assert COINCIDENT.index("int time=") < COINCIDENT.index("for(int source=")


def test_changed_duplicated_and_plausible_unverified_parents_fail_closed():
    for text in (ORIGINAL + ORIGINAL, ORIGINAL.replace("s.count", "other_count", 1)):
        with pytest.raises(ValueError):
            compose_text(text)
    with pytest.raises(ValueError, match="Exact a939"):
        transformed({name: ORIGINAL.encode() for name in PARENT_HASHES})


def test_actual_extracted_cpp_matches_and_missing_fact_has_negative_witness(tmp_path):
    source = Path(".cache/cuda-five-hour-2026-11-13/source-pending-paired-on-v12/source")
    if not source.exists() or not shutil.which("c++"):
        pytest.skip("Frozen source and scalar C++ compiler required")
    # Script supports direct CLI use; give its explicit sibling import the same
    # path for this isolated unit test, without importing package GPU fixtures.
    scripts = str(Path(__file__).resolve().parents[2] / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("source_geometry_validation", Path(scripts) / "validate_cuda_source_geometry.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.validate(source, tmp_path / "proof")
    finally:
        sys.path.remove(scripts)
    assert result["checks"] > 40_000 and result["errors"] == 0
    assert result["noncoincident_negative_witnesses"] == 384
    assert result["empty_null_pointer_checks"] > 10_000
