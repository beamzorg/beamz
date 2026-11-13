"""Single-grid composition retains the paired source's independent template fact."""

import shutil
from pathlib import Path

import pytest

from scripts.patch_cuda_v12_noio_axis_bundle import (
    EDITS,
    PARENT_HASHES,
    compose_text,
    dispatch_axis,
    transformed,
)
from scripts.validate_cuda_v12_axis_bundle import validate


def skeleton():
    # Test the compositional mechanics without requiring cached source files.
    return "\n".join(old for old, _ in EDITS) + "\n// untouched source body\nEnvironment<Layout,Fast,IsotropicH,Policy,UniformE,20> source;\n"


def test_axis_argument_is_appended_after_source_mask_and_default_remains_generic():
    result = compose_text(skeleton())
    assert "int SourceMask=-1,int ActiveAxes=-1> struct Environment" in result
    assert "UniformE,-1,ActiveAxes>env" in result
    assert "UniformE,20> source" in result
    assert "default:UniformBody<Depth,1,false,true,1,true>" in result


def test_composition_has_exact_inverse_and_does_not_rewrite_arithmetic_or_sources():
    original = skeleton() + "\nvalue=__fmaf_rn(a,b,__fmul_rn(c,d));\n"
    result = compose_text(original)
    for old, new in reversed(EDITS):
        assert result.count(new) == 1
        result = result.replace(new, old, 1)
    assert result == original


@pytest.mark.parametrize("index", range(5))
def test_changed_or_duplicated_anchors_fail_closed(index):
    text = skeleton()
    with pytest.raises(ValueError):
        compose_text(text.replace(EDITS[index][0], "changed anchor", 1))
    with pytest.raises(ValueError):
        compose_text(text + EDITS[index][0])


def test_parent_hashes_fail_closed_even_when_anchors_are_plausible():
    with pytest.raises(ValueError, match="Exact a939"):
        transformed({name: skeleton().encode() for name in PARENT_HASHES})


def test_all_256_flags_use_one_hot_typed_axis_or_exact_generic_fallback():
    for flags in range(256):
        actual = dispatch_axis(flags)
        expected = (flags & 14) >> 1 if flags & 14 in (2, 4, 8) else -1
        assert actual == expected
        for bit in (1, 16, 32, 64, 128):
            assert actual == dispatch_axis(flags ^ bit)
        for axis in range(3):
            active = bool(flags & (2 << axis))
            assert active == (bool(actual & (1 << axis)) if actual >= 0 else active)


def test_actual_frozen_cpp_guard_bodies_and_wrong_axis_negative(tmp_path):
    source = Path(".cache/cuda-five-hour-2026-11-13/source-pending-paired-on-v12/source")
    if not source.exists() or not shutil.which("c++"):
        pytest.skip("Frozen research source and host compiler required")
    proof = validate(source, tmp_path)
    assert proof["checks"] > 1_000_000 and proof["errors"] == 0
    assert proof["wrong_axis_witnesses"] > 0
