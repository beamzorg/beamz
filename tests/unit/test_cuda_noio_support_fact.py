"""Only certified interior NoIO tiles may erase native support predicates."""

from pathlib import Path

import pytest

from scripts.patch_cuda_noio_support_fact import PARENT, transformed
from scripts.prove_cuda_noio_support_fact import axis_certificate, prove


SOURCE = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source")


def parent():
    if not SOURCE.exists():
        pytest.skip("Frozen selected52865 required")
    return {n: (SOURCE / "src" / n).read_bytes() for n in PARENT}


def test_wrong_source_cannot_inherit_native_certificate():
    with pytest.raises(ValueError, match="Exact selected52865"):
        transformed({n: b"wrong" for n in PARENT})


def test_new_variant_only_selects_bit32_clear_certified_noio_and_preserves_fallbacks():
    before = parent()
    after, proof = transformed(before)
    old, new = before["asymmetric_native.cuh"].decode(), after["asymmetric_native.cuh"].decode()
    assert "if(!(p.flags[tiles[blockIdx.x]]&32))" in new
    assert "bool SafeNoIO=false> struct Environment" in new
    assert "Policy==1&&!Fast&&UniformE&&IsotropicH&&Layout==1" in new
    assert "static constexpr bool HasCpml=Policy==1||Policy==5;" in new
    assert new.count("__syncthreads()") == old.count("__syncthreads()")
    for axis in (1, 2, 4):
        assert f"UniformBody<Depth,1,false,true,1,true,{axis}>(p,tiles);" in new
        assert f"UniformBody<Depth,1,false,true,1,true,{axis},true>(p,tiles);" in new
    marker = "// Private source-only storage schedule; source-bound"
    assert new[new.index(marker):] == old[old.index(marker):]
    assert proof["exact_source_inverse"] and proof["expected_moving_kernels"] == 62
    assert proof["startup_Initial_Contains_checks_kept"] and proof["CPML_kept_active"]
    for name in PARENT:
        if name != "asymmetric_native.cuh":
            assert after[name] == before[name]


def test_native_boundary_strict_comparisons_cannot_be_weakened():
    assert not axis_certificate(16, 7, 16, 64)  # Last imported low is -1.
    assert axis_certificate(21, 7, 16, 64)
    assert not axis_certificate(57, 7, 4, 63)  # Startup high equals N.
    assert axis_certificate(56, 7, 4, 63)


def test_trace_geometry_all_heads_fields_imports_and_both_phases():
    result = prove((25, 31, 32, 64, 136, 512))
    assert result["errors"] == 0
    assert result["safe_CPML_axis_tiles"] > 0
    assert result["traced_field_read_checks"] > 100000
    assert result["native_wall_checks"] > 0
    assert result["wrong_low_comparison_counterexamples"] > 0
    assert result["wrong_high_comparison_counterexamples"] > 0
    assert [len(result["axis_cache_heads"][str(a)]) for a in range(3)] == [8, 16, 32]
