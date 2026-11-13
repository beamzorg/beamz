"""Actual generated native runtime-fact point body and guarded source rewrites."""

import re
import shutil

import pytest

from scripts.cuda_space_time_ir import Operator, yee_operator
from scripts.cuda_stencil_trace import Expr, Read, Update
from scripts.cuda_trace_runtime_uniform import (
    UniformElectricRegionClass,
    specialize_runtime_e,
)
from scripts.cuda_trace_specialize import MagneticScalarClass, specialize_scalar_class
from scripts.generate_cuda_native_cell import generate_native_cell
from scripts.generate_cuda_runtime_fact_cell import generate, verify_rounded_rewrite
from scripts.validate_cuda_runtime_fact_host import validate


def test_default_physics_header_stays_exact_and_new_kernel_identity_is_deterministic():
    assert generate_native_cell()["source_sha256"] == "bef1f611ddedfe171806ded4faa0e6ec205e1d5bd97abde1040b22e3ed4865c5"
    a, b = generate(), generate()
    assert a["source"] == b["source"] and a["source_sha256"] == b["source_sha256"]
    assert a["leaf_storage"]["runtime_scalar_sources"] == ["source_Ex", "source_Ey", "source_Ez", "source_Hx"]
    assert a["leaf_storage"]["spatial_electric_source_leaves"] == 0
    for field in (0, 3, 4, 5):
        assert f"env.Coefficient(1,{field},0,0,0)" in a["source"]
    assert a["rounded_rewrite"]["all_expression_edges_checked"]


def test_profile_aliases_change_only_readonly_bindings_not_mutable_ids():
    ordinary, aliased = generate(), generate(profiles=True)
    for pattern in (r"env.LoadLocal\((\d+),", r"env.StoreLocal\((\d+),"):
        assert re.findall(pattern, ordinary["source"]) == re.findall(pattern, aliased["source"])
    assert aliased["leaf_storage"]["mutable_psi_outputs"] == 12
    assert set(map(int, re.findall(r"env.Coefficient\(2,(\d+),", aliased["source"]))) == {0, 1, 3, 6, 7, 9}


def test_unadmitted_rounding_and_operand_changes_are_rejected():
    old = yee_operator(cpml_axes=(0, 1, 2))
    new, _ = specialize_scalar_class(old, MagneticScalarClass())
    new, _ = specialize_runtime_e(new, UniformElectricRegionClass())
    verify_rounded_rewrite(old, new)
    first = new.updates[0]
    bad = Expr(first.value.op, (first.value.args[1], first.value.args[0], *first.value.args[2:]))
    corrupt = Operator((Update(first.output, bad), *new.updates[1:]), new.tensors)
    with pytest.raises(ValueError):
        verify_rounded_rewrite(old, corrupt)
    signed_old = Operator((Update(Read("Hx", 1), Expr("constant", (-0.,))),), ())
    signed_bad = Operator((Update(Read("Hx", 1), Expr("constant", (0.,))),), ())
    with pytest.raises(ValueError, match="exact FP32 constant"):
        verify_rounded_rewrite(signed_old, signed_bad)


@pytest.mark.parametrize("profiles,inactive,zero,metallic", [(False, False, False, 0),
                                                           (True, False, False, 63),
                                                           (True, True, True, 63)])
def test_actual_emitted_native_body_matches_independent_cpml_wall_source_oracle(tmp_path, profiles, inactive, zero, metallic):
    if not shutil.which("c++"):
        pytest.skip("C++20 compiler unavailable")
    result = validate(tmp_path, profiles=profiles, inactive=inactive, zero=zero, metallic=metallic)
    assert result["run_exit_code"] == 0 and '"errors":0' in result["stdout"]
