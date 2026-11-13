"""Phase-level profile commoning, independent state lifetimes and output order."""

import shutil

import pytest

from scripts.cuda_readonly_profile_bindings import EqualAxisProfileClass, bind_profiles
from scripts.cuda_space_time_ir import IllegalSchedule, Operator, yee_operator
from scripts.cuda_stencil_trace import Expr, Read, Update
from scripts.generate_cuda_native_phase import generate, plan_phase
from scripts.validate_cuda_native_phase_host import validate


def test_deterministic_plan_shares_profiles_but_not_mutable_state():
    a, b = generate(), generate()
    assert a["source"] == b["source"]
    assert a["proof"]["proof_sha256"] == b["proof"]["proof_sha256"]
    assert a["source"].count("env.LocalActive(") == 6
    for kind in (2, 3, 4):
        assert a["source"].count(f"env.Coefficient({kind},") == 6
    assert a["source"].count("env.LoadLocal(") == 12
    assert a["source"].count("env.StoreLocal(") == 12
    assert a["proof"]["peak_scoped_profile_values"] == 3
    assert a["proof"]["phases"][0]["profile_pairs"] == {"0": [0, 5], "1": [1, 2], "3": [3, 4]}
    assert all(p["per_output_rounding_reconstruction"] for p in a["proof"]["phases"])
    assert a["proof"]["phases"][0]["executed_profile_order"] == [1, 0, 3]
    assert a["source"].index("result.x=env.Finish(0,") < a["source"].index("env.LocalActive(3,")
    assert "__fadd_rn(correction_0, correction_1)" in a["source"]


def test_cross_component_mutable_old_state_edge_rejects_reordering():
    operator = yee_operator(cpml_axes=(0, 1, 2))

    def alias(value):
        if value.op == "read" and value.args[0].field == "psi_H2":
            old = value.args[0]
            return Expr("read", (Read("psi_H0", old.version, old.offset),))
        return Expr(value.op, tuple(alias(arg) if isinstance(arg, Expr) else arg for arg in value.args))

    changed = Operator(tuple(Update(u.output, alias(u.value)) for u in operator.updates), operator.tensors)
    with pytest.raises(IllegalSchedule):
        plan_phase(changed, 0, bind_profiles(changed, EqualAxisProfileClass()))


def test_changed_original_term_order_is_not_silently_recanonicalized():
    operator = yee_operator(cpml_axes=(0, 1, 2))
    updates = list(operator.updates)
    index = next(i for i, u in enumerate(updates) if u.output.field == "Hx")
    old = updates[index]
    coefficient, terms, own = old.value.args
    changed = Expr("fma_rn", (coefficient, Expr("add", tuple(reversed(terms.args))), own))
    updates[index] = Update(old.output, changed)
    bad = Operator(tuple(updates), operator.tensors)
    with pytest.raises(IllegalSchedule):
        plan_phase(bad, 0, bind_profiles(bad, EqualAxisProfileClass()))


@pytest.mark.parametrize("inactive,zero,metallic,depth", [(False, False, 0, 3),
                                                        (False, False, 63, 7),
                                                        (True, True, 63, 3)])
def test_actual_phase_codegen_matches_independent_oracle_and_profile_load_count(tmp_path, inactive, zero, metallic, depth):
    if not shutil.which("c++"):
        pytest.skip("C++20 compiler required")
    result = validate(tmp_path, inactive=inactive, zero=zero, metallic=metallic, depth=depth)
    assert result["run_exit_code"] == 0 and '"errors":0' in result["stdout"]
    assert "expected_shared_pair_loads" in result["stdout"]
