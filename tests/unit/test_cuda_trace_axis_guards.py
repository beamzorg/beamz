"""Guard pruning is independently checked against frontend active-region traces."""

import itertools
import shutil
from dataclasses import replace

import pytest

from scripts.cuda_space_time_ir import IllegalSchedule, Operator, yee_operator
from scripts.cuda_stencil_trace import Expr, Update, serialize_updates
from scripts.cuda_trace_axis_guards import (
    AxisGuardClass,
    guard_from_native_flags,
    prune_inactive_axes,
)
from scripts.cuda_trace_runtime_uniform import (
    UniformElectricRegionClass,
    specialize_runtime_e,
)
from scripts.cuda_trace_specialize import MagneticScalarClass, specialize_scalar_class
from scripts.generate_cuda_native_phase import generate
from scripts.validate_cuda_native_phase_host import validate


@pytest.mark.parametrize("axes", [axes for n in range(4) for axes in itertools.combinations(range(3), n)])
def test_pruned_actual_graph_equals_independently_traced_active_region(axes):
    operator = yee_operator(cpml_axes=(0, 1, 2))
    expected = yee_operator(cpml_axes=axes)
    # Coefficient facts compose without baking their runtime values.
    for specialize, fact in ((specialize_scalar_class, MagneticScalarClass()),
                             (specialize_runtime_e, UniformElectricRegionClass())):
        operator, _ = specialize(operator, fact)
        expected, _ = specialize(expected, fact)
    actual, proof = prune_inactive_axes(operator, AxisGuardClass(axes))
    assert serialize_updates(actual.updates) == serialize_updates(expected.updates)
    assert actual.tensors == expected.tensors
    assert len(proof["removed_local_states"]) == 12 - 4 * len(axes)
    assert proof["proof_sha256"] == prune_inactive_axes(operator, AxisGuardClass(axes))[1]["proof_sha256"]


def test_actual_flag_classes_are_exhaustive_and_ignore_only_orthogonal_bits():
    for flags in range(256):
        fact = guard_from_native_flags(flags)
        fact.verify_flags(flags)
        assert fact.native_mask == flags & 14
        for bit in (1, 16, 32, 64, 128):
            fact.verify_flags(flags ^ bit)
        with pytest.raises(IllegalSchedule):
            fact.verify_flags(flags ^ 2)


@pytest.mark.parametrize("axes", [(1, 0), (0, 0), (True,), (3,)])
def test_axis_facts_reject_ambiguous_or_noncanonical_axes(axes):
    with pytest.raises(IllegalSchedule):
        AxisGuardClass(axes)


def test_input_graph_must_be_complete_before_pruning():
    with pytest.raises(IllegalSchedule, match="complete"):
        prune_inactive_axes(yee_operator(cpml_axes=(0,)), AxisGuardClass((0,)))


def test_profile_axis_must_match_actual_derivative_geometry_not_just_its_label():
    original = yee_operator(cpml_axes=(0, 1, 2))

    def wrong_axis(value):
        if value.op == "coefficient" and value.args[0].name.endswith("psi_H0"):
            return Expr("coefficient", (replace(value.args[0], axes=(0,)),))
        return Expr(value.op, tuple(wrong_axis(arg) if isinstance(arg, Expr) else arg for arg in value.args))

    changed = Operator(tuple(Update(u.output, wrong_axis(u.value)) for u in original.updates), original.tensors)
    with pytest.raises(IllegalSchedule, match="physical axis"):
        prune_inactive_axes(changed, AxisGuardClass((0,)))


def test_default_frozen_phase_header_and_proof_remain_identical():
    result = generate()
    assert result["source_sha256"] == "cdc020dcc633a592c3e8f84c2608a6ce9a7c5e520f0cdabc1e95df77e01811b0"
    assert result["proof"]["proof_sha256"] == "27476064e00944b0c0593cee660183e02786118ebb497574c925fce0c3d1d5ff"


@pytest.mark.parametrize("axes", [(), (0,), (1,), (2,), (0, 1), (0, 1, 2)])
def test_emitted_phase_has_only_reachable_profiles_and_mutable_states(axes):
    result = generate(axis_guard=AxisGuardClass(axes))
    assert result["source"].count("env.LocalActive(") == 2 * len(axes)
    assert result["source"].count("env.LoadLocal(") == 4 * len(axes)
    assert result["source"].count("env.StoreLocal(") == 4 * len(axes)
    assert result["proof"]["profile_load_sites"] == 6 * len(axes)
    assert f"AdmittedNativeAxisFlags=={AxisGuardClass(axes).native_mask}" in result["source"]


@pytest.mark.parametrize("axes,zero", [((), False), ((0,), False), ((1,), False), ((2,), False),
                                      ((0, 1, 2), False), ((0,), True), ((1,), True), ((2,), True)])
def test_actual_axis_pruned_emitter_matches_independent_all_state_oracle(tmp_path, axes, zero):
    if not shutil.which("c++"):
        pytest.skip("C++20 compiler required")
    result = validate(tmp_path, metallic=63, depth=5, zero=zero, axis_guard=AxisGuardClass(axes))
    assert result["run_exit_code"] == 0 and '"errors":0' in result["stdout"]
