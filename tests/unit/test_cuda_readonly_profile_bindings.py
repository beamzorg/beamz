"""Readonly profile equality witnesses and logical/physical binding separation."""

import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA
from scripts.cuda_readonly_profile_bindings import (
    AXES,
    EqualAxisProfileClass,
    bind_profiles,
    equal_axis_profile_class,
    profile_input_index,
    verify_profile_class,
)
from scripts.cuda_space_time_ir import IllegalSchedule, StreamShape, yee_operator
from scripts.cuda_stencil_trace import coefficient_reads, serialize_updates
from scripts.generate_cuda_native_cell import generate_native_cell
from scripts.validate_cuda_streamed_host import host_source


def packet(scale=1.):
    inputs = [None] * 62
    for field, delta in enumerate(SUPPORT_DELTA):
        inputs[field] = np.empty(tuple(n + d for n, d in zip((32, 35, 37), delta, strict=True)), np.float32)
    for phase in (0, 1):
        for term, axis in enumerate(AXES):
            shape = tuple(24 if a == axis else 1 for a in range(3))
            for kind in range(3):
                values = np.arange(24, dtype=np.float32) * np.float32(scale) + np.float32(phase * 20 + axis * 3 + kind)
                inputs[profile_input_index(phase, term, kind)] = values.reshape(shape).copy()
    return inputs


def test_value_independent_equal_axis_class_preserves_logical_operator():
    operator = yee_operator(cpml_axes=(0, 1, 2))
    before = serialize_updates(operator.updates)
    a, b = packet(), packet(.25)
    fact_a, fact_b = equal_axis_profile_class(a), equal_axis_profile_class(b)
    assert fact_a == fact_b == EqualAxisProfileClass()
    verify_profile_class(fact_a, a)
    bindings = bind_profiles(operator, fact_a)
    assert bindings == bind_profiles(operator, fact_b)
    assert len(bindings.bindings) == 36
    generated = generate_native_cell(operator, profile_bindings=bindings)
    assert serialize_updates(operator.updates) == before
    assert generated["source"].count("if(env.LocalActive(") == 12
    assert "env.LoadLocal(2," in generated["source"]
    assert "env.StoreLocal(2," in generated["source"]
    assert "env.Coefficient(2,2," not in generated["source"]
    assert "env.Coefficient(2,1," in generated["source"]
    assert generated["local_term_ids"]["psi_H2"] == 2


@pytest.mark.parametrize("mutation", ("bit", "dtype", "shape", "signed-zero", "nonfinite", "field-shape"))
def test_profile_class_rejects_unsafe_or_stale_aliases(mutation):
    inputs = packet()
    index = profile_input_index(0, 2, 0)
    if mutation == "bit":
        inputs[index].view(np.uint32).flat[0] ^= 1
    elif mutation == "dtype":
        inputs[index] = inputs[index].astype(np.float64)
    elif mutation == "shape":
        inputs[index] = inputs[index].reshape(24)
    elif mutation == "signed-zero":
        inputs[profile_input_index(0, 1, 0)].flat[0] = 0.
        inputs[index].flat[0] = -0.
    elif mutation == "nonfinite":
        inputs[index].flat[0] = np.inf
    else:
        inputs[4] = np.empty((33, 36, 38), np.float32)
    assert equal_axis_profile_class(inputs) is None
    with pytest.raises(IllegalSchedule):
        verify_profile_class(EqualAxisProfileClass(), inputs)


def test_forged_cross_phase_binding_rejected_before_codegen():
    operator = yee_operator(cpml_axes=(0, 1, 2))
    bindings = bind_profiles(operator, EqualAxisProfileClass())
    pairs = list(bindings.bindings)
    pairs[0] = (pairs[0][0], pairs[0][1] + 6)
    with pytest.raises(IllegalSchedule):
        generate_native_cell(operator, profile_bindings=replace(bindings, bindings=tuple(pairs)))
    logical = next(r for u in operator.updates for r in coefficient_reads(u.value) if r.name == "a_psi_H2")
    with pytest.raises(IllegalSchedule):
        bindings.slot(replace(logical, offset=(1, 0, 0)))


@pytest.mark.parametrize("half", (False, True))
def test_actual_aliased_native_functor_matches_independent_scalar_oracle(tmp_path, half):
    if not shutil.which("c++"):
        pytest.skip("C++ compiler required")
    operator = yee_operator(cpml_axes=(0, 1, 2), precision="bfloat16" if half else "float32")
    result = generate_native_cell(operator, profile_bindings=bind_profiles(operator, EqualAxisProfileClass()))
    source = host_source(result, StreamShape((2, 2, 3), 2, 96), cpml=True, half=half,
                         native=True, metallic=63, native_cell=True)
    path, binary = tmp_path / "host.cc", tmp_path / "host"
    path.write_text(source)
    compiled = subprocess.run(["c++", "-std=c++20", "-O1", "-pthread", "-ffp-contract=off", str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert '"errors":0' in executed.stdout
