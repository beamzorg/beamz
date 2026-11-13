"""Dispatch evidence must prevent constants from surviving packet changes."""
import numpy as np
import pytest

from scripts.cuda_space_time_codegen import generate
from scripts.cuda_space_time_ir import compile_stream, yee_operator
from scripts.cuda_trace_specialize import (
    ScalarFact,
    magnetic_facts,
    magnetic_scalar_class,
    specialize,
    specialize_scalar_class,
    verify,
    verify_scalar_class,
)


def packet():
    return [None] * 6 + [np.float32(1)] * 3 + [np.float32(0.125)] * 3


@pytest.mark.parametrize("cpml", (False, True))
def test_constants_remove_callbacks_preserving_field_dependencies(cpml):
    inputs = packet()
    facts = magnetic_facts(inputs)
    operator = yee_operator(cpml_axes=(0, 1, 2) if cpml else None)
    result, evidence = specialize(operator, facts)
    verify(facts, inputs)
    assert len(evidence["substituted"]) == 6
    assert [u.reads for u in result.updates] == [u.reads for u in operator.updates]
    assert compile_stream(result)["frontier"] == compile_stream(operator)["frontier"]
    code = generate(result)["source"]
    assert "env.Coefficient(0,0," not in code
    assert "env.Coefficient(1,0," not in code
    assert "__fmaf_rn" in code
    assert "__fmul_rn" in code


@pytest.mark.parametrize("replacement", (np.float32(0.25), np.float64(0.125),
                                         np.array([0.125], dtype=np.float32),
                                         np.float32(np.nan)))
def test_changed_value_shape_precision_or_finiteness_rejects_dispatch(replacement):
    inputs = packet()
    facts = magnetic_facts(inputs)
    inputs[9] = replacement
    with pytest.raises(ValueError):
        verify(facts, inputs)


def test_signed_zero_and_dynamic_coefficients_have_distinct_evidence():
    a, b = packet(), packet()
    a[9], b[9] = np.float32(0.), np.float32(-0.)
    fa, fb = magnetic_facts(a), magnetic_facts(b)
    assert specialize(yee_operator(), fa)[1]["specialization_sha256"] != specialize(yee_operator(), fb)[1]["specialization_sha256"]
    with pytest.raises(ValueError):
        verify(fa, b)
    a[6] = np.ones((2, 3, 4), dtype=np.float32)
    assert "decay_Hx" not in {f.name for f in magnetic_facts(a)}


def test_fact_cannot_bind_an_unrelated_input_or_evolving_state():
    with pytest.raises(ValueError, match="prepared ABI"):
        ScalarFact("decay_Hx", 0, 0x3f800000)
    with pytest.raises(ValueError, match="prepared ABI"):
        ScalarFact("Hx", 6, 0x3f800000)


def test_scalar_class_reuses_kernel_across_dt_values_but_keeps_runtime_source():
    a, b = packet(), packet()
    b[9:12] = [np.float32(.0625)] * 3
    fact = magnetic_scalar_class(a)
    verify_scalar_class(fact, b)
    operator = yee_operator(cpml_axes=(0, 1, 2))
    result, evidence = specialize_scalar_class(operator, fact)
    assert evidence["specialization_sha256"] == specialize_scalar_class(operator, magnetic_scalar_class(b))[1]["specialization_sha256"]
    assert [u.reads for u in result.updates] == [u.reads for u in operator.updates]
    coefficients = compile_stream(result)["legality"]["coefficients"]
    assert {c["name"] for c in coefficients if c["name"].startswith("source_H")} == {"source_Hx"}
    assert next(c for c in coefficients if c["name"] == "source_Hx")["scope"] == "scalar"
    assert "env.Coefficient(1,0," in generate(result)["source"]


def test_scalar_class_rejects_loss_anisotropy_shape_and_signed_zero_mismatch():
    original = packet()
    fact = magnetic_scalar_class(original)
    for index, value in ((6, np.float32(.99)), (10, np.float32(.25)),
                         (9, np.array([.125], dtype=np.float32))):
        changed = list(original)
        changed[index] = value
        assert magnetic_scalar_class(changed) is None
        with pytest.raises(ValueError):
            verify_scalar_class(fact, changed)
    changed = packet()
    changed[9:12] = [np.float32(x) for x in (0., -0., 0.)]
    assert magnetic_scalar_class(changed) is None
