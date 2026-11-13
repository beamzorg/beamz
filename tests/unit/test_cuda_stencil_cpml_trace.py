"""Independent local CPML and typed-IR contracts for experimental CUDA tracing."""

import itertools
import json

import numpy as np
import pytest

from scripts import cuda_stencil_trace as trace

FIELDS = ("Hx", "Hy", "Hz", "Ex", "Ey", "Ez")


def _bf16(value):
    values = np.asarray(value, np.float32)
    words = values.view(np.uint32)
    rounded = words + np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32).astype(np.float64)


def _evaluate(expression, values, coefficients):
    op = expression.op
    if op == "read":
        read = expression.args[0]
        return np.roll(
            values[read.field, read.version],
            tuple(-v for v in read.offset),
            axis=(0, 1, 2),
        )
    if op == "coefficient":
        read = expression.args[0]
        return np.roll(
            np.broadcast_to(coefficients[read.name], (4, 5, 6)),
            tuple(-v for v in read.offset),
            axis=(0, 1, 2),
        )
    if op == "constant":
        return expression.args[0]
    args = [_evaluate(arg, values, coefficients) for arg in expression.args]
    if op == "add":
        return args[0] + args[1]
    if op == "sub":
        return args[0] - args[1]
    if op == "mul_rn":
        return args[0] * args[1]
    if op == "fma_rn":
        return args[0] * args[1] + args[2]
    if op == "neg":
        return -args[0]
    if op == "round_bf16":
        return _bf16(args[0])
    raise AssertionError(op)


def _fixture(active_axes):
    shape = (4, 5, 6)
    rng = np.random.default_rng(77)
    fields = {name: rng.normal(size=shape) for name in FIELDS}
    coefficients = {"inv_spacing": 0.71}
    axes = (1, 0, 0, 2, 2, 1)
    for field in FIELDS:
        coefficients["source_" + field] = rng.uniform(0.03, 0.09, shape)
        coefficients["decay_" + field] = rng.uniform(0.95, 1.02, shape)
    for phase, term in itertools.product("HE", range(6)):
        axis = axes[term]
        if axis not in active_axes:
            continue
        name = f"psi_{phase}{term}"
        fields[name] = rng.normal(scale=0.01, size=shape)
        profile_shape = tuple(n if a == axis else 1 for a, n in enumerate(shape))
        coefficients["a_" + name] = rng.uniform(-0.1, 0, profile_shape)
        coefficients["b_" + name] = rng.uniform(0.7, 0.99, profile_shape)
        coefficients["inv_kappa_" + name] = rng.uniform(0.6, 1, profile_shape)
    return fields, coefficients


def _reference_step(fields, coefficients, active_axes, precision):
    """Six curls spelled out separately; no tracer component/axis table."""
    result = fields.copy()
    inv = coefficients["inv_spacing"]
    axes = (1, 0, 0, 2, 2, 1)

    def forward(value, axis):
        return (np.roll(value, -1, axis=axis) - value) * inv

    def backward(value, axis):
        return (value - np.roll(value, 1, axis=axis)) * inv

    for family in "HE":
        if family == "H":
            ex, ey, ez = (result[name] for name in ("Ex", "Ey", "Ez"))
            derivatives = (
                forward(ez, 1),
                forward(ey, 0),
                forward(ex, 0),
                forward(ez, 2),
                forward(ey, 2),
                forward(ex, 1),
            )
        else:
            hx, hy, hz = (result[name] for name in ("Hx", "Hy", "Hz"))
            derivatives = (
                backward(hz, 1),
                backward(hy, 0),
                backward(hx, 0),
                backward(hz, 2),
                backward(hy, 2),
                backward(hx, 1),
            )
        corrected = []
        for term, derivative in enumerate(derivatives):
            value = derivative
            if axes[term] in active_axes:
                name = f"psi_{family}{term}"
                next_psi = (
                    coefficients["b_" + name] * fields[name]
                    + coefficients["a_" + name] * derivative
                )
                result[name] = _bf16(next_psi) if precision == "bfloat16" else next_psi
                value = derivative * coefficients["inv_kappa_" + name] + next_psi
            corrected.append(value if term % 2 == 0 else -value)
        for c, component in enumerate("xyz"):
            name = family + component
            source = coefficients["source_" + name]
            result[name] = coefficients["decay_" + name] * fields[name] + (
                -source if family == "H" else source
            ) * (corrected[2 * c] + corrected[2 * c + 1])
    return result


@pytest.mark.parametrize(
    "active_axes", [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
)
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_cpml_region_matches_independent_equations_and_storage_rounding(
    active_axes, precision
):
    fields, coefficients = _fixture(active_axes)
    expected = fields.copy()
    values = {(name, 0): value.copy() for name, value in fields.items()}
    for version in range(3):
        expected = _reference_step(expected, coefficients, active_axes, precision)
        updates = trace.trace_cpml_step(
            version, active_axes=active_axes, precision=precision
        )
        for update in updates:
            values[update.output.field, update.output.version] = _evaluate(
                update.value, values, coefficients
            )
    for name, value in expected.items():
        np.testing.assert_allclose(values[name, 3], value, rtol=1e-13, atol=1e-13)


def test_readonly_coefficients_are_distinct_from_evolving_state():
    coefficient = trace.Coefficient("a_profile", (1,))[0, 1, 0]
    state = trace.Field("psi_H0", 4)[0, 0, 0]
    value = coefficient * state
    assert trace.reads(value) == {trace.Read("psi_H0", 4)}
    assert trace.coefficient_reads(value) == {
        trace.CoefficientRead("a_profile", (1,), (0, 1, 0))
    }
    assert trace.emit_cuda(coefficient) == "coefficient_a_profile(0, 1, 0)"
    assert (
        trace.emit_cuda(
            coefficient, coefficient_name=lambda read: f"profile_{read.axes[0]}_load"
        )
        == "profile_1_load"
    )
    assert trace.emit_cuda(trace.Coefficient("scalar")[0, 0, 0]) == "scalar"


@pytest.mark.parametrize("axes", [(2, 0), (0, 0), (3,), (0, 1), (True,), [1]])
def test_unsupported_coefficient_layouts_fail(axes):
    with pytest.raises(ValueError, match="Coefficients require"):
        trace.Coefficient("coefficient", axes)


def test_coefficient_index_cannot_offset_a_broadcast_dimension():
    with pytest.raises(TypeError, match="varying axes"):
        trace.Coefficient("profile", (1,))[1, 0, 0]


def test_field_and_constant_metadata_rejects_invalid_values():
    with pytest.raises(ValueError, match="Field version"):
        trace.Field("Hx", -1)
    with pytest.raises(ValueError, match="Field name"):
        trace.Field("not a field", 0)
    with pytest.raises(ValueError, match="float32"):
        trace.expression(1e100)


def test_streamed_update_generator_cannot_reuse_dead_expression_identities():
    updates = (
        trace.Update(trace.Read("out", i), trace.parameter(f"input_{i}") + i)
        for i in range(100)
    )
    graph = trace.serialize_updates(updates)
    for i, output in enumerate(graph["outputs"]):
        add = graph["nodes"][output["value"]]
        assert add["op"] == "add"
        assert graph["nodes"][add["inputs"][0]]["value"] == f"input_{i}"
        assert graph["nodes"][add["inputs"][1]]["value"] == float(i)


def test_constants_are_typed_as_float32_before_serialization():
    rounded = trace.expression(0.1)
    assert rounded.args[0] == float(np.float32(0.1))
    assert trace.expression(1e-50).args[0] == 0.0
    assert np.signbit(trace.expression(-0.0).args[0])
    with pytest.raises(ValueError, match="float32"):
        trace.expression(10**1000)


def test_structural_cse_preserves_order_versions_offsets_and_signed_zero():
    old = trace.Field("Ex", 0)
    duplicated = old[0, 0, 0] + old[0, 0, 0]
    graph = trace.serialize_updates((trace.Update(trace.Read("Ex", 1), duplicated),))
    assert len(graph["nodes"]) == 2
    assert graph["nodes"][1]["inputs"] == [0, 0]
    unmerged = trace.serialize_updates(
        (trace.Update(trace.Read("Ex", 1), duplicated),), cse=False
    )
    assert len(unmerged["nodes"]) == 3
    updates = tuple(
        trace.Update(trace.Read("out", i), value)
        for i, value in enumerate(
            (
                old[0, 0, 0],
                old[1, 0, 0],
                trace.Field("Ex", 1)[0, 0, 0],
                trace.expression(0.0),
                trace.expression(-0.0),
            )
        )
    )
    graph = trace.serialize_updates(updates)
    assert len({output["value"] for output in graph["outputs"]}) == len(updates)
    assert json.dumps(graph) == json.dumps(trace.serialize_updates(updates))


def test_ssa_emitter_shares_cpml_next_without_reassociation():
    derivative = trace.parameter("derivative")
    psi, corrected = trace.cpml_term(
        derivative,
        trace.parameter("old"),
        trace.parameter("a"),
        trace.parameter("b"),
        trace.parameter("inv_kappa"),
        -1,
        precision="bfloat16",
    )
    updates = (
        trace.Update(trace.Read("psi", 1), psi),
        trace.Update(trace.Read("corrected", 1), corrected),
    )
    emitted = trace.emit_cuda_block(updates)
    assert emitted["source"].count("__fmaf_rn(") == 2
    assert emitted["source"].count("__float2bfloat16_rn(") == 1
    graph = emitted["graph"]
    stored = graph["nodes"][graph["outputs"][0]["value"]]
    assert stored["op"] == "round_bf16"
    next_node = stored["inputs"][0]
    correction_nodes = [node for node in graph["nodes"] if node["op"] == "fma_rn"]
    assert next_node in correction_nodes[-1]["inputs"]
    assert emitted["liveness"]["use_counts"][next_node] == 2


def test_liveness_accounts_for_outputs_and_rejects_forward_references():
    value = trace.parameter("input")
    graph = trace.serialize_updates(
        (trace.Update(trace.Read("out", 1), value + value),)
    )
    live = trace.cell_liveness(graph)
    assert live["last_use"] == [1, 2]
    assert live["peak_live_scalars"] == 2
    graph["nodes"][1]["inputs"] = [2]
    with pytest.raises(ValueError, match="topologically"):
        trace.cell_liveness(graph)
