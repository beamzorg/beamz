"""Independent equation and dependency checks for the experimental cell tracer.

These host checks establish neither GPU execution nor a production boundary
implementation. The numerical oracle uses periodic common-shape arrays and
spells out the six curls independently of the tracer's component table.
"""

import itertools
import json

import numpy as np
import pytest

from scripts import cuda_stencil_trace as trace

FIELDS = ("Hx", "Hy", "Hz", "Ex", "Ey", "Ez")


def _reference_step(fields, parameters):
    hx, hy, hz, ex, ey, ez = (fields[name] for name in FIELDS)
    inv = parameters["inv_spacing"]

    def forward(value, axis):
        return (np.roll(value, -1, axis=axis) - value) * inv

    def backward(value, axis):
        return (value - np.roll(value, 1, axis=axis)) * inv

    hx = parameters["decay_Hx"] * hx - parameters["source_Hx"] * (
        forward(ez, 1) - forward(ey, 0)
    )
    hy = parameters["decay_Hy"] * hy - parameters["source_Hy"] * (
        forward(ex, 0) - forward(ez, 2)
    )
    hz = parameters["decay_Hz"] * hz - parameters["source_Hz"] * (
        forward(ey, 2) - forward(ex, 1)
    )
    ex = parameters["decay_Ex"] * ex + parameters["source_Ex"] * (
        backward(hz, 1) - backward(hy, 0)
    )
    ey = parameters["decay_Ey"] * ey + parameters["source_Ey"] * (
        backward(hx, 0) - backward(hz, 2)
    )
    ez = parameters["decay_Ez"] * ez + parameters["source_Ez"] * (
        backward(hy, 2) - backward(hx, 1)
    )
    return dict(zip(FIELDS, (hx, hy, hz, ex, ey, ez), strict=True))


def _parameters(shape, seed=19):
    rng = np.random.default_rng(seed)
    return {
        "inv_spacing": 0.73,
        **{"decay_" + name: rng.uniform(0.95, 1.05, shape) for name in FIELDS},
        **{"source_" + name: rng.uniform(0.03, 0.09, shape) for name in FIELDS},
    }


def _evaluate(expression, values, parameters):
    """Interpret arithmetic in float64; this does not emulate CUDA FMA bits."""
    if expression.op == "read":
        read = expression.args[0]
        return np.roll(
            values[read.field, read.version],
            tuple(-offset for offset in read.offset),
            axis=(0, 1, 2),
        )
    if expression.op == "parameter":
        return parameters[expression.args[0]]
    if expression.op == "constant":
        return expression.args[0]
    args = [_evaluate(arg, values, parameters) for arg in expression.args]
    if expression.op == "neg":
        return -args[0]
    if expression.op == "add":
        return args[0] + args[1]
    if expression.op == "sub":
        return args[0] - args[1]
    if expression.op == "mul_rn":
        return args[0] * args[1]
    if expression.op == "fma_rn":
        return args[0] * args[1] + args[2]
    raise AssertionError(f"Unexpected expression opcode: {expression.op}")


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_traced_equations_match_independent_periodic_yee_updates(steps):
    shape = (7, 8, 9)
    rng = np.random.default_rng(771)
    expected = {name: rng.normal(size=shape) for name in FIELDS}
    parameters = _parameters(shape)
    values = {(name, 0): value.copy() for name, value in expected.items()}
    for version in range(steps):
        expected = _reference_step(expected, parameters)
        for update in trace.trace_yee_step(version):
            values[update.output.field, update.output.version] = _evaluate(
                update.value, values, parameters
            )
    for name in FIELDS:
        np.testing.assert_allclose(
            values[name, steps], expected[name], rtol=1e-13, atol=1e-13
        )


@pytest.mark.parametrize("version", [0, 3])
def test_reads_retain_staggered_versions_and_component_axes(version):
    # Independent explicit read contracts: axes are z,y,x; components are x,y,z.
    opposite = {
        "Hx": (("Ez", (0, 1, 0)), ("Ey", (1, 0, 0))),
        "Hy": (("Ex", (1, 0, 0)), ("Ez", (0, 0, 1))),
        "Hz": (("Ey", (0, 0, 1)), ("Ex", (0, 1, 0))),
        "Ex": (("Hz", (0, -1, 0)), ("Hy", (-1, 0, 0))),
        "Ey": (("Hx", (-1, 0, 0)), ("Hz", (0, 0, -1))),
        "Ez": (("Hy", (0, 0, -1)), ("Hx", (0, -1, 0))),
    }
    updates = trace.trace_yee_step(version)
    assert tuple(update.output.field for update in updates) == FIELDS
    for update in updates:
        name = update.output.field
        dependency_version = version if name.startswith("H") else version + 1
        expected = {trace.Read(name, version)}
        for field, offset in opposite[name]:
            expected |= {
                trace.Read(field, dependency_version),
                trace.Read(field, dependency_version, offset),
            }
        assert update.output == trace.Read(name, version + 1)
        assert update.reads == expected


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("output", FIELDS)
def test_dependency_cone_matches_independent_impulse_propagation(output, depth):
    # Spatially varying independent coefficients avoid cancellations specific
    # to homogeneous curl identities. Every impulse is evaluated by the oracle.
    shape = (2 * depth + 5,) * 3
    center = np.asarray((depth + 2,) * 3)
    parameters = _parameters(shape)
    cone = trace.dependency_cone(output, depth)
    assert cone and all(read.version == 0 for read in cone)
    assert all(max(map(abs, read.offset)) <= depth for read in cone)
    rng = np.random.default_rng(42)
    candidates = sorted(cone)
    selected = rng.choice(len(candidates), size=min(16, len(candidates)), replace=False)
    probes = [candidates[index] for index in selected]
    # Sample unsupported locations within the bounding cube as well as all six
    # beyond-halo axial positions. This catches overlarge/underlarge support.
    possible = [
        trace.Read(name, 0, offset)
        for name in FIELDS
        for offset in itertools.product(range(-depth, depth + 1), repeat=3)
    ]
    outside = [read for read in possible if read not in cone]
    if outside:
        probes.extend(
            outside[index]
            for index in rng.choice(len(outside), min(16, len(outside)), replace=False)
        )
    probes.extend(
        trace.Read(
            output, 0, tuple(sign * (depth + 1) if a == axis else 0 for a in range(3))
        )
        for axis in range(3)
        for sign in (-1, 1)
    )
    for read in probes:
        state = {name: np.zeros(shape) for name in FIELDS}
        state[read.field][tuple(center + read.offset)] = 1.0
        for _ in range(depth):
            state = _reference_step(state, parameters)
        influence = state[output][tuple(center)]
        if read in cone:
            assert abs(influence) > 1e-18, read
        else:
            assert influence == 0.0, read


def test_cuda_emission_preserves_explicit_rounding_and_load_metadata():
    for update in trace.trace_yee_step():
        source = trace.emit_cuda(update.value)
        assert source.startswith("__fmaf_rn(")
        assert source.count("__fmul_rn(") == 3
        assert source.count("__fsub_rn(") == 3
        assert f"decay_{update.output.field}" in source
        assert f"source_{update.output.field}" in source
        assert source.startswith(
            "__fmaf_rn((-source_"
        ) == update.output.field.startswith("H")
    seen = []

    def load(read):
        seen.append(read)
        return "selected_load"

    update = trace.trace_yee_step()[3]
    emitted = trace.emit_cuda(update.value, load)
    assert "load_" not in emitted
    assert "selected_load" in emitted
    assert frozenset(seen) == update.reads
    assert trace.emit_cuda(trace.expression(2)) == "2.0f"
    assert trace.emit_cuda(trace.expression(2) + 3) == "__fadd_rn(2.0f, 3.0f)"


@pytest.mark.parametrize(
    "offset",
    [(1, 2), [0, 0, 0], (0.0, 0, 0), (True, 0, 0), (trace.parameter("index"), 0, 0)],
)
def test_dynamic_or_malformed_stencil_indices_fail(offset):
    with pytest.raises(TypeError, match="constant integer offsets"):
        trace.Field("Hx", 0)[offset]


def test_data_dependent_branch_fails_instead_of_silently_specializing():
    with pytest.raises(TypeError, match="Data-dependent branching"):
        bool(trace.Field("Hx", 0)[0, 0, 0])


@pytest.mark.parametrize("comparison", [lambda x: x == 0, lambda x: x != 0])
def test_comparison_cannot_silently_specialize_a_branch(comparison):
    with pytest.raises(TypeError, match="Data-dependent comparisons"):
        comparison(trace.Field("Hx", 0)[0, 0, 0])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_constants_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        trace.expression(value)


def test_scalar_left_operands_are_recorded():
    value = trace.parameter("coefficient")
    assert trace.emit_cuda(2 + value) == "__fadd_rn(2.0f, coefficient)"
    assert trace.emit_cuda(2 - value) == "__fsub_rn(2.0f, coefficient)"
    assert trace.emit_cuda(2 * value) == "__fmul_rn(2.0f, coefficient)"


def test_serialized_dag_is_deterministic_topological_and_keeps_shared_nodes():
    graph = trace.serialize_updates(trace.trace_yee_step())
    assert json.dumps(graph) == json.dumps(
        trace.serialize_updates(trace.trace_yee_step())
    )
    for index, node in enumerate(graph["nodes"]):
        assert all(parent < index for parent in node.get("inputs", ()))
    shared = trace.parameter("shared")
    graph = trace.serialize_updates(
        (trace.Update(trace.Read("custom", 1), shared + shared),)
    )
    assert len(graph["nodes"]) == 2
    assert graph["nodes"][1]["inputs"] == [0, 0]


@pytest.mark.parametrize("version", [-1, 1.0, True])
def test_invalid_versions_fail(version):
    with pytest.raises(ValueError, match="nonnegative integer"):
        trace.trace_yee_step(version)


@pytest.mark.parametrize("depth", [0, -1, 2.0, True])
def test_invalid_depths_fail(depth):
    with pytest.raises(ValueError, match="positive integer"):
        trace.dependency_cone("Ex", depth)


def test_unknown_field_and_opcode_fail():
    with pytest.raises(ValueError, match="Unknown Yee field"):
        trace.dependency_cone("Jx", 2)
    with pytest.raises(ValueError, match="Unsupported operation"):
        trace.emit_cuda(trace.Expr("arbitrary_python", ()))


def test_manifest_reports_actual_bounds_and_scope():
    result = trace.manifest(2)
    assert result["axis_order"] == ["z", "y", "x"]
    assert "no boundaries, sources or monitors" in result["scope"]
    for name in FIELDS:
        cone = trace.dependency_cone(name, 2)
        assert result["fields"][name] == {
            "initial_sample_count": len(cone),
            "lower": [min(read.offset[a] for read in cone) for a in range(3)],
            "upper": [max(read.offset[a] for read in cone) for a in range(3)],
        }
