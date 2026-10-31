"""CPU contracts for CUDA's Python VJP orchestration.

Only native launches are replaced by a small linear leapfrog model. These tests
exercise the real custom VJPs, active-column pruning, replay and CPML bank parity;
physical CUDA/JAX kernel parity remains in tests/hardware/test_cuda_autodiff.py.
"""

from collections import namedtuple
from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.simulation.cuda import differentiation as diff
from beamz.simulation.cuda import runtime
from beamz.simulation.execute import _known_primal

State = namedtuple(
    "State",
    "ex ey ez hx hy hz cpml_psi_h_terms cpml_psi_e_terms "
    "dft_vec_re dft_vec_im dft_weight_sum t current_step",
)
Coefficients = namedtuple(
    "Coefficients",
    [
        f"{phase}_{kind}_{axis}"
        for phase in "eh"
        for kind in ("decay", "source")
        for axis in "xyz"
    ]
    + ["unused"],
)


@dataclass(frozen=True)
class Context:
    config: object
    source_batches: object = None
    dt_scalar: object = None


def _observe(_program, state, _index, time, _dt):
    fields = tuple(getattr(state, p + a) for p in "eh" for a in "xyz")
    return state._replace(
        dft_vec_re=state.dft_vec_re + sum(fields) * time,
        dft_vec_im=state.dft_vec_im - sum(fields),
        dft_weight_sum=state.dft_weight_sum + 1,
    )


def _transition(state, coeffs, source=0.0, known=None):
    updates = {}
    for axis in "xyz":
        h = getattr(coeffs, f"h_decay_{axis}") * getattr(state, f"h{axis}")
        h += getattr(coeffs, f"h_source_{axis}") * getattr(state, f"e{axis}")
        if known is not None:
            h = _known_primal(h, getattr(known, f"h{axis}"))
        e = getattr(coeffs, f"e_decay_{axis}") * getattr(state, f"e{axis}")
        e += getattr(coeffs, f"e_source_{axis}") * h + source
        if known is not None:
            e = _known_primal(e, getattr(known, f"e{axis}"))
        updates[f"h{axis}"], updates[f"e{axis}"] = h, e
    # Independent linear CPML leaves expose incorrect bank selection.
    updates["cpml_psi_h_terms"] = tuple(v * 0.7 for v in state.cpml_psi_h_terms)
    updates["cpml_psi_e_terms"] = tuple(v * 0.8 for v in state.cpml_psi_e_terms)
    return state._replace(**updates)


@pytest.fixture
def native_model(monkeypatch):
    state = State(
        *(jnp.float32(v / 10) for v in range(1, 7)),
        (jnp.float32(0.2),),
        (jnp.float32(0.3),),
        jnp.float32(0.1),
        jnp.float32(0.2),
        jnp.float32(0.3),
        jnp.float32(0.4),
        jnp.int32(2),
    )
    coeffs = Coefficients(*(jnp.float32(0.6 + i / 100) for i in range(13)))
    group = SimpleNamespace(
        waveforms=jnp.ones((1, 8), jnp.float32),
        coeffs=jnp.ones((1, 1, 1, 1), jnp.float32) * 0.1,
        starts=jnp.zeros((1, 3), jnp.int32),
        n=1,
        max_sizes=(1, 1, 1),
        starts_tuple=((0, 0, 0),),
    )
    batches = {"empty": (None, ()), "source": (group, ())}
    context = Context(
        SimpleNamespace(backend="cuda_streamed"), batches, jnp.float32(0.1)
    )
    monitor = SimpleNamespace(
        recorder_index=-1,
        dft_enabled=True,
        freq_count=1,
        dft_point_count=1,
    )
    program = SimpleNamespace(sources=(), monitors=(monitor,))
    monkeypatch.setattr(diff, "compiled_source_batches", lambda _sources: batches)
    monkeypatch.setattr(diff, "pack_dft_monitors", lambda _monitors: ("packed",))
    monkeypatch.setattr(diff, "update_monitors", _observe)

    def source_value(ctx):
        g = ctx.source_batches["source"][0]
        return jnp.sum(g.coeffs) * g.waveforms[0, 0]

    def forward(
        state,
        *,
        ctx,
        coeffs,
        program,
        update_kernel,
        known_fields=None,
        observation_time=None,
    ):
        out = _transition(state, coeffs, source_value(ctx), known_fields)
        out = _observe(
            program,
            out,
            state.current_step,
            observation_time,
            ctx.dt_scalar,
        )
        return out._replace(t=observation_time, current_step=state.current_step + 1)

    def launch(state, ctx, coeffs, groups, nsteps):
        assert groups[0] is None and groups[1].starts_tuple == ((0, 0, 0),)
        for _ in range(nsteps):
            state = _transition(state, coeffs, source_value(ctx))
        return state

    def monitored(
        state,
        ctx,
        coeffs,
        groups,
        monitors,
        nsteps,
        *,
        observation_origin,
        observation_step_offset,
    ):
        for i in range(nsteps):
            time = (
                observation_origin + (observation_step_offset + i + 1) * ctx.dt_scalar
            )
            state = forward(
                state,
                ctx=ctx,
                coeffs=coeffs,
                program=program,
                update_kernel=None,
                observation_time=time,
            )
        return state

    def reverse(dy, known_h, old_psi, ctx, coeffs, gradients, axes, banks, iteration):
        # The E-source Jacobian needs post-H values; state Jacobians are linear.
        zero = jax.tree.map(jnp.zeros_like, state)
        known = zero._replace(
            **dict(
                zip(
                    ("hx", "hy", "hz"),
                    tuple(v if v.size else jnp.float32(0) for v in known_h),
                    strict=True,
                )
            )
        )

        def phase(values, sources):
            dynamic = coeffs._replace(
                **dict(zip((f"e_source_{a}" for a in "xyz"), sources, strict=True))
            )
            return _transition(values, dynamic, known=known)

        sources = tuple(getattr(coeffs, f"e_source_{a}") for a in "xyz")
        _, pullback = jax.vjp(phase, zero, sources)
        # Float0 clock leaves must retain their tangent dtype.
        dy = dy._replace(current_step=jnp.zeros((), dtype=jax.dtypes.float0))
        ds, dc = pullback(dy)
        old_bank = iteration & 1
        new_bank = 1 - old_bank
        banks = tuple(
            bank.at[new_bank].set(bank[old_bank] * factor)
            for bank, factor in zip(banks, (0.7, 0.8), strict=True)
        )
        return (
            ds,
            tuple(
                g + d if active else runtime._EMPTY
                for g, d, active in zip(gradients, dc, axes, strict=True)
            ),
            banks,
        )

    monkeypatch.setattr(diff, "forward_step", forward)
    monkeypatch.setattr(diff, "run_source_group_steps", launch)
    monkeypatch.setattr(diff, "run_program_steps", monitored)
    monkeypatch.setattr(diff, "reverse_material_step", reverse)
    return program, state, context, coeffs, forward


@pytest.mark.parametrize("nsteps", [1, 2, 3])
@pytest.mark.parametrize("train", ["state", "material", "all", "source", "clock"])
def test_native_step_and_chunk_vjps_match_unrolled_model(native_model, nsteps, train):
    program, state, ctx, coeffs, forward = native_model
    stepper = diff.build_native_step(program, ctx, None, lambda s: s, lambda s: s)
    origin, offset = state.t, jnp.int32(0)

    def loss(value, native):
        initial, dynamic, context, clock = state, coeffs, ctx, origin
        if train == "state":
            initial = state._replace(
                ex=value, cpml_psi_h_terms=(value,), cpml_psi_e_terms=(value,)
            )
        elif train == "material":
            dynamic = coeffs._replace(e_source_x=value)
        elif train == "all":
            dynamic = Coefficients(*(v * value for v in coeffs))
        elif train == "source":
            group, rest = ctx.source_batches["source"]
            group = SimpleNamespace(**{**vars(group), "coeffs": group.coeffs * value})
            context = replace(
                ctx, source_batches={**ctx.source_batches, "source": (group, rest)}
            )
        else:
            clock = value
        if native:
            run = stepper if nsteps == 1 else stepper.for_steps(nsteps)
            out = run(initial, dynamic, context, clock, offset)
        else:
            out = initial
            for i in range(nsteps):
                out = forward(
                    out,
                    ctx=context,
                    coeffs=dynamic,
                    program=program,
                    update_kernel=None,
                    observation_time=clock + (i + 1) * ctx.dt_scalar,
                )
        return (
            sum(jnp.sum(getattr(out, p + a) ** 2) for p in "eh" for a in "xyz")
            + sum(jnp.sum(v**2) for v in (*out.cpml_psi_h_terms, *out.cpml_psi_e_terms))
            + out.dft_vec_re**2
            + out.dft_vec_im**2
        )

    value = jnp.float32(0.43)
    expected = jax.value_and_grad(lambda v: loss(v, False))(value)
    actual = jax.jit(jax.value_and_grad(lambda v: loss(v, True)))(value)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-6)
    assert abs(float(expected[1])) > 0


def test_native_step_eligibility(native_model):
    program, _state, ctx, _coeffs, _forward = native_model

    def build(p, c):
        return diff.build_native_step(p, c, None, lambda s: s, lambda s: s)

    assert build(program, replace(ctx, config=SimpleNamespace(backend="jax"))) is None
    previous_x64 = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", True)
        assert build(program, ctx) is None
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
    for field, value in (
        ("recorder_index", 0),
        ("dft_enabled", False),
        ("freq_count", 0),
        ("dft_point_count", 0),
    ):
        monitor = SimpleNamespace(**{**vars(program.monitors[0]), field: value})
        assert build(SimpleNamespace(sources=(), monitors=(monitor,)), ctx) is None


@pytest.mark.parametrize("train_all", [False, True])
@pytest.mark.parametrize("nterms", [0, 6])
def test_phase_vjp_masks_and_symbolic_cotangents(monkeypatch, train_all, nterms):
    """Check active argument ordering and broadcast coefficient reductions."""
    fields = tuple(jnp.full((2, 3, 4), 0.1 * (i + 1), jnp.float32) for i in range(6))
    material = tuple(jnp.float32(0.5 + i / 10) for i in range(6))
    metadata = jnp.zeros((nterms, 5), jnp.int32)
    parameters = tuple(jnp.float32(0.1) for _ in range(3 * nterms))
    psi = tuple(jnp.ones((2, 3, 4), jnp.float32) for _ in range(nterms))
    metrics = (jnp.float32(1),) * 3
    arguments = (*fields, *material, metadata, *parameters, *psi, *metrics)
    psi_start = 13 + 3 * nterms
    results = (*fields[:3], *psi)
    calls = []

    def model(inputs):
        return tuple(
            inputs[i] * inputs[6 + i] + inputs[3 + i] * inputs[9 + i] for i in range(3)
        ) + tuple(inputs[psi_start : psi_start + nterms])

    def ffi(target, result_values, aliases):
        assert aliases == {}

        def call(*inputs, **attrs):
            calls.append((target, attrs))
            if target == runtime.abi.CUDA_STREAMED_TARGET:
                return model(inputs)
            saved, cotangents = inputs[: len(arguments)], inputs[len(arguments) :]
            # Reconstruct dead residuals with their original shape. The model's
            # only nonlinear terms use residuals required by the active mask.
            saved = tuple(
                v if v.size else jnp.zeros_like(original)
                for v, original in zip(saved, arguments, strict=True)
            )
            _, pullback = jax.vjp(model, saved)
            gradients = pullback(cotangents)[0]
            return tuple(
                gradient
                if int(attrs["gradient_mask"]) & (1 << i)
                else jnp.zeros_like(result_values[i if i < 12 else i - 1])
                for i, gradient in enumerate(gradients)
                if i != 12
            )

        return call

    monkeypatch.setattr(runtime, "_ffi_call", ffi)
    attrs = dict(nterms=nterms, phase=0, boundary_code=0, metric_kind=1)

    def objective(scale, native):
        inputs = tuple(
            v * scale if i != 12 and (train_all or i == 0) else v
            for i, v in enumerate(arguments)
        )
        outputs = (
            runtime._phase_with_vjp(inputs, results, attrs) if native else model(inputs)
        )
        # Leaving other outputs unused supplies SymbolicZero cotangents.
        return jnp.sum(outputs[0] ** 2)

    actual = jax.value_and_grad(lambda s: objective(s, True))(jnp.float32(0.7))
    expected = jax.value_and_grad(lambda s: objective(s, False))(jnp.float32(0.7))
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    mask = [
        attrs["gradient_mask"]
        for target, attrs in calls
        if target == runtime.abi.CUDA_ADJOINT_TARGET
    ][0]
    assert int(mask) & 7 == 0
    assert bool(int(mask) & (1 << 6)) == train_all


def test_phase_rejects_non_float32_inputs():
    with pytest.raises(ValueError, match="float32"):
        runtime._phase_with_vjp((jnp.int32(1),), (), {})


@pytest.mark.parametrize("phase", [0, 1])
@pytest.mark.parametrize("component", [0, 1, 2])
def test_target_cotangent_applies_pec_faces(phase, component):
    cotangent = jnp.ones((3, 4, 5), jnp.float32)
    result = runtime._target_cotangent(
        cotangent,
        jnp.float32(0.9),
        component,
        dict(phase=phase, boundary_code=63),
    )
    expected = np.full((3, 4, 5), 0.9, np.float32)
    for axis in range(3):
        if (phase == 0) == (axis == 2 - component):
            index = [slice(None)] * 3
            index[axis] = [0, -1]
            expected[tuple(index)] = 0
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("cpml", [False, True])
@pytest.mark.parametrize("axes", [(False, False, False), (True, False, True)])
def test_material_transpose_ffi_layout(monkeypatch, cpml, axes):
    from tests.unit.test_cuda_runtime_contract import _program_and_state

    program, state, ctx = _program_and_state(cpml=cpml)
    nterms = len(ctx.boundary.cpml.h_terms)
    banks = tuple(
        jnp.stack((v, v)) for v in (*state.cpml_psi_h_terms, *state.cpml_psi_e_terms)
    )
    known_h = (state.hx, state.hy, state.hz)
    gradients = tuple(
        getattr(program.coefficients, f"e_source_{a}") if active else runtime._EMPTY
        for a, active in zip("xyz", axes, strict=True)
    )
    captured = []

    def ffi(target, results, aliases):
        def call(*inputs, **attrs):
            captured.append((target, results, aliases, inputs, attrs))
            return tuple(v + jnp.float32(i + 1) for i, v in enumerate(results))

        return call

    monkeypatch.setattr(runtime, "_ffi_call", ffi)
    dy, accumulated, next_banks = runtime.reverse_material_step(
        state,
        known_h,
        state.cpml_psi_e_terms,
        ctx,
        program.coefficients,
        gradients,
        axes,
        banks,
        jnp.int32(1),
    )
    target, results, aliases, inputs, attrs = captured[0]
    assert target == runtime.abi.CUDA_MATERIAL_ADJOINT_TARGET
    start = 23 + 7 * nterms
    material_start = 6 + 2 * nterms
    assert aliases == {
        **{start + i: i for i in range(material_start)},
        **{
            start + material_start + i: material_start + i
            for i, active in enumerate(axes)
            if active
        },
    }
    assert attrs["nterms"] == nterms
    assert attrs["e_gradient_mask"] == sum(
        1 << i for i, active in enumerate(axes) if active
    )
    assert int(inputs[-1]) == 1
    np.testing.assert_allclose(dy.ex, state.ex + 1)
    for i, (actual, initial) in enumerate(zip(next_banks, banks, strict=True)):
        np.testing.assert_allclose(actual, initial + i + 7)
    for i, (value, active) in enumerate(zip(accumulated, axes, strict=True)):
        if active:
            np.testing.assert_allclose(
                value, results[material_start + i] + material_start + i + 1
            )
        else:
            assert value.size == 0


def test_material_transpose_rejects_mismatched_cpml():
    ctx = SimpleNamespace(
        boundary=SimpleNamespace(cpml=SimpleNamespace(h_terms=(), e_terms=(1,)))
    )
    with pytest.raises(ValueError, match="matching H/E CPML"):
        runtime.reverse_material_step(None, (), (), ctx, None, (), (), (), 0)


def test_source_only_primal_advances_relative_clock(native_model):
    program, state, ctx, coeffs, _forward = native_model
    program = SimpleNamespace(sources=program.sources, monitors=())
    stepper = diff.build_native_step(program, ctx, None, lambda s: s, lambda s: s)
    for nsteps in (1, 3):
        run = stepper if nsteps == 1 else stepper.for_steps(nsteps)
        actual = run(state, coeffs, ctx, jnp.float32(1.7), jnp.int32(4))
        expected = state
        for _ in range(nsteps):
            expected = _transition(expected, coeffs, source=0.1)
        np.testing.assert_allclose(actual.ex, expected.ex)
        np.testing.assert_allclose(actual.t, 1.7 + (4 + nsteps) * 0.1)
        assert int(actual.current_step) == int(state.current_step) + nsteps
        np.testing.assert_array_equal(actual.dft_vec_re, state.dft_vec_re)


def test_native_step_rejects_unbatched_sources(native_model, monkeypatch):
    program, _state, ctx, _coeffs, _forward = native_model
    monkeypatch.setattr(
        diff, "compiled_source_batches", lambda _: {"source": (None, (object(),))}
    )
    assert diff.build_native_step(program, ctx, None, lambda s: s, lambda s: s) is None
