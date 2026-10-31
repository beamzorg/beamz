"""Native timestep primals with the exact, existing streamed/JAX transpose.

The linearization substitutes already computed post-source fields. Phase VJPs
need only their inputs; additive sources and field DFTs are linear in fields.
Consequently the unused forward phase launches disappear from the backward HLO.
"""

from dataclasses import dataclass, replace
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.typing import DTypeLike

from beamz.devices.sources.compiler import BatchedSlabGroup
from beamz.simulation.cuda.runtime import (
    _EMPTY,
    pack_dft_monitors,
    reverse_material_step,
    run_program_steps,
    run_source_group_steps,
)
from beamz.simulation.execute import compiled_source_batches, forward_step
from beamz.simulation.observe import is_vector_dft_monitor, update_monitors


@jax.tree_util.register_static
@dataclass(frozen=True)
class _Columns:
    flags: tuple[bool, ...]


@dataclass(frozen=True)
class _NativeStepper:
    step: Callable[..., Any]
    chunk_factory: Callable[[int], Callable[..., Any]]

    def __call__(self, *arguments):
        return self.step(*arguments)

    def for_steps(self, nsteps):
        return self.chunk_factory(nsteps)


def _materialize_cotangents(cotangents, values):
    return jax.tree.map(
        lambda gradient, _value: (
            jnp.zeros(
                cast(ShapedArray, gradient.aval).shape,
                cast(DTypeLike, cast(ShapedArray, gradient.aval).dtype),
            )
            if isinstance(gradient, jax.custom_derivatives.SymbolicZero)
            else gradient
        ),
        cotangents,
        values,
    )


def build_native_step(program, context, update_kernel, split_dft, pack_dft):
    """Return a differentiable native step, or None for unsupported recorders."""
    if context.config.backend != "cuda_streamed" or bool(
        jax.config.read("jax_enable_x64")
    ):
        return None
    batches = compiled_source_batches(program.sources)
    if any(rest for _group, rest in batches.values()):
        return None
    if any(not is_vector_dft_monitor(mon) for mon in program.monitors):
        return None
    keys = tuple(batches)
    templates = tuple(batches[key][0] for key in keys)
    packed_monitors = pack_dft_monitors(program.monitors) if program.monitors else None
    base_context = replace(context, source_batches=None)

    def source_values(ctx):
        return tuple(
            (group.waveforms, group.coeffs, group.starts)
            for key in keys
            if (group := ctx.source_batches[key][0]) is not None
        )

    def restore_context(values):
        iterator = iter(values)
        groups = []
        for template in templates:
            if template is None:
                groups.append(None)
            else:
                waveforms, coeffs, starts = next(iterator)
                groups.append(
                    BatchedSlabGroup(
                        waveforms=waveforms,
                        coeffs=coeffs,
                        starts=starts,
                        n=template.n,
                        max_sizes=template.max_sizes,
                        starts_tuple=template.starts_tuple,
                    )
                )
        ctx = replace(
            base_context,
            source_batches=dict(
                zip(keys, ((group, ()) for group in groups), strict=True)
            ),
        )
        return ctx, tuple(groups)

    def native_forward(inputs, nsteps=1, *, observe=True):
        state, coeffs, sources, clock = inputs
        ctx, groups = restore_context(sources)
        packed = pack_dft(state) if observe else state
        out = (
            run_program_steps(
                packed,
                ctx,
                coeffs,
                groups,
                packed_monitors,
                nsteps,
                observation_origin=clock[0],
                observation_step_offset=clock[1],
            )
            if observe and packed_monitors is not None
            else run_source_group_steps(packed, ctx, coeffs, groups, nsteps)
        )
        out = out._replace(
            t=clock[0] + (clock[1] + nsteps).astype(jnp.float32) * ctx.dt_scalar,
            current_step=state.current_step + jnp.int32(nsteps),
        )
        return split_dft(out) if observe else out

    @jax.custom_vjp
    def native_step(inputs):
        return native_forward(inputs)

    def step_forward(primal_inputs):
        values = jax.tree.map(lambda primal: primal.value, primal_inputs)
        flags = jax.tree.map(
            lambda primal: (
                bool(primal.perturbed)
                and jnp.issubdtype(primal.value.dtype, jnp.inexact)
            ),
            primal_inputs,
        )
        state_flags, coeff_flags, _, clock_flags = flags
        clock_active = clock_flags[0]
        # CUDA consumes precomputed decay/source coefficients. Other physical
        # planner leaves have zero columns in this timestep's Jacobian.
        used_coefficients = {
            f"{phase}_{kind}_{axis}"
            for phase in ("e", "h")
            for kind in ("decay", "source")
            for axis in "xyz"
        }
        coeff_flags = coeff_flags._replace(
            **{
                name: False
                for name in coeff_flags._fields
                if name not in used_coefficients
            }
        )
        flags = (state_flags, coeff_flags, flags[2], clock_flags)
        # Keep observation arithmetic outside the opaque FFI during tape replay.
        # It is linear in accumulated DFT state; XLA can remove its unused
        # primals while retaining the exact source/monitor transpose.
        out = native_forward(values, observe=False)
        old, coeffs, sources, clock = values
        out = update_monitors(
            program,
            out,
            old.current_step,
            out.t,
            context.dt_scalar,
            out.ex,
            out.ey,
            out.ez,
            out.hx,
            out.hy,
            out.hz,
        )
        old, coeffs, sources, clock = values
        h_source = any(getattr(coeff_flags, f"h_source_{axis}") for axis in "xyz")
        e_source = any(getattr(coeff_flags, f"e_source_{axis}") for axis in "xyz")
        e_decay = any(getattr(coeff_flags, f"e_decay_{axis}") for axis in "xyz")
        saved_old = old._replace(
            **{
                f"e{axis}": getattr(old, f"e{axis}")
                if h_source or e_decay
                else jnp.zeros_like(getattr(old, f"e{axis}"))
                for axis in "xyz"
            },
            **{
                f"h{axis}": getattr(old, f"h{axis}")
                if getattr(coeff_flags, f"h_decay_{axis}")
                else jnp.zeros_like(getattr(old, f"h{axis}"))
                for axis in "xyz"
            },
            cpml_psi_h_terms=old.cpml_psi_h_terms
            if h_source
            else jax.tree.map(jnp.zeros_like, old.cpml_psi_h_terms),
            cpml_psi_e_terms=old.cpml_psi_e_terms
            if e_source
            else jax.tree.map(jnp.zeros_like, old.cpml_psi_e_terms),
            dft_vec_re=jax.tree.map(jnp.zeros_like, old.dft_vec_re),
            dft_vec_im=jax.tree.map(jnp.zeros_like, old.dft_vec_im),
            dft_weight_sum=jax.tree.map(jnp.zeros_like, old.dft_weight_sum),
        )
        # Only material-source VJPs and clock derivatives need post-H values;
        # only clock derivatives need post-E values. Field/DFT Jacobians are
        # linear in these fields and do not need their numeric primals.
        known = jax.tree.map(jnp.zeros_like, out)._replace(
            **{
                f"h{axis}": getattr(out, f"h{axis}")
                if e_source or clock_active
                else jnp.zeros_like(getattr(out, f"h{axis}"))
                for axis in "xyz"
            },
            **{
                f"e{axis}": getattr(out, f"e{axis}")
                if clock_active
                else jnp.zeros_like(getattr(out, f"e{axis}"))
                for axis in "xyz"
            },
        )
        return out, (
            (saved_old, coeffs, sources, clock),
            known,
            _Columns(tuple(jax.tree.leaves(flags))),
        )

    def step_backward(residuals, cotangents):
        inputs, known, columns = residuals
        leaves, tree = jax.tree.flatten(inputs)
        active = tuple(
            value for value, flag in zip(leaves, columns.flags, strict=True) if flag
        )

        def linearize(variables):
            iterator = iter(variables)
            state, coeffs, sources, clock = tree.unflatten(
                [
                    next(iterator) if flag else jax.lax.stop_gradient(value)
                    for value, flag in zip(leaves, columns.flags, strict=True)
                ]
            )
            ctx, _groups = restore_context(sources)
            return forward_step(
                state,
                ctx=ctx,
                coeffs=coeffs,
                program=program,
                update_kernel=update_kernel,
                known_fields=known,
                observation_time=clock[0]
                + (clock[1] + 1).astype(jnp.float32) * ctx.dt_scalar,
            )

        _, pullback = jax.vjp(linearize, active)
        cotangents = _materialize_cotangents(cotangents, known)
        gradients = iter(pullback(cotangents)[0])
        return (
            tree.unflatten(
                [next(gradients) if flag else None for flag in columns.flags]
            ),
        )

    native_step.defvjp(step_forward, step_backward, symbolic_zeros=True)

    def step(state, coefficients, ctx, observation_origin, observation_step_offset):
        return native_step(
            (
                state,
                coefficients,
                source_values(ctx),
                (observation_origin, observation_step_offset),
            )
        )

    def material_backward(inputs, cotangents, flags, nsteps):
        state, coeffs, sources, clock = inputs
        ctx, _groups = restore_context(sources)
        axes = tuple(getattr(flags[1], f"e_source_{axis}") for axis in "xyz")
        train_material = any(axes)
        initial_step = state.current_step

        def replay(carry, index):
            old_psi = carry.cpml_psi_e_terms
            next_state = native_forward(
                (carry, coeffs, sources, (clock[0], clock[1] + index)), observe=False
            )
            tape = (tuple(getattr(next_state, f"h{axis}") for axis in "xyz"), old_psi)
            return next_state, tape

        indices = jnp.arange(nsteps, dtype=jnp.int32)
        if train_material:
            _final, tape = jax.lax.scan(replay, state, indices)
            records = (indices, *tape)
        else:
            # Fixed-coefficient state Jacobians are linear: no field/psi
            # trajectory is needed at all.
            records = indices
        gradients = tuple(
            jnp.zeros_like(getattr(coeffs, f"e_source_{axis}")) if active else _EMPTY
            for axis, active in zip("xyz", axes, strict=True)
        )
        initial_cotangents = _materialize_cotangents(cotangents, state)
        psi_banks = tuple(
            jnp.stack((value, jnp.zeros_like(value)))
            for value in (
                *initial_cotangents.cpml_psi_h_terms,
                *initial_cotangents.cpml_psi_e_terms,
            )
        )
        # Observations do not depend on CPML state. Keep these cotangent leaves
        # out of their transpose; native ping-pong banks carry the real values.
        initial_cotangents = initial_cotangents._replace(
            cpml_psi_h_terms=jax.tree.map(jnp.zeros_like, state.cpml_psi_h_terms),
            cpml_psi_e_terms=jax.tree.map(jnp.zeros_like, state.cpml_psi_e_terms),
        )

        def transpose(carry, record):
            dy, accumulated, banks = carry
            if train_material:
                index, known_h, old_psi_e = record
            else:
                index = record
                known_h, old_psi_e = (
                    (_EMPTY,) * 3,
                    (_EMPTY,) * len(state.cpml_psi_e_terms),
                )
            observation_time = (
                clock[0] + (clock[1] + index + 1).astype(jnp.float32) * ctx.dt_scalar
            )

            def observe(primal):
                return update_monitors(
                    program,
                    primal,
                    initial_step + index,
                    observation_time,
                    ctx.dt_scalar,
                    primal.ex,
                    primal.ey,
                    primal.ez,
                    primal.hx,
                    primal.hy,
                    primal.hz,
                )

            # Only vector DFT recorders enter this path; their field Jacobian
            # is independent of the numeric field and accumulator primals.
            _observed, pullback = jax.vjp(observe, state)
            observed_cotangents = pullback(dy)[0]
            next_dy, next_accumulated, next_banks = reverse_material_step(
                observed_cotangents,
                known_h,
                old_psi_e,
                ctx,
                coeffs,
                accumulated,
                axes,
                banks,
                jnp.int32(nsteps - 1) - index,
            )
            return (next_dy, next_accumulated, next_banks), None

        (state_gradient, material_gradient, final_banks), _history = jax.lax.scan(
            transpose, (initial_cotangents, gradients, psi_banks), records, reverse=True
        )
        nterms = len(state.cpml_psi_h_terms)
        state_gradient = state_gradient._replace(
            t=None,
            current_step=None,
            cpml_psi_h_terms=tuple(value[nsteps & 1] for value in final_banks[:nterms]),
            cpml_psi_e_terms=tuple(value[nsteps & 1] for value in final_banks[nterms:]),
        )
        coefficient_gradient = coeffs._replace(
            **{
                name: (
                    material_gradient["xyz".index(name[-1])]
                    if name in {"e_source_x", "e_source_y", "e_source_z"}
                    else None
                )
                for name in coeffs._fields
            }
        )
        source_gradient = jax.tree.map(lambda _value: None, sources)
        gradients = (
            state_gradient,
            coefficient_gradient,
            source_gradient,
            (None, None),
        )
        tree = jax.tree.structure(inputs)
        values = tree.flatten_up_to(gradients)
        return tree.unflatten(
            [
                gradient if active else None
                for gradient, active in zip(values, jax.tree.leaves(flags), strict=True)
            ]
        )

    def build_chunk(nsteps):
        @jax.custom_vjp
        def native_chunk(inputs):
            return native_forward(inputs, nsteps)

        def chunk_forward(primal_inputs):
            values = jax.tree.map(lambda primal: primal.value, primal_inputs)
            flags = jax.tree.map(
                lambda primal: (
                    bool(primal.perturbed)
                    and jnp.issubdtype(primal.value.dtype, jnp.inexact)
                ),
                primal_inputs,
            )
            # Only the native decay/source leaves participate in this map.
            used = {
                f"{phase}_{kind}_{axis}"
                for phase in ("e", "h")
                for kind in ("decay", "source")
                for axis in "xyz"
            }
            flags = (
                flags[0],
                flags[1]._replace(
                    **{name: False for name in flags[1]._fields if name not in used}
                ),
                flags[2],
                flags[3],
            )
            return native_forward(values, nsteps), (
                values,
                _Columns(tuple(jax.tree.leaves(flags))),
            )

        def chunk_backward(residuals, cotangents):
            inputs, columns = residuals
            leaves, tree = jax.tree.flatten(inputs)
            flags = tree.unflatten(list(columns.flags))
            if (
                not any(jax.tree.leaves(flags[2]))
                and not any(jax.tree.leaves(flags[3]))
                and not any(
                    getattr(flags[1], name)
                    for name in flags[1]._fields
                    if name not in {"e_source_x", "e_source_y", "e_source_z"}
                )
            ):
                return (material_backward(inputs, cotangents, flags, nsteps),)
            active = tuple(
                value for value, flag in zip(leaves, columns.flags, strict=True) if flag
            )

            def replay(variables):
                iterator = iter(variables)
                state, coeffs, sources, clock = tree.unflatten(
                    [
                        next(iterator) if flag else jax.lax.stop_gradient(value)
                        for value, flag in zip(leaves, columns.flags, strict=True)
                    ]
                )
                initial_step = state.current_step

                def body(carry, _unused):
                    return native_step(
                        (
                            carry,
                            coeffs,
                            sources,
                            (
                                clock[0],
                                clock[1] + carry.current_step - initial_step,
                            ),
                        )
                    ), None

                return jax.lax.scan(body, state, xs=None, length=nsteps)[0]

            _, pullback = jax.vjp(replay, active)
            gradients = iter(
                pullback(_materialize_cotangents(cotangents, inputs[0]))[0]
            )
            return (
                tree.unflatten(
                    [next(gradients) if flag else None for flag in columns.flags]
                ),
            )

        native_chunk.defvjp(chunk_forward, chunk_backward, symbolic_zeros=True)

        def chunk(
            state, coefficients, ctx, observation_origin, observation_step_offset
        ):
            return native_chunk(
                (
                    state,
                    coefficients,
                    source_values(ctx),
                    (observation_origin, observation_step_offset),
                )
            )

        return chunk

    return _NativeStepper(step, build_chunk)
