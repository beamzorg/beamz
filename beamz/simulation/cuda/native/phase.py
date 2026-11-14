"""Lower fused 3D Yee phases to the optional BeamZ CUDA typed FFI targets."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0
from beamz.simulation.compile.backend import CUDA_ABI_VERSION
from beamz.simulation.compile.material_blocks import TensorRegion
from beamz.simulation.cuda.native import abi
from beamz.simulation.cuda.native.base import (
    _EMPTY,
    _PHASE_E,
    _PHASE_H,
    _boundary_code,
    _cpml_payload,
    _metric_kind_code,
    _phase_materials,
)
from beamz.simulation.model import SimulationState


@jax.tree_util.register_static
@dataclass(frozen=True)
class _GradientMask:
    bits: int


def _phase_with_vjp(arguments, result_values, attributes):
    """Keep primal buffers immutable and transpose the native phase on CUDA."""
    from beamz.simulation.cuda import runtime as runtime_module

    for index, value in enumerate(arguments):
        if index != 12 and value.dtype != jnp.float32:
            raise ValueError(
                "CUDA autodiff requires dense float32 fields, coefficients, and CPML state"
            )
    forward = runtime_module._ffi_call(abi.CUDA_STREAMED_TARGET, result_values, {})

    @jax.custom_vjp
    def phase_call(*inputs):
        return tuple(forward(*inputs, **attributes))

    def phase_forward(*inputs):
        values = tuple(primal.value for primal in inputs)
        mask = sum(
            1 << i for i, primal in enumerate(inputs) if i != 12 and primal.perturbed
        )
        # Field/psi cotangents depend on the coefficients and output cotangents,
        # not on old fields. Save primal arrays only for coefficient derivatives
        # that actually need them. Empty slots let the adjoint decoder recover
        # unused Yee/psi shapes from cotangents without full-sized zero arrays.
        nterms = int(attributes["nterms"])
        needed = [True] * len(values)

        def active(index):
            return bool(mask & (1 << index))

        for component in range(3):
            needed[component] = active(6 + component)
            needed[3 + component] = False
        for term, (source, axis) in enumerate(
            zip((2, 1, 0, 2, 1, 0), (1, 0, 0, 2, 2, 1), strict=True)
        ):
            component, base = term // 2, 13 + 3 * term
            raw_needed = active(9 + component) or (
                int(attributes["metric_kind"]) != 0 and active(13 + 4 * nterms + axis)
            )
            if nterms:
                raw_needed |= active(base) or active(base + 2)
                needed[13 + 3 * nterms + term] = active(9 + component) or active(
                    base + 1
                )
            needed[3 + source] |= raw_needed
        residuals = tuple(
            value if save else _EMPTY
            for value, save in zip(values, needed, strict=True)
        )
        return tuple(forward(*values, **attributes)), (residuals, _GradientMask(mask))

    def phase_backward(residuals, cotangents):
        inputs, mask = residuals
        # Keep the diagonal target-field transpose visible to XLA. In lossless
        # media this becomes an identity, allowing scan to reuse the cotangent
        # instead of allocating/copying three new native output grids per phase.
        native_mask = mask.bits & ~7
        # Fixed coefficients have zero Jacobian columns. Do not allocate or
        # compute their often highly contended broadcast reductions on CUDA.
        gradient_values = tuple(
            value if native_mask & (1 << i) else _EMPTY
            for i, value in enumerate(arguments)
            if i != 12
        )
        backward = runtime_module._ffi_call(
            abi.CUDA_ADJOINT_TARGET, gradient_values, {}
        )
        cotangents = tuple(
            jnp.zeros(value.shape, value.dtype)
            if isinstance(g, jax.custom_derivatives.SymbolicZero)
            else g
            for g, value in zip(cotangents, result_values, strict=True)
        )
        gradients = iter(
            backward(
                *inputs, *cotangents, gradient_mask=np.int64(native_mask), **attributes
            )
        )
        result = []
        for i in range(len(inputs)):
            if i == 12:
                result.append(None)
            elif i < 3:
                _discard(gradients)
                result.append(
                    _target_cotangent(cotangents[i], inputs[6 + i], i, attributes)
                    if mask.bits & (1 << i)
                    else None
                )
            else:
                result.append(
                    next(gradients) if native_mask & (1 << i) else _discard(gradients)
                )
        return tuple(result)

    phase_call.defvjp(phase_forward, phase_backward, symbolic_zeros=True)
    return phase_call(*arguments)


def reverse_material_step(
    cotangent, known_h, old_psi_e, ctx, coeffs, gradients, axes, psi_banks, iteration
):
    """Transpose E then H, reusing field and coefficient cotangent buffers.

    Source profiles, clocks, CPML coefficients, and metrics are fixed on this
    specialization. The generic VJP remains available for those trainable leaves.
    """
    from beamz.simulation.cuda import runtime as runtime_module

    h_terms, e_terms = ctx.boundary.cpml.h_terms, ctx.boundary.cpml.e_terms
    nterms = len(h_terms)
    if len(e_terms) != nterms or nterms not in (0, 6):
        raise ValueError(
            "The native material transpose requires matching H/E CPML plans"
        )
    fields = tuple(
        getattr(cotangent, name) for name in ("ex", "ey", "ez", "hx", "hy", "hz")
    )
    sources = tuple(getattr(coeffs, f"e_source_{axis}") for axis in "xyz")
    arguments = (
        *(
            getattr(coeffs, f"{phase}_{kind}_{axis}")
            for phase in ("h", "e")
            for kind in ("decay", "source")
            for axis in "xyz"
        ),
        *_cpml_payload(h_terms),
        *_cpml_payload(e_terms),
        *_phase_metrics(ctx, _PHASE_H),
        *_phase_metrics(ctx, _PHASE_E),
        *known_h,
        *old_psi_e,
        *fields,
        *psi_banks,
        *gradients,
        iteration,
    )
    cotangent_start = 23 + 7 * nterms
    material_start = 6 + 2 * nterms
    accumulated_start = cotangent_start + 6 + 2 * nterms
    results = (
        *fields,
        *psi_banks,
        *(
            source if active else _EMPTY
            for source, active in zip(sources, axes, strict=True)
        ),
        *fields[3:],
    )
    aliases = {cotangent_start + index: index for index in range(6 + 2 * nterms)}
    aliases.update(
        {
            accumulated_start + index: material_start + index
            for index, active in enumerate(axes)
            if active
        }
    )
    call = runtime_module._ffi_call(abi.CUDA_MATERIAL_ADJOINT_TARGET, results, aliases)
    outputs = call(
        *arguments,
        abi_version=np.int32(CUDA_ABI_VERSION),
        cuda_flags=np.int32(ctx.config.cuda_flags),
        nterms=np.int32(nterms),
        dt=np.float32(ctx.dt),
        resolution=np.float32(ctx.resolution),
        boundary_code_h=np.int32(
            _boundary_code(ctx.boundary.cpml.metallic_edges, h_terms)
        ),
        boundary_code_e=np.int32(
            _boundary_code(ctx.boundary.cpml.metallic_edges, e_terms)
        ),
        metric_kind=_metric_kind_code(ctx),
        e_gradient_mask=np.int32(
            sum(1 << index for index, active in enumerate(axes) if active)
        ),
    )
    state = cotangent._replace(
        **dict(zip(("ex", "ey", "ez", "hx", "hy", "hz"), outputs[:6], strict=True)),
        cpml_psi_h_terms=cotangent.cpml_psi_h_terms,
        cpml_psi_e_terms=cotangent.cpml_psi_e_terms,
    )
    return (
        state,
        tuple(outputs[material_start : material_start + 3]),
        tuple(outputs[6 : 6 + 2 * nterms]),
    )


def _discard(values):
    next(values)
    return None


def _target_cotangent(cotangent, decay, component, attributes):
    phase, edges = int(attributes["phase"]), int(attributes["boundary_code"]) & 63
    normal = 2 - component
    for axis, size in enumerate(cotangent.shape):
        if (phase == _PHASE_H) != (axis == normal):
            continue
        low, high = edges & (1 << (2 * axis)), edges & (1 << (2 * axis + 1))
        if low or high:
            coordinate = jnp.arange(size).reshape(
                tuple(size if a == axis else 1 for a in range(3))
            )
            mask = ((coordinate == 0) & bool(low)) | (
                (coordinate == size - 1) & bool(high)
            )
            cotangent = jnp.where(mask, jnp.float32(0), cotangent)
    return decay * cotangent


def _phase_metrics(ctx, phase: int):
    """Return CUDA-axis-ordered derivative metrics for one Yee phase."""
    metrics = ctx.metrics
    if phase == _PHASE_H:
        return (metrics.e_to_h_z, metrics.e_to_h_y, metrics.e_to_h_x)
    return (metrics.h_to_e_z, metrics.h_to_e_y, metrics.h_to_e_x)


def _ffi_phase(
    target: str,
    phase: int,
    targets,
    sources,
    materials,
    terms,
    psi_terms,
    metrics,
    *,
    metric_kind,
    dt,
    resolution,
    cuda_flags,
    metallic_edges,
    differentiable=False,
    shard_geometry=None,
    shard_halos=(),
    tensor_payload=None,
):
    from beamz.simulation.cuda import runtime as runtime_module

    if len(terms) != len(psi_terms):
        raise ValueError(
            "CUDA CPML terms and recurrence buffers must have equal length"
        )
    if terms and len(terms) != 6:
        raise ValueError("3D CUDA CPML requires exactly six derivative terms per phase")
    arguments = (
        *targets,
        *sources,
        *materials,
        *_cpml_payload(terms),
        *psi_terms,
        *metrics,
        *((shard_geometry,) if shard_geometry is not None else ()),
        *shard_halos,
    )
    psi_start = 13 + 3 * len(terms)
    aliases = {0: 0, 1: 1, 2: 2}
    aliases.update({psi_start + index: 3 + index for index in range(len(psi_terms))})
    attributes = dict(
        abi_version=np.int32(CUDA_ABI_VERSION),
        cuda_flags=np.int32(cuda_flags),
        phase=np.int32(phase),
        nterms=np.int32(len(terms)),
        dt=np.float32(dt),
        resolution=np.float32(resolution),
        boundary_code=np.int32(_boundary_code(metallic_edges, terms)),
        metric_kind=np.int32(metric_kind),
    )
    if differentiable:
        return _phase_with_vjp(arguments, (*targets, *psi_terms), attributes)
    results = (*targets, *psi_terms)
    if tensor_payload is not None:
        arguments, results, aliases = append_io(
            arguments, results, aliases, tensor_payload
        )
    call = runtime_module._ffi_call(target, results, aliases)
    outputs = call(*arguments, **attributes)
    return tuple(outputs[: 3 + len(terms)])


def update_h(state, ctx, coeffs) -> SimulationState:
    """Advance the three magnetic fields and optional CPML memory on CUDA."""
    return _update_phase(state, ctx, coeffs, phase=_PHASE_H)


def update_e(state, ctx, coeffs) -> SimulationState:
    """Advance the three electric fields and optional CPML memory on CUDA."""
    return _update_phase(state, ctx, coeffs, phase=_PHASE_E)


def _update_phase(state, ctx, coeffs, *, phase):
    # Phase is a Python constant chosen before tracing, including the tensor ABI.
    prefix, other = ("h", "e") if phase == _PHASE_H else ("e", "h")
    terms = getattr(ctx.boundary.cpml, prefix + "_terms")
    tensor_payload = None
    if phase == _PHASE_E and coeffs.e_inverse_offdiagonal.size:
        tensor_payload = payload(ctx, coeffs)
        coeffs = coefficients(ctx, coeffs)
    outputs = _ffi_phase(
        abi.CUDA_STREAMED_TARGET,
        phase,
        tuple(getattr(state, prefix + axis) for axis in "xyz"),
        tuple(getattr(state, other + axis) for axis in "xyz"),
        _phase_materials(coeffs, prefix),
        terms,
        getattr(state, f"cpml_psi_{prefix}_terms"),
        _phase_metrics(ctx, phase),
        metric_kind=_metric_kind_code(ctx),
        dt=ctx.dt,
        resolution=ctx.resolution,
        cuda_flags=ctx.config.cuda_flags,
        metallic_edges=ctx.boundary.cpml.metallic_edges,
        differentiable=ctx.differentiable,
        tensor_payload=tensor_payload,
    )
    return state._replace(
        **dict(zip((prefix + axis for axis in "xyz"), outputs[:3], strict=True)),
        **{f"cpml_psi_{prefix}_terms": outputs[3:]},
    )


def coefficients(ctx, coeffs):
    """Form the true inverse-diagonal update once per compiled invocation."""
    scale = jnp.float32(ctx.dt) / jnp.float32(EPS_0)
    if coeffs.e_source_x.ndim == 2 and coeffs.e_source_x.dtype == jnp.int32:
        return coeffs._replace(
            **{
                f"e_decay_{c}": scale * getattr(coeffs, f"e_inverse_diagonal_{c}")
                for c in "xyz"
            }
        )
    return coeffs._replace(
        **{
            f"e_{kind}_{c}": jnp.float32(1)
            if kind == "decay"
            else scale * getattr(coeffs, f"e_inverse_diagonal_{c}")
            for c in "xyz"
            for kind in ("decay", "source")
        }
    )


def payload(ctx, coeffs):
    if ctx.differentiable:
        raise ValueError("Full-tensor CUDA autodiff requires backend='jax'.")
    cross = coeffs.e_inverse_offdiagonal
    if isinstance(cross, TensorRegion):
        outer = tuple(
            (max(0, lo - 1), min(n, hi + 1))
            for (lo, hi), n in zip(cross.bounds, cross.node_shape, strict=True)
        )
        values = jnp.pad(
            cross.values,
            tuple(
                (lo - start, stop - hi)
                for (lo, hi), (start, stop) in zip(cross.bounds, outer, strict=True)
            )
            + ((0, 0),),
        )
        origin = tuple(lo for lo, _ in outer)
    else:
        values, origin = cross, (0, 0, 0)
    widths = tuple(
        jnp.float32(ctx.resolution)
        if ctx.config.metric_kind == "isotropic_uniform"
        else jnp.reciprocal(getattr(ctx.metrics, "e_to_h_" + axis))
        for axis in "zyx"
    )
    scratch = tuple(
        jnp.empty(
            tuple(n - int(a == 2 - c) for a, n in enumerate(values.shape[:3])),
            dtype=jnp.float32,
        )
        for c in range(3)
    )
    return (values, jnp.asarray(origin, jnp.int32), *widths, *scratch)


def append_io(arguments, results, aliases, tensor):
    aliases = dict(aliases)
    for c in range(3):
        aliases[len(arguments) + abi.TENSOR_INPUT_COUNT - 3 + c] = len(results) + c
    return (*arguments, *tensor), (*results, *tensor[-3:]), aliases


def run_steps(
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
    """Reuse the in-place CUDA graph's source/DFT ordering with a tensor payload."""
    from beamz.simulation.cuda import runtime as rt

    tensor = payload(ctx, coeffs)
    coeffs = coefficients(ctx, coeffs)
    args, results, aliases = rt._graph_io(state, ctx, coeffs)
    args = (*args, *rt._source_group_arguments(groups))
    args, results, aliases, dft_start, count = rt._monitor_io(
        state,
        monitors,
        args,
        results,
        aliases,
        observation_origin=observation_origin,
        observation_step_offset=observation_step_offset,
    )
    args, results, aliases = append_io(args, results, aliases, tensor)
    plan = rt.NativeSchedulePlan(
        layout=abi.PROGRAM_LAYOUT_MONITOR_IN_PLACE,
        flags=rt._native_schedule_flags(ctx, coeffs, sources=True, monitors=count),
        monitor_count=count,
        coincident_source_group_mask=rt._coincident_source_group_mask(groups),
        disjoint_source_group_mask=rt._disjoint_source_group_mask(groups),
    )
    outputs = rt._ffi_call(abi.CUDA_PROGRAM_TARGET, results, aliases)(
        *args, **rt._program_attributes(ctx, nsteps, plan)
    )
    result = (
        rt._replace_graph_outputs(state, outputs)
        if ctx.boundary.cpml.enabled
        else rt._replace_fields(state, outputs)
    )
    return rt._replace_dft(result, outputs, dft_start) if count else result
