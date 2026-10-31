"""Lower fused 3D Yee phases to the optional BeamZ CUDA typed FFI targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from beamz.simulation import _cuda_abi as abi
from beamz.simulation.backend import CUDA_ABI_VERSION
from beamz.simulation.model import SimulationState

_PHASE_H = 0
_PHASE_E = 1
_COMPONENT_CODE = {name: index for index, name in enumerate(("Hx", "Hy", "Hz"))}
_COMPONENT_CODE.update({name: index for index, name in enumerate(("Ex", "Ey", "Ez"))})
_EMPTY = jnp.empty((0,), dtype=jnp.float32)
_EMPTY_SOURCE_GROUP = (
    jnp.zeros((0, 1, 1, 1), dtype=jnp.float32),
    jnp.zeros((0, 1), dtype=jnp.float32),
    jnp.zeros((0, 3), dtype=jnp.int32),
)
_METRIC_KIND_CODE = {
    "isotropic_uniform": 0,
    "axis_uniform": 1,
    "rectilinear": 2,
}

_FIELD_COUNT = abi.FIELD_COUNT
_CPML_TERM_COUNT = abi.CPML_TERM_COUNT
_SOURCE_GROUP_COUNT = abi.SOURCE_GROUP_COUNT
_CPML_H_PSI_INPUT = abi.CPML_H_PSI_INPUT_OFFSET
_CPML_E_PSI_INPUT = abi.CPML_E_PSI_INPUT_OFFSET
_TEMPORAL_FIELD_WORKSPACE_INPUT = abi.TEMPORAL_FIELD_WORKSPACE_INPUT
_TEMPORAL_PSI_WORKSPACE_INPUT = abi.TEMPORAL_PSI_WORKSPACE_INPUT


@dataclass(frozen=True, slots=True)
class NativeSchedulePlan:
    """Validated compilation-time choices consumed by the native program ABI.

    The native launcher revalidates these capability bits against decoded buffer
    shapes before capture.  Keeping the decision here prevents Python, FFI
    decoding, program scheduling, and leaf kernels from independently guessing
    which specialization is sound.
    """

    layout: int
    flags: int
    monitor_count: int = 0
    coincident_source_group_mask: int = 0
    disjoint_source_group_mask: int = 0
    logical_shape: tuple[int, int, int] = (0, 0, 0)
    temporal_steps: int = 1

    @property
    def uses_temporal_fields(self) -> bool:
        return bool(self.flags & abi.NATIVE_SCHEDULE_TEMPORAL)


def _cpml_payload(terms):
    """Metadata and profile arrays in the shared forward/transpose ABI order."""
    return (
        _term_metadata(terms),
        *(value for term in terms for value in (term.a, term.b, term.inv_kappa)),
    )


def _phase_materials(coeffs, phase):
    """Native ABI order: x/y/z decay, then x/y/z source; arrays stay dynamic."""
    return tuple(
        getattr(coeffs, f"{phase}_{kind}_{axis}")
        for kind in ("decay", "source")
        for axis in "xyz"
    )


def _metallic_edge_mask(edges: frozenset[str]) -> int:
    order = ("front", "back", "bottom", "top", "left", "right")
    return sum(1 << index for index, name in enumerate(order) if name in edges)


def _boundary_code(edges: frozenset[str], terms=()) -> int:
    """Pack PEC faces and a CPML thickness safe for every supplied phase.

    A non-zero CPML thickness selects a CUDA specialization which deliberately
    skips the per-term packed-slab descriptor.  It is therefore valid only when
    *all* supplied H and E recurrences have the same symmetric slab.  Phase-only
    FFI calls pass one phase; native program calls pass both phases.
    """
    thickness = 0
    if terms:
        first = int(terms[0].slab.low)
        if first > 0 and all(
            int(term.slab.low) == first and int(term.slab.high) == first
            for term in terms
        ):
            thickness = first
    return _metallic_edge_mask(edges) | (thickness << 8)


def _term_metadata(terms) -> jnp.ndarray:
    return jnp.asarray(
        [
            (
                _COMPONENT_CODE[term.component],
                int(term.axis),
                int(term.slab.low),
                int(term.slab.high),
                1 if float(term.sign) > 0.0 else -1,
            )
            for term in terms
        ],
        dtype=jnp.int32,
    ).reshape((len(terms), 5))


def _shape(value):
    return jax.ShapeDtypeStruct(value.shape, value.dtype)


def _fields(state):
    return (state.hx, state.hy, state.hz, state.ex, state.ey, state.ez)


def _ffi_call(target, result_values, aliases):
    return jax.ffi.ffi_call(
        target,
        tuple(_shape(value) for value in result_values),
        input_output_aliases=aliases,
        vmap_method="sequential",
    )


@jax.tree_util.register_static
@dataclass(frozen=True)
class _GradientMask:
    bits: int


def _phase_with_vjp(arguments, result_values, attributes):
    """Keep primal buffers immutable and transpose the native phase on CUDA."""
    for index, value in enumerate(arguments):
        if index != 12 and value.dtype != jnp.float32:
            raise ValueError(
                "CUDA autodiff requires dense float32 fields, coefficients, and CPML state"
            )
    forward = _ffi_call(abi.CUDA_STREAMED_TARGET, result_values, {})

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
        backward = _ffi_call(abi.CUDA_ADJOINT_TARGET, gradient_values, {})
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
    call = _ffi_call(abi.CUDA_MATERIAL_ADJOINT_TARGET, results, aliases)
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


def _metric_kind_code(ctx) -> np.int32:
    try:
        return np.int32(_METRIC_KIND_CODE[ctx.config.metric_kind])
    except KeyError as exc:
        raise ValueError(
            f"Unsupported CUDA derivative metric kind: {ctx.config.metric_kind!r}"
        ) from exc


def _program_attributes(ctx, nsteps: int, plan: NativeSchedulePlan):
    return {
        "abi_version": np.int32(CUDA_ABI_VERSION),
        "cuda_flags": np.int32(ctx.config.cuda_flags),
        "graph_cache_capacity": np.int32(ctx.config.cuda_graph_cache_capacity),
        "nsteps": np.int32(nsteps),
        "dt": np.float32(ctx.dt),
        "resolution": np.float32(ctx.resolution),
        "boundary_code": np.int32(
            _boundary_code(
                ctx.boundary.cpml.metallic_edges,
                (*ctx.boundary.cpml.h_terms, *ctx.boundary.cpml.e_terms),
            )
        ),
        "metric_kind": _metric_kind_code(ctx),
        "program_layout": np.int32(plan.layout),
        "cpml_enabled": np.int32(ctx.boundary.cpml.enabled),
        "monitor_count": np.int32(plan.monitor_count),
        "coincident_source_group_mask": np.int32(plan.coincident_source_group_mask),
        "disjoint_source_group_mask": np.int32(plan.disjoint_source_group_mask),
        "schedule_flags": np.int32(plan.flags),
        "logical_z": np.int32(plan.logical_shape[0]),
        "logical_y": np.int32(plan.logical_shape[1]),
        "logical_x": np.int32(plan.logical_shape[2]),
        "temporal_steps": np.int32(plan.temporal_steps),
    }


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
        from .tensor import append_io

        arguments, results, aliases = append_io(
            arguments, results, aliases, tensor_payload
        )
    call = _ffi_call(target, results, aliases)
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
        from .tensor import coefficients, payload

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


def _cpml_graph_io(state, ctx, coeffs):
    fields = _fields(state)
    h_terms = ctx.boundary.cpml.h_terms
    e_terms = ctx.boundary.cpml.e_terms
    h_materials = _phase_materials(coeffs, "h")
    e_materials = _phase_materials(coeffs, "e")
    h_payload = (
        *h_materials,
        *_cpml_payload(h_terms),
        *state.cpml_psi_h_terms,
    )
    e_payload = (
        *e_materials,
        *_cpml_payload(e_terms),
        *state.cpml_psi_e_terms,
    )
    arguments = (
        *fields,
        *h_payload,
        *e_payload,
        *_phase_metrics(ctx, _PHASE_H),
        *_phase_metrics(ctx, _PHASE_E),
    )
    result_values = (*fields, *state.cpml_psi_h_terms, *state.cpml_psi_e_terms)
    aliases = {index: index for index in range(_FIELD_COUNT)}
    aliases.update(
        {
            _CPML_H_PSI_INPUT + index: _FIELD_COUNT + index
            for index in range(_CPML_TERM_COUNT)
        }
    )
    aliases.update(
        {
            _CPML_E_PSI_INPUT + index: _FIELD_COUNT + _CPML_TERM_COUNT + index
            for index in range(_CPML_TERM_COUNT)
        }
    )
    return arguments, result_values, aliases


def _yee_graph_io(state, ctx, coeffs):
    fields = _fields(state)
    materials = (*_phase_materials(coeffs, "h"), *_phase_materials(coeffs, "e"))
    arguments = (
        *fields,
        *materials,
        *_phase_metrics(ctx, _PHASE_H),
        *_phase_metrics(ctx, _PHASE_E),
    )
    return arguments, fields, {index: index for index in range(_FIELD_COUNT)}


def _graph_io(state, ctx, coeffs):
    return (
        _cpml_graph_io(state, ctx, coeffs)
        if ctx.boundary.cpml.enabled
        else _yee_graph_io(state, ctx, coeffs)
    )


def _replace_fields(state, outputs, start=0) -> SimulationState:
    return state._replace(
        hx=outputs[start],
        hy=outputs[start + 1],
        hz=outputs[start + 2],
        ex=outputs[start + 3],
        ey=outputs[start + 4],
        ez=outputs[start + 5],
    )


def _replace_graph_outputs(state, outputs) -> SimulationState:
    return _replace_fields(state, outputs)._replace(
        cpml_psi_h_terms=outputs[_FIELD_COUNT : _FIELD_COUNT + _CPML_TERM_COUNT],
        cpml_psi_e_terms=outputs[
            _FIELD_COUNT + _CPML_TERM_COUNT : _FIELD_COUNT + 2 * _CPML_TERM_COUNT
        ],
    )


def _field_padding(ctx):
    flags = ctx.config.cuda_flags
    x = (
        64
        if flags & abi.CUDA_FIELD_PAD64
        else 32
        if flags & abi.CUDA_FIELD_PAD32
        else 1
    )
    y = 8 if x > 1 and flags & abi.CUDA_FIELD_PAD_Y8 else 1
    return y, x


def _temporal_cpml_graph_io(state, ctx, coeffs, temporal_steps=1):
    arguments, _, _ = _cpml_graph_io(state, ctx, coeffs)
    fields = _fields(state)
    py, px = _field_padding(ctx)
    if px > 1:
        fields = tuple(
            jnp.pad(
                value, ((0, 0), (0, -value.shape[1] % py), (0, -value.shape[2] % px))
            )
            for value in fields
        )
        arguments = (*fields, *arguments[_FIELD_COUNT:])
    workspace = tuple(jnp.empty_like(value) for value in fields)
    psi = (*state.cpml_psi_h_terms, *state.cpml_psi_e_terms)
    psi_workspace = tuple(jnp.empty_like(value) for value in psi)
    result_values = (*fields, *workspace, *psi, *psi_workspace)
    aliases = {index: index for index in range(_FIELD_COUNT)}
    aliases.update(
        {
            _TEMPORAL_FIELD_WORKSPACE_INPUT + index: _FIELD_COUNT + index
            for index in range(_FIELD_COUNT)
        }
    )
    aliases.update(
        {
            _CPML_H_PSI_INPUT + index: 2 * _FIELD_COUNT + index
            for index in range(_CPML_TERM_COUNT)
        }
    )
    aliases.update(
        {
            _CPML_E_PSI_INPUT + index: 2 * _FIELD_COUNT + _CPML_TERM_COUNT + index
            for index in range(_CPML_TERM_COUNT)
        }
    )
    aliases.update(
        {
            _TEMPORAL_PSI_WORKSPACE_INPUT + index: 2 * _FIELD_COUNT
            + 2 * _CPML_TERM_COUNT
            + index
            for index in range(2 * _CPML_TERM_COUNT)
        }
    )
    arguments = (*arguments, *workspace, *psi_workspace)
    if temporal_steps == 2:
        pair_fields = tuple(jnp.empty_like(value) for value in fields)
        aliases.update(
            {len(arguments) + i: len(result_values) + i for i in range(_FIELD_COUNT)}
        )
        arguments = (*arguments, *pair_fields)
        result_values = (*result_values, *pair_fields)
    return arguments, result_values, aliases


def _replace_temporal_cpml_outputs(
    state, outputs, nsteps, temporal_steps=1, cpml_pair=False
):
    field_start = 0 if nsteps % 2 == 0 else _FIELD_COUNT
    if temporal_steps == 2 and nsteps % 4 == 2:
        field_start = 2 * _FIELD_COUNT + 4 * _CPML_TERM_COUNT
    psi_start = 2 * _FIELD_COUNT
    psi_swaps = (
        nsteps // 2 + nsteps % 2 if cpml_pair and temporal_steps == 2 else nsteps
    )
    if psi_swaps % 2:
        psi_start += 2 * _CPML_TERM_COUNT
    fields = tuple(
        value[tuple(slice(0, n) for n in original.shape)]
        for value, original in zip(
            outputs[field_start : field_start + _FIELD_COUNT],
            _fields(state),
            strict=True,
        )
    )
    return _replace_fields(state, fields)._replace(
        cpml_psi_h_terms=outputs[psi_start : psi_start + _CPML_TERM_COUNT],
        cpml_psi_e_terms=outputs[
            psi_start + _CPML_TERM_COUNT : psi_start + 2 * _CPML_TERM_COUNT
        ],
    )


def _source_group_arguments(groups):
    if len(groups) != _SOURCE_GROUP_COUNT:
        raise ValueError("CUDA source graph requires nine phase/component groups")
    return tuple(
        value
        for group in groups
        for value in (
            _EMPTY_SOURCE_GROUP
            if group is None
            else (group.coeffs, group.waveforms, group.starts)
        )
    )


def _coincident_source_group_mask(groups) -> int:
    """Encode equal-shape source groups whose slabs share one origin."""
    mask = 0
    for index, group in enumerate(groups):
        if group is None:
            continue
        starts = getattr(group, "starts_tuple", ())
        if starts and all(start == starts[0] for start in starts[1:]):
            mask |= 1 << index
    return mask


def _disjoint_source_group_mask(groups: tuple[Any, ...]) -> int:
    """Encode groups whose static slabs cannot write the same cell.

    The source starts are compiler-owned literals.  A group earns this flag only
    when every pair of equal-shaped slabs is separated along at least one axis;
    dynamic or unavailable origins deliberately retain the atomic implementation.
    """
    mask = 0
    for index, group in enumerate(groups):
        if group is None:
            continue
        raw_starts: Any = getattr(group, "starts_tuple", ())
        starts: tuple[Any, ...] = tuple(raw_starts)
        if len(starts) < 2 or any(len(start) != 3 for start in starts):
            continue
        extents = tuple(int(size) for size in group.coeffs.shape[1:])
        if len(extents) != 3:
            continue
        if all(
            any(
                first[axis] + extents[axis] <= second[axis]
                or second[axis] + extents[axis] <= first[axis]
                for axis in range(3)
            )
            for first_index, first in enumerate(starts)
            for second in starts[first_index + 1 :]
        ):
            mask |= 1 << index
    return mask


def _uniform_cpml_thickness(ctx) -> int:
    """Return the one thickness shared by every H/E recurrence, or zero."""
    return (
        _boundary_code(
            ctx.boundary.cpml.metallic_edges,
            (*ctx.boundary.cpml.h_terms, *ctx.boundary.cpml.e_terms),
        )
        >> 8
    )


def _packed_e_material(coeffs) -> bool:
    values = _phase_materials(coeffs, "e")
    return all(value.ndim == 1 for value in values[:3]) and all(
        value.ndim in (1, 2) and value.dtype == jnp.int32
        for value in (coeffs.e_source_x, coeffs.e_source_y, coeffs.e_source_z)
    )


def _combined_cpml_core_supported(state, ctx, coeffs) -> bool:
    """Prove the combined CPML queue has a non-empty regular-grid core.

    Material encoding is independent of the spatial partition: scalar, dense,
    and codebook E coefficients all use the same disjoint core/shell boxes.
    Dense H coefficients retain the general schedule until they benefit.
    """
    if (
        not ctx.boundary.cpml.enabled
        or ctx.config.metric_kind != "isotropic_uniform"
        or _uniform_cpml_thickness(ctx) <= 0
    ):
        return False
    # Dense H coefficients did not benefit from the combined queue or a
    # second bank in the conductive benchmark; retain the general schedule.
    if any(
        getattr(coeffs, f"h_{kind}_{axis}").ndim != 0
        for kind in ("decay", "source")
        for axis in "xyz"
    ):
        return False
    thickness = _uniform_cpml_thickness(ctx)
    return all(
        min(int(field.shape[axis]) for field in _fields(state)[:3]) > 2 * thickness
        for axis in range(3)
    )


def _temporal_cpml_fields_supported(ctx, coeffs, nsteps: int) -> bool:
    """Prove that a second field bank can preserve the CPML timestep order."""
    if (
        getattr(ctx, "low_memory", False)
        or nsteps < 2
        or not ctx.boundary.cpml.enabled
        or ctx.config.metric_kind != "isotropic_uniform"
        or _uniform_cpml_thickness(ctx) <= 0
    ):
        return False
    h_values = _phase_materials(coeffs, "h")
    e_sources = (coeffs.e_source_x, coeffs.e_source_y, coeffs.e_source_z)
    return all(value.ndim == 0 for value in h_values) and all(
        value.size > 0
        and (
            value.ndim in (0, 3) or (value.ndim in (1, 2) and value.dtype == jnp.int32)
        )
        for value in e_sources
    )


def _temporal_yee_supported(ctx, coeffs, nsteps: int) -> bool:
    """Prove that the frozen-field Yee kernel supports this material layout."""
    if (
        getattr(ctx, "low_memory", False)
        or nsteps < 4
        or _metallic_edge_mask(ctx.boundary.cpml.metallic_edges) != 63
    ):
        return False
    values = (*_phase_materials(coeffs, "h"), *_phase_materials(coeffs, "e"))
    return all(value.ndim in (0, 3) for value in values)


def _native_schedule_flags(
    ctx, coeffs, *, sources=False, monitors=0, temporal=False, combined=False
):
    flags = 0
    if ctx.boundary.cpml.enabled:
        flags |= abi.NATIVE_SCHEDULE_CPML
    if temporal:
        flags |= abi.NATIVE_SCHEDULE_TEMPORAL
    if _uniform_cpml_thickness(ctx) > 0:
        flags |= abi.NATIVE_SCHEDULE_UNIFORM_CPML
    if _packed_e_material(coeffs):
        flags |= abi.NATIVE_SCHEDULE_PACKED_MATERIAL
    if combined:
        flags |= abi.NATIVE_SCHEDULE_COMBINED_CPML_CORE
    if sources:
        flags |= abi.NATIVE_SCHEDULE_SOURCES
    if monitors:
        flags |= abi.NATIVE_SCHEDULE_MONITORS
    if ctx.config.cuda_flags & abi.CUDA_GRAPH_CACHE:
        flags |= abi.NATIVE_SCHEDULE_GRAPH_CACHE
    return flags


def _native_schedule_plan(
    state,
    ctx,
    coeffs,
    nsteps: int,
    *,
    kind: str,
    groups=(),
    monitor_count: int = 0,
) -> NativeSchedulePlan:
    """Choose every native fast path once and serialize that proof through FFI."""
    cpml = bool(ctx.boundary.cpml.enabled)
    combined_cpml_core = _combined_cpml_core_supported(state, ctx, coeffs)
    temporal_cpml = _temporal_cpml_fields_supported(ctx, coeffs, nsteps)
    if kind == "steps":
        if temporal_cpml:
            layout = abi.PROGRAM_LAYOUT_SOURCE_TEMPORAL_CPML
        elif cpml:
            layout = abi.PROGRAM_LAYOUT_CPML_IN_PLACE
        elif _temporal_yee_supported(ctx, coeffs, nsteps):
            layout = abi.PROGRAM_LAYOUT_YEE_TEMPORAL
        else:
            layout = abi.PROGRAM_LAYOUT_YEE_IN_PLACE
    elif kind == "source":
        layout = (
            abi.PROGRAM_LAYOUT_SOURCE_TEMPORAL_CPML
            if temporal_cpml
            else abi.PROGRAM_LAYOUT_SOURCE_IN_PLACE
        )
    elif kind == "monitor":
        layout = (
            abi.PROGRAM_LAYOUT_MONITOR_TEMPORAL_CPML
            if temporal_cpml
            else abi.PROGRAM_LAYOUT_MONITOR_IN_PLACE
        )
    else:
        raise ValueError(f"unknown native schedule family: {kind!r}")

    flags = _native_schedule_flags(
        ctx,
        coeffs,
        sources=(
            kind in {"source", "monitor"}
            or layout == abi.PROGRAM_LAYOUT_SOURCE_TEMPORAL_CPML
        ),
        monitors=monitor_count,
        temporal=layout
        in {
            abi.PROGRAM_LAYOUT_YEE_TEMPORAL,
            abi.PROGRAM_LAYOUT_SOURCE_TEMPORAL_CPML,
            abi.PROGRAM_LAYOUT_MONITOR_TEMPORAL_CPML,
        },
        combined=combined_cpml_core,
    )
    return NativeSchedulePlan(
        layout=layout,
        flags=flags,
        monitor_count=monitor_count,
        coincident_source_group_mask=_coincident_source_group_mask(groups),
        disjoint_source_group_mask=_disjoint_source_group_mask(groups),
        logical_shape=(state.hx.shape[0], state.hx.shape[1], state.hy.shape[2])
        if temporal_cpml and _field_padding(ctx)[1] > 1
        else (0, 0, 0),
        temporal_steps=2
        if (
            temporal_cpml
            and combined_cpml_core
            and ctx.config.cuda_flags & abi.CUDA_TEMPORAL_PAIR
            and all(group is None for group in groups[:3])
            and min(state.hx.shape[0], state.hx.shape[1], state.hy.shape[2])
            > 2 * (_uniform_cpml_thickness(ctx) + 2)
        )
        else 1,
    )


def _scheduled_graph_io(state, ctx, coeffs, plan):
    return (
        _temporal_cpml_graph_io(state, ctx, coeffs, plan.temporal_steps)
        if plan.uses_temporal_fields and ctx.boundary.cpml.enabled
        else _graph_io(state, ctx, coeffs)
    )


def _replace_native_outputs(state, ctx, outputs, nsteps, plan):
    if plan.uses_temporal_fields and ctx.boundary.cpml.enabled:
        return _replace_temporal_cpml_outputs(
            state,
            outputs,
            nsteps,
            plan.temporal_steps,
            bool(ctx.config.cuda_flags & abi.CUDA_CPML_PAIR),
        )
    if ctx.boundary.cpml.enabled:
        return _replace_graph_outputs(state, outputs)
    return _replace_fields(state, outputs)


def run_source_group_steps(state, ctx, coeffs, groups, nsteps: int) -> SimulationState:
    """Advance packed slab-source groups in every leapfrog phase on CUDA."""
    if nsteps < 1:
        raise ValueError("CUDA step count must be positive")
    source_arguments = _source_group_arguments(groups)
    plan = _native_schedule_plan(
        state, ctx, coeffs, nsteps, kind="source", groups=groups
    )
    arguments, result_values, aliases = _scheduled_graph_io(state, ctx, coeffs, plan)
    call = _ffi_call(abi.CUDA_PROGRAM_TARGET, result_values, aliases)
    outputs = call(
        *arguments,
        *source_arguments,
        state.current_step,
        **_program_attributes(ctx, nsteps, plan),
    )
    return _replace_native_outputs(state, ctx, outputs, nsteps, plan)


def pack_dft_monitors(monitors):
    """Pack DFT gather plans with offsets into ragged accumulator arenas."""
    if not monitors:
        raise ValueError("CUDA DFT graph requires at least one monitor")
    if any(
        np.any(np.asarray(indices) > np.iinfo(np.int32).max)
        for monitor in monitors
        for indices in monitor.dft_flat_idx
    ):
        raise ValueError("Native DFT graph indices exceed int32; use a sharded run")
    max_points = max(int(monitor.dft_point_count) for monitor in monitors)
    max_frequencies = max(int(monitor.freq_count) for monitor in monitors)
    max_neighbors = max(
        int(indices.shape[1])
        for monitor in monitors
        for indices in monitor.dft_flat_idx
    )

    def pad_plan(value, points, neighbors, *, fill=0):
        return jnp.pad(
            value,
            ((0, points - value.shape[0]), (0, neighbors - value.shape[1])),
            constant_values=fill,
        )

    def pack_plans(attribute, dtype):
        return jnp.stack(
            tuple(
                jnp.stack(
                    tuple(
                        pad_plan(value, max_points, max_neighbors)
                        for value in getattr(monitor, attribute)
                    )
                )
                for monitor in monitors
            )
        ).astype(dtype)

    indices = pack_plans("dft_flat_idx", jnp.int32)
    weights = pack_plans("dft_weights", jnp.float32)
    frequencies = jnp.stack(
        tuple(
            jnp.pad(
                monitor.freq_hz,
                (0, max_frequencies - monitor.freq_count),
            )
            for monitor in monitors
        )
    ).astype(jnp.float32)
    component_masks = jnp.stack(
        tuple(monitor.dft_component_mask for monitor in monitors)
    ).astype(jnp.float32)
    counts = jnp.asarray(
        tuple(
            (
                monitor.freq_count,
                monitor.dft_point_count,
                monitor.dft_record_interval,
                monitor.dft_value_offset,
                monitor.dft_weight_offset,
            )
            for monitor in monitors
        ),
        dtype=jnp.int32,
    )
    codes = jnp.asarray(
        tuple(
            (monitor.dft_window_code, monitor.dft_normalization_code)
            for monitor in monitors
        ),
        dtype=jnp.int32,
    )
    windows = jnp.asarray(
        tuple(
            (monitor.dft_t_start, monitor.dft_t_end, monitor.dft_length_unit)
            for monitor in monitors
        ),
        dtype=jnp.float32,
    )
    return (
        indices,
        weights,
        frequencies,
        component_masks,
        counts,
        codes,
        windows,
    )


def _pair_publication_mask(state, packed_monitors):
    """Conservative 16x8x1 map of intermediate fields read by any DFT gather.

    Includes every component and interpolation neighbor, independent of record
    interval/window. Static gather plans let XLA fold this once at compilation.
    Logical coordinates deliberately exclude storage padding.
    """
    fields = (state.ex, state.ey, state.ez, state.hx, state.hy, state.hz)
    nz, ny, nx = (max(field.shape[a] for field in fields) for a in range(3))
    mask = jnp.zeros((nz, (ny + 7) // 8, (nx + 15) // 16), dtype=jnp.int32)
    for component, field in enumerate(fields):
        indices = packed_monitors[0][:, component].reshape(-1)
        z = indices // (field.shape[1] * field.shape[2])
        y = (indices // field.shape[2]) % field.shape[1]
        x = indices % field.shape[2]
        valid = (indices >= 0) & (indices < field.size)
        # Invalid indices are dropped; duplicate updates all write the same 1.
        mask = mask.at[jnp.where(valid, z, nz), y // 8, x // 16].set(1, mode="drop")
    return mask


def _monitor_io(
    state,
    packed,
    arguments,
    values,
    aliases,
    *,
    observation_origin,
    observation_step_offset,
    phase_steps=1,
):
    """Lower the common DFT arena, phase workspace and absolute clocks."""
    if packed is None:
        packed = (
            jnp.zeros((0, 6, 1, 1), jnp.int32),
            jnp.zeros((0, 6, 1, 1), jnp.float32),
            jnp.zeros((0, 1), jnp.float32),
            jnp.zeros((0, 6), jnp.float32),
            jnp.zeros((0, 5), jnp.int32),
            jnp.zeros((0, 2), jnp.int32),
            jnp.zeros((0, 3), jnp.float32),
        )
        dft = (jnp.zeros((0,), jnp.float32),) * 3
    else:
        dft = (state.dft_vec_re, state.dft_vec_im, state.dft_weight_sum)
        if any(value.dtype != jnp.float32 for value in dft):
            raise ValueError("CUDA program graph requires float32 DFT accumulators")
    count, frequencies = packed[2].shape
    phase = tuple(
        jnp.empty((max(1, count) * phase_steps, frequencies), jnp.float32)
        for _ in range(3)
    )
    arguments = (*arguments, *packed)
    monitor_start = len(values)
    aliases = dict(aliases)
    for value in (*dft, *phase):
        aliases[len(arguments)] = len(values)
        arguments = (*arguments, value)
        values = (*values, value)
    return (
        (
            *arguments,
            observation_origin,
            state.current_step,
            jnp.asarray(observation_step_offset, jnp.int32),
        ),
        values,
        aliases,
        monitor_start,
        count,
    )


def _replace_dft(state, outputs, start):
    return state._replace(
        dft_vec_re=outputs[start],
        dft_vec_im=outputs[start + 1],
        dft_weight_sum=outputs[start + 2],
    )


def run_native_steps(
    state,
    ctx,
    coeffs,
    groups,
    packed_monitors,
    nsteps,
    *,
    observation_origin,
    observation_step_offset,
    native_calls=None,
):
    """Dispatch shared CUDA execution, including wrapped storage layouts."""
    if coeffs.e_inverse_offdiagonal.size:
        from .tensor import run_steps as run_tensor_steps

        return run_tensor_steps(
            state,
            ctx,
            coeffs,
            groups,
            packed_monitors,
            nsteps,
            observation_origin=observation_origin,
            observation_step_offset=observation_step_offset,
        )
    if native_calls is None:
        from beamz.simulation import cuda

        native_calls = (
            cuda.run_steps,
            cuda.run_program_steps,
            cuda.run_source_group_steps,
        )
    ordinary, observed, sourced = native_calls
    if packed_monitors is not None:
        return observed(
            state,
            ctx,
            coeffs,
            groups,
            packed_monitors,
            nsteps,
            observation_origin=observation_origin,
            observation_step_offset=observation_step_offset,
        )
    if any(group is not None for group in groups):
        return sourced(state, ctx, coeffs, groups, nsteps)
    return ordinary(state, ctx, coeffs, nsteps)


def run_program_steps(
    state,
    ctx,
    coeffs,
    groups,
    packed_monitors,
    nsteps: int,
    *,
    observation_origin=None,
    observation_step_offset: int | jax.Array = 0,
) -> SimulationState:
    """Advance arbitrary slab sources and packed vector DFTs in one CUDA graph."""
    if nsteps < 1:
        raise ValueError("CUDA step count must be positive")
    if state.dft_vec_re.dtype != jnp.float32:
        raise ValueError("CUDA program graph requires float32 DFT accumulators")
    source_arguments = _source_group_arguments(groups)
    plan = _native_schedule_plan(
        state,
        ctx,
        coeffs,
        nsteps,
        kind="monitor",
        groups=groups,
        monitor_count=int(packed_monitors[0].shape[0]),
    )
    arguments, result_values, aliases = _scheduled_graph_io(state, ctx, coeffs, plan)
    arguments = (*arguments, *source_arguments)
    arguments, result_values, aliases, state_output_count, _ = _monitor_io(
        state,
        packed_monitors,
        arguments,
        result_values,
        aliases,
        observation_origin=state.t
        if observation_origin is None
        else observation_origin,
        observation_step_offset=observation_step_offset,
        phase_steps=plan.temporal_steps,
    )
    if plan.temporal_steps == 2:
        result_values = (
            *result_values,
            jnp.empty(
                (
                    int(packed_monitors[0].shape[0]),
                    6,
                    int(packed_monitors[0].shape[2]),
                    2,
                ),
                dtype=jnp.float32,
            ),
        )
        arguments = (*arguments, _pair_publication_mask(state, packed_monitors))
    outputs = _ffi_call(abi.CUDA_PROGRAM_TARGET, result_values, aliases)(
        *arguments,
        **_program_attributes(ctx, nsteps, plan),
    )
    next_state = _replace_native_outputs(state, ctx, outputs, nsteps, plan)
    return _replace_dft(next_state, outputs, state_output_count)


def run_steps(state, ctx, coeffs, nsteps: int) -> SimulationState:
    """Advance a source-free, monitor-free Yee run through one CUDA FFI call."""
    if nsteps < 1:
        raise ValueError("CUDA step count must be positive")
    plan = _native_schedule_plan(state, ctx, coeffs, nsteps, kind="steps")
    if plan.layout == abi.PROGRAM_LAYOUT_SOURCE_TEMPORAL_CPML:
        # The source-group target's empty groups add no graph nodes. Reusing it
        # gives plain CPML runs the same frozen-input field banks without
        # maintaining a second native handler for an identical update graph.
        return run_source_group_steps(
            state, ctx, coeffs, (None,) * _SOURCE_GROUP_COUNT, nsteps
        )
    arguments, results, aliases = _graph_io(state, ctx, coeffs)
    if plan.layout == abi.PROGRAM_LAYOUT_YEE_TEMPORAL:
        fields = _fields(state)
        workspace = tuple(jnp.empty_like(value) for value in fields)
        arguments = (*fields, *workspace, *arguments[_FIELD_COUNT:])
        results = (*fields, *workspace)
        aliases = {index: index for index in range(2 * _FIELD_COUNT)}
    outputs = _ffi_call(abi.CUDA_PROGRAM_TARGET, results, aliases)(
        *arguments,
        **_program_attributes(ctx, nsteps, plan),
    )
    return _replace_native_outputs(state, ctx, outputs, nsteps, plan)
