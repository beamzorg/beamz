"""Lower fused 3D Yee phases to the optional BeamZ CUDA typed FFI targets."""

from __future__ import annotations

import os
from typing import Any

import jax.numpy as jnp
import numpy as np

from beamz.simulation import _cuda_abi as abi
from beamz.simulation.model import SimulationState

from .base import (
    _CPML_E_PSI_INPUT,
    _CPML_H_PSI_INPUT,
    _CPML_TERM_COUNT,
    _EMPTY_SOURCE_GROUP,
    _FIELD_COUNT,
    _PHASE_E,
    _PHASE_H,
    _SOURCE_GROUP_COUNT,
    _TEMPORAL_FIELD_WORKSPACE_INPUT,
    _TEMPORAL_PSI_WORKSPACE_INPUT,
    NativeSchedulePlan,
    _boundary_code,
    _cpml_payload,
    _fields,
    _metallic_edge_mask,
    _metric_kind_code,  # noqa: F401
    _phase_materials,
)
from .phase import (  # noqa: F401
    _discard,
    _ffi_phase,
    _GradientMask,
    _phase_metrics,
    _phase_with_vjp,
    _target_cotangent,
    _update_phase,
    reverse_material_step,
    update_e,
    update_h,
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


def _temporal_cpml_fields_supported(ctx, coeffs, nsteps: int, state=None) -> bool:
    """Prove that a second field bank can preserve the CPML timestep order."""
    if (
        getattr(ctx, "low_memory", False)
        or nsteps < 2
        or not ctx.boundary.cpml.enabled
        or ctx.config.metric_kind != "isotropic_uniform"
        or _uniform_cpml_thickness(ctx) <= 0
    ):
        return False
    if (
        state is not None
        and ctx.config.cuda_memory_policy == "auto"
        and ctx.config.cuda_storage_axes == (0, 1, 2)
        and not ctx.config.cuda_flags & (abi.CUDA_FIELD_PAD32 | abi.CUDA_FIELD_PAD64)
    ):
        shape = tuple(min(f.shape[a] for f in _fields(state)[3:]) for a in range(3))
        forced = os.environ.get("BEAMZ_CUDA_CPML_CORE_FUSION")
        # Match the native fusion gate before reserving an alternate field bank.
        # Explicit temporal experiments and speed policy retain their workspace.
        if not ctx.config.cuda_flags & abi.CUDA_TEMPORAL_PAIR and (
            forced == "0"
            or (
                forced != "1"
                and (
                    shape[2] < abi.CPML_FUSION_MIN_WIDTH
                    or np.prod(shape) < abi.CPML_FUSION_MIN_CELLS
                )
            )
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
    # The frozen-field layout supports scalar damping only; dense damping uses
    # the general schedule so conductive media retain correct field updates.
    return all(
        getattr(coeffs, f"{phase}_{kind}_{axis}").ndim
        in ((0,) if kind == "decay" else (0, 3))
        for phase in "he"
        for kind in ("decay", "source")
        for axis in "xyz"
    )


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
    temporal_cpml = _temporal_cpml_fields_supported(ctx, coeffs, nsteps, state)
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
