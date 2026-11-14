"""Lower fused 3D Yee phases to the optional BeamZ CUDA typed FFI targets."""

from __future__ import annotations

# Retain the names available from this module before its implementation was split.
import os as os
from dataclasses import dataclass as dataclass
from typing import Any as Any

import jax
import jax.numpy as jnp
import numpy as np

from beamz.simulation.cuda.native import abi
from beamz.simulation.cuda.native.base import (
    _COMPONENT_CODE,  # noqa: F401
    _CPML_E_PSI_INPUT,  # noqa: F401
    _CPML_H_PSI_INPUT,  # noqa: F401
    _CPML_TERM_COUNT,  # noqa: F401
    _EMPTY,  # noqa: F401
    _EMPTY_SOURCE_GROUP,  # noqa: F401
    _FIELD_COUNT,
    _METRIC_KIND_CODE,  # noqa: F401
    _PHASE_E,  # noqa: F401
    _PHASE_H,  # noqa: F401
    _SOURCE_GROUP_COUNT,
    _TEMPORAL_FIELD_WORKSPACE_INPUT,  # noqa: F401
    _TEMPORAL_PSI_WORKSPACE_INPUT,  # noqa: F401
    CUDA_ABI_VERSION,  # noqa: F401
    NativeSchedulePlan,  # noqa: F401
    _boundary_code,  # noqa: F401
    _cpml_payload,  # noqa: F401
    _ffi_call,
    _fields,
    _metallic_edge_mask,  # noqa: F401
    _metric_kind_code,  # noqa: F401
    _phase_materials,  # noqa: F401
    _program_attributes,
    _shape,  # noqa: F401
    _term_metadata,  # noqa: F401
)
from beamz.simulation.cuda.native.graph import (
    _coincident_source_group_mask,  # noqa: F401
    _combined_cpml_core_supported,  # noqa: F401
    _cpml_graph_io,  # noqa: F401
    _disjoint_source_group_mask,  # noqa: F401
    _field_padding,  # noqa: F401
    _graph_io,
    _native_schedule_flags,  # noqa: F401
    _native_schedule_plan,
    _packed_e_material,  # noqa: F401
    _replace_fields,  # noqa: F401
    _replace_graph_outputs,  # noqa: F401
    _replace_native_outputs,
    _replace_temporal_cpml_outputs,  # noqa: F401
    _scheduled_graph_io,
    _source_group_arguments,
    _temporal_cpml_fields_supported,  # noqa: F401
    _temporal_cpml_graph_io,  # noqa: F401
    _temporal_yee_supported,  # noqa: F401
    _uniform_cpml_thickness,  # noqa: F401
    _yee_graph_io,  # noqa: F401
)
from beamz.simulation.cuda.native.phase import (
    _discard,  # noqa: F401
    _ffi_phase,  # noqa: F401
    _GradientMask,  # noqa: F401
    _phase_metrics,  # noqa: F401
    _phase_with_vjp,  # noqa: F401
    _target_cotangent,  # noqa: F401
    _update_phase,  # noqa: F401
    reverse_material_step,  # noqa: F401
    update_e,  # noqa: F401
    update_h,  # noqa: F401
)
from beamz.simulation.model import SimulationState


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
        from beamz.simulation.cuda.native.phase import run_steps as run_tensor_steps

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
