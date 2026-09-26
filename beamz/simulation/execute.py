"""Scan execution, caching, continuation, and runtime placement."""

from __future__ import annotations

import os
import pathlib
import platform
import sys
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from beamz._helpers import (
    _finish_inline_progress,
    _print_inline_progress,
    _print_inline_status,
)
from beamz.const import EPS_0, MU_0
from beamz.devices.sources.compiler import (
    BatchedSlabGroup,
    CompiledSourceSpec,
    batch_slab_specs,
)
from beamz.simulation.backend import CUDA_BF16_PSI
from beamz.simulation.boundary_masks import compact_boundary_masks
from beamz.simulation.model import (
    AutoTermination,
    CompiledProgram,
    SimulationState,
    UpdateCoefficients,
)

from . import kernels as update_runtime
from . import observe as monitor_runtime
from . import sharding as sharding_runtime
from .results import (
    MonitorResults,
    RunTermination,
    SimulationPerformance,
    SimulationResults,
    SimulationRun,
)

SOURCE_PHASE_COMPONENTS = {
    "pre_e": ("Ex", "Ey", "Ez"),
    "h": ("Hx", "Hy", "Hz"),
    "e": ("Ex", "Ey", "Ez"),
}

# Partition sources by leapfrog phase: magnetic and electric injections must sit next to
# the update whose temporal staggering was used when their amplitudes were normalized.
SourceBatchMap = dict[
    tuple[str, str],
    tuple[BatchedSlabGroup | None, tuple[CompiledSourceSpec, ...]],
]

# Native CUDA calls capture one graph node sequence per step. Keep captures small
# enough to instantiate predictably, then replay the same executable from an XLA loop.
# The even bound also keeps the temporal ping-pong kernel on its fast return layout.
CUDA_GRAPH_MAX_STEPS = 256


_COMPONENT_OFFSETS_2D = {
    "Ex": (0.0, 0.5),
    "Ey": (0.5, 0.0),
    "Ez": (0.0, 0.0),
    "Hx": (0.5, 0.0),
    "Hy": (0.0, 0.5),
    "Hz": (0.5, 0.5),
}


def _axis_integration_weights(edges, offset: float, count: int) -> np.ndarray:
    """Return control-volume widths for one edge- or center-aligned Yee axis."""
    edges = np.asarray(edges, dtype=np.float64)
    widths = np.diff(edges)
    count = int(count)
    if offset == 0.5:
        if count != widths.size:
            raise ValueError("Center-aligned Yee support does not match the grid.")
        return widths
    if offset != 0.0 or count != edges.size:
        raise ValueError("Edge-aligned Yee support does not match the grid.")
    out = np.empty(count, dtype=np.float64)
    out[0] = 0.5 * widths[0]
    out[-1] = 0.5 * widths[-1]
    if count > 2:
        out[1:-1] = 0.5 * (widths[:-1] + widths[1:])
    return out


def _component_integration_weights(program: CompiledProgram, component: str):
    """Return physical integration weights on one native Yee support."""
    shape = tuple(int(value) for value in getattr(program.grid, component).shape)
    geometry = program.grid.geometry
    if program.config.is_3d:
        from beamz.lattice import component_axis_offsets_3d

        offsets = component_axis_offsets_3d(component)
        z = _axis_integration_weights(geometry.z_edges, offsets["z"], shape[0])
        y = _axis_integration_weights(geometry.y_edges, offsets["y"], shape[1])
        x = _axis_integration_weights(geometry.x_edges, offsets["x"], shape[2])
        return jnp.asarray(z[:, None, None] * y[None, :, None] * x[None, None, :])
    y_offset, x_offset = _COMPONENT_OFFSETS_2D[component]
    y = _axis_integration_weights(geometry.y_edges, y_offset, shape[0])
    x = _axis_integration_weights(geometry.x_edges, x_offset, shape[1])
    return jnp.asarray(y[:, None] * x[None, :])


def _node_integration_weights(program: CompiledProgram, shape: tuple[int, ...]):
    """Return physical integration weights on the shared Yee-node support."""
    geometry = program.grid.geometry
    if program.config.is_3d:
        z = _axis_integration_weights(geometry.z_edges, 0.0, shape[0])
        y = _axis_integration_weights(geometry.y_edges, 0.0, shape[1])
        x = _axis_integration_weights(geometry.x_edges, 0.0, shape[2])
        return jnp.asarray(z[:, None, None] * y[None, :, None] * x[None, None, :])
    y = _axis_integration_weights(geometry.y_edges, 0.0, shape[0])
    x = _axis_integration_weights(geometry.x_edges, 0.0, shape[1])
    return jnp.asarray(y[:, None] * x[None, :])


def _energy_terms(program: CompiledProgram):
    """Precompute field, material, and measure names for energy diagnostics."""
    active = (
        {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
        if program.config.is_3d
        else (
            {"Ez", "Hx", "Hy"}
            if program.config.polarization_2d == "tm"
            else {"Ex", "Ey", "Hz"}
        )
    )
    extent = program.grid.geometry.extent
    domain_measure = float(
        extent[0] * extent[1] * (extent[2] if program.config.is_3d else 1.0)
    )
    terms = tuple(
        (
            component.lower(),
            getattr(program.grid, material),
            _component_integration_weights(program, component) / domain_measure,
            constant,
        )
        for component, material, constant in (
            ("Ex", "eps_ex", EPS_0),
            ("Ey", "eps_ey", EPS_0),
            ("Ez", "eps_ez", EPS_0),
            ("Hx", "mu_hx", MU_0),
            ("Hy", "mu_hy", MU_0),
            ("Hz", "mu_hz", MU_0),
        )
        if component in active
    )
    cross_terms = ()
    material_grid = program.grid.material_grid
    if material_grid.uses_full_permittivity:
        node_tensor = np.asarray(material_grid.yee_tensors["eps_node"])
        node_shape = tuple(int(value) for value in node_tensor.shape[1:])
        node_measure = _node_integration_weights(program, node_shape) / domain_measure
        cross_terms = tuple(
            (
                left.lower(),
                left,
                right.lower(),
                right,
                jnp.asarray(node_tensor[tensor_index]),
                node_measure,
            )
            for left, right, tensor_index in (
                ("Ex", "Ey", 3),
                ("Ex", "Ez", 4),
                ("Ey", "Ez", 5),
            )
            if left in active and right in active
        )
    return terms, cross_terms, domain_measure


def _field_diagnostics(state: SimulationState, plan) -> tuple[float, float, bool]:
    """Compute integrated electromagnetic energy, maximum field, and finiteness."""
    terms, cross_terms, domain_measure = plan
    energy_density = jnp.asarray(0.0, dtype=jnp.float32)
    max_field = jnp.asarray(0.0, dtype=jnp.float32)
    finite = jnp.asarray(True)
    for field_name, material, measure, constant in terms:
        values = jnp.asarray(getattr(state, field_name))
        finite = finite & jnp.all(jnp.isfinite(values))
        max_field = jnp.maximum(max_field, jnp.max(jnp.abs(values), initial=0.0))
        density = jnp.asarray(material) * values * values
        energy_density = energy_density + 0.5 * float(constant) * jnp.sum(
            density * jnp.asarray(measure)
        )
    centered = {}
    for (
        left_name,
        left_component,
        right_name,
        right_component,
        material,
        measure,
    ) in cross_terms:
        node_shape = tuple(int(value) for value in material.shape)
        for field_name, component in (
            (left_name, left_component),
            (right_name, right_component),
        ):
            if component not in centered:
                centered[component] = update_runtime.collocate_yee_component(
                    getattr(state, field_name), component, "Node", node_shape
                )
        energy_density = energy_density + float(EPS_0) * jnp.sum(
            material
            * centered[left_component]
            * centered[right_component]
            * jnp.asarray(measure)
        )
    return float(energy_density) * domain_measure, float(max_field), bool(finite)


def _remaining_source_activity(
    program: CompiledProgram, total_steps: int
) -> np.ndarray:
    """Return the maximum future source amplitude relative to each term's peak."""
    total_steps = int(total_steps)
    activity = np.zeros(total_steps, dtype=np.float64)
    for source in program.sources:
        waveform = np.abs(np.asarray(source.waveform, dtype=np.float64).reshape(-1))
        if waveform.size == 0:
            continue
        peak = float(np.max(waveform, initial=0.0))
        if peak > 0.0:
            sampled = np.empty(total_steps, dtype=np.float64)
            copied = min(total_steps, waveform.size)
            sampled[:copied] = waveform[:copied]
            if copied < total_steps:
                sampled[copied:] = waveform[-1]
            activity = np.maximum(activity, sampled / peak)
    return np.maximum.accumulate(activity[::-1])[::-1] if activity.size else activity


def _source_residual(remaining_activity: np.ndarray, current_step: int) -> float:
    """Return the largest normalized source drive at or after ``current_step``."""
    index = int(current_step)
    if index < 0:
        raise ValueError("current_step must be non-negative.")
    return float(remaining_activity[index]) if index < remaining_activity.size else 0.0


def _selected_monitor_names(program: CompiledProgram, policy: AutoTermination):
    available = {
        spec.name
        for spec in program.monitors
        if spec.dft_enabled
        and spec.freq_count > 0
        and spec.dft_point_count > 0
        and np.any(np.asarray(spec.dft_component_mask) > 0.0)
    }
    if policy.monitor_names:
        missing = set(policy.monitor_names) - available
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                "Automatic termination monitors must be applicable "
                f"frequency-domain monitors: {names}."
            )
        return policy.monitor_names
    return tuple(spec.name for spec in program.monitors if spec.name in available)


def _monitor_vectors(
    results: SimulationResults, names: tuple[str, ...]
) -> dict[tuple[str, str, int], np.ndarray]:
    """Return one raw DFT convergence vector per monitor, component, and frequency."""
    values = {}
    for name in names:
        monitor = results.monitors[name]
        fields = monitor._raw_dft_fields or monitor.dft_fields
        for component in sorted(fields):
            field = np.asarray(fields[component], dtype=np.complex128)
            for frequency, vector in enumerate(field.reshape(field.shape[0], -1)):
                values[(name, component, frequency)] = vector
    if names and not values:
        raise ValueError("Automatic termination monitors produced no DFT data.")
    return values


def _relative_monitor_change(
    current: dict[tuple[str, str, int], np.ndarray],
    previous: dict[tuple[str, str, int], np.ndarray] | None,
):
    """Return the largest scale-safe change among cumulative monitor values."""
    if previous is None:
        return None
    if current.keys() != previous.keys():
        raise ValueError("Automatic termination monitor values changed structure.")
    changes = []
    for key, current_value in current.items():
        previous_value = previous[key]
        if current_value.shape != previous_value.shape:
            raise ValueError("Automatic termination monitor values changed shape.")
        numerator = float(np.linalg.norm(current_value - previous_value))
        denominator = max(
            float(np.linalg.norm(current_value)),
            float(np.linalg.norm(previous_value)),
        )
        changes.append(
            0.0
            if denominator <= np.finfo(float).tiny and numerator <= np.finfo(float).tiny
            else numerator / denominator
        )
    return max(changes, default=0.0)


def _configured_dft_weight_sum(simulation, spec) -> float:
    """Return a monitor's DFT window weight through the configured time limit."""
    steps = np.arange(int(simulation.num_steps), dtype=np.int64)
    times = float(simulation.time[0]) + (steps.astype(np.float64) + 1.0) * float(
        simulation.dt
    )
    selected = (
        (steps % max(1, int(spec.dft_record_interval)) == 0)
        & (times >= float(spec.dft_t_start))
        & (times <= float(spec.dft_t_end))
    )
    if not np.any(selected):
        return 0.0
    if int(spec.dft_window_code) != 1 or not np.isfinite(spec.dft_t_end):
        return float(np.count_nonzero(selected))
    span = max(float(spec.dft_t_end) - float(spec.dft_t_start), 1e-30)
    tau = np.clip((times[selected] - float(spec.dft_t_start)) / span, 0.0, 1.0)
    return float(np.sum(0.5 * (1.0 - np.cos(2.0 * np.pi * tau))))


def _complete_converged_dft_weights(
    results: SimulationResults, simulation, program: CompiledProgram
) -> SimulationResults:
    """Normalize converged DFTs as though their negligible tail had been sampled."""
    monitors = dict(results.monitors)
    for spec in program.monitors:
        if not spec.dft_enabled or spec.freq_count <= 0:
            continue
        monitor = simulation.monitors[int(spec.monitor_index)]
        name = str(getattr(monitor, "name", None) or f"monitor_{spec.monitor_index}")
        result = monitors[name]
        total_weight = _configured_dft_weight_sum(simulation, spec)
        weights = np.full(result.dft_weight_sum.shape, total_weight, dtype=np.float64)
        monitors[name] = replace(result, dft_weight_sum=weights)
    return replace(results, monitors=monitors)


def compiled_source_batches(
    source_specs: tuple[CompiledSourceSpec, ...],
) -> SourceBatchMap:
    # Apply sources in their scheduled leapfrog phase so amplitude normalization
    # matches field time.
    return {
        (timing, component): batch_slab_specs(
            tuple(
                spec
                for spec in source_specs
                if spec.timing == timing and spec.component == component
            )
        )
        for timing, components in SOURCE_PHASE_COMPONENTS.items()
        for component in components
    }


def _apply_specs(
    arr: jnp.ndarray,
    abs_step: jnp.ndarray,
    specs: tuple[CompiledSourceSpec, ...],
) -> jnp.ndarray:
    # Apply irregular specs in declaration order because overlapping injections add.
    out = arr
    for spec in specs:
        safe_idx = jnp.clip(abs_step, 0, spec.waveform.shape[0] - 1)
        amp = spec.waveform[safe_idx]
        patch = (spec.coeff * amp).astype(out.dtype)
        if (
            spec.is_slab
            and spec.slab_starts is not None
            and spec.slab_sizes is not None
        ):
            cur = jax.lax.dynamic_slice(out, spec.slab_starts, spec.slab_sizes)
            out = jax.lax.dynamic_update_slice(out, cur + patch, spec.slab_starts)
        else:
            out = out.at[spec.index].add(patch)
    return out


def _apply_batched_slabs(
    arr: jnp.ndarray,
    abs_step: jnp.ndarray,
    group: BatchedSlabGroup,
    *,
    dense_single_slab: bool,
) -> jnp.ndarray:
    # Batch equal-shaped slabs so source count does not linearly grow generated code.
    safe_idx = jnp.clip(abs_step, 0, group.waveforms.shape[1] - 1)
    ndim = len(group.max_sizes)

    if group.n == 1:
        amp = group.waveforms[0, safe_idx]
        starts_0 = group.starts_tuple[0]
        if dense_single_slab:
            pad_width = tuple(
                (
                    starts_0[d],
                    int(arr.shape[d]) - starts_0[d] - group.max_sizes[d],
                )
                for d in range(ndim)
            )
            dense_coeff = jnp.pad(group.coeffs[0], pad_width)
            return arr + (dense_coeff * amp).astype(arr.dtype)
    if group.n <= 2:
        # Static scatter-add keeps a small source plane local to its field shard.
        # A dynamic read/modify/write can gather the entire distributed field.
        out = arr
        for index, starts in enumerate(group.starts_tuple):
            clamped_starts = tuple(
                min(
                    max(start if start >= 0 else start + int(arr.shape[axis]), 0),
                    int(arr.shape[axis]) - group.max_sizes[axis],
                )
                for axis, start in enumerate(starts)
            )
            region = tuple(
                slice(start, start + size)
                for start, size in zip(clamped_starts, group.max_sizes, strict=True)
            )
            patch = (group.coeffs[index] * group.waveforms[index, safe_idx]).astype(
                out.dtype
            )
            out = out.at[region].add(patch)
        return out

    def body(i, out):
        # Carry prior additions so overlapping slabs accumulate rather than overwrite.
        amp = group.waveforms[i, safe_idx]
        patch = (group.coeffs[i] * amp).astype(out.dtype)
        starts_i = [group.starts[i, d] for d in range(ndim)]
        cur = jax.lax.dynamic_slice(out, starts_i, group.max_sizes)
        return jax.lax.dynamic_update_slice(out, cur + patch, starts_i)

    return jax.lax.fori_loop(0, group.n, body, arr)


def apply_source_phase(
    eng: SimulationState,
    abs_step: jnp.ndarray,
    batches: SourceBatchMap,
    timing: str,
    *,
    dense_single_slab: bool,
) -> SimulationState:
    # Apply sources in their scheduled leapfrog phase so amplitude normalization
    # matches field time.
    updates = {}
    for component in SOURCE_PHASE_COMPONENTS[timing]:
        field_name = component.lower()
        value = getattr(eng, field_name)
        batch, rest = batches[(timing, component)]
        if batch is not None:
            value = _apply_batched_slabs(
                value,
                abs_step,
                batch,
                dense_single_slab=dense_single_slab,
            )
        if rest:
            value = _apply_specs(value, abs_step, rest)
        updates[field_name] = value.astype(getattr(eng, field_name).dtype)
    return eng._replace(**updates)


ScanResult = SimulationState


def forward_step(
    carry,
    *,
    ctx: update_runtime.CompiledStepContext,
    coeffs: UpdateCoefficients,
    program,
    update_kernel: update_runtime.StepUpdateKernel,
):
    """Advance one compiled timestep."""
    cfg = ctx.config
    metallic = ctx.boundary.metallic

    # 1. Pre-E sources precede H because their waveform is normalized at this leapfrog
    # phase; the selected kernel then advances the canonical magnetic fields.
    state = apply_source_phase(
        carry,
        carry.current_step,
        ctx.source_batches,
        "pre_e",
        dense_single_slab=cfg.source_single_slab_dense,
    )
    if ctx.boundary.periodic_axes:
        ex, ey, ez = update_runtime.apply_post_source_boundaries(
            (state.ex, state.ey, state.ez),
            (metallic.ex_mask, metallic.ey_mask, metallic.ez_mask),
            components=("Ex", "Ey", "Ez"),
            periodic_axes=ctx.boundary.periodic_axes,
            material_shape=ctx.boundary.material_shape,
            logical_shapes=ctx.boundary.logical_component_shapes,
        )
        state = state._replace(ex=ex, ey=ey, ez=ez)
    state = update_kernel.update_h(state, ctx, coeffs)

    # 2. H-phase sources may overwrite constrained cells, so reapply the compiled masks
    # before E consumes the half-step magnetic fields.
    state = apply_source_phase(
        state,
        state.current_step,
        ctx.source_batches,
        "h",
        dense_single_slab=cfg.source_single_slab_dense,
    )
    cuda_owns_pec = (
        cfg.backend == "cuda_streamed"
        and not cfg.sharding.enabled
        and not program.sources
    )
    if not cuda_owns_pec:
        hx, hy, hz = update_runtime.apply_post_source_boundaries(
            (state.hx, state.hy, state.hz),
            (metallic.hx_mask, metallic.hy_mask, metallic.hz_mask),
            components=("Hx", "Hy", "Hz"),
            periodic_axes=ctx.boundary.periodic_axes,
            material_shape=ctx.boundary.material_shape,
            logical_shapes=ctx.boundary.logical_component_shapes,
        )
        state = state._replace(hx=hx, hy=hy, hz=hz)

    # 3. Advance E, inject its sources, and restore its masks before observation.
    old_e_state = state
    state = update_kernel.update_e(state, ctx, coeffs)
    state = apply_source_phase(
        state,
        state.current_step,
        ctx.source_batches,
        "e",
        dense_single_slab=cfg.source_single_slab_dense,
    )
    from beamz.simulation.dispersion import update_dispersion

    if program.dispersion is not None:
        state = update_dispersion(old_e_state, state, program.dispersion)
    if not cuda_owns_pec:
        ex, ey, ez = update_runtime.apply_post_source_boundaries(
            (state.ex, state.ey, state.ez),
            (metallic.ex_mask, metallic.ey_mask, metallic.ez_mask),
            components=("Ex", "Ey", "Ez"),
            periodic_axes=ctx.boundary.periodic_axes,
            material_shape=ctx.boundary.material_shape,
            logical_shapes=ctx.boundary.logical_component_shapes,
        )
        state = state._replace(ex=ex, ey=ey, ez=ez)

    # 4. Observe only fully constrained end-of-step fields, then advance both clocks.
    t_phys = state.t + ctx.dt_scalar
    state = monitor_runtime.update_monitors(
        program,
        state,
        state.current_step,
        t_phys,
        ctx.dt_scalar,
        state.ex,
        state.ey,
        state.ez,
        state.hx,
        state.hy,
        state.hz,
    )
    return state._replace(
        t=t_phys,
        current_step=state.current_step + jnp.array(1, dtype=jnp.int32),
    )


def build_scan(program, *, donate_state: bool = False):
    """Build the jitted compiled scan for a program."""

    # 1. Pull immutable configuration out of the program before tracing. These values
    # select shapes and kernels, so they should remain static for executable reuse.
    # TODO(adjoint-checkpointing): Once trainable material arrays are dynamic
    # runtime inputs, expose chunked scan transitions here and rematerialize at
    # chunk boundaries. Keep parameter *values* out of the compiled-program key
    # so optimization iterations reuse one executable.

    cfg = program.config
    boundary = program.boundary
    if cfg.backend == "jax":
        boundary = compact_boundary_masks(boundary)
    resolution = float(cfg.resolution)
    dt = float(cfg.dt)
    dt_scalar = jnp.asarray(dt, dtype=jnp.float32)
    is_3d = cfg.is_3d

    # 2. Batch sources once; monitors are already canonical executable plans.
    source_batches = compiled_source_batches(program.sources)

    # 3. Assemble the shared step context and select the specialized update kernel before
    # JIT compilation begins.
    step_context = update_runtime.CompiledStepContext(
        config=cfg,
        boundary=boundary,
        source_batches=source_batches,
        metrics=program.metrics,
        resolution=resolution,
        dt=dt,
        dt_scalar=dt_scalar,
        is_3d=is_3d,
        sharding_plan=program.sharding,
    )
    update_kernel = update_runtime.select_update_kernel(step_context)
    graph_source_groups = tuple(
        source_batches[(timing, component)][0]
        for timing, components in SOURCE_PHASE_COMPONENTS.items()
        for component in components
    )
    source_groups_supported = bool(program.sources) and all(
        not rest for _group, rest in source_batches.values()
    )
    graph_monitors_supported = bool(program.monitors) and all(
        monitor.recorder_index < 0
        and not monitor.accumulate_power
        and not monitor.accumulate_frequency
        and monitor.dft_enabled
        and monitor.freq_count > 0
        and monitor.dft_point_count > 0
        for monitor in program.monitors
    )
    packed_graph_monitors = None
    if (
        cfg.backend == "cuda_streamed"
        and not cfg.sharding.enabled
        and graph_monitors_supported
        and not bool(jax.config.read("jax_enable_x64"))
    ):
        from beamz.simulation.cuda import pack_dft_monitors

        packed_graph_monitors = pack_dft_monitors(program.monitors)
    cuda_multi_step = (
        cfg.backend == "cuda_streamed"
        and not cfg.sharding.enabled
        and (not program.monitors or packed_graph_monitors is not None)
        and (not program.sources or source_groups_supported)
    )
    native_graph_calls = None
    if cfg.cuda_storage_axes != (0, 1, 2):
        if not (
            cuda_multi_step
            and cfg.is_3d
            and cfg.metric_kind == "isotropic_uniform"
            and boundary.cpml.enabled
        ):
            raise ValueError(
                "CUDA storage-axis permutations require a 3D uniform-grid CPML "
                "native graph with supported slab sources and DFT monitors"
            )
        from beamz.simulation.cuda import (
            run_program_steps,
            run_source_group_steps,
            run_steps,
        )
        from beamz.simulation.cuda.storage import wrap_native_calls

        native_graph_calls = wrap_native_calls(
            run_steps, run_program_steps, run_source_group_steps, cfg.cuda_storage_axes
        )

    def run_scan(
        state: SimulationState,
        coeffs: UpdateCoefficients,
    ):
        # 4. Run the same transition through scan or fori_loop. The choice changes the
        # lowering strategy, not timestep semantics.
        if cuda_multi_step:
            from beamz.simulation.cuda import (
                run_program_steps,
                run_source_group_steps,
                run_steps,
            )

            if native_graph_calls is not None:
                run_steps, run_program_steps, run_source_group_steps = (
                    native_graph_calls
                )

            def advance_native_chunk(chunk_state, chunk_steps: int, elapsed_steps):
                elapsed_steps = jnp.asarray(elapsed_steps, dtype=jnp.int32)
                chunk_state = chunk_state._replace(
                    # Derive clocks from the immutable run origin. Incrementally
                    # accumulating float32 chunk times would perturb long-run DFT
                    # phases relative to one unbounded native launch.
                    t=state.t + dt_scalar * elapsed_steps,
                    current_step=state.current_step + elapsed_steps,
                )
                chunk_out = (
                    run_steps(chunk_state, step_context, coeffs, chunk_steps)
                    if not program.sources and not program.monitors
                    else run_program_steps(
                        chunk_state,
                        step_context,
                        coeffs,
                        graph_source_groups,
                        packed_graph_monitors,
                        chunk_steps,
                    )
                    if program.monitors
                    else run_source_group_steps(
                        chunk_state,
                        step_context,
                        coeffs,
                        graph_source_groups,
                        chunk_steps,
                    )
                )
                completed_steps = elapsed_steps + jnp.asarray(
                    chunk_steps, dtype=jnp.int32
                )
                return chunk_out._replace(
                    t=state.t + dt_scalar * completed_steps,
                    current_step=state.current_step + completed_steps,
                )

            if cfg.num_steps <= CUDA_GRAPH_MAX_STEPS:
                scan_out = advance_native_chunk(state, cfg.num_steps, 0)
            else:
                full_chunks, tail_steps = divmod(cfg.num_steps, CUDA_GRAPH_MAX_STEPS)
                # The loop body has one fixed native call signature. XLA therefore
                # reuses its buffers and the CUDA layer hits one bounded graph-cache
                # entry instead of capturing a graph proportional to the entire run.
                scan_out = jax.lax.fori_loop(
                    0,
                    full_chunks,
                    lambda _i, chunk_state: advance_native_chunk(
                        chunk_state,
                        CUDA_GRAPH_MAX_STEPS,
                        _i * CUDA_GRAPH_MAX_STEPS,
                    ),
                    state,
                )
                if tail_steps:
                    scan_out = advance_native_chunk(
                        scan_out,
                        tail_steps,
                        full_chunks * CUDA_GRAPH_MAX_STEPS,
                    )
        elif cfg.loop_kind == "scan":

            def _scan_body(carry, _unused):
                # Emit no per-step output because final state and explicit buffers hold results.
                return (
                    forward_step(
                        carry,
                        ctx=step_context,
                        coeffs=coeffs,
                        program=program,
                        update_kernel=update_kernel,
                    ),
                    None,
                )

            scan_out, _ = jax.lax.scan(
                _scan_body,
                state,
                xs=None,
                length=cfg.num_steps,
            )
        else:
            scan_out = jax.lax.fori_loop(
                0,
                cfg.num_steps,
                lambda _i, c: forward_step(
                    c,
                    ctx=step_context,
                    coeffs=coeffs,
                    program=program,
                    update_kernel=update_kernel,
                ),
                state,
            )
        return scan_out

    # 8. Buffer donation is an explicit ownership transfer. The default executable
    # preserves its input state; callers may opt into the lower-memory variant when
    # they no longer need that continuation value.
    donate_argnums = (0,) if donate_state else ()
    return jax.jit(run_scan, donate_argnums=donate_argnums)


# Runtime owns executable caching and device placement; immutable plans own numerical
# meaning. Separating those lifetimes makes continuation and cache reuse safe.


def _init_persistent_cache():
    # 1. Honor explicit opt-in and opt-out environment controls before changing JAX's
    # process-wide compilation-cache configuration.
    if os.environ.get("BEAMZ_ENABLE_JAX_PERSISTENT_CACHE", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    if os.environ.get("BEAMZ_DISABLE_JAX_PERSISTENT_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    # 2. Derive a cache namespace from JAX version, backend, architecture, and Python ABI
    # so incompatible executables are never reused.
    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    backend = jax.default_backend()
    arch = platform.machine() or "unknown"
    cache_dir = os.environ.get(
        "BEAMZ_JAX_CACHE_DIR",
        str(
            pathlib.Path.home()
            / ".cache"
            / "beamz"
            / "jax_cache"
            / f"jax-{jax.__version__}"
            / backend
            / arch
            / py_tag
        ),
    )
    # 3. Preserve a user-provided JAX cache directory; otherwise configure the derived
    # Beamz location through the newest available JAX API with a compatibility fallback.
    if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        return
    try:
        from jax.experimental.compilation_cache import compilation_cache as cc

        cc.set_cache_dir(cache_dir)
    except Exception:
        jax.config.update("jax_compilation_cache_dir", cache_dir)


@dataclass(slots=True)
class _ExecutionCache:
    # Cache lowered/compiled functions per runtime signature. Keeping it outside immutable
    # plan equality prevents compilation artifacts from changing semantic identity.
    compiled_scan: Callable[..., ScanResult] | None = None
    compiled_scan_donating: Callable[..., ScanResult] | None = None
    executable_ready: bool = False
    executable_donating_ready: bool = False


_MAX_EXECUTION_CACHES = 8
_EXECUTION_CACHES: OrderedDict[int, tuple[object, _ExecutionCache]] = OrderedDict()


def execution_cache(program) -> _ExecutionCache:
    """Return bounded private executable state for an immutable plan."""
    # Key cached work by semantic execution inputs so equivalent states reuse
    # compilation safely.
    key = id(program)
    entry = _EXECUTION_CACHES.get(key)
    if entry is not None and entry[0] is program:
        _EXECUTION_CACHES.move_to_end(key)
        return entry[1]
    cache = _ExecutionCache()
    _EXECUTION_CACHES[key] = (program, cache)
    if len(_EXECUTION_CACHES) > _MAX_EXECUTION_CACHES:
        _EXECUTION_CACHES.popitem(last=False)
    return cache


def clear_execution_cache() -> None:
    # Key cached work by semantic execution inputs so equivalent states reuse
    # compilation safely.
    _EXECUTION_CACHES.clear()


def initial_program_state(
    program: CompiledProgram,
    *,
    t: float,
    current_step: int,
    continuation: SimulationState | None = None,
    monitor_steps: int | None = None,
) -> SimulationState:
    """Allocate or restore every runtime buffer required by a compiled plan."""
    cpml, layout = program.boundary.cpml, program.sharding.layout
    psi_dtype = None
    if (
        program.config.backend == "cuda_streamed"
        and cpml.enabled
        and program.config.cuda_flags & CUDA_BF16_PSI
    ):
        psi_dtype = jnp.bfloat16

    def field(name):
        # Fresh runs use the compiled lattice; continuations supply evolved canonical
        # arrays without reconstructing a mutable field container.
        return (
            jnp.array(getattr(program.grid, name))
            if continuation is None
            else getattr(continuation, name.lower())
        )

    def zeros(shape, dtype):
        shape = tuple(int(value) for value in shape)
        return (
            np.zeros(shape, dtype=np.dtype(dtype))
            if layout.enabled
            else jnp.zeros(shape, dtype=dtype)
        )

    # Continue compatible packed CPML memories; a changed boundary plan starts clean.
    def restore_psi(old, terms, dtype):
        dtype = dtype if psi_dtype is None else psi_dtype
        shapes = tuple(term.slab.shape for term in terms)
        if len(old) == len(shapes) and all(
            tuple(value.shape)
            in (shape, sharding_runtime.logical_cpml_shape(layout, term))
            for value, shape, term in zip(old, shapes, terms, strict=True)
        ):
            converter = np.asarray if layout.enabled else jnp.asarray
            return tuple(
                sharding_runtime._pad_high_to_shape(
                    converter(value, dtype=dtype), shape, pad_value=0.0
                )
                for value, shape in zip(old, shapes, strict=True)
            )
        return tuple(zeros(shape, dtype) for shape in shapes)

    old_h = () if continuation is None else continuation.cpml_psi_h_terms
    old_e = () if continuation is None else continuation.cpml_psi_e_terms
    monitor_values = monitor_runtime.empty_monitor_values(
        program,
        num_steps=max(
            1,
            int(program.config.num_steps if monitor_steps is None else monitor_steps),
        ),
    )
    recorder_buffers = monitor_values["recorded_fields"]
    recorder_compatible = (
        continuation is not None
        and len(continuation.recorded_fields) == len(recorder_buffers)
        and all(
            old.shape[1:] == new.shape[1:]
            for old, new in zip(
                continuation.recorded_fields, recorder_buffers, strict=True
            )
        )
    )
    # Reuse monitor history only when its compiled shapes still match this program.
    if (
        program.monitors
        and continuation is not None
        and continuation.counts.shape == monitor_values["counts"].shape
        and recorder_compatible
    ):
        monitor_values = {
            name: getattr(continuation, name) for name in monitor_runtime.MONITOR_FIELDS
        }
    from beamz.simulation.dispersion import initial_polarization

    return SimulationState(
        polarization=initial_polarization(program.dispersion, continuation)
        if program.dispersion is not None
        else (),
        ex=field("Ex"),
        ey=field("Ey"),
        ez=field("Ez"),
        hx=field("Hx"),
        hy=field("Hy"),
        hz=field("Hz"),
        cpml_psi_h_terms=restore_psi(old_h, cpml.h_terms, field("Hx").dtype),
        cpml_psi_e_terms=restore_psi(old_e, cpml.e_terms, field("Ez").dtype),
        **monitor_values,
        t=jnp.asarray(t, dtype=jnp.float32),
        current_step=jnp.asarray(current_step, dtype=jnp.int32),
    )


def build_program_scan(program: CompiledProgram, *, donate_state: bool = False):
    """Build and cache the JIT scan only when execution first needs it."""
    cache = execution_cache(program)
    scan = build_scan(program, donate_state=bool(donate_state))
    if donate_state:
        cache.compiled_scan_donating = scan
        cache.executable_donating_ready = False
    else:
        cache.compiled_scan = scan
        cache.executable_ready = False
    return scan


def program_is_compiled(
    program: CompiledProgram, *, donate_state: bool = False
) -> bool:
    """Return whether the requested ownership variant is already cached."""
    cache = execution_cache(program)
    return (
        cache.compiled_scan_donating if donate_state else cache.compiled_scan
    ) is not None


def run_program(
    program: CompiledProgram,
    state: SimulationState,
    *,
    donate_state: bool = False,
) -> SimulationState:
    """Place state and coefficients, then execute the program's cached scan."""
    state = sharding_runtime.prepare_state(
        program,
        state,
        replicated_fields=(*monitor_runtime.MONITOR_FIELDS, "t", "current_step"),
    )
    coeffs = sharding_runtime.place_tree(program, program.coefficients)
    cache = execution_cache(program)
    compiled_scan = (
        cache.compiled_scan_donating if donate_state else cache.compiled_scan
    ) or build_program_scan(program, donate_state=donate_state)
    return compiled_scan(state, coeffs)


def compile_program_execution(
    program: CompiledProgram,
    state: SimulationState,
    *,
    donate_state: bool = False,
) -> None:
    """Compile and cache a program executable without advancing its state."""
    state = sharding_runtime.prepare_state(
        program,
        state,
        replicated_fields=(*monitor_runtime.MONITOR_FIELDS, "t", "current_step"),
    )
    coeffs = sharding_runtime.place_tree(program, program.coefficients)
    _compiled_program_execution(program, state, coeffs, donate_state=donate_state)


def _compiled_program_execution(
    program: CompiledProgram,
    state: SimulationState,
    coeffs,
    *,
    donate_state: bool,
):
    """Return an executable after compiling it outside the performance timer."""
    cache = execution_cache(program)
    ready = cache.executable_donating_ready if donate_state else cache.executable_ready
    compiled_scan = (
        cache.compiled_scan_donating if donate_state else cache.compiled_scan
    ) or build_program_scan(program, donate_state=donate_state)
    if ready:
        return compiled_scan
    compiled = compiled_scan.lower(state, coeffs).compile()
    if donate_state:
        cache.compiled_scan_donating = compiled
        cache.executable_donating_ready = True
    else:
        cache.compiled_scan = compiled
        cache.executable_ready = True
    return compiled


def step_program(
    program: CompiledProgram,
    state: SimulationState,
    *,
    donate_state: bool = False,
) -> SimulationState:
    """Execute a program compiled for exactly one canonical timestep."""
    if program.config.num_steps != 1:
        raise ValueError("CompiledProgram.step() requires num_steps=1.")
    return run_program(program, state, donate_state=donate_state)


def _decode_monitor_results(sim, program, state) -> dict[str, MonitorResults]:
    """Detach packed monitor rows into the user-visible named mapping."""
    return {
        str(
            getattr(monitor, "name", None) or f"monitor_{spec.monitor_index}"
        ): MonitorResults.from_compiled_state(monitor, spec, state, program.config)
        for spec in program.monitors
        for monitor in (sim.monitors[spec.monitor_index],)
    }


def _compiled_source_launch_powers(program, source_count: int):
    """Return a source power only when all of its compiled terms agree."""
    values: list[list[float]] = [[] for _ in range(int(source_count))]
    for spec in program.sources:
        source_index = int(getattr(spec, "source_index", -1))
        power = getattr(spec, "launched_power", None)
        if 0 <= source_index < len(values) and power is not None:
            values[source_index].append(float(power))
    return tuple(
        None
        if not powers
        or not all(
            abs(value - powers[0]) <= 1e-10 * max(1.0, abs(powers[0]))
            for value in powers
        )
        else powers[0]
        for powers in values
    )


def runtime_inputs(
    program: CompiledProgram,
    state: SimulationState,
    *,
    monitor_steps: int,
) -> SimulationState:
    """Restore continuation buffers required by an already compiled program."""
    return initial_program_state(
        program,
        t=state.t,
        current_step=state.current_step,
        continuation=state,
        monitor_steps=monitor_steps,
    )


def execute_step(
    program: CompiledProgram,
    state: SimulationState,
    *,
    monitor_steps: int,
    donate_state: bool = False,
) -> SimulationState:
    """Advance one explicit state through a one-step program."""
    return sharding_runtime.crop_state(
        program,
        step_program(
            program,
            runtime_inputs(program, state, monitor_steps=monitor_steps),
            donate_state=donate_state,
        ),
    )


def _run_program_state(
    program: CompiledProgram,
    state: SimulationState,
    *,
    monitor_steps: int,
    donate_state: bool,
) -> SimulationState:
    """Execute a program and return its cropped continuation state."""
    state = run_program(
        program,
        runtime_inputs(program, state, monitor_steps=monitor_steps),
        donate_state=donate_state,
    )
    state.ez.block_until_ready()
    return sharding_runtime.crop_state(program, state)


def _run_program_state_timed(
    program: CompiledProgram,
    state: SimulationState,
    *,
    monitor_steps: int,
    donate_state: bool,
) -> tuple[SimulationState, float]:
    """Run a ready executable and measure only its device-complete execution."""
    state = runtime_inputs(program, state, monitor_steps=monitor_steps)
    state = sharding_runtime.prepare_state(
        program,
        state,
        replicated_fields=(*monitor_runtime.MONITOR_FIELDS, "t", "current_step"),
    )
    coeffs = sharding_runtime.place_tree(program, program.coefficients)
    executable = _compiled_program_execution(
        program, state, coeffs, donate_state=donate_state
    )
    started = perf_counter()
    state = executable(state, coeffs)
    state.ez.block_until_ready()
    runtime_s = max(perf_counter() - started, np.finfo(float).tiny)
    return sharding_runtime.crop_state(program, state), runtime_s


def _performance_for(
    program: CompiledProgram, *, runtime_s: float, steps: int
) -> SimulationPerformance:
    return SimulationPerformance(
        runtime_s=runtime_s,
        cells=int(np.prod(program.grid.permittivity.shape)),
        steps=steps,
    )


def _print_performance(performance: SimulationPerformance) -> None:
    """Print the two execution-only statistics reported for a finished run."""
    print(f"Simulation runtime: {performance.runtime_s:.2f} s")
    print(f"GCUPS: {performance.gcups:.3f}")


def compiled_xla_memory_analysis(
    program: CompiledProgram,
    state: SimulationState,
    *,
    monitor_steps: int,
) -> dict:
    """Return backend memory data for the exact placed program and state."""
    state = sharding_runtime.prepare_state(
        program,
        runtime_inputs(program, state, monitor_steps=monitor_steps),
        replicated_fields=(*monitor_runtime.MONITOR_FIELDS, "t", "current_step"),
    )
    coeffs = sharding_runtime.place_tree(program, program.coefficients)
    cache = execution_cache(program)
    compiled_scan = cache.compiled_scan or build_program_scan(program)
    compiled = compiled_scan.lower(state, coeffs).compile()
    analysis = getattr(compiled, "memory_analysis", lambda: None)()
    if analysis is None:
        return {"available": False}
    return {
        "available": True,
        **{
            name: value
            for name in dir(analysis)
            if not name.startswith("_")
            and isinstance(
                (value := getattr(analysis, name)),
                (int, float, str, bool, type(None)),
            )
        },
    }


def run_simulation_program(
    simulation,
    program: CompiledProgram,
    state: SimulationState,
    *,
    progress: bool,
    store_full_materials: bool,
    monitor_steps: int,
    donate_state: bool,
    performance: bool,
    report_performance: bool,
) -> SimulationRun:
    """Execute one continuation and separate durable results from runtime state."""
    compiling = progress and not program_is_compiled(program, donate_state=donate_state)
    if compiling:
        _print_inline_status("Compiling simulation...")
    runtime_s = None
    if performance:
        state, runtime_s = _run_program_state_timed(
            program, state, monitor_steps=monitor_steps, donate_state=donate_state
        )
    else:
        state = _run_program_state(
            program, state, monitor_steps=monitor_steps, donate_state=donate_state
        )
    if progress:
        if compiling:
            _finish_inline_progress()
        _print_inline_progress(
            program.config.num_steps,
            program.config.num_steps,
            label="Running simulation",
        )
        _finish_inline_progress()
    stats = (
        _performance_for(
            program, runtime_s=runtime_s, steps=int(program.config.num_steps)
        )
        if runtime_s is not None
        else None
    )
    results = SimulationResults.from_run(
        simulation,
        runtime_fields=program.grid,
        monitor_results=_decode_monitor_results(simulation, program, state),
        completed_steps=int(state.current_step),
        store_full_materials=store_full_materials,
        source_launch_powers=_compiled_source_launch_powers(
            program, len(simulation.sources)
        ),
        performance=stats,
    )
    if stats is not None and report_performance:
        _print_performance(stats)
    return SimulationRun(results=results, state=state)


_PROGRESS_CHUNK_COUNT = 20


def _progress_chunk_lengths(total_steps: int) -> tuple[int, ...]:
    """Split a run into a bounded number of visible progress updates."""
    chunk_steps = max(1, int(np.ceil(total_steps / _PROGRESS_CHUNK_COUNT)))
    full_chunks, remainder = divmod(total_steps, chunk_steps)
    return (chunk_steps,) * full_chunks + ((remainder,) if remainder else ())


def run_simulation_with_progress(
    simulation,
    state: SimulationState | None,
    *,
    num_steps: int,
    sharding,
    store_full_materials: bool,
    monitor_steps: int,
    donate_state: bool,
    backend: str,
    performance: bool,
    report_performance: bool,
) -> SimulationRun:
    """Run a segment in compiled chunks and report actual completed timesteps."""
    chunk_lengths = _progress_chunk_lengths(num_steps)
    _print_inline_status("Compiling simulation...")
    try:
        programs = {
            length: simulation.compile(
                num_steps=length,
                sharding=sharding,
                backend=backend,
                progress=False,
            )
            for length in set(chunk_lengths)
        }
        if state is None:
            state = SimulationState.initial(
                programs[chunk_lengths[0]].grid, t=float(simulation.time[0])
            )
        for program in programs.values():
            compile_program_execution(
                program,
                runtime_inputs(program, state, monitor_steps=monitor_steps),
                donate_state=donate_state,
            )
    finally:
        _finish_inline_progress()

    completed = 0
    runtime_s = 0.0
    program = programs[chunk_lengths[-1]]
    _print_inline_progress(0, num_steps, label="Running simulation")
    try:
        for length in chunk_lengths:
            program = programs[length]
            if performance:
                state, elapsed = _run_program_state_timed(
                    program,
                    state,
                    monitor_steps=monitor_steps,
                    donate_state=donate_state,
                )
                runtime_s += elapsed
            else:
                state = _run_program_state(
                    program,
                    state,
                    monitor_steps=monitor_steps,
                    donate_state=donate_state,
                )
            completed += length
            _print_inline_progress(completed, num_steps, label="Running simulation")
    finally:
        _finish_inline_progress()

    stats = (
        _performance_for(program, runtime_s=runtime_s, steps=num_steps)
        if performance
        else None
    )
    results = SimulationResults.from_run(
        simulation,
        runtime_fields=program.grid,
        monitor_results=_decode_monitor_results(simulation, program, state),
        completed_steps=int(state.current_step),
        store_full_materials=store_full_materials,
        source_launch_powers=_compiled_source_launch_powers(
            program, len(simulation.sources)
        ),
        performance=stats,
    )
    if stats is not None and report_performance:
        _print_performance(stats)
    return SimulationRun(results=results, state=state)


def run_until_terminated(
    simulation,
    policy: AutoTermination,
    *,
    progress: bool,
    store_full_materials: bool,
    sharding,
    backend: str,
    performance: bool,
) -> SimulationResults:
    """Run reusable chunks until convergence, failure, or the configured time limit."""
    if not isinstance(policy, AutoTermination):
        raise TypeError("termination must be an AutoTermination instance or None.")
    chunk_steps = min(int(policy.chunk_steps), int(simulation.num_steps))
    first_program = simulation.compile(
        num_steps=chunk_steps,
        sharding=sharding,
        backend=backend,
        progress=progress,
    )
    if first_program.grid.material_grid.dispersion:
        raise ValueError(
            "Automatic energy termination does not yet include dispersive material energy. Use a fixed run_time and check spectral convergence with advance()."
        )
    monitor_names = _selected_monitor_names(first_program, policy)
    monitor_tolerance = (
        None if policy.monitor_change is None else float(policy.monitor_change)
    )
    use_monitor = monitor_tolerance is not None and bool(monitor_names)
    if policy.field_decay == 0.0 and not use_monitor:
        raise ValueError(
            "Automatic termination has no applicable field or monitor criterion."
        )

    source_activity = _remaining_source_activity(first_program, simulation.num_steps)
    terms = _energy_terms(first_program)
    state = SimulationState.initial(first_program.grid, t=float(simulation.time[0]))
    previous_energy: float | None = None
    previous_monitor: dict[tuple[str, str, int], np.ndarray] | None = None
    energy = peak_energy = max_field = 0.0
    field_decay = monitor_change = None
    source_decay = _source_residual(source_activity, 0)
    successful_checks = growth_checks = 0
    reason = "time_limit"
    last_run: SimulationRun | None = None
    runtime_s = 0.0
    compiling = progress and not program_is_compiled(first_program, donate_state=True)
    if compiling:
        _print_inline_status("Compiling simulation...")
    elif progress:
        _print_inline_progress(0, int(simulation.num_steps), label="Running simulation")

    try:
        while int(state.current_step) < int(simulation.num_steps):
            current_step = int(state.current_step)
            remaining = int(simulation.num_steps) - current_step
            steps = min(chunk_steps, remaining)
            program = (
                first_program
                if steps == chunk_steps
                else simulation.compile(
                    num_steps=steps,
                    sharding=sharding,
                    backend=backend,
                    progress=False,
                )
            )
            last_run = run_simulation_program(
                simulation,
                program,
                state,
                progress=False,
                store_full_materials=store_full_materials,
                monitor_steps=remaining,
                donate_state=True,
                performance=performance,
                report_performance=False,
            )
            state = last_run.state
            if last_run.results.performance is not None:
                runtime_s += last_run.results.performance.runtime_s
            current_step = int(state.current_step)
            if progress:
                if compiling:
                    _finish_inline_progress()
                    compiling = False
                    _print_inline_progress(
                        0, int(simulation.num_steps), label="Running simulation"
                    )
                _print_inline_progress(
                    current_step,
                    int(simulation.num_steps),
                    label="Running simulation",
                )

            energy, max_field, fields_finite = _field_diagnostics(state, terms)
            current_monitor = _monitor_vectors(last_run.results, monitor_names)
            monitors_finite = all(
                np.isfinite(value).all() for value in current_monitor.values()
            )
            if not fields_finite or not np.isfinite(energy) or not monitors_finite:
                reason = "nonfinite"
                break

            peak_energy = max(peak_energy, energy)
            field_decay = (
                energy / peak_energy if peak_energy > np.finfo(float).tiny else 0.0
            )
            source_decay = _source_residual(source_activity, current_step)
            source_off = source_decay <= policy.source_decay
            monitor_change = _relative_monitor_change(current_monitor, previous_monitor)

            if (
                source_off
                and previous_energy is not None
                and previous_energy > np.finfo(float).tiny
                and energy > policy.growth_factor * previous_energy
            ):
                growth_checks += 1
            else:
                growth_checks = 0
            if growth_checks >= policy.growth_checks:
                reason = "diverged"
                break

            eligible = source_off and current_step >= policy.min_steps
            energy_stable = (
                policy.field_decay == 0.0 or field_decay <= policy.field_decay
            )
            monitor_stable = not use_monitor or (
                monitor_change is not None
                and monitor_tolerance is not None
                and monitor_change <= monitor_tolerance
            )
            if eligible and energy_stable and monitor_stable:
                successful_checks += 1
            else:
                successful_checks = 0
            if successful_checks >= policy.consecutive_checks:
                reason = "converged"
                break

            previous_energy = energy
            previous_monitor = current_monitor
    finally:
        if progress:
            _finish_inline_progress()

    if last_run is None:
        raise RuntimeError("Automatic termination executed no simulation steps.")
    report = RunTermination(
        reason=reason,
        steps=int(state.current_step),
        time=float(state.t),
        converged=reason == "converged",
        field_decay=field_decay,
        monitor_change=monitor_change if use_monitor else None,
        source_decay=source_decay,
        energy=energy,
        peak_energy=peak_energy,
        max_field=max_field,
        consecutive_checks=successful_checks,
    )
    results = last_run.results
    if reason == "converged":
        results = _complete_converged_dft_weights(results, simulation, first_program)
    stats = (
        _performance_for(
            first_program, runtime_s=runtime_s, steps=int(state.current_step)
        )
        if performance
        else None
    )
    results = replace(results, termination=report, performance=stats)
    if stats is not None:
        _print_performance(stats)
    return results


_init_persistent_cache()
