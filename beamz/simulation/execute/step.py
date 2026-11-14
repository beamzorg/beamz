"""Source injection and one-step field evolution."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from beamz.devices.sources.compiler import (
    BatchedSlabGroup,
    CompiledSourceSpec,
    batch_slab_specs,
)
from beamz.simulation.compile import physics as update_runtime
from beamz.simulation.compile.backend import CUDA_BF16_PSI, trace_preparation
from beamz.simulation.execute import sharding as sharding_runtime
from beamz.simulation.model import CompiledProgram, SimulationState, UpdateCoefficients
from beamz.simulation.results import observe as monitor_runtime

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
            out = _add_source_slab(out, spec.slab_starts, patch)
        else:
            out = out.at[spec.index].add(patch)
    return out


def _add_source_slab(arr, starts, patch):
    # Scatter-add expresses the exact identity derivative with respect to the
    # incoming field. Slice/read/update has the same forward value but its VJP
    # unnecessarily slices, zeroes and rebuilds whole field grids each step.
    ndim = arr.ndim
    # Match dynamic_slice's negative-index normalization and origin clamping
    # before promising an in-bounds scatter, including clipped source slabs.
    starts = tuple(
        jnp.clip(jnp.where(start < 0, start + size, start), 0, size - patch.shape[axis])
        for axis, (start, size) in enumerate(zip(starts, arr.shape, strict=True))
    )
    return jax.lax.scatter_add(
        arr,
        jnp.asarray(starts, dtype=jnp.int32),
        patch,
        dimension_numbers=jax.lax.ScatterDimensionNumbers(
            update_window_dims=tuple(range(ndim)),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=tuple(range(ndim)),
        ),
        indices_are_sorted=True,
        unique_indices=True,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


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
        return _add_source_slab(out, starts_i, patch)

    return jax.lax.fori_loop(0, group.n, body, arr)


def apply_source_phase(
    eng: SimulationState,
    abs_step: jnp.ndarray,
    batches: SourceBatchMap,
    timing: str,
    *,
    dense_single_slab: bool,
    sharding_plan=None,
    local_single_owner: bool = False,
) -> SimulationState:
    # Apply sources in their scheduled leapfrog phase so amplitude normalization
    # matches field time.
    updates = {}
    for component in SOURCE_PHASE_COMPONENTS[timing]:
        field_name = component.lower()
        value = getattr(eng, field_name)
        batch, rest = batches[(timing, component)]
        if batch is not None:
            use_local = sharding_plan is not None and sharding_plan.layout.enabled
            if use_local:
                from beamz.simulation.execute.distributed import (
                    apply_batched_slabs,
                    requires_local_injection,
                )

                use_local = local_single_owner or requires_local_injection(
                    value, batch, sharding_plan
                )
            if use_local:
                value = apply_batched_slabs(value, abs_step, batch, sharding_plan)
            else:
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


@jax.custom_vjp
def _known_primal(value, known):
    # Preserve the VJP while reusing an already computed primal.
    return known


def _known_primal_forward(value, known):
    return known, ()


def _known_primal_backward(_residuals, cotangent):
    return cotangent, None


_known_primal.defvjp(_known_primal_forward, _known_primal_backward)


def forward_step(
    carry,
    *,
    ctx: update_runtime.CompiledStepContext,
    coeffs: UpdateCoefficients,
    program,
    update_kernel: update_runtime.StepUpdateKernel,
    known_fields=None,
    observation_time=None,
):
    """Advance one compiled timestep."""
    cfg = ctx.config
    metallic = ctx.boundary.metallic

    def constrain(current: Any = (), family: str = "") -> SimulationState:
        names = tuple(f"{family}{axis}" for axis in "xyz")
        values = update_runtime.apply_post_source_boundaries(
            tuple(getattr(current, name) for name in names),
            tuple(getattr(metallic, f"{name}_mask") for name in names),
            components=tuple(name.capitalize() for name in names),
            periodic_axes=ctx.boundary.periodic_axes,
            periodic_phases=ctx.boundary.periodic_phases,
            material_shape=ctx.boundary.material_shape,
            logical_shapes=ctx.boundary.logical_component_shapes,
        )
        return current._replace(**dict(zip(names, values, strict=True)))

    # 1. Pre-E sources precede H because their waveform is normalized at this leapfrog
    # phase; the selected kernel then advances the canonical magnetic fields.
    state = apply_source_phase(
        carry,
        carry.current_step,
        ctx.source_batches,
        "pre_e",
        dense_single_slab=cfg.source_single_slab_dense,
        sharding_plan=program.sharding,
        local_single_owner=cfg.backend == "cuda_streamed",
    )
    if ctx.boundary.periodic_axes:
        state = constrain(state, "e")
    state = update_kernel.update_h(state, ctx, coeffs)

    # 2. H-phase sources may overwrite constrained cells, so reapply the compiled masks
    # before E consumes the half-step magnetic fields.
    state = apply_source_phase(
        state,
        state.current_step,
        ctx.source_batches,
        "h",
        dense_single_slab=cfg.source_single_slab_dense,
        sharding_plan=program.sharding,
        local_single_owner=cfg.backend == "cuda_streamed",
    )
    kernel_owns_pec = (
        cfg.backend in {"cuda_streamed", "cuda_mstep"}
        and not cfg.sharding.enabled
        and not program.sources
    )
    if (
        update_kernel.kind in {"cuda_streamed_sharded", "jax_local_cpml"}
        and ctx.boundary.cpml.enabled
        and not coeffs.e_inverse_offdiagonal.size
    ):
        from beamz.simulation.execute.distributed import sources_are_interior

        # These kernels already constrain their output. Interior source patches
        # cannot undo that, so avoid six redundant full-volume mask passes.
        kernel_owns_pec = sources_are_interior(
            state, ctx.source_batches, program.sharding
        )
    if not kernel_owns_pec:
        state = constrain(state, "h")
    if known_fields is not None:
        state = state._replace(
            **{
                name: _known_primal(getattr(state, name), getattr(known_fields, name))
                for name in ("hx", "hy", "hz")
            }
        )

    # 3. Advance E, inject its sources, and restore its masks before observation.
    old_e_state = state
    state = update_kernel.update_e(state, ctx, coeffs)
    state = apply_source_phase(
        state,
        state.current_step,
        ctx.source_batches,
        "e",
        dense_single_slab=cfg.source_single_slab_dense,
        sharding_plan=program.sharding,
        local_single_owner=cfg.backend == "cuda_streamed",
    )
    from beamz.simulation.compile.material_blocks import update_dispersion

    if program.dispersion is not None:
        state = update_dispersion(old_e_state, state, program.dispersion)
    if not kernel_owns_pec:
        state = constrain(state, "e")
    if known_fields is not None:
        state = state._replace(
            **{
                name: _known_primal(getattr(state, name), getattr(known_fields, name))
                for name in ("ex", "ey", "ez")
            }
        )

    # 4. Observe only fully constrained end-of-step fields, then advance both clocks.
    # Derive the default from the absolute source index, retaining explicit
    # observation times for streamed schedules.
    t_phys = (
        jnp.asarray(cfg.t0, dtype=jnp.float32)
        + (state.current_step + 1).astype(jnp.float32) * ctx.dt_scalar
        if observation_time is None
        else observation_time
    )
    state = monitor_runtime.update_monitors(
        program,
        state,
        state.current_step,
        t_phys,
        ctx.dt_scalar,
    )
    if cfg.backend == "jax" and cfg.is_3d and not cfg.sharding.enabled:
        # Keep a common field layout across scan iterations. Recent JAX can
        # otherwise choose conflicting stencil/monitor layouts and insert copies.
        # JAX < 0.7 names the device-local type DeviceLocalLayout; its Layout
        # pairs a device-local layout with sharding instead.
        try:
            from jax.experimental import layout
            from jax.experimental.layout import with_layout_constraint
        except ImportError:
            pass
        else:
            layout_type = getattr(layout, "DeviceLocalLayout", layout.Layout)
            state = state._replace(
                **{
                    name: with_layout_constraint(
                        getattr(state, name), layout_type((0, 1, 2))
                    )
                    for name in ("ex", "ey", "ez", "hx", "hy", "hz")
                }
            )
    return state._replace(
        t=t_phys,
        current_step=state.current_step + jnp.array(1, dtype=jnp.int32),
    )


@trace_preparation("initial_state")
def initial_program_state(
    program: CompiledProgram,
    *,
    t: float,
    current_step: int,
    continuation: SimulationState | None = None,
    monitor_steps: int | None = None,
) -> SimulationState:
    """Allocate or restore every runtime buffer required by a compiled plan."""
    from beamz.simulation import execute as execution_runtime

    cpml, layout = program.boundary.cpml, program.sharding.layout
    psi_dtype = None
    if (
        program.config.backend in {"cuda_streamed", "cuda_mstep"}
        and cpml.enabled
        and program.config.cuda_flags & CUDA_BF16_PSI
    ):
        psi_dtype = jnp.bfloat16

    def field(name):
        if continuation is None and layout.enabled:
            value = getattr(program.grid, name)
            if isinstance(value, np.ndarray) and all(
                stride == 0 for stride in value.strides
            ):
                shape = layout.padded_shapes[name]
                target = sharding_runtime._array_sharding(
                    program,
                    jax.ShapeDtypeStruct(shape, value.dtype),
                    program.sharding.mesh,
                )
                return jax.jit(
                    lambda: jnp.zeros(shape, dtype=value.dtype), out_shardings=target
                )()
        # Fresh runs use the compiled lattice; continuations supply evolved canonical
        # arrays without reconstructing a mutable field container.
        if continuation is None:
            return execution_runtime._copy_initial_field(getattr(program.grid, name))
        value = getattr(continuation, name.lower())
        dtype = field_dtype(name)
        return value if value.dtype == dtype else value.astype(dtype)

    def field_dtype(name):
        # Query metadata without copying two additional full fields just to
        # choose the CPML dtype during fresh-state construction.
        dtype = getattr(program.grid, name).dtype
        if continuation is not None:
            dtype = jnp.result_type(dtype, getattr(continuation, name.lower()).dtype)
        return dtype

    def zeros(shape, dtype):
        shape = tuple(int(value) for value in shape)
        return (
            np.broadcast_to(np.zeros((), dtype=np.dtype(dtype)), shape)
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
            return tuple(
                sharding_runtime._pad_high_to_shape(
                    # Host-created setup slabs stay on the host, but evolved
                    # device slabs must not round-trip through NumPy at every
                    # continuation boundary.
                    (np.asarray if isinstance(value, np.ndarray) else jnp.asarray)(
                        value, dtype=dtype
                    ),
                    shape,
                    pad_value=0.0,
                )
                for value, shape in zip(old, shapes, strict=True)
            )
        return tuple(zeros(shape, dtype) for shape in shapes)

    old_h = () if continuation is None else continuation.cpml_psi_h_terms
    old_e = () if continuation is None else continuation.cpml_psi_e_terms
    recorder_shapes = tuple(
        shape
        for spec in program.monitors
        if spec.recorder_index >= 0
        for shape in spec.field_shapes
    )
    # Check metadata before allocating: compatible histories can be much larger
    # than the fields, and must not need a second bank just to restore a state.
    if (
        program.monitors
        and continuation is not None
        and continuation.dft_vec_re.shape
        == (monitor_runtime.monitor_dft_value_size(program.monitors),)
        and continuation.dft_vec_im.shape
        == (monitor_runtime.monitor_dft_value_size(program.monitors),)
        and continuation.dft_weight_sum.shape
        == (monitor_runtime.monitor_dft_weight_size(program.monitors),)
        and tuple(value.shape[1:] for value in continuation.recorded_fields)
        == recorder_shapes
    ):
        monitor_values = {
            name: getattr(continuation, name) for name in monitor_runtime.MONITOR_FIELDS
        }
    else:
        monitor_values = monitor_runtime.empty_monitor_values(
            program,
            num_steps=max(
                1,
                int(
                    program.config.num_steps if monitor_steps is None else monitor_steps
                ),
            ),
        )
    # A continuation can promote the fields (for example, Periodic -> Bloch).
    # Promote recorder storage too, retaining all previous samples and metadata.
    recorded_fields = list(monitor_values["recorded_fields"])
    for spec in program.monitors:
        if spec.recorder_index < 0:
            continue
        for component, index in zip(
            spec.canonical_components, spec.field_buffer_indices, strict=True
        ):
            value = recorded_fields[index]
            dtype = jnp.result_type(value.dtype, field_dtype(component))
            if value.dtype != dtype:
                recorded_fields[index] = value.astype(dtype)
    monitor_values["recorded_fields"] = tuple(recorded_fields)
    from beamz.simulation.compile.material_blocks import initial_polarization

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
        cpml_psi_h_terms=restore_psi(old_h, cpml.h_terms, field_dtype("Hx")),
        cpml_psi_e_terms=restore_psi(old_e, cpml.e_terms, field_dtype("Ez")),
        **monitor_values,
        t=jnp.asarray(t, dtype=jnp.float32),
        current_step=jnp.asarray(current_step, dtype=jnp.int32),
    )
