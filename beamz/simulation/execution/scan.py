"""Execution policy and compiled scan construction."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.core import Tracer

from beamz.const import EPS_0, MU_0
from beamz.simulation import kernels as update_runtime
from beamz.simulation import observe as monitor_runtime
from beamz.simulation.memory import cuda_capacity_schedule
from beamz.simulation.model import SimulationState, UpdateCoefficients

from .step import SOURCE_PHASE_COMPONENTS, compiled_source_batches, forward_step


class _BoundedCompileScan:
    # Per-jit options cannot be nested under jit/grad/vmap; let the outer
    # executable own compilation when any input is a tracer.

    def __init__(self, body, compiled):
        self.body = body
        self.compiled = compiled

    def __call__(self, *arguments):
        if any(isinstance(value, Tracer) for value in jax.tree.leaves(arguments)):
            return self.body(*arguments)
        return self.compiled(*arguments)

    def __getattr__(self, name):
        return getattr(self.compiled, name)


def build_step_context(program, *, donate_state: bool = False):
    """Bind the shared source, boundary, and memory plan for one timestep."""
    from beamz.simulation import execute as execution_runtime

    cfg = program.config
    boundary = program.boundary
    if cfg.backend == "jax" or cfg.sharding.enabled:
        boundary = execution_runtime.compact_boundary_masks(boundary)
    return update_runtime.CompiledStepContext(
        config=cfg,
        boundary=boundary,
        source_batches=compiled_source_batches(program.sources),
        metrics=program.metrics,
        resolution=float(cfg.resolution),
        dt=float(cfg.dt),
        dt_scalar=jnp.asarray(cfg.dt, dtype=jnp.float32),
        is_3d=cfg.is_3d,
        sharding_plan=program.sharding,
        low_memory=cuda_capacity_schedule(program, donate_state=donate_state),
    )


def scan_steps(step, state, num_steps, *, checkpoint_interval=0, remat_steps=False):
    """Run indexed transitions with one bounded checkpoint/tail lifecycle.

    AD material binding retains each chunk's local trajectory during replay;
    ordinary checkpointed scans additionally rematerialize individual steps.
    These policies differ deliberately while sharing chunk indices and tails.
    """
    transition = jax.checkpoint(step) if remat_steps else step

    def advance(carry, start, length):
        indices = start + jnp.arange(length, dtype=jnp.int32)
        return jax.lax.scan(transition, carry, indices)[0]

    if not checkpoint_interval:
        return advance(state, 0, num_steps)
    chunks, tail = divmod(num_steps, checkpoint_interval)

    def chunk(carry, index):
        return advance(carry, index * checkpoint_interval, checkpoint_interval), None

    out = jax.lax.scan(
        jax.checkpoint(chunk, prevent_cse=False),
        state,
        jnp.arange(chunks, dtype=jnp.int32),
    )[0]
    if tail:
        out = jax.checkpoint(
            lambda carry: advance(carry, chunks * checkpoint_interval, tail),
            prevent_cse=False,
        )(out)
    return out


@dataclass(frozen=True)
class ExecutionDecision:
    """One invocation policy, resolved before tracing and owned by its scan cache.

    Context freezes backend, physics, precision, storage, sharding and capacity;
    the remaining fields bind the selected algorithm and its dynamic ABI layout.
    Donation participates in construction and in the executable cache key.
    """

    context: Any
    kernel: Any
    source_groups: Any
    monitors: Any
    mstep: Any
    report: Any
    native_calls: Any
    native: Any
    local_cpml: Any
    local_dft: Any


def execution_decision(program, *, donate_state=False):
    """Resolve kernel, schedule, operands, workspace and diagnostics together."""
    cfg = program.config
    step_context = build_step_context(program, donate_state=donate_state)
    source_batches = step_context.source_batches

    update_kernel = update_runtime.select_update_kernel(step_context)
    from beamz.simulation import distributed_monitors, jax_sharding

    local_jax_cpml = jax_sharding.supported(program)
    local_dft = distributed_monitors.supported(program)
    if local_jax_cpml:
        update_kernel = jax_sharding.select_kernel()
    graph_source_groups = tuple(
        source_batches[(timing, component)][0]
        for timing, components in SOURCE_PHASE_COMPONENTS.items()
        for component in components
    )
    source_groups_supported = bool(program.sources) and all(
        not rest for _group, rest in source_batches.values()
    )
    graph_monitors_supported = bool(program.monitors) and all(
        monitor_runtime.is_vector_dft_monitor(monitor) for monitor in program.monitors
    )
    packed_graph_monitors = None
    if (
        cfg.backend in {"cuda_streamed", "cuda_mstep"}
        and not cfg.sharding.enabled
        and graph_monitors_supported
        and not bool(jax.config.read("jax_enable_x64"))
    ):
        from beamz.simulation.cuda import pack_dft_monitors

        packed_graph_monitors = pack_dft_monitors(program.monitors)
    cuda_multi_step = (
        cfg.backend in {"cuda_streamed", "cuda_mstep"}
        and not cfg.sharding.enabled
        and (not program.monitors or packed_graph_monitors is not None)
        and (not program.sources or source_groups_supported)
    )
    from beamz.simulation.cuda.schedule import choose_execution_schedule

    mstep_plan, schedule_decision = choose_execution_schedule(
        program,
        graph_source_groups,
        packed_graph_monitors,
        native_supported=cuda_multi_step,
        donate_state=donate_state,
    )
    if mstep_plan is not None:
        from beamz.simulation.cuda.mstep_config import ALLOWED_FLAGS, variant_summary

        step_context = replace(
            step_context,
            config=replace(
                cfg,
                cuda_flags=schedule_decision.get("mstep_cuda_flags", cfg.cuda_flags)
                & ALLOWED_FLAGS,
            ),
        )
        schedule_decision.update(variant_summary(step_context.config.cuda_flags))
    elif cfg.backend == "cuda_mstep" or cfg.cuda_schedule == "mstep":
        step_context = replace(step_context, low_memory=True)
    if cfg.backend in {"cuda_streamed", "cuda_mstep"} and mstep_plan is None:
        from beamz.simulation.memory import cuda_workspace_estimate

        schedule_decision["workspace_bytes"] = cuda_workspace_estimate(
            program,
            donate_state=donate_state,
            low_memory=step_context.low_memory,
        )
        schedule_decision["capacity_schedule"] = step_context.low_memory
    native_graph_calls = None
    if (
        cfg.cuda_storage_axes != (0, 1, 2)
        and not cfg.sharding.enabled
        and not step_context.low_memory
    ):
        if not (
            cuda_multi_step and cfg.is_3d and cfg.metric_kind == "isotropic_uniform"
        ):
            raise ValueError(
                "CUDA storage-axis permutations require a 3D uniform-grid "
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

    return ExecutionDecision(
        context=step_context,
        kernel=update_kernel,
        source_groups=graph_source_groups,
        monitors=packed_graph_monitors,
        mstep=mstep_plan,
        report=schedule_decision,
        native_calls=native_graph_calls,
        native=cuda_multi_step,
        local_cpml=local_jax_cpml,
        local_dft=local_dft,
    )


def build_scan(program, *, donate_state: bool = False, checkpoint_interval=None) -> Any:
    """Build the compiled scan, optionally rematerializing chunks for reverse AD.

    Checkpointed JAX execution uses the ordinary timestep, source, boundary,
    dispersion, and absolute observation-clock transition.
    """
    from beamz.simulation import execute as execution_runtime

    if checkpoint_interval is not None:
        if (
            isinstance(checkpoint_interval, bool)
            or int(checkpoint_interval) != checkpoint_interval
            or checkpoint_interval < 1
        ):
            raise ValueError("checkpoint_interval must be a positive integer.")
        if program.config.backend != "jax":
            raise ValueError("Checkpointed differentiation requires backend='jax'.")
        checkpoint_interval = int(checkpoint_interval)

    from beamz.simulation import distributed_monitors

    decision = execution_decision(program, donate_state=donate_state)
    cfg = program.config
    step_context = decision.context
    boundary = step_context.boundary
    dt_scalar = step_context.dt_scalar
    update_kernel = decision.kernel
    graph_source_groups = decision.source_groups
    packed_graph_monitors = decision.monitors
    mstep_plan = decision.mstep
    schedule_decision = decision.report
    native_graph_calls = decision.native_calls
    cuda_multi_step = decision.native
    local_jax_cpml = decision.local_cpml
    local_dft = decision.local_dft

    def run_scan(
        state: SimulationState,
        coeffs: UpdateCoefficients,
        dispersion_coefficients=None,
        mstep_buffers=None,
    ):
        runtime_program = program
        if dispersion_coefficients is not None:
            runtime_program = replace(
                program,
                dispersion=replace(
                    program.dispersion,
                    coefficients=tuple(
                        (record[0], *values)
                        for record, values in zip(
                            program.dispersion.coefficients,
                            dispersion_coefficients,
                            strict=True,
                        )
                    ),
                ),
            )
        # Normal continuations keep one absolute integer clock. Explicit states
        # may intentionally supply a different time origin; preserve that API
        # contract without accumulating rounding at ordinary chunk boundaries.
        t0 = jnp.asarray(cfg.t0, dtype=jnp.float32)
        grid_time = t0 + dt_scalar * state.current_step.astype(jnp.float32)
        clock_tolerance = jnp.finfo(jnp.float32).eps * jnp.maximum(
            jnp.abs(grid_time), jnp.finfo(jnp.float32).tiny
        )
        on_grid = jnp.abs(state.t - grid_time) <= clock_tolerance
        observation_origin = cast(jax.Array, jnp.where(on_grid, t0, state.t))
        observation_offset = cast(jax.Array, jnp.where(on_grid, state.current_step, 0))
        if local_dft:
            state = distributed_monitors.scan_local_dft(state, program)
        local_cuda_cpml = (
            cfg.backend == "cuda_streamed" and program.sharding.layout.enabled
        )
        if local_cuda_cpml or local_jax_cpml:
            from beamz.simulation.distributed import scan_local_cpml

            state = scan_local_cpml(state, program)
        # Compute differentiable material scales once per invocation. Expressing
        # the lossy update as an increment avoids a field-sized division each step.
        if (
            cfg.backend == "jax"
            and cfg.is_3d
            and boundary.cpml.enabled
            and cfg.metric_kind == "isotropic_uniform"
            and not coeffs.e_inverse_offdiagonal.size
        ):
            from beamz.simulation.material_blocks import MaterialBlocks, material_values

            dt_mu, dt_eps = dt_scalar / MU_0, dt_scalar / EPS_0
            scales = {}
            for component in "xyz":
                sigma_h = getattr(coeffs, f"h_sigma_m_{component}")
                sigma_e = getattr(coeffs, f"e_conductivity_{component}")
                epsilon = getattr(coeffs, f"e_permittivity_{component}")
                scales[f"h_source_{component}"] = dt_mu / (1 + 0.5 * dt_mu * sigma_h)
                if isinstance(epsilon, MaterialBlocks) and sigma_e.ndim == 0:
                    scales[f"e_source_{component}"] = replace(
                        epsilon,
                        values=dt_eps / (epsilon.values + 0.5 * dt_eps * sigma_e),
                    )
                else:
                    scales[f"e_source_{component}"] = dt_eps / (
                        material_values(epsilon) + 0.5 * dt_eps * sigma_e
                    )
            coeffs = coeffs._replace(**scales)

        def step(carry, step_index):
            return forward_step(
                carry,
                ctx=step_context,
                coeffs=coeffs,
                program=runtime_program,
                update_kernel=update_kernel,
                observation_time=observation_origin
                + dt_scalar * (observation_offset + step_index + 1).astype(jnp.float32),
            ), None

        # 4. Run the same transition through scan or fori_loop. The choice changes the
        # lowering strategy, not timestep semantics.
        if cuda_multi_step:
            from beamz.simulation.cuda.mstep import run_mstep
            from beamz.simulation.cuda.runtime import run_native_steps

            def invoke_mstep(
                chunk_state, ctx, coefficients, groups, monitors, steps, **clock
            ):
                return run_mstep(
                    chunk_state,
                    ctx,
                    coefficients,
                    groups,
                    monitors,
                    mstep_plan,
                    steps,
                    plan_buffers=mstep_buffers,
                    **clock,
                )

            mstep_call: Callable = invoke_mstep
            if mstep_plan is not None and cfg.cuda_storage_axes != (0, 1, 2):
                from beamz.simulation.cuda.storage import wrap_program_call

                mstep_call = wrap_program_call(mstep_call, cfg.cuda_storage_axes)

            def advance_native_chunk(chunk_state, chunk_steps: int, elapsed_steps):
                elapsed_steps = jnp.asarray(elapsed_steps, dtype=jnp.int32)
                chunk_state = chunk_state._replace(
                    # Derive clocks from the simulation origin, including across
                    # separate advance() calls and automatic-termination chunks.
                    t=observation_origin
                    + dt_scalar
                    * (observation_offset + elapsed_steps).astype(jnp.float32),
                    current_step=state.current_step + elapsed_steps,
                )
                chunk_out = (
                    mstep_call(
                        chunk_state,
                        step_context,
                        coeffs,
                        graph_source_groups,
                        packed_graph_monitors,
                        chunk_steps,
                        observation_origin=observation_origin,
                        observation_step_offset=observation_offset + elapsed_steps,
                    )
                    if mstep_plan is not None
                    else run_native_steps(
                        chunk_state,
                        step_context,
                        coeffs,
                        graph_source_groups,
                        packed_graph_monitors,
                        chunk_steps,
                        observation_origin=observation_origin,
                        observation_step_offset=observation_offset + elapsed_steps,
                        native_calls=native_graph_calls,
                    )
                )
                completed_steps = elapsed_steps + jnp.asarray(
                    chunk_steps, dtype=jnp.int32
                )
                return chunk_out._replace(
                    t=observation_origin
                    + dt_scalar
                    * (observation_offset + completed_steps).astype(jnp.float32),
                    current_step=state.current_step + completed_steps,
                )

            if cfg.num_steps <= execution_runtime.CUDA_GRAPH_MAX_STEPS:
                scan_out = advance_native_chunk(state, cfg.num_steps, 0)
            else:
                full_chunks, tail_steps = divmod(
                    cfg.num_steps, execution_runtime.CUDA_GRAPH_MAX_STEPS
                )
                # The loop body has one fixed native call signature. XLA therefore
                # reuses its buffers and the CUDA layer hits one bounded graph-cache
                # entry instead of capturing a graph proportional to the entire run.
                scan_out = jax.lax.fori_loop(
                    0,
                    full_chunks,
                    lambda _i, chunk_state: advance_native_chunk(
                        chunk_state,
                        execution_runtime.CUDA_GRAPH_MAX_STEPS,
                        _i * execution_runtime.CUDA_GRAPH_MAX_STEPS,
                    ),
                    state,
                )
                if tail_steps:
                    scan_out = advance_native_chunk(
                        scan_out,
                        tail_steps,
                        full_chunks * execution_runtime.CUDA_GRAPH_MAX_STEPS,
                    )
        elif checkpoint_interval is not None or cfg.loop_kind == "scan":
            scan_out = scan_steps(
                step,
                state,
                cfg.num_steps,
                checkpoint_interval=checkpoint_interval or 0,
                remat_steps=checkpoint_interval is not None,
            )
        else:
            scan_out = jax.lax.fori_loop(
                0,
                cfg.num_steps,
                lambda index, carry: step(carry, index)[0],
                state,
            )
        if local_cuda_cpml or local_jax_cpml:
            scan_out = scan_local_cpml(scan_out, program, assemble=True)
        if local_dft:
            scan_out = distributed_monitors.scan_local_dft(
                scan_out, program, assemble=True
            )
        return scan_out

    # 8. Buffer donation is an explicit ownership transfer. The default executable
    # preserves its input state; callers may opt into the lower-memory variant when
    # they no longer need that continuation value.
    donate_argnums = (0,) if donate_state else ()
    # Empirical fusion autotuning allocates full-sized trial inputs/outputs during
    # compilation, outside executable.memory_analysis(). Use default emitters for
    # both FDTD backends so an otherwise fitting run can compile near capacity.
    # JAX and the plan adapters expose a dynamic callable/lowering interface.
    scan: Any
    if "compiler_options" in inspect.signature(jax.jit).parameters:
        scan = _BoundedCompileScan(
            run_scan,
            jax.jit(
                run_scan,
                donate_argnums=donate_argnums,
                compiler_options={"xla_gpu_autotune_level": 0},
            ),
        )
    # Older supported JAX releases lack per-executable options. Preserve their
    # existing behavior without changing process-wide XLA_FLAGS after startup.
    else:
        scan = jax.jit(run_scan, donate_argnums=donate_argnums)
    if mstep_plan is not None:
        from beamz.simulation.cuda.mstep import bind_plan_buffers

        scan = bind_plan_buffers(scan, mstep_plan)
    scan.cuda_schedule_decision = schedule_decision
    return scan
