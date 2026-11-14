"""Scan execution, caching, continuation, and runtime placement."""

from __future__ import annotations

import os
import pathlib
import platform
import sys
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace  # noqa: F401
from time import perf_counter
from typing import Any, cast  # noqa: F401

import jax
import jax.numpy as jnp  # noqa: F401
import numpy as np
from jax.core import Tracer  # noqa: F401

from beamz._helpers import (
    _finish_inline_progress,
    _print_inline_progress,
    _print_inline_status,
)
from beamz.const import EPS_0, MU_0  # noqa: F401
from beamz.devices.sources.compiler import (  # noqa: F401
    BatchedSlabGroup,
    CompiledSourceSpec,
    batch_slab_specs,
)
from beamz.simulation.compile import kernels as update_runtime  # noqa: F401
from beamz.simulation.compile.backend import CUDA_BF16_PSI, trace_preparation
from beamz.simulation.compile.backend import (
    compact_boundary_masks as compact_boundary_masks,
)
from beamz.simulation.compile.storage import cuda_capacity_schedule  # noqa: F401
from beamz.simulation.execute import sharding as sharding_runtime
from beamz.simulation.execute.automatic import (
    _energy_terms,
    _field_diagnostics,
    _relative_monitor_change,
)
from beamz.simulation.execute.automatic import (
    run_until_terminated as run_until_terminated,
)
from beamz.simulation.execute.scan import (
    ExecutionDecision,
    _BoundedCompileScan,
    build_scan,
    build_step_context,
    execution_decision,
    scan_steps,
)
from beamz.simulation.execute.step import (
    SOURCE_PHASE_COMPONENTS,
    ScanResult,
    SourceBatchMap,
    _add_source_slab,
    _apply_batched_slabs,
    _apply_specs,
    _known_primal,
    apply_source_phase,
    compiled_source_batches,
    forward_step,
)
from beamz.simulation.execute.step import initial_program_state as initial_program_state
from beamz.simulation.model import (
    AutoTermination,
    CompiledProgram,
    SimulationState,
    UpdateCoefficients,
)
from beamz.simulation.model import _copy_initial_field as _copy_initial_field
from beamz.simulation.results import (
    MonitorResults,
    RunTermination,
    SimulationPerformance,
    SimulationResults,
    SimulationRun,
)
from beamz.simulation.results import observe as monitor_runtime

# Native CUDA calls capture one graph node sequence per step. Keep captures small
# enough to instantiate predictably, then replay the same executable from an XLA loop.
# The even bound also keeps the temporal ping-pong kernel on its fast return layout.
CUDA_GRAPH_MAX_STEPS = 256


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
    executable_ready: bool = False
    cuda_schedule_decision: dict | None = None


_MAX_EXECUTION_CACHES = 8
_EXECUTION_CACHES: OrderedDict[int, tuple[object, dict[bool, _ExecutionCache]]] = (
    OrderedDict()
)


def execution_cache(program, *, donate_state=False) -> _ExecutionCache:
    """Return one ownership variant while bounding the number of retained programs."""
    key = id(program)
    entry = _EXECUTION_CACHES.get(key)
    if entry is None or entry[0] is not program:
        entry = (program, {})
        _EXECUTION_CACHES[key] = entry
        if len(_EXECUTION_CACHES) > _MAX_EXECUTION_CACHES:
            _EXECUTION_CACHES.popitem(last=False)
    _EXECUTION_CACHES.move_to_end(key)
    return entry[1].setdefault(bool(donate_state), _ExecutionCache())


def clear_execution_cache() -> None:
    _EXECUTION_CACHES.clear()


def build_program_scan(program: CompiledProgram, *, donate_state: bool = False):
    """Build and cache the JIT scan only when execution first needs it."""
    cache = execution_cache(program, donate_state=donate_state)
    scan = build_scan(program, donate_state=bool(donate_state))
    cache.compiled_scan = scan
    cache.cuda_schedule_decision = scan.cuda_schedule_decision
    cache.executable_ready = False
    return scan


def program_is_compiled(
    program: CompiledProgram, *, donate_state: bool = False
) -> bool:
    """Return whether the requested ownership variant is already cached."""
    cache = execution_cache(program, donate_state=donate_state)
    return cache.compiled_scan is not None


def _place_program_inputs(program, state):
    """Place fields, monitor state, and coefficients before execution or inspection."""
    state = sharding_runtime.prepare_state(
        program,
        state,
        replicated_fields=(*monitor_runtime.MONITOR_FIELDS, "t", "current_step"),
    )
    coeffs = sharding_runtime.place_tree(program, program.coefficients)
    return state, coeffs


def run_program(
    program: CompiledProgram,
    state: SimulationState,
    *,
    donate_state: bool = False,
) -> SimulationState:
    """Place state and coefficients, then execute the program's cached scan."""
    state, coeffs = _place_program_inputs(program, state)
    cache = execution_cache(program, donate_state=donate_state)
    compiled_scan = cache.compiled_scan or build_program_scan(
        program, donate_state=donate_state
    )
    return compiled_scan(*_execution_arguments(program, state, coeffs))


def _execution_arguments(program, state, coeffs):
    """Keep grid-sized ADE arrays out of the compiled loop's constant payload."""
    if program.dispersion is not None and program.dispersion.coefficients:
        return (
            state,
            coeffs,
            tuple(record[1:] for record in program.dispersion.coefficients),
        )
    return state, coeffs


def cuda_schedule_report(program, *, donate_state=False):
    """Return the frozen schedule decision without executing a timing trial."""
    from copy import deepcopy

    cache = execution_cache(program, donate_state=donate_state)
    decision = cache.cuda_schedule_decision
    if decision is None:
        scan = build_program_scan(program, donate_state=donate_state)
        decision = scan.cuda_schedule_decision
    return deepcopy(decision)


def compile_program_execution(
    program: CompiledProgram,
    state: SimulationState,
    *,
    donate_state: bool = False,
) -> None:
    """Compile and cache a program executable without advancing its state."""
    state, coeffs = _place_program_inputs(program, state)
    _compiled_program_execution(program, state, coeffs, donate_state=donate_state)


def _compiled_program_execution(
    program: CompiledProgram,
    state: SimulationState,
    coeffs,
    *,
    donate_state: bool,
):
    """Return an executable after compiling it outside the performance timer."""
    cache = execution_cache(program, donate_state=donate_state)
    compiled_scan = cache.compiled_scan or build_program_scan(
        program, donate_state=donate_state
    )
    if cache.executable_ready:
        return compiled_scan
    compiled = (
        cast(Any, compiled_scan)
        .lower(*_execution_arguments(program, state, coeffs))
        .compile()
    )
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


def _execute_chunk(program, state, *, monitor_steps, donate_state, timed):
    """Restore, place, execute, synchronize and crop one owned continuation.

    Compilation precedes the timer; the untimed path retains lazy compilation.
    Neither path derives clocks on the host or copies a donated input afterward.
    """
    state = runtime_inputs(program, state, monitor_steps=monitor_steps)
    if timed:
        state, coeffs = _place_program_inputs(program, state)
        executable = _compiled_program_execution(
            program, state, coeffs, donate_state=donate_state
        )
        started = perf_counter()
        state = executable(*_execution_arguments(program, state, coeffs))
    else:
        state = run_program(program, state, donate_state=donate_state)
    state.ez.block_until_ready()
    elapsed = max(perf_counter() - started, np.finfo(float).tiny) if timed else 0.0
    return sharding_runtime.crop_state(program, state), elapsed


def _performance_for(
    program: CompiledProgram, *, runtime_s: float, steps: int
) -> SimulationPerformance:
    return SimulationPerformance(
        runtime_s=runtime_s,
        cells=int(np.prod(program.grid.permittivity.shape)),
        steps=steps,
    )


def compiled_xla_memory_analysis(
    program: CompiledProgram,
    state: SimulationState,
    *,
    monitor_steps: int,
) -> dict:
    """Return backend memory data for the exact placed program and state."""
    state, coeffs = _place_program_inputs(
        program, runtime_inputs(program, state, monitor_steps=monitor_steps)
    )
    cache = execution_cache(program)
    compiled_scan = cache.compiled_scan or build_program_scan(program)
    compiled = (
        cast(Any, compiled_scan)
        .lower(*_execution_arguments(program, state, coeffs))
        .compile()
    )
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


def _run_results(
    simulation,
    program,
    state,
    store_full_materials,
    *,
    monitor_results=None,
    **metadata,
):
    """Detach acquisitions and source/material metadata at the run boundary."""
    return SimulationResults.from_run(
        simulation,
        runtime_fields=program.grid,
        monitor_results=(
            _decode_monitor_results(simulation, program, state)
            if monitor_results is None
            else monitor_results
        ),
        store_full_materials=store_full_materials,
        source_launch_powers=_compiled_source_launch_powers(
            program, len(simulation.sources)
        ),
        **metadata,
    )


_PROGRESS_CHUNK_COUNT = 20


def run_simulation_program(
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
    progress: bool,
    schedule: str | None = None,
    expected_runs: int = 1,
) -> SimulationRun:
    """Run one segment; optional progress changes chunking, not lifecycle ownership."""
    from beamz.simulation.cuda.schedule import normalize_expected_runs

    expected_runs = normalize_expected_runs(expected_runs)
    if progress:
        chunk_steps = max(1, int(np.ceil(num_steps / _PROGRESS_CHUNK_COUNT)))
        full_chunks, remainder = divmod(num_steps, chunk_steps)
        chunk_lengths = (chunk_steps,) * full_chunks + (
            (remainder,) if remainder else ()
        )
    else:
        chunk_lengths = (num_steps,)
    # This API already knows how often each executable will run. Account for
    # that work even when visible progress divides a long run into short calls.
    chunk_reuse = Counter(chunk_lengths)
    if progress:
        _print_inline_status("Compiling simulation...")
    try:
        programs = {
            length: simulation.compile(
                num_steps=length,
                sharding=sharding,
                backend=backend,
                schedule=schedule,
                expected_runs=expected_runs * chunk_reuse[length],
                progress=False,
            )
            for length in set(chunk_lengths)
        }
        if state is None:
            state = SimulationState.initial(
                programs[chunk_lengths[0]].grid, t=float(simulation.time[0])
            )
        if progress:
            for program in programs.values():
                compile_program_execution(
                    program,
                    runtime_inputs(program, state, monitor_steps=monitor_steps),
                    donate_state=donate_state,
                )
    finally:
        if progress:
            _finish_inline_progress()

    completed = 0
    runtime_s = 0.0
    program = programs[chunk_lengths[-1]]
    if progress:
        _print_inline_progress(0, num_steps, label="Running simulation")
    try:
        for length in chunk_lengths:
            program = programs[length]
            state, elapsed = _execute_chunk(
                program,
                state,
                monitor_steps=monitor_steps,
                donate_state=donate_state,
                timed=performance,
            )
            runtime_s += elapsed
            completed += length
            if progress:
                _print_inline_progress(completed, num_steps, label="Running simulation")
    finally:
        if progress:
            _finish_inline_progress()

    stats = (
        _performance_for(program, runtime_s=runtime_s, steps=num_steps)
        if performance
        else None
    )
    state = cast(SimulationState, state)
    results = _run_results(
        simulation,
        program,
        state,
        store_full_materials,
        completed_steps=int(state.current_step),
        performance=stats,
    )
    if stats is not None and performance:
        print(f"Simulation runtime: {stats.runtime_s:.2f} s")
        print(f"GCUPS: {stats.gcups:.3f}")
    return SimulationRun(results=results, state=state)


_init_persistent_cache()
