"""Automatic execution termination and convergence checks."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from beamz._helpers import (
    _finish_inline_progress,
    _print_inline_progress,
    _print_inline_status,
)
from beamz.simulation.model import AutoTermination, SimulationState
from beamz.simulation.results import MonitorResults, RunTermination, SimulationResults

from .termination import (
    _complete_converged_dft_weights,
    _energy_terms,
    _monitor_vectors,
    _relative_monitor_change,
    _remaining_source_activity,
    _selected_monitor_names,
    _source_residual,
)


def run_until_terminated(
    simulation,
    policy: AutoTermination,
    *,
    progress: bool,
    store_full_materials: bool,
    sharding,
    backend: str,
    performance: bool,
    schedule: str | None = None,
    expected_runs: int = 1,
) -> SimulationResults:
    """Run reusable chunks until convergence, failure, or the configured time limit."""
    from beamz.simulation import execute as execution_runtime

    if not isinstance(policy, AutoTermination):
        raise TypeError("termination must be an AutoTermination instance or None.")
    wall_start = execution_runtime.perf_counter()
    chunk_steps = min(int(policy.chunk_steps), int(simulation.num_steps))
    first_program = simulation.compile(
        num_steps=chunk_steps,
        sharding=sharding,
        backend=backend,
        schedule=schedule,
        expected_runs=expected_runs,
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
    monitor_results: dict[str, MonitorResults] | None = None
    program = first_program
    runtime_s = 0.0
    compiling = progress and not execution_runtime.program_is_compiled(
        first_program, donate_state=True
    )
    if compiling:
        _print_inline_status("Compiling simulation...")
    elif progress:
        _print_inline_progress(0, int(simulation.num_steps), label="Running simulation")

    try:
        while int(state.current_step) < int(simulation.num_steps):
            if (
                monitor_results is not None
                and policy.max_wall_time is not None
                and execution_runtime.perf_counter() - wall_start
                >= policy.max_wall_time
            ):
                reason = "wall_time_limit"
                break
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
                    schedule=schedule,
                    expected_runs=expected_runs,
                    progress=False,
                )
            )
            state, elapsed = execution_runtime._execute_chunk(
                program,
                state,
                monitor_steps=remaining,
                donate_state=True,
                timed=performance,
            )
            runtime_s += elapsed
            # Convergence uses raw acquisitions. Building a durable result here
            # repeatedly normalizes the full source record and copies metadata.
            monitor_results = execution_runtime._decode_monitor_results(
                simulation, program, state
            )
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

            energy, max_field, fields_finite = execution_runtime._field_diagnostics(
                state, terms
            )
            current_monitor = _monitor_vectors(monitor_results, monitor_names)
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

    if monitor_results is None:
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
    if reason == "converged":
        monitor_results = _complete_converged_dft_weights(
            monitor_results, simulation, first_program
        )
    results = execution_runtime._run_results(
        simulation,
        program,
        state,
        store_full_materials,
        monitor_results=monitor_results,
        completed_steps=int(state.current_step)
        if reason == "wall_time_limit"
        else None,
    )
    stats = (
        execution_runtime._performance_for(
            first_program, runtime_s=runtime_s, steps=int(state.current_step)
        )
        if performance
        else None
    )
    results = replace(results, termination=report, performance=stats)
    if stats is not None:
        print(f"Simulation runtime: {stats.runtime_s:.2f} s")
        print(f"GCUPS: {stats.gcups:.3f}")
    return results
