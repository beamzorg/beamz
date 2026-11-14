"""Automatic execution termination and convergence checks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from beamz._helpers import (
    _finish_inline_progress,
    _print_inline_progress,
    _print_inline_status,
)
from beamz.const import EPS_0, MU_0
from beamz.simulation.compile import physics as update_runtime
from beamz.simulation.model import AutoTermination, CompiledProgram, SimulationState
from beamz.simulation.results import MonitorResults, RunTermination, SimulationResults


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


_COMPONENT_OFFSETS_2D = {
    "Ex": (0.0, 0.5),
    "Ey": (0.5, 0.0),
    "Ez": (0.0, 0.0),
    "Hx": (0.5, 0.0),
    "Hy": (0.0, 0.5),
    "Hz": (0.5, 0.5),
}


def _axis_integration_weights(edges: Any, offset: float, count: int) -> np.ndarray:
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


def _component_integration_weights(
    program: CompiledProgram, component: str
) -> jax.Array:
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


def _node_integration_weights(
    program: CompiledProgram, shape: tuple[int, ...]
) -> jax.Array:
    geometry = program.grid.geometry
    if program.config.is_3d:
        z = _axis_integration_weights(geometry.z_edges, 0.0, shape[0])
        y = _axis_integration_weights(geometry.y_edges, 0.0, shape[1])
        x = _axis_integration_weights(geometry.x_edges, 0.0, shape[2])
        return jnp.asarray(z[:, None, None] * y[None, :, None] * x[None, None, :])
    y = _axis_integration_weights(geometry.y_edges, 0.0, shape[0])
    x = _axis_integration_weights(geometry.x_edges, 0.0, shape[1])
    return jnp.asarray(y[:, None] * x[None, :])


def _energy_terms(program: CompiledProgram) -> tuple[Any, ...]:
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
    cross_terms: tuple[tuple[Any, ...], ...] = ()
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
    widths = tuple(
        jnp.asarray(program.grid.geometry.cell_widths(cast(Literal["x", "y", "z"], a)))
        for a in ("x", "y", "z")
    )
    return terms, cross_terms, domain_measure, widths


def _field_diagnostics(state: SimulationState, plan: Any) -> tuple[float, float, bool]:
    terms, cross_terms, domain_measure, widths = plan
    energy_density = jnp.asarray(0.0, dtype=jnp.float32)
    max_field = jnp.asarray(0.0, dtype=jnp.float32)
    finite = jnp.asarray(True)
    for field_name, material, measure, constant in terms:
        values = jnp.asarray(getattr(state, field_name))
        finite = finite & jnp.all(jnp.isfinite(values))
        max_field = jnp.maximum(max_field, jnp.max(jnp.abs(values), initial=0.0))
        density = jnp.asarray(material) * jnp.abs(values) ** 2
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
                    getattr(state, field_name),
                    component,
                    "Node",
                    node_shape,
                    cell_widths=widths,
                )
        energy_density = energy_density + float(EPS_0) * jnp.sum(
            material
            * jnp.real(centered[left_component] * jnp.conj(centered[right_component]))
            * jnp.asarray(measure)
        )
    return float(energy_density) * domain_measure, float(max_field), bool(finite)


def _remaining_source_activity(
    program: CompiledProgram, total_steps: int
) -> np.ndarray:
    total_steps = int(total_steps)
    activity = np.zeros(total_steps, dtype=np.float64)
    for source in program.sources:
        waveform = np.abs(np.asarray(source.waveform).reshape(-1))
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
    index = int(current_step)
    if index < 0:
        raise ValueError("current_step must be non-negative.")
    return float(remaining_activity[index]) if index < remaining_activity.size else 0.0


def _selected_monitor_names(
    program: CompiledProgram, policy: AutoTermination
) -> tuple[str, ...]:
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
    monitors: dict[str, MonitorResults], names: tuple[str, ...]
) -> dict[tuple[str, str, int], np.ndarray]:
    values = {}
    for name in names:
        monitor = monitors[name]
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
) -> float | None:
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


def _configured_dft_weight_sum(simulation: Any, spec: Any) -> float:
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
    monitors: dict[str, MonitorResults], simulation: Any, program: CompiledProgram
) -> dict[str, MonitorResults]:
    monitors = dict(monitors)
    for spec in program.monitors:
        if not spec.dft_enabled or spec.freq_count <= 0:
            continue
        monitor = simulation.monitors[int(spec.monitor_index)]
        name = str(getattr(monitor, "name", None) or f"monitor_{spec.monitor_index}")
        result = monitors[name]
        total_weight = _configured_dft_weight_sum(simulation, spec)
        weights = np.full(result.dft_weight_sum.shape, total_weight, dtype=np.float64)
        monitors[name] = replace(result, dft_weight_sum=weights)
    return monitors
