"""Automatic-termination energy, source and monitor measurements."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0
from beamz.simulation import kernels as update_runtime
from beamz.simulation.model import AutoTermination, CompiledProgram, SimulationState
from beamz.simulation.results import MonitorResults

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
    widths = tuple(
        jnp.asarray(program.grid.geometry.cell_widths(a)) for a in ("x", "y", "z")
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
