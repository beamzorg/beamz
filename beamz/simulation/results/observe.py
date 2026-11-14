"""Generic monitor and field-recorder sampling and accumulation."""

from __future__ import annotations

# Retain the names available from this module before its implementation was split.
from dataclasses import dataclass as dataclass
from dataclasses import replace as replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import LIGHT_SPEED
from beamz.devices._immutable import readonly_array as readonly_array
from beamz.devices.monitors.compiler import (
    CompiledMonitorSpec,
    SamplingPlan,
)
from beamz.devices.sources.specs import GaussianBeamSource as GaussianBeamSource
from beamz.devices.sources.specs import GaussianSource as GaussianSource
from beamz.devices.sources.specs import ModeSource as ModeSource
from beamz.simulation.model import SimulationState
from beamz.simulation.results.analysis import (
    SourceNormalization,  # noqa: F401
    _flux_component,  # noqa: F401
    _optional_flux_component,  # noqa: F401
    _sampled_source_spectrum,  # noqa: F401
    _scale_frequency_axis,  # noqa: F401
    _signal_dft_spectrum,  # noqa: F401
    _source_launch_power_ratio,  # noqa: F401
    _source_normalization,  # noqa: F401
    monitor_dft_component,  # noqa: F401
    monitor_dft_flux,  # noqa: F401
    monitor_flux,  # noqa: F401
    monitor_frequencies,  # noqa: F401
    normalization_from_result,  # noqa: F401
    renormalize_result,  # noqa: F401
    source_field_amplitude_normalization,  # noqa: F401
    source_normalization,  # noqa: F401
)


def step_hits_interval(step, interval):
    """Return whether a zero-based step lands on a positive modulo interval."""
    step = jnp.asarray(step, dtype=jnp.int32)
    interval = jnp.maximum(jnp.asarray(interval, dtype=jnp.int32), 1)
    return (step % interval) == 0


def is_vector_dft_monitor(mon):
    """Whether a monitor contains only nonempty vector DFT acquisitions."""
    return (
        mon.recorder_index < 0
        and mon.dft_enabled
        and mon.freq_count > 0
        and mon.dft_point_count > 0
    )


def monitor_records_on_step(step, interval):
    """Return whether a monitor emits after this zero-based simulation step."""
    return step_hits_interval(jnp.asarray(step, dtype=jnp.int32) + 1, interval)


def monitor_dft_should_accumulate(enabled, step, t, t_start, t_end, interval):
    """Apply the compiled time window and sampling interval for one DFT sample."""
    t = jnp.asarray(t, dtype=jnp.float32)
    return (
        jnp.asarray(enabled)
        & (t >= jnp.asarray(t_start, dtype=jnp.float32))
        & (t <= jnp.asarray(t_end, dtype=jnp.float32))
        & step_hits_interval(step, interval)
    )


def monitor_dft_window_weight(t, t_start, t_end, use_hann):
    """Return one or the Hann weight selected by a compiled monitor plan."""
    t, t0, t1 = (jnp.asarray(value, dtype=jnp.float32) for value in (t, t_start, t_end))
    span = jnp.maximum(t1 - t0, jnp.asarray(1e-30, dtype=jnp.float32))
    tau = jnp.clip((t - t0) / span, 0.0, 1.0)
    hann = 0.5 * (1.0 - jnp.cos(jnp.asarray(2.0 * np.pi, dtype=jnp.float32) * tau))
    return jnp.where(jnp.asarray(use_hann) & jnp.isfinite(t1) & (t1 > t0), hann, 1.0)


def monitor_dft_sample_scale(
    weight, *, normalization_code, base_dt, record_interval, length_unit
):
    """Scale a nonnegative DFT weight in native or physical units."""
    dtype = jnp.result_type(weight, base_dt, record_interval)
    native = jnp.maximum(jnp.asarray(weight, dtype=dtype), 0.0)
    physical = native * (
        jnp.asarray(base_dt * record_interval * LIGHT_SPEED / length_unit, dtype=dtype)
        / jnp.sqrt(jnp.asarray(2.0 * np.pi, dtype=dtype))
    )
    return jnp.where(jnp.asarray(normalization_code) == 1, physical, native)


def monitor_dft_accumulator_dtype():
    """Return the real dtype used for compiled monitor DFT accumulators."""

    # Match global precision so long accumulations gain x64 accuracy when enabled.
    return jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32


def monitor_dft_value_size(specs: tuple[CompiledMonitorSpec, ...]) -> int:
    """Return the exact number of real values in one vector-DFT arena."""
    return sum(6 * int(spec.freq_count) * int(spec.dft_point_count) for spec in specs)


def monitor_dft_weight_size(specs: tuple[CompiledMonitorSpec, ...]) -> int:
    """Return the exact number of normalization weights in the DFT arena."""
    return sum(int(spec.freq_count) for spec in specs)


MONITOR_FIELDS = (
    "dft_vec_re",
    "dft_vec_im",
    "dft_weight_sum",
    "recorded_fields",
    "recorded_steps",
    "recorded_times",
    "recorded_counts",
)


def empty_monitor_values(program, num_steps: int | None = None) -> dict[str, Any]:
    specs = program.monitors
    dft_dtype = monitor_dft_accumulator_dtype()
    # Flux is derived from acquired field DFTs; only produced state is allocated.
    dft_value_count = monitor_dft_value_size(specs)
    dft_weight_count = monitor_dft_weight_size(specs)
    recorder_specs = tuple(spec for spec in specs if spec.recorder_index >= 0)
    recorder_steps = max(
        1, int(program.config.num_steps if num_steps is None else num_steps)
    )
    # 3. Initialize every accumulated quantity at zero.
    values = (
        jnp.zeros((dft_value_count,), dtype=dft_dtype),
        jnp.zeros((dft_value_count,), dtype=dft_dtype),
        jnp.zeros((dft_weight_count,), dtype=dft_dtype),
        tuple(
            jnp.zeros(
                (
                    int(np.ceil(recorder_steps / spec.record_interval)),
                    *shape,
                ),
                dtype=program.grid.Ex.dtype,
            )
            for spec in recorder_specs
            for shape in spec.field_shapes
        ),
        tuple(
            jnp.zeros(
                (int(np.ceil(recorder_steps / spec.record_interval)),),
                dtype=jnp.int32,
            )
            for spec in recorder_specs
        ),
        tuple(
            jnp.zeros(
                (int(np.ceil(recorder_steps / spec.record_interval)),),
                dtype=jnp.float32,
            )
            for spec in recorder_specs
        ),
        tuple(jnp.zeros((), dtype=jnp.int32) for _ in recorder_specs),
    )
    return dict(zip(MONITOR_FIELDS, values, strict=True))


def _dft_phase_angle(frequencies, time, dtype):
    """Round angular frequency and observation time before their product.

    XLA otherwise reassociates constant frequencies with the timestep arithmetic.
    That changes optical phases by many float32 ULPs over long runs relative to
    the native monitor kernel, even when the fields and clocks agree.
    """
    omega = jax.lax.optimization_barrier(
        jnp.asarray(2.0 * np.pi, dtype=dtype) * jnp.asarray(frequencies, dtype=dtype)
    )
    time = jax.lax.optimization_barrier(jnp.asarray(time, dtype=dtype))
    return omega * time


def _accumulate_dft(
    mon,
    carry,
    field_arrays,
    t_phys,
    dt_scalar,
    *,
    sampled_vectors=None,
    local_offsets=False,
):
    """Gather and accumulate one monitor's compiled field-vector DFT."""
    d_re, d_im, d_w = carry
    dtype = d_re.dtype
    theta = _dft_phase_angle(mon.freq_hz, t_phys, dtype)
    phase_re, phase_im = jnp.cos(theta), jnp.sin(theta)
    window = jnp.asarray(
        monitor_dft_window_weight(
            t_phys,
            mon.dft_t_start,
            mon.dft_t_end,
            mon.dft_window_code == 1,
        ),
        dtype=jnp.float32,
    )
    scale = jnp.asarray(
        monitor_dft_sample_scale(
            window,
            normalization_code=mon.dft_normalization_code,
            base_dt=dt_scalar,
            record_interval=mon.dft_record_interval,
            length_unit=mon.dft_length_unit,
        ),
        dtype=dtype,
    )

    # Compilation has already encoded dimensional colocation in these weighted gathers.
    vectors = sampled_vectors
    if vectors is None:
        vectors = mon.dft_sampling.sample(
            field_arrays,
            active_mask=np.asarray(mon.dft_component_mask),
        )
    component_mask = mon.dft_component_mask.astype(dtype)[:, None, None]
    vector_re = jnp.real(vectors).astype(dtype)
    vector_im = jnp.imag(vectors).astype(dtype)
    delta_re = (
        scale
        * component_mask
        * (
            jnp.einsum("f,cp->cfp", phase_re, vector_re)
            - jnp.einsum("f,cp->cfp", phase_im, vector_im)
        )
    )
    delta_im = (
        scale
        * component_mask
        * (
            jnp.einsum("f,cp->cfp", phase_im, vector_re)
            + jnp.einsum("f,cp->cfp", phase_re, vector_im)
        )
    )
    nf, npnt = mon.freq_count, mon.dft_point_count
    values, weights = mon.dft_slices(local=local_offsets)
    d_re = d_re.at[values].add(delta_re[:, :nf, :npnt].reshape(-1))
    d_im = d_im.at[values].add(delta_im[:, :nf, :npnt].reshape(-1))
    d_w = d_w.at[weights].add(jnp.full((nf,), window, dtype=dtype))
    return d_re, d_im, d_w


def _record_fields(program, state, abs_step, t_phys, fields):
    """Update every compiled FieldRecorder and return replacement state members."""
    recorded_fields = list(state.recorded_fields)
    recorded_steps = list(state.recorded_steps)
    recorded_times = list(state.recorded_times)
    recorded_counts = list(state.recorded_counts)
    for spec in program.monitors:
        if spec.recorder_index < 0:
            continue
        recorder_index = spec.recorder_index
        count = recorded_counts[recorder_index]
        capacity = recorded_steps[recorder_index].shape[0]
        should_record = monitor_records_on_step(abs_step, spec.record_interval) & (
            count < capacity
        )
        slot = jnp.minimum(count, capacity - 1)
        for component, sign, buffer_index, shape, flat_idx, weights in zip(
            spec.canonical_components,
            spec.component_signs,
            spec.field_buffer_indices,
            spec.field_shapes,
            spec.field_interp_flat_idx,
            spec.field_interp_weights,
            strict=True,
        ):
            value = (
                SamplingPlan((flat_idx,), (weights,))
                .sample((fields[component],))[0]
                .reshape(shape)
                if flat_idx.size
                else fields[component][tuple(slice(0, size) for size in shape)]
            )
            recorded_fields[buffer_index] = jax.lax.cond(
                should_record,
                lambda buffer, value=value * sign, slot=slot: buffer.at[slot].set(
                    value
                ),
                lambda buffer: buffer,
                recorded_fields[buffer_index],
            )
        recorded_steps[recorder_index] = jax.lax.cond(
            should_record,
            lambda buffer, slot=slot: buffer.at[slot].set(abs_step + 1),
            lambda buffer: buffer,
            recorded_steps[recorder_index],
        )
        recorded_times[recorder_index] = jax.lax.cond(
            should_record,
            lambda buffer, slot=slot: buffer.at[slot].set(t_phys),
            lambda buffer: buffer,
            recorded_times[recorder_index],
        )
        recorded_counts[recorder_index] = count + should_record.astype(jnp.int32)
    return {
        "recorded_fields": tuple(recorded_fields),
        "recorded_steps": tuple(recorded_steps),
        "recorded_times": tuple(recorded_times),
        "recorded_counts": tuple(recorded_counts),
    }


def update_monitors(
    program,
    state: SimulationState,
    abs_step: jnp.ndarray,
    t_phys: jnp.ndarray,
    dt_scalar: jnp.ndarray,
) -> SimulationState:
    # 1. Return immediately when no monitor plan exists; touching empty buffers would add
    # needless operations to every compiled timestep.
    if not program.monitors:
        return state

    dft_vec_re = state.dft_vec_re
    dft_vec_im = state.dft_vec_im
    dft_weight_sum = state.dft_weight_sum
    # Differentiable scans keep monitors as independent pytree leaves, allowing
    # AD to discard the transpose of a monitor unused by the objective. Ordinary
    # execution and the public state retain the existing packed-array layout.
    segmented_dft = isinstance(dft_vec_re, tuple)
    if segmented_dft:
        dft_vec_re, dft_vec_im, dft_weight_sum = (
            list(dft_vec_re),
            list(dft_vec_im),
            list(dft_weight_sum),
        )
    components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    field_arrays = tuple(getattr(state, name.lower()) for name in components)

    # 3. Every scalar and spectral monitor follows the same sampling and accumulator path;
    # geometry and dimensional choices are already encoded in its weighted gathers.
    for monitor_position, mon in enumerate(program.monitors):
        if mon.recorder_index >= 0:
            continue
        if mon.dft_enabled and mon.freq_count > 0 and mon.dft_point_count > 0:
            do_dft = monitor_dft_should_accumulate(
                mon.dft_enabled and mon.freq_count > 0 and mon.dft_point_count > 0,
                abs_step,
                t_phys,
                mon.dft_t_start,
                mon.dft_t_end,
                mon.dft_record_interval,
            )

            carry = (
                (
                    dft_vec_re[monitor_position],
                    dft_vec_im[monitor_position],
                    dft_weight_sum[monitor_position],
                )
                if segmented_dft
                else (dft_vec_re, dft_vec_im, dft_weight_sum)
            )

            def accumulate(carry, mon=mon):
                if carry[0].ndim == 2:
                    from beamz.simulation.execute.distributed import accumulate_dft

                    return accumulate_dft(
                        program, mon, carry, field_arrays, t_phys, dt_scalar
                    )
                return _accumulate_dft(
                    mon,
                    carry,
                    field_arrays,
                    t_phys,
                    dt_scalar,
                    local_offsets=segmented_dft,
                )

            if (
                mon.dft_record_interval == 1
                and mon.dft_t_start == 0.0
                and np.isinf(mon.dft_t_end)
            ):
                # Continuous monitors do not benefit from a CPU-controlled GPU
                # conditional each timestep. Predicate the accumulator instead;
                # this also preserves the gate for a negative initial clock.
                updated_dft = tuple(
                    jnp.where(do_dft, cast(jax.Array, new), cast(jax.Array, old))
                    for new, old in zip(accumulate(carry), carry, strict=True)
                )
            else:
                updated_dft = jax.lax.cond(
                    do_dft, accumulate, lambda carry: carry, carry
                )
            if segmented_dft:
                (
                    dft_vec_re[monitor_position],
                    dft_vec_im[monitor_position],
                    dft_weight_sum[monitor_position],
                ) = updated_dft
            else:
                dft_vec_re, dft_vec_im, dft_weight_sum = updated_dft

    recorder_updates = _record_fields(
        program,
        state,
        abs_step,
        t_phys,
        dict(zip(components, field_arrays, strict=True)),
    )
    return state._replace(
        dft_vec_re=tuple(dft_vec_re) if segmented_dft else dft_vec_re,
        dft_vec_im=tuple(dft_vec_im) if segmented_dft else dft_vec_im,
        dft_weight_sum=tuple(dft_weight_sum) if segmented_dft else dft_weight_sum,
        **recorder_updates,
    )
