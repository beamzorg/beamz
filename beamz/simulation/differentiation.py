"""Pure, checkpointed FDTD execution with dynamic inverse-design parameters.

Build the solver outside ``jit``/``grad``. Compilation, rasterization, mode
solving, and analysis-result decoding remain setup operations; fields and raw
monitor accumulators stay in JAX throughout the differentiable execution.
"""

from __future__ import annotations

import math
import operator
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0
from beamz.lattice import (
    _material_conductivities,
    sample_voxel_grid_at_component_2d,
    sample_voxel_grid_at_component_3d,
)
from beamz.simulation import distributed_monitors, jax_sharding, kernels, sharding
from beamz.simulation._cuda_abi import (
    CUDA_BF16_PSI,
    CUDA_CPML_PAIR,
    CUDA_FIELD_PAD32,
    CUDA_FIELD_PAD64,
    CUDA_FIELD_PAD_Y8,
    CUDA_TEMPORAL_PAIR,
)
from beamz.simulation.execute import (
    build_step_context,
    compiled_source_batches,
    forward_step,
    initial_program_state,
)
from beamz.simulation.model import CompiledProgram, SimulationState, UpdateCoefficients


def _checkpoint_interval(value, num_steps):
    if value is None:
        return max(1, math.isqrt(num_steps))
    if isinstance(value, bool):
        raise ValueError("checkpoint_interval must be an integer >= 0 or None")
    try:
        interval = operator.index(value)
    except TypeError as exc:
        raise ValueError("checkpoint_interval must be an integer >= 0 or None") from exc
    if interval < 0:
        raise ValueError("checkpoint_interval must be an integer >= 0 or None")
    return min(interval, num_steps)


def build_differentiable_scan(program: CompiledProgram, *, checkpoint_interval=None):
    """Return ``run(state, coefficients, source_amplitudes=None)`` for ``jax.grad``.

    ``None`` selects approximately sqrt(T) steps per checkpoint. Zero disables
    rematerialization for small reference problems. Every chunk follows the same
    discrete transition as ordinary execution, including sources and monitors.
    CUDA uses float32 native timestep graphs where source/monitor plans permit,
    with out-of-place phase calls as the fallback and native first-order VJPs.
    Material-codebook compression is bypassed for trainable coefficients.
    """
    cfg = program.config
    if cfg.backend != "jax" and program.coefficients.e_inverse_offdiagonal.size:
        raise ValueError("Full-tensor CUDA autodiff requires backend='jax'.")
    distributed = program.sharding.layout.enabled
    if distributed:
        from beamz.simulation.distributed_differentiation import validate

        validate(program)
        if checkpoint_interval == 0:
            raise ValueError("Sharded autodiff requires checkpointing")
    if cfg.backend != "jax" and cfg.cuda_flags & CUDA_BF16_PSI:
        raise ValueError("CUDA autodiff requires BEAMZ_CUDA_CPML_PSI_PRECISION=fp32")
    if cfg.num_steps < 1:
        raise ValueError("Differentiable execution requires at least one timestep")
    interval = _checkpoint_interval(checkpoint_interval, cfg.num_steps)
    if program.dispersion is not None and (
        program.dispersion.regions or program.dispersion.interfaces
    ):
        raise ValueError(
            "Differentiable material binding requires nondispersive materials"
        )
    if cfg.backend != "jax" and cfg.cuda_storage_axes != (0, 1, 2) and not distributed:
        raise ValueError("CUDA autodiff requires BEAMZ_CUDA_STORAGE_AXES=012")
    # Differentiable fields use contiguous logical Yee supports. Forward-only
    # padding and temporal schedules do not define native adjoint layouts.
    forward_only_flags = (
        CUDA_FIELD_PAD32
        | CUDA_FIELD_PAD64
        | CUDA_FIELD_PAD_Y8
        | CUDA_TEMPORAL_PAIR
        | CUDA_CPML_PAIR
    )
    config = (
        replace(cfg, cuda_flags=cfg.cuda_flags & ~forward_only_flags)
        if cfg.backend != "jax"
        else cfg
    )
    context = replace(
        build_step_context(program),
        config=config,
        source_batches=None,
        differentiable=True,
    )
    update_kernel = (
        jax_sharding.select_kernel()
        if jax_sharding.supported(program, differentiable=True)
        else kernels.select_update_kernel(context)
    )
    source_count = max((spec.source_index for spec in program.sources), default=-1) + 1

    def split_dft(state):
        if distributed:
            return state
        if not any(mon.dft_enabled for mon in program.monitors):
            return state
        values, weights = [], []
        for mon in program.monitors:
            count = 6 * mon.freq_count * mon.dft_point_count if mon.dft_enabled else 0
            values.append(slice(mon.dft_value_offset, mon.dft_value_offset + count))
            weight_count = mon.freq_count if mon.dft_enabled else 0
            weights.append(
                slice(mon.dft_weight_offset, mon.dft_weight_offset + weight_count)
            )
        return state._replace(
            dft_vec_re=tuple(state.dft_vec_re[key] for key in values),
            dft_vec_im=tuple(state.dft_vec_im[key] for key in values),
            dft_weight_sum=tuple(state.dft_weight_sum[key] for key in weights),
        )

    def pack_dft(state):
        if not isinstance(state.dft_vec_re, tuple):
            return state
        return state._replace(
            dft_vec_re=jnp.concatenate(state.dft_vec_re),
            dft_vec_im=jnp.concatenate(state.dft_vec_im),
            dft_weight_sum=jnp.concatenate(state.dft_weight_sum),
        )

    native_step = None
    if cfg.backend != "jax" and not distributed:
        from beamz.simulation.cuda.differentiation import build_native_step

        native_step = build_native_step(
            program, context, update_kernel, split_dft, pack_dft
        )

    def run(state, coefficients, source_amplitudes=None):
        if cfg.backend != "jax" and not distributed:
            from beamz.simulation.cuda.material_blocks import (
                differentiable_coefficients,
            )

            coefficients = differentiable_coefficients(coefficients, state)
        if distributed:
            from beamz.simulation.distributed import scan_local_cpml

            coefficients = sharding.place_tree(
                program,
                sharding._lower_coefficients(coefficients, program.sharding.layout),
            )
            state = scan_local_cpml(state, program)
            if program.monitors:
                state = distributed_monitors.scan_local_dft(state, program)
        state = split_dft(state)
        if source_amplitudes is None:
            sources = program.sources
        else:
            amplitudes = jnp.asarray(source_amplitudes)
            if amplitudes.shape != (source_count,):
                raise ValueError(f"source_amplitudes must have shape {(source_count,)}")
            sources = tuple(
                replace(spec, coeff=spec.coeff * amplitudes[spec.source_index])
                for spec in program.sources
            )
        ctx = replace(context, source_batches=compiled_source_batches(sources))
        observation_origin, initial_step = state.t, state.current_step

        def step(carry, _unused):
            if native_step is not None:
                return native_step(
                    carry,
                    coefficients,
                    ctx,
                    observation_origin,
                    carry.current_step - initial_step,
                ), None
            return forward_step(
                carry,
                ctx=ctx,
                coeffs=coefficients,
                program=program,
                update_kernel=update_kernel,
                observation_time=observation_origin
                + (carry.current_step - initial_step + 1).astype(jnp.float32)
                * ctx.dt_scalar,
            ), None

        if interval == 0:
            return pack_dft(jax.lax.scan(step, state, xs=None, length=cfg.num_steps)[0])

        if native_step is not None:
            # One native forward graph per checkpoint. Its custom VJP retains
            # the boundary state, replays the local trajectory, and applies the
            # same complete discrete timestep transposes as the phase path.
            native_chunk = native_step.for_steps(interval)

            def chunk(carry, _unused):
                return native_chunk(
                    carry,
                    coefficients,
                    ctx,
                    observation_origin,
                    carry.current_step - initial_step,
                ), None

            chunks, tail = divmod(cfg.num_steps, interval)
            out = jax.lax.scan(chunk, state, xs=None, length=chunks)[0]
            if tail:
                out = native_step.for_steps(tail)(
                    out,
                    coefficients,
                    ctx,
                    observation_origin,
                    out.current_step - initial_step,
                )
            return pack_dft(out)

        def chunk(carry, _unused):
            return jax.lax.scan(step, carry, xs=None, length=interval)[0], None

        # Only chunk boundary states survive the forward pass. The backward pass
        # replays a chunk and retains its local trajectory, then releases it.
        chunks, tail = divmod(cfg.num_steps, interval)
        out = jax.lax.scan(
            jax.checkpoint(chunk, prevent_cse=False), state, xs=None, length=chunks
        )[0]
        if tail:
            out = jax.checkpoint(
                lambda carry: jax.lax.scan(step, carry, xs=None, length=tail)[0],
                prevent_cse=False,
            )(out)
        if distributed:
            out = scan_local_cpml(out, program, assemble=True)
            if program.monitors:
                out = distributed_monitors.scan_local_dft(out, program, assemble=True)
        return pack_dft(out)

    return jax.jit(run)


class DifferentiableSimulation:
    """Reusable FDTD function with dynamic materials and JAX-native outputs.

    Construct through ``simulation.differentiable(...)`` outside transformations.
    Call with a cell-centered relative-permittivity array or a tuple of three
    arrays on Ex/Ey/Ez supports. ``None`` preserves the compiled materials.
    Conductivity accepts the same layouts, in S/m. Geometry, boundaries, time
    grid, source profiles, and monitor definitions remain fixed.
    """

    def __init__(self, program: CompiledProgram, *, t0=0.0, checkpoint_interval=None):
        self.program = program
        self._fixed_lossless_e = tuple(
            bool(np.all(np.asarray(getattr(program.grid, f"sig_{axis}")) == 0))
            for axis in "xyz"
        )
        self.run_coefficients = build_differentiable_scan(
            program, checkpoint_interval=checkpoint_interval
        )
        self.initial_state = initial_program_state(program, t=t0, current_step=0)
        if program.sharding.layout.enabled:
            from beamz.simulation.observe import MONITOR_FIELDS

            self.initial_state = sharding.prepare_state(
                program, self.initial_state, replicated_fields=MONITOR_FIELDS
            )
        self._run = jax.jit(self._advance)

    def prepare_permittivity(self, values):
        """Place scalar or host cell parameters before applying ``jit``/``grad``.

        Distributed arrays include storage padding; returned material gradients
        use the same placement and have zero sensitivity in that padding.
        """
        if isinstance(values, tuple):
            if len(values) != 3:
                raise ValueError("Yee materials require three component arrays")
            return tuple(
                sharding.place_tree(
                    self.program,
                    sharding._prepare_component(self.program, f"E{a}", value),
                )
                if self.program.sharding.layout.enabled
                else jnp.asarray(value, dtype=jnp.float32)
                for a, value in zip("xyz", values, strict=True)
            )
        if self.program.sharding.layout.enabled:
            from beamz.simulation.distributed_differentiation import place_parameter

            if not isinstance(values, jax.Array):
                values = np.asarray(values, dtype=np.float32)
            return place_parameter(self.program, values)
        return jnp.asarray(values, dtype=jnp.float32)

    def _sample(self, values, components):
        grid = self.program.grid
        if self.program.sharding.layout.enabled:
            from beamz.simulation.distributed_differentiation import sample_material

            return tuple(sample_material(self.program, values, c) for c in components)
        if self.program.config.is_3d:
            return tuple(
                sample_voxel_grid_at_component_3d(values, component)
                for component in components
            )
        return tuple(
            sample_voxel_grid_at_component_2d(
                values, component, "xy", grid.polarization_2d
            )
            for component in components
        )

    def _electric_material(self, value, prefix):
        grid = self.program.grid
        if value is None:
            if self.program.sharding.layout.enabled:
                return sharding.place_tree(
                    self.program,
                    tuple(
                        sharding._pad_high_to_shape(
                            getattr(grid, f"{prefix}_{a}"),
                            self.program.sharding.layout.padded_shapes[f"E{a}"],
                            pad_value=1 if prefix == "eps" else 0,
                        )
                        for a in "xyz"
                    ),
                )
            return tuple(jnp.asarray(getattr(grid, f"{prefix}_{a}")) for a in "xyz")
        if isinstance(value, tuple):
            if len(value) != 3:
                raise ValueError(
                    "Yee materials require a tuple of three component arrays"
                )
            values = tuple(
                jnp.asarray(component, dtype=jnp.float32) for component in value
            )
            expected = tuple(
                self.program.sharding.layout.padded_shapes[f"E{a}"] for a in "xyz"
            )
            if tuple(component.shape for component in values) != expected:
                raise ValueError(f"Yee material component shapes must be {expected}")
            return values
        if prefix == "eps" and grid.material_grid.uses_direct_yee_materials:
            raise ValueError(
                "This grid has directly rasterized Yee materials; supply an Ex/Ey/Ez "
                "tuple to preserve its material sampling"
            )
        values = jnp.asarray(value, dtype=jnp.float32)
        if self.program.sharding.layout.enabled:
            return self._sample(values, ("Ex", "Ey", "Ez"))
        if values.ndim == 0:
            values = jnp.broadcast_to(values, grid.permittivity.shape)
        if values.shape != grid.permittivity.shape:
            raise ValueError(f"Cell material shape must be {grid.permittivity.shape}")
        return self._sample(values, ("Ex", "Ey", "Ez"))

    def coefficients(
        self, permittivity=None, *, conductivity=None
    ) -> UpdateCoefficients:
        """Form dynamic Yee coefficients without recompiling or host conversion.

        Scalar/cell conductivity is physical electric loss, combined with the
        compiled absorber using the ordinary material rule. Only the sponge
        contribution has matched magnetic loss. A Yee tuple directly replaces
        total E losses and preserves compiled H losses; use ``run_coefficients``
        for independent H parameters.
        """
        program, grid = self.program, self.program.grid
        base, dt = program.coefficients, program.config.dt
        if program.sharding.layout.enabled and conductivity is not None:
            raise ValueError("Sharded autodiff supports fixed zero conductivity only")
        if (
            permittivity is not None or conductivity is not None
        ) and base.e_inverse_offdiagonal.size:
            raise ValueError(
                "Full-tensor material optimization requires dynamic inverse tensor "
                "coefficients through run_coefficients"
            )
        eps = self._electric_material(permittivity, "eps")
        if program.sharding.layout.enabled:
            updates = {}
            for axis, epsilon in zip("xyz", eps, strict=True):
                valid = jnp.asarray(True)
                for d, size in enumerate(
                    program.sharding.layout.logical_shapes[f"E{axis}"]
                ):
                    shape = [1, 1, 1]
                    shape[d] = epsilon.shape[d]
                    valid = valid & (jnp.arange(epsilon.shape[d]).reshape(shape) < size)
                epsilon = jnp.where(valid, epsilon, 1)
                updates.update(
                    {
                        f"e_decay_{axis}": jnp.float32(1),
                        f"e_source_{axis}": dt / (EPS_0 * epsilon),
                        f"e_permittivity_{axis}": epsilon,
                        f"h_decay_{axis}": jnp.float32(1),
                        f"h_source_{axis}": jnp.float32(dt / MU_0),
                    }
                )
            return base._replace(**updates)
        sig = self._electric_material(conductivity, "sig")
        h_sigma = tuple(getattr(grid, f"sigma_m_h{axis}") for axis in "xyz")
        if conductivity is not None and not isinstance(conductivity, tuple):
            cell_sigma = jnp.broadcast_to(
                jnp.asarray(conductivity, dtype=jnp.float32), grid.permittivity.shape
            )
            total_sigma, sponge_sigma = _material_conductivities(
                replace(grid, conductivity=cell_sigma)
            )
            sig = self._sample(total_sigma, ("Ex", "Ey", "Ez"))
            if grid.has_pml and not grid.has_cpml:
                sigma_m = sponge_sigma * grid.permeability * (MU_0 / EPS_0)
                h_sigma = self._sample(sigma_m, ("Hx", "Hy", "Hz"))
        updates = {}
        for axis, epsilon, sigma, magnetic_sigma, fixed_lossless in zip(
            "xyz", eps, sig, h_sigma, self._fixed_lossless_e, strict=True
        ):
            e_decay, e_source = kernels.precompute_e_update_coefficients(
                shape=getattr(grid, f"E{axis}").shape,
                conductivity=sigma,
                permittivity=epsilon,
                dt=dt,
                region=getattr(grid, f"region_{axis}"),
            )
            if conductivity is None and fixed_lossless:
                # With fixed zero conductivity the decay is identically one.
                # Express that independence explicitly so AD does not retain
                # old E fields to form a coefficient VJP that is later zeroed.
                e_decay = jnp.float32(1.0)
            h_decay, h_source = kernels.precompute_h_update_coefficients(
                magnetic_sigma, dt
            )
            updates.update(
                {
                    f"e_decay_{axis}": e_decay,
                    f"e_source_{axis}": e_source,
                    f"e_conductivity_{axis}": sigma,
                    f"e_permittivity_{axis}": epsilon,
                    f"h_decay_{axis}": h_decay,
                    f"h_source_{axis}": h_source,
                    f"h_sigma_m_{axis}": magnetic_sigma,
                }
            )
        return base._replace(**updates)

    def _advance(self, permittivity, conductivity, state, source_amplitudes):
        coefficients = self.coefficients(permittivity, conductivity=conductivity)
        return self.run_coefficients(state, coefficients, source_amplitudes)

    def _monitor(self, monitor):
        for spec in self.program.monitors:
            if (isinstance(monitor, str) and spec.name == monitor) or (
                not isinstance(monitor, str) and spec.monitor_index == monitor
            ):
                if not spec.dft_enabled or spec.freq_count <= 0:
                    raise ValueError(
                        "The selected monitor has no frequency-domain fields"
                    )
                return spec
        raise ValueError(f"Unknown monitor {monitor!r}")

    def monitor_fields(self, state, monitor=0):
        """Return normalized complex DFT fields with shape (6, frequencies, points).

        Component order is Ex, Ey, Ez, Hx, Hy, Hz on the canonical solver axes.
        H phasors are aligned with the electric leapfrog time. The monitor's
        configured native/physical units are preserved; source normalization is
        an explicit part of the caller's objective.
        """
        spec = self._monitor(monitor)
        count = 6 * spec.freq_count * spec.dft_point_count
        start = spec.dft_value_offset
        values = (
            state.dft_vec_re[start : start + count]
            + 1j * state.dft_vec_im[start : start + count]
        ).reshape((6, spec.freq_count, spec.dft_point_count))
        start = spec.dft_weight_offset
        weights = state.dft_weight_sum[start : start + spec.freq_count]
        values = values * (2.0 / jnp.maximum(weights, 1e-18))[None, :, None]
        h_phase = jnp.exp(-1j * jnp.pi * spec.freq_hz * self.program.config.dt)
        return values.at[3:].multiply(h_phase[None, :, None])

    def monitor_flux(self, state, monitor=0):
        """Return differentiable signed power per frequency through a planar monitor."""
        spec = self._monitor(monitor)
        axis = spec.normal_axis
        if axis not in (0, 1, 2):
            raise ValueError(
                "Flux objectives require a planar monitor with a normal axis"
            )
        fields = self.monitor_fields(state, monitor)
        first, second = ((1, 2), (2, 0), (0, 1))[axis]
        poynting = fields[first] * jnp.conj(fields[3 + second]) - (
            fields[second] * jnp.conj(fields[3 + first])
        )
        measure = spec.integration_weights
        if measure.size != spec.dft_point_count:
            measure = spec.power_scale
        return 0.5 * spec.normal_sign * jnp.real(jnp.sum(poynting * measure, axis=-1))

    def __call__(
        self,
        permittivity=None,
        *,
        conductivity=None,
        state: SimulationState | None = None,
        source_amplitudes=None,
    ) -> SimulationState:
        """Advance the fixed time segment, returning differentiable raw state.

        ``state`` must match ``initial_state`` including CPML and monitor buffers.
        Its fields can themselves be differentiation inputs. ``source_amplitudes``
        scales each original source by one dynamic real amplitude (default one).
        """
        return self._run(
            permittivity,
            conductivity,
            self.initial_state if state is None else state,
            source_amplitudes,
        )
