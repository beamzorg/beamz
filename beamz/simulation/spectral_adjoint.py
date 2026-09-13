"""Spectral Maxwell adjoint for the lossless 2D/3D Yee/CPML topology problem.

The forward solve is ordinary pulsed FDTD. Only electric DFTs in the design
region are retained. A custom simulation VJP constructs pulsed adjoint sources
from monitor sensitivities, runs a transposed Maxwell/CPML FDTD update, and
overlaps electric DFTs. No AD traverses either time loop. JAX transposes one
linear Maxwell step and differentiates the objective and material map.

The spectral equations omit terminal-state terms: the pulse response must decay.
This is not the exact derivative of a truncated time-domain experiment. Sources
can be grouped by frequency or a numerically verified broadband spatial basis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0
from beamz.simulation import kernels
from beamz.simulation.differentiable import (
    _cell_centers_to_yee,
    topology_coefficients,
    topology_yee_permittivity,
)
from beamz.simulation.execute import build_step_context, forward_step
from beamz.simulation.observe import monitor_dft_sample_scale

_IMPEDANCE = np.sqrt(MU_0 / EPS_0)
_COMPONENTS = ("ex", "ey", "ez", "hx", "hy", "hz")


class SpectralAdjoint:
    """Internal backend bound to a validated fixed-grid TopologyProblem."""

    def __init__(self, problem, reduce_objective, *, decay_tolerance):
        self.problem = problem
        self.program = program = problem.program
        self.decay_tolerance = decay_tolerance
        self.last_diagnostics = None
        if program.config.is_3d:
            self.e_names, self.h_names = ("ex", "ey", "ez"), ("hx", "hy", "hz")
        else:
            self.e_names = (
                ("ez",) if program.config.polarization_2d == "tm" else ("ex", "ey")
            )
            self.h_names = ("hx", "hy") if self.e_names == ("ez",) else ("hz",)
        self.names = self.e_names + self.h_names
        self.shapes = {
            name: getattr(problem._initial, name).shape for name in self.names
        }
        self.slices = {}
        offset = 0
        for name, shape in self.shapes.items():
            count = int(np.prod(shape))
            self.slices[name] = slice(offset, offset + count)
            offset += count
        self.size = offset
        self._validate()
        # Only frequencies that influence the objective require stored fields or
        # adjoint solves. The ordinary monitors can still record other frequencies.
        frequencies = []
        for name, requested in problem.frequency_requests:
            mon = next(m for m in program.monitors if m.name == name)
            selected = (
                np.asarray(mon.freq_hz)
                if requested is None
                else np.asarray(requested, dtype=np.float32)
            )
            frequencies.extend(selected.tolist())
        self.frequencies = np.unique(frequencies)
        self.design_indices = tuple(
            np.flatnonzero(
                np.asarray(
                    _cell_centers_to_yee(
                        problem._mask.astype(jnp.float32),
                        name.capitalize(),
                        program.config.polarization_2d,
                    )
                )
            )
            for name in self.e_names
        )
        self._build_forward()

        def objective(re, im, weights):
            state = problem._initial._replace(
                dft_vec_re=re, dft_vec_im=im, dft_weight_sum=weights
            )
            return reduce_objective(state)

        def material_map(density, beta):
            eps = topology_yee_permittivity(
                program, problem._permittivity(density, beta), problem._mask
            )
            return tuple(
                eps["xyz".index(name[-1])].ravel()[idx]
                for name, idx in zip(self.e_names, self.design_indices, strict=True)
            )

        self._material_map = jax.jit(material_map)
        self._pullback = jax.jit(
            lambda density, beta, cotangent: jax.vjp(material_map, density, beta)[1](
                cotangent
            )
        )
        self._build_adjoint()

        def monitor_forward(density, beta):
            state, electric_dfts = self._checked_forward(density, beta)
            data = (state.dft_vec_re, state.dft_vec_im, state.dft_weight_sum)
            return data, (density, beta, electric_dfts)

        @jax.custom_vjp
        def monitor_run(density, beta):
            return monitor_forward(density, beta)[0]

        def monitor_backward(residual, cotangents):
            density, beta, electric_dfts = residual
            grad_re, grad_im, _weight_gradient = cotangents
            # Monitor normalization weights depend on the fixed sampling plan,
            # not material values; their material derivative is zero.
            return self._backward(density, beta, electric_dfts, grad_re, grad_im)

        monitor_run.defvjp(monitor_forward, monitor_backward)
        self._monitor_run = monitor_run
        # Host orchestration permits explicit decay failures and skips zero
        # sources. Each FDTD run is JIT compiled; the whole driver is not jittable.
        self.value_and_grad = jax.value_and_grad(
            lambda density, beta: objective(*monitor_run(density, beta))
        )

    def _validate(self):
        program = self.program
        for mon in program.monitors:
            if (
                mon.dft_window_code != 0
                or mon.dft_record_interval != 1
                or mon.dft_t_start > float(self.problem.simulation.time[0])
                or mon.dft_t_end < float(self.problem.simulation.time[-1])
            ):
                raise ValueError(
                    "adjoint requires full-run rectangular DFT monitors sampled every timestep."
                )

    def _build_forward(self):
        problem, program = self.problem, self.program
        ctx = build_step_context(program)
        update_kernel = kernels.select_update_kernel(ctx)
        frequencies = jnp.asarray(self.frequencies)
        indices = tuple(jnp.asarray(idx) for idx in self.design_indices)

        def run(density, beta):
            eps = problem._permittivity(density, beta)
            coefficients = topology_coefficients(program, eps, problem._mask)
            dfts = tuple(
                jnp.zeros((frequencies.size, idx.size), dtype=jnp.complex64)
                for idx in indices
            )

            def step(carry, _):
                state, dfts, peak = carry
                state = forward_step(
                    state,
                    ctx=ctx,
                    coeffs=coefficients,
                    program=program,
                    update_kernel=update_kernel,
                    time_origin=problem._initial.t,
                )
                phase = jnp.exp(2j * jnp.pi * frequencies * state.t)
                dfts = tuple(
                    acc + phase[:, None] * getattr(state, name).ravel()[idx][None, :]
                    for acc, name, idx in zip(dfts, self.e_names, indices, strict=True)
                )
                energy = sum(
                    jnp.sum(getattr(state, name) ** 2)
                    * (1 if name in self.e_names else _IMPEDANCE**2)
                    for name in self.names
                )
                return (state, dfts, jnp.maximum(peak, energy)), None

            (state, dfts, peak), _ = jax.lax.scan(
                step,
                (problem._initial, dfts, jnp.asarray(0.0)),
                None,
                length=program.config.num_steps,
            )
            terminal = sum(
                jnp.sum(getattr(state, name) ** 2)
                * (1 if name in self.e_names else _IMPEDANCE**2)
                for name in self.names
            )
            return state, dfts, jnp.sqrt(terminal / jnp.maximum(peak, 1e-30))

        self._forward = jax.jit(run)

    def _build_adjoint(self):
        problem, program = self.problem, self.program
        ctx = build_step_context(program)
        kernel = kernels.select_update_kernel(ctx)
        initial = problem._initial
        dx = program.config.resolution
        # Balance the state units before transposition: E, Z0 H, dx psi_H,
        # Z0 dx psi_E. Adjoint arrays are dual variables in this scaled basis.
        zero = tuple(
            jnp.zeros_like(getattr(initial, name), dtype=jnp.complex64)
            for name in self.names
        )
        zero += tuple(
            jnp.zeros_like(v, dtype=jnp.complex64) for v in initial.cpml_psi_h_terms
        )
        zero += tuple(
            jnp.zeros_like(v, dtype=jnp.complex64) for v in initial.cpml_psi_e_terms
        )
        nfields = len(self.names)
        nhpsi = len(initial.cpml_psi_h_terms)
        scales = tuple(
            1.0 if name in self.e_names else _IMPEDANCE for name in self.names
        )
        scales += (dx,) * nhpsi + (_IMPEDANCE * dx,) * len(initial.cpml_psi_e_terms)
        self._zero_physics = zero
        indices = tuple(jnp.asarray(idx) for idx in self.design_indices)

        def transition(physics, coefficients):
            values = tuple(v / scale for v, scale in zip(physics, scales, strict=True))
            state = initial._replace(
                **dict(zip(self.names, values[:nfields], strict=True)),
                cpml_psi_h_terms=values[nfields : nfields + nhpsi],
                cpml_psi_e_terms=values[nfields + nhpsi :],
            )
            state = kernel.update_h(state, ctx, coefficients)
            metallic = ctx.boundary.metallic
            hx, hy, hz = kernels.apply_post_source_boundaries(
                (state.hx, state.hy, state.hz),
                (metallic.hx_mask, metallic.hy_mask, metallic.hz_mask),
            )
            state = state._replace(hx=hx, hy=hy, hz=hz)
            state = kernel.update_e(state, ctx, coefficients)
            ex, ey, ez = kernels.apply_post_source_boundaries(
                (state.ex, state.ey, state.ez),
                (metallic.ex_mask, metallic.ey_mask, metallic.ez_mask),
            )
            state = state._replace(ex=ex, ey=ey, ez=ez)
            fields = tuple(getattr(state, name) for name in self.names)
            values = fields + state.cpml_psi_h_terms + state.cpml_psi_e_terms
            return tuple(v * scale for v, scale in zip(values, scales, strict=True))

        def run(coefficients, source, frequencies):
            frequency = jnp.mean(frequencies)
            # One complex simulation is two real quadrature simulations sharing
            # the same real Maxwell operator. Normalize the pulse's measured DFT.
            transpose_step = jax.linear_transpose(
                lambda x: transition(x, coefficients), zero
            )
            dfts = tuple(
                jnp.zeros((frequencies.size, idx.size), dtype=jnp.complex64)
                for idx in indices
            )
            # Several cycles and a negligible leading tail reduce off-band/DC
            # excitation of slowly decaying dual CPML states. The measured pulse
            # DFT below preserves the same frequency-domain source normalization.
            sigma = 3.0 / frequency
            center = 8.0 * sigma

            def carrier(t):
                return jnp.exp(-0.5 * ((t - center) / sigma) ** 2) * jnp.exp(
                    -2j * jnp.pi * frequency * t
                )

            # Stop after 512 quiet steps beyond the pulse, checking the complete
            # dual state. Long forward windows need not force long adjoint runs.
            stopping_ratio = min(1e-7, self.decay_tolerance * 1e-3)

            def condition(carry):
                i, _, _, _, _, quiet = carry
                return (i < program.config.num_steps) & (quiet < 512)

            def advance(carry):
                i, physics, dfts, pulse_dft, peak, quiet = carry
                t = (i + 1) * ctx.dt_scalar
                phase = jnp.exp(2j * jnp.pi * frequencies * t)
                # A discrete temporal difference suppresses the zero-frequency
                # component that can excite stationary dual Maxwell/CPML states.
                # Its measured DFT still normalizes the desired frequency exactly.
                pulse = carrier(t) - jnp.where(i == 0, 0.0j, carrier(i * ctx.dt_scalar))
                physics = tuple(
                    v + pulse * src
                    for v, src in zip(transpose_step(physics)[0], source, strict=True)
                )
                dfts = tuple(
                    acc + phase[:, None] * field.ravel()[idx][None, :]
                    for acc, field, idx in zip(
                        dfts, physics[: len(self.e_names)], indices, strict=True
                    )
                )
                energy = sum(jnp.sum(jnp.abs(v) ** 2) for v in physics)
                peak = jnp.maximum(peak, energy)
                small = (energy < stopping_ratio**2 * peak) & (t > center + 8 * sigma)
                return (
                    i + 1,
                    physics,
                    dfts,
                    pulse_dft + pulse * phase,
                    peak,
                    jnp.where(small, quiet + 1, 0),
                )

            steps, physics, dfts, pulse_dft, peak, _ = jax.lax.while_loop(
                condition,
                advance,
                (
                    jnp.asarray(0),
                    zero,
                    dfts,
                    jnp.zeros(frequencies.size, dtype=jnp.complex64),
                    jnp.asarray(0.0),
                    jnp.asarray(0),
                ),
            )
            terminal = sum(jnp.sum(jnp.abs(v) ** 2) for v in physics)
            return (
                tuple(v / pulse_dft[:, None] for v in dfts),
                jnp.sqrt(terminal / jnp.maximum(peak, 1e-30)),
                steps,
            )

        self._adjoint_run = jax.jit(run)

    def _adjoint_sources(self, grad_re, grad_im):
        plans = []
        for mon in self.program.monitors:
            count = 6 * mon.freq_count * mon.dft_point_count
            start = mon.dft_value_offset
            gradients = (
                np.asarray(grad_re[start : start + count])
                - 1j * np.asarray(grad_im[start : start + count])
            ).reshape(6, mon.freq_count, mon.dft_point_count)
            scale = float(
                monitor_dft_sample_scale(
                    1.0,
                    normalization_code=mon.dft_normalization_code,
                    base_dt=self.program.config.dt,
                    record_interval=1,
                    length_unit=mon.dft_length_unit,
                )
            )
            plans.append((mon, gradients, scale))
        # Materialize only one frequency's source grid at a time.
        for frequency in self.frequencies:
            rhs = np.zeros(self.size, dtype=np.complex128)
            for mon, gradients, scale in plans:
                match = np.flatnonzero(np.asarray(mon.freq_hz) == frequency)
                if not match.size:
                    continue
                for name in self.names:
                    ci = _COMPONENTS.index(name)
                    indices = np.asarray(mon.dft_flat_idx[ci])
                    weights = np.asarray(mon.dft_weights[ci])
                    values = (
                        gradients[ci, match[0], :, None]
                        * weights
                        * scale
                        * float(mon.dft_component_mask[ci])
                    )
                    if name in self.h_names:
                        values /= _IMPEDANCE
                    np.add.at(rhs[self.slices[name]], indices.ravel(), values.ravel())
            yield rhs

    def _source_groups(self, sources):
        """Use fewer broadband basis solves only after checking every source row.

        A basis vector has one fixed spatial profile. Its complex weights at
        each frequency are applied after the solve. SVD keeps frequency-varying
        modal profiles rather than assuming they are identical across a band.
        """
        assert self.last_diagnostics is not None
        # The shared three-cycle pulse must have useful spectrum throughout
        # the group. Wide bands retain independent, individually centered pulses.
        center_frequency = np.mean(self.frequencies)
        narrow_band = (
            np.max(np.abs(self.frequencies - center_frequency))
            <= 0.1 * center_frequency
        )
        if (
            self.problem.adjoint_source_grouping == "auto"
            and len(self.frequencies) > 1
            and narrow_band
        ):
            support = np.unique(
                np.concatenate(
                    [
                        self.slices[name].start
                        + np.asarray(mon.dft_flat_idx[_COMPONENTS.index(name)]).ravel()
                        for mon in self.program.monitors
                        for name in self.names
                    ]
                )
            )
            compact = np.stack([source[support] for source in sources])
            norms = np.linalg.norm(compact, axis=1)
            active = np.flatnonzero(norms > 0)
            if active.size:
                normalized = compact[active] / norms[active, None]
                u, singular, vh = np.linalg.svd(normalized, full_matrices=False)
                for rank in range(1, active.size):
                    reconstructed = (u[:, :rank] * singular[:rank]) @ vh[:rank]
                    error = float(
                        np.max(np.linalg.norm(normalized - reconstructed, axis=1))
                    )
                    if error <= 1e-7:
                        self.last_diagnostics.update(
                            source_grouping="broadband_basis",
                            source_basis_rank=rank,
                            source_reconstruction_error=error,
                            active_frequencies=int(active.size),
                        )
                        for k in range(rank):
                            source = np.zeros(self.size, dtype=np.complex128)
                            source[support] = vh[k]
                            yield active, source, norms[active] * u[:, k] * singular[k]
                        return
            # No smaller accurate basis: retain independent frequency solves.
            self.last_diagnostics.update(
                source_grouping="frequency", active_frequencies=int(active.size)
            )
            for fi in active:
                source = np.zeros(self.size, dtype=np.complex128)
                source[support] = compact[fi] / norms[fi]
                yield np.array([fi]), source, np.array([norms[fi]])
            return
        self.last_diagnostics.update(source_grouping="frequency")
        for fi, source in enumerate(sources):
            norm = np.linalg.norm(source)
            if norm:
                yield np.array([fi]), source / norm, np.array([norm])

    def _checked_forward(self, density, beta):
        state, electric_dfts, tail = self._forward(density, beta)
        tail = float(tail)
        self.last_diagnostics = {
            "terminal_field_ratio": tail,
            "decay_tolerance": self.decay_tolerance,
            "adjoint_solves": 0,
        }
        if not np.isfinite(tail) or tail > self.decay_tolerance:
            raise ValueError(
                f"adjoint forward pulse has not decayed: terminal/peak field ratio {tail:.3g} exceeds {self.decay_tolerance:.3g}. Increase run_time or use gradient_backend='autodiff'."
            )
        return state, electric_dfts

    def _backward(self, density, beta, electric_dfts, grad_re, grad_im):
        """Pull monitor sensitivities back through independent adjoint FDTD runs.

        Let x_n = A(eps) x_(n-1) + s_n and q = exp(+i omega dt). Once the
        terminal state decays, (I - q A) X = S for the field DFT X. The dual
        solve is (I - q A.T) lambda = g, implemented with pulsed A.T stepping.
        In a source-free, lossless design cell, only the final electric update
        depends on eps, giving dJ/deps = Re[-lambda_E (1-q) E_hat / eps].
        The material-map VJP then transposes Yee sampling and density transforms.
        Complex cotangents use Re(g.T dX), with no conjugation in the overlap.
        """
        assert self.last_diagnostics is not None  # populated by the custom VJP forward
        sources = self._adjoint_sources(grad_re, grad_im)
        eps = self.problem._permittivity(density, beta)
        coefficients = topology_coefficients(self.program, eps, self.problem._mask)
        material = tuple(np.asarray(v) for v in self._material_map(density, beta))
        cotangent = tuple(np.zeros_like(v, dtype=np.float64) for v in material)
        adjoint_tails = []
        adjoint_steps = []
        for freq_indices, source, weights in self._source_groups(sources):
            fields = tuple(
                jnp.asarray(
                    source[self.slices[name]].reshape(self.shapes[name]),
                    dtype=jnp.complex64,
                )
                for name in self.names
            )
            adjoint_source = fields + self._zero_physics[len(fields) :]
            frequencies = self.frequencies[freq_indices]
            adjoint, adjoint_tail, steps = self._adjoint_run(
                coefficients,
                adjoint_source,
                jnp.asarray(frequencies, dtype=jnp.float32),
            )
            adjoint_tail = float(adjoint_tail)
            adjoint_tails.append(adjoint_tail)
            adjoint_steps.append(int(steps))
            self.last_diagnostics.update(
                adjoint_solves=len(adjoint_tails),
                adjoint_terminal_field_ratios=tuple(adjoint_tails),
                adjoint_timesteps=tuple(adjoint_steps),
            )
            if not np.isfinite(adjoint_tail) or adjoint_tail > self.decay_tolerance:
                raise ValueError(
                    f"adjoint pulse has not decayed near {np.mean(frequencies):g} Hz: terminal/peak ratio {adjoint_tail:.3g} exceeds {self.decay_tolerance:.3g}. Increase run_time or use gradient_backend='autodiff'."
                )
            q = np.exp(2j * np.pi * frequencies * self.program.config.dt)
            for adj, efield, eps_e, grad in zip(
                adjoint, electric_dfts, material, cotangent, strict=True
            ):
                grad += np.real(
                    np.sum(
                        -np.asarray(adj)
                        * (weights * (1 - q))[:, None]
                        * np.asarray(efield)[freq_indices]
                        / eps_e[None, :],
                        axis=0,
                    )
                )
        self.last_diagnostics.update(
            stored_design_dft_values=sum(v.size for v in electric_dfts)
        )
        return self._pullback(
            density, beta, tuple(jnp.asarray(v, dtype=density.dtype) for v in cotangent)
        )
