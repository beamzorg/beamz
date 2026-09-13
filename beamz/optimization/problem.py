"""A complete fixed-grid 2D topology optimization workflow."""

from __future__ import annotations

import hashlib
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

from beamz._cache_tokens import cache_token
from beamz.const import LIGHT_SPEED
from beamz.design.materials import Material
from beamz.devices.monitors import FieldMonitor, ModeMonitor
from beamz.devices.sources import ModeSource
from beamz.simulation.differentiable import (
    TOPOLOGY_NUMERICS_VERSION,
    topology_coefficients,
    topology_material_grid,
)
from beamz.simulation.execute import (
    _compiled_source_launch_powers,
    _decode_monitor_results,
    build_scan,
    initial_program_state,
)
from beamz.simulation.results import SimulationResults

from .autodiff import transform_density
from .objectives import ModePower, WeightedObjective
from .polygonize import density_to_polygons
from .result import OptimizationStep, TopologyResult, load_result
from .topology import TopologySpec


class TopologyProblem:
    """Optimize a topology region against a scalar modal objective.

    Both backends support uniform 2D xy grids, scalar lossless dielectrics,
    a fixed mode source, and spectral monitors. The default uses the exact JAX
    derivative with chunk checkpointing, including CPML and DFT recurrences. Geometry,
    port modes, boundaries, timestep, and run length stay fixed across iterations.

    Parameters
    ----------
    simulation : Simulation
        Base simulation containing the fixed geometry, source, and monitors.
    topology : TopologySpec
        Existing density transform, design mask, and optimizer configuration.
    objective : ModePower, SoftMinModePower, or WeightedObjective
        Modal power averaged across selected frequencies, or a weighted combination
        rewarding transmission and penalizing reflection/unwanted-port power.
        SoftMinModePower targets the weakest selected frequencies instead.
    checkpoint_interval : int, default=32
        Timestep chunk length for reverse-mode rematerialization. ``None`` retains
        ordinary reverse-mode history and is intended for small reference tests.
    gradient_backend : {"autodiff", "adjoint"}, default="autodiff"
        Exact checkpointed differentiation through FDTD, or a spectral Maxwell
        adjoint after a pulsed FDTD forward run. The latter runs adjoint FDTD
        simulations and requires decayed fields and full-run rectangular monitors.
    adjoint_decay_tolerance : float, default=1e-4
        Maximum terminal/peak field norm for the spectral backend. This check is
        necessary, but run-duration convergence must still be verified.
    """

    def __init__(
        self,
        simulation,
        topology: TopologySpec,
        objective: ModePower | WeightedObjective,
        *,
        checkpoint_interval: int | None = 32,
        gradient_backend="autodiff",
        adjoint_decay_tolerance=1e-4,
    ):
        if gradient_backend not in ("autodiff", "adjoint"):
            raise ValueError("gradient_backend must be 'autodiff' or 'adjoint'.")
        if (
            isinstance(adjoint_decay_tolerance, bool)
            or not isinstance(adjoint_decay_tolerance, (int, float))
            or not np.isfinite(adjoint_decay_tolerance)
            or not 0 < adjoint_decay_tolerance < 1
        ):
            raise ValueError(
                "adjoint_decay_tolerance must be finite and between zero and one."
            )
        self.gradient_backend = gradient_backend
        self.adjoint_decay_tolerance = float(adjoint_decay_tolerance)
        self.simulation = simulation
        self.topology = topology
        self.objective = objective
        self.checkpoint_interval = checkpoint_interval
        self.program = simulation.compile(backend="jax")
        self._validate()
        self._mask = jnp.asarray(topology.region_mask)
        self._base_eps = jnp.asarray(self.program.grid.material_grid.permittivity)
        self._initial = initial_program_state(
            self.program, t=float(simulation.time[0]), current_step=0
        )
        empty_results = SimulationResults.from_run(
            simulation,
            runtime_fields=self.program.grid,
            monitor_results=_decode_monitor_results(
                simulation, self.program, self._initial
            ),
            source_launch_powers=_compiled_source_launch_powers(
                self.program, len(simulation.sources)
            ),
        )
        cache = {}
        reduce_objective = objective.bind(empty_results, self.program, cache)
        spectra = [
            term.bind_spectrum(empty_results, self.program, cache)
            for term in objective.terms
        ]
        scan = build_scan(self.program, checkpoint_interval=checkpoint_interval)

        def simulate(density, beta):
            eps = self._permittivity(density, beta)
            coefficients = topology_coefficients(self.program, eps, self._mask)
            return scan(self._initial, coefficients)

        def evaluate(density, beta):
            return reduce_objective(simulate(density, beta))

        def evaluate_spectra(density, beta):
            state = simulate(density, beta)
            return tuple(spectrum(state) for spectrum in spectra)

        self._evaluate = jax.jit(evaluate)
        self._value_and_grad = jax.jit(jax.value_and_grad(evaluate))
        self._spectra = jax.jit(evaluate_spectra)
        self._simulate = jax.jit(simulate)
        self._modal_data_bindings = {}
        self._binding_results = empty_results
        self._spectral_adjoint = None
        if gradient_backend == "adjoint":
            from beamz.simulation.spectral_adjoint import SpectralAdjoint

            self._spectral_adjoint = SpectralAdjoint(
                self, reduce_objective, decay_tolerance=self.adjoint_decay_tolerance
            )
            self._value_and_grad = self._spectral_adjoint.value_and_grad
        # Normalize the legacy ModePower identity, then version the numerics so
        # histories cannot silently resume under a changed physical objective.
        if (
            type(objective) is ModePower
            and objective.frequencies is None
            and objective.reference_mode_index == 0
            and objective.reference_direction == "+"
        ):
            identity = (
                simulation,
                topology,
                objective.monitor,
                objective.mode_index,
                objective.direction,
                objective.reference_monitor,
            )
        else:
            identity = (simulation, topology, objective)
        if gradient_backend != "autodiff":
            identity = (identity, gradient_backend, self.adjoint_decay_tolerance)
        token = cache_token((TOPOLOGY_NUMERICS_VERSION, identity))
        self.fingerprint = hashlib.sha256(repr(token).encode()).hexdigest()

    def simulation_data(self, density, *, beta=1.0, monitor_names=None):
        """Run and return differentiable ``data[name].amps.sel(...).values``.

        Use inside a JAX scalar objective and call ``jax.value_and_grad``. Names,
        port modes, and coordinates are static; complex amplitude values retain
        the selected simulation derivative. This modal subset is not xarray and
        does not support arbitrary field-monitor or interpolation operations.
        """
        from .data import bind_modal_data

        names = (
            tuple(monitor_names)
            if monitor_names is not None
            else tuple(
                m.name for m in self.simulation.monitors if isinstance(m, ModeMonitor)
            )
        )
        if not names or len(set(names)) != len(names):
            raise ValueError("monitor_names must be nonempty and unique.")
        if names not in self._modal_data_bindings:
            self._modal_data_bindings[names] = bind_modal_data(
                self._binding_results, self.program, names
            )
        values = jnp.asarray(density, dtype=jnp.float32)
        if values.shape != self.topology.region_mask.shape:
            raise ValueError("Density must match the topology mask shape.")
        if not isinstance(values, Tracer):
            values = self._density(values)
        if self._spectral_adjoint is None:
            state = self._simulate(values, jnp.asarray(beta, dtype=jnp.float32))
        else:
            # The low-level problem plans frequencies from its objective. A
            # data-returning caller must not silently access unplanned outputs.
            available = self._spectral_adjoint.frequencies
            for name in names:
                mon = next(m for m in self.program.monitors if m.name == name)
                if not np.isin(np.asarray(mon.freq_hz), available).all():
                    raise ValueError(
                        "Requested monitor frequencies are not included in this adjoint problem's objective."
                    )
            re, im, weights = self._spectral_adjoint._monitor_run(
                values, jnp.asarray(beta, dtype=jnp.float32)
            )
            state = self._initial._replace(
                dft_vec_re=re, dft_vec_im=im, dft_weight_sum=weights
            )
        return self._modal_data_bindings[names](state)

    def _validate(self):
        sim, spec, program = self.simulation, self.topology, self.program
        grid = program.grid
        if program.config.is_3d or sim.plane_2d != "xy":
            raise ValueError("TopologyProblem currently supports 2D xy simulations.")
        if (
            grid.geometry.metric_kind_for(("x", "y")) != "isotropic_uniform"
            or program.sharding.layout.enabled
        ):
            raise ValueError("TopologyProblem requires an unsharded uniform grid.")
        if not isinstance(self.objective, (ModePower, WeightedObjective)):
            raise TypeError(
                "TopologyProblem supports ModePower and weighted modal objectives."
            )
        if sim.design != spec.design:
            raise ValueError(
                "TopologySpec.design must match the base simulation design."
            )
        if sim.material_grid is not None:
            raise ValueError(
                "TopologyProblem currently requires design-rasterized materials."
            )
        materials = [
            sim.design.background,
            *(s.material for s in sim.design.structures),
        ]
        if any(
            not isinstance(m, Material) or np.ndim(m.permittivity) != 0
            for m in materials
        ):
            raise ValueError("TopologyProblem requires scalar Material permittivity.")
        if spec.region_mask.shape != grid.material_grid.shape or not np.any(
            spec.region_mask
        ):
            raise ValueError(
                "Topology mask must be nonempty and match the simulation material grid."
            )
        if spec.resolution is None or not np.isclose(
            spec.resolution, sim.resolution, rtol=1e-10, atol=0
        ):
            raise ValueError("Topology and simulation resolution must agree.")
        if (
            not np.isfinite([spec.eps_min, spec.eps_max]).all()
            or not 0 < spec.eps_min < spec.eps_max
        ):
            raise ValueError("Require finite 0 < eps_min < eps_max.")
        if not np.isfinite(spec.beta_schedule).all() or min(spec.beta_schedule) <= 0:
            raise ValueError(
                "The optimization beta schedule must be finite and positive."
            )
        if not np.isfinite(spec.learning_rate) or spec.learning_rate <= 0:
            raise ValueError(
                "The optimization learning rate must be finite and positive."
            )
        if not np.isfinite(spec.filter_radius) or spec.filter_radius < 0:
            raise ValueError("filter_radius must be finite and nonnegative.")
        if not 0 < spec.projection_eta < 1:
            raise ValueError("projection_eta must lie strictly between zero and one.")
        if grid.material_grid.uses_full_permittivity:
            raise ValueError("TopologyProblem requires scalar permittivity.")
        for axis in "xyz":
            if np.any(np.asarray(getattr(grid, f"sig_{axis}")) != 0):
                raise ValueError(
                    "TopologyProblem requires lossless materials; use CPML instead of sponge PML."
                )
            if not np.allclose(getattr(grid, f"mu_h{axis}"), 1):
                raise ValueError("TopologyProblem requires relative permeability one.")
        if len(sim.sources) != 1 or not isinstance(sim.sources[0], ModeSource):
            raise ValueError("TopologyProblem requires exactly one fixed ModeSource.")
        if not all(
            isinstance(monitor, (ModeMonitor, FieldMonitor)) for monitor in sim.monitors
        ):
            raise ValueError(
                "TopologyProblem accepts frequency-domain ModeMonitor/FieldMonitor only."
            )
        if any(np.argmin(m.size) == 2 for m in sim.monitors):
            raise ValueError("2D spectral monitors must be lines normal to x or y.")
        monitor_names = {monitor.name for monitor in sim.monitors}
        for term in self.objective.terms:
            if term.monitor not in monitor_names:
                raise ValueError("Objective monitor is not present in the simulation.")
            if (
                term.reference_monitor is not None
                and term.reference_monitor not in monitor_names
            ):
                raise ValueError("Reference monitor is not present in the simulation.")
        min_eps = min(spec.eps_min, float(np.min(grid.permittivity)))
        cfl = sim.resolution * np.sqrt(min_eps / 2) / LIGHT_SPEED
        if program.config.dt > cfl * (1 + 1e-6):
            raise ValueError(
                "Simulation timestep violates the CFL bound for the topology material range."
            )
        # A fixed source or modal basis is valid only if its entire transverse
        # material slice is outside the design, with a two-cell sampling margin.
        ys, xs = np.nonzero(spec.region_mask)
        if np.any(spec.region_mask[[0, -1], :]) or np.any(spec.region_mask[:, [0, -1]]):
            raise ValueError("The design region must stay inside the domain boundary.")
        for device in (
            *sim.sources,
            *(m for m in sim.monitors if isinstance(m, ModeMonitor)),
        ):
            axis = int(np.argmin(np.asarray(device.size)))
            if axis not in (0, 1):
                raise ValueError("Optimization mode planes must be normal to x or y.")
            position = float(device.center[axis]) / sim.resolution
            indices = xs if axis == 0 else ys
            if np.any(np.abs(indices - position) <= 2):
                raise ValueError(
                    "Keep the design at least two cells away from source and mode-monitor planes."
                )
        if grid.pml_data is not None:
            pml_mask = np.asarray(grid.pml_data["mask"])
            if pml_mask.shape != spec.region_mask.shape:
                raise ValueError("Unsupported absorber mask for topology optimization.")
            if np.any(pml_mask & spec.region_mask):
                raise ValueError("The design region must not overlap the absorber.")

    def _physical_density(self, density, beta):
        spec = self.topology
        return transform_density(
            jnp.where(self._mask, density, 0.0),
            self._mask,
            beta,
            spec.projection_eta,
            spec.filter_radius_cells,
            filter_type=spec.filter_type,
            morphology_operation=spec.morphology_operation,
            morphology_tau=spec.morphology_smooth_tau,
            fixed_structure_mask=jnp.asarray(spec.fixed_structure_mask),
            projection_type=spec.projection_type,
            ssp_smoothing_radius=spec.ssp_smoothing_radius,
        )

    def _permittivity(self, density, beta):
        physical = self._physical_density(density, beta)
        spec = self.topology
        return jnp.where(
            self._mask,
            spec.eps_min + (spec.eps_max - spec.eps_min) * physical,
            self._base_eps,
        )

    def _density(self, density):
        density = np.asarray(density, dtype=np.float32)
        if (
            density.shape != self.topology.region_mask.shape
            or not np.isfinite(density).all()
        ):
            raise ValueError(
                "Density must be finite and match the topology mask shape."
            )
        if np.any((density < 0) | (density > 1)):
            raise ValueError("Density must lie in [0, 1].")
        return jnp.asarray(density)

    def value(self, density, *, beta=1.0):
        """Evaluate the reduced scalar objective for latent design parameters."""
        value = float(self._evaluate(self._density(density), self._beta(beta)))
        if not np.isfinite(value):
            raise FloatingPointError(
                "Nonfinite objective; check stability and incoming modal power."
            )
        return value

    @staticmethod
    def _beta(beta: float):
        if (
            not isinstance(beta, (int, float, np.integer, np.floating))
            or not np.isfinite(beta)
            or beta <= 0
        ):
            raise ValueError("beta must be a finite positive scalar.")
        return jnp.asarray(beta, dtype=jnp.float32)

    def value_and_grad(self, density, *, beta=1.0):
        """Return the scalar objective and the selected backend's density gradient."""
        value, gradient = self._value_and_grad(self._density(density), self._beta(beta))
        value, gradient = float(value), np.asarray(gradient)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError(
                "Nonfinite objective/gradient; check FDTD stability and source spectrum."
            )
        return value, gradient

    @property
    def gradient_diagnostics(self):
        """Return spectral decay/solve diagnostics from the last gradient call."""
        if (
            self._spectral_adjoint is None
            or self._spectral_adjoint.last_diagnostics is None
        ):
            return None
        return dict(self._spectral_adjoint.last_diagnostics)

    def spectra(self, density, *, beta=1.0):
        """Return unweighted per-frequency powers for each objective term.

        The tuple follows ``objective.terms``; each array follows that term's
        requested frequencies, or its monitor's order when no subset is supplied.
        Ratios include reference normalization. One forward solve produces every
        spectrum. Keeping these physical values separate from the weighted score
        makes transmission, reflection, and crosstalk individually inspectable.
        """
        values = tuple(
            np.asarray(v)
            for v in self._spectra(self._density(density), self._beta(beta))
        )
        if not all(np.isfinite(v).all() for v in values):
            raise FloatingPointError(
                "Nonfinite modal spectrum; check stability and incoming modal power."
            )
        return values

    def physical_density(self, result: TopologyResult):
        """Return final physical density at the recorded projection strength."""
        self._check_result(result)
        return np.asarray(
            self._physical_density(
                self._density(result.state.density), self._beta(result.beta)
            )
        )

    def _check_result(self, result):
        if result.problem_fingerprint != self.fingerprint:
            raise ValueError("Result belongs to a different optimization problem.")

    def run(
        self,
        num_steps,
        *,
        initial_density=None,
        resume=None,
        stop_after=None,
        checkpoint=None,
        callback=None,
    ):
        """Run gradient ascent, with optional checkpoints and exact continuation.

        ``num_steps`` defines the full beta schedule. ``stop_after`` is an absolute
        completed-step limit for interrupted runs. Resume with the same total
        schedule and a result returned by this method or :meth:`load`.
        ``callback(result)`` is called after each completed and evaluated update.
        """
        if isinstance(num_steps, bool) or int(num_steps) != num_steps or num_steps < 1:
            raise ValueError("num_steps must be a positive integer.")
        num_steps = int(num_steps)
        limit = num_steps if stop_after is None else stop_after
        if (
            isinstance(limit, bool)
            or int(limit) != limit
            or not 0 <= limit <= num_steps
        ):
            raise ValueError(
                "stop_after must be an integer between zero and num_steps."
            )
        if resume is not None:
            if initial_density is not None:
                raise ValueError("Use initial_density or resume, not both.")
            self._check_result(resume)
            if resume.total_steps != num_steps or limit < resume.completed_steps:
                raise ValueError(
                    "Resume must preserve the total beta schedule and completed steps."
                )
            result = resume
        else:
            state = self.topology.initial_state(initial_density)
            density = self._density(state.density)
            state = replace(
                state,
                density=np.asarray(density),
                optimizer_state=self.topology._optimizer().init(density),
            )
            beta = self.topology.beta(0, num_steps)
            initial = self.value(state.density, beta=beta)
            result = TopologyResult(
                state, (), initial, beta, num_steps, self.fingerprint
            )
        for step in range(result.completed_steps, int(limit)):
            start = time.perf_counter()
            beta = self.topology.beta(step, num_steps)
            before, gradient = self.value_and_grad(result.state.density, beta=beta)
            state, max_update = self.topology.apply_parameter_gradient(
                result.state, gradient
            )
            value = self.value(state.density, beta=beta)
            if not np.isfinite(value):
                raise FloatingPointError(
                    "Nonfinite objective after the optimizer update."
                )
            state = state.with_objective(value)
            record = OptimizationStep(
                step + 1,
                beta,
                before,
                value,
                float(np.linalg.norm(gradient)),
                max_update,
                time.perf_counter() - start,
            )
            result = TopologyResult(
                state,
                (*result.history, record),
                result.initial_objective,
                beta,
                num_steps,
                self.fingerprint,
            )
            if checkpoint is not None:
                result.save(checkpoint)
            if callback is not None:
                callback(result)
        if checkpoint is not None and not result.history:
            result.save(checkpoint)
        return result

    def load(self, path):
        """Restore a checkpoint after checking its simulation/design identity."""
        return load_result(path, topology=self.topology, fingerprint=self.fingerprint)

    def material_simulation(self, density, *, beta=1.0):
        """Create an ordinary simulation with the identical density material map."""
        eps = self._permittivity(self._density(density), self._beta(beta))
        materials = topology_material_grid(self.program, eps, self._mask)
        return self.simulation.updated_copy(material_grid=materials)

    def export_design(self, result, *, threshold=0.5):
        """Return fixed geometry plus thresholded topology polygons, for re-solving.

        The base design must contain only fixed structures in the design region's
        background. A low-index region polygon overwrites any initial geometry in
        that region before adding solid topology contours.
        """
        if not 0 < threshold < 1:
            raise ValueError("threshold must lie strictly between zero and one.")
        physical = self.physical_density(result)
        spec = self.topology
        origin = self.program.grid.material_grid.origin
        dx = float(self.simulation.resolution)
        kwargs = {"dx": dx, "x0": origin[0], "y0": origin[1]}
        design = spec.design
        for polygon in density_to_polygons(
            spec.region_mask.astype(float),
            level=0.5,
            material=Material(permittivity=spec.eps_min),
            **kwargs,
        ):
            design += polygon
        for polygon in density_to_polygons(
            np.where(spec.region_mask, physical, 0.0),
            level=threshold,
            material=Material(permittivity=spec.eps_max),
            **kwargs,
        ):
            design += polygon
        return design
