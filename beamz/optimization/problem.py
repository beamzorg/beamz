"""A complete fixed-grid topology optimization workflow."""

from __future__ import annotations

import hashlib
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

from ._validation import check_density, step_count
from .autodiff import transform_density
from .objectives import ModePower, WeightedObjective
from .polygonize import density_to_polygons
from .result import TopologyResult, load_result
from .topology import TopologySpec, TopologyState


class TopologyProblem:
    """Optimize a topology region against a scalar modal objective.

    Both backends support uniform 2D xy or 3D grids, scalar lossless dielectrics,
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
        objective: ModePower | WeightedObjective | None = None,
        *,
        checkpoint_interval: int | None = 32,
        gradient_backend="autodiff",
        adjoint_decay_tolerance=1e-4,
        adjoint_source_grouping="frequency",
        output_monitor_names=None,
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
        if adjoint_source_grouping not in ("frequency", "auto"):
            raise ValueError("adjoint_source_grouping must be frequency or auto.")
        self.adjoint_source_grouping = adjoint_source_grouping
        self.gradient_backend = gradient_backend
        self.adjoint_decay_tolerance = float(adjoint_decay_tolerance)
        self.simulation = simulation
        self.topology = topology
        self.objective = objective
        if objective is not None and not isinstance(
            objective, (ModePower, WeightedObjective)
        ):
            raise TypeError(
                "TopologyProblem supports ModePower and weighted modal objectives."
            )
        if objective is None and not output_monitor_names:
            raise ValueError("Supply an objective or output_monitor_names.")
        self.frequency_requests = (
            tuple((term.monitor, term.frequencies) for term in objective.terms)
            if objective is not None
            else tuple((name, None) for name in output_monitor_names or ())
        )
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
        reduce_objective = (
            objective.bind(empty_results, self.program, cache)
            if objective is not None
            else lambda _: jnp.asarray(0.0)
        )
        spectra = [
            term.bind_spectrum(empty_results, self.program, cache)
            for term in (() if objective is None else objective.terms)
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
        token = cache_token(
            (
                TOPOLOGY_NUMERICS_VERSION,
                simulation,
                topology,
                objective or self.frequency_requests,
                gradient_backend,
                self.adjoint_decay_tolerance,
                adjoint_source_grouping,
            )
        )
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
        if not program.config.is_3d and sim.plane_2d != "xy":
            raise ValueError("TopologyProblem supports 2D xy or 3D simulations.")
        if (
            grid.geometry.metric_kind_for(
                tuple("xyz" if program.config.is_3d else "xy")
            )
            != "isotropic_uniform"
            or program.sharding.layout.enabled
        ):
            raise ValueError("TopologyProblem requires an unsharded uniform grid.")
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
        if program.config.is_3d and (
            spec.filter_radius != 0 or spec.projection_type == "ssp"
        ):
            raise ValueError(
                "3D low-level topology requires zero filter radius and no SSP; apply planar transforms through TopologyDesignRegion."
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
        if not program.config.is_3d and any(
            np.argmin(m.size) == 2 for m in sim.monitors
        ):
            raise ValueError("2D spectral monitors must be lines normal to x or y.")
        monitor_names = {monitor.name for monitor in sim.monitors}
        for term in () if self.objective is None else self.objective.terms:
            if term.monitor not in monitor_names:
                raise ValueError("Objective monitor is not present in the simulation.")
            if (
                term.reference_monitor is not None
                and term.reference_monitor not in monitor_names
            ):
                raise ValueError("Reference monitor is not present in the simulation.")
        min_eps = min(spec.eps_min, float(np.min(grid.permittivity)))
        cfl = (
            sim.resolution
            * np.sqrt(min_eps / (3 if program.config.is_3d else 2))
            / LIGHT_SPEED
        )
        if program.config.dt > cfl * (1 + 1e-6):
            raise ValueError(
                "Simulation timestep violates the CFL bound for the topology material range."
            )
        # A fixed source or modal basis is valid only if its entire transverse
        # material slice is outside the design, with a two-cell sampling margin.
        coordinates = np.nonzero(spec.region_mask)[::-1]
        if any(
            np.any(np.take(spec.region_mask, [0, -1], axis=a))
            for a in range(spec.region_mask.ndim)
        ):
            raise ValueError("The design region must stay inside the domain boundary.")
        for device in (
            *sim.sources,
            *(m for m in sim.monitors if isinstance(m, ModeMonitor)),
        ):
            axis = int(np.argmin(np.asarray(device.size)))
            if axis >= len(coordinates):
                raise ValueError(
                    "Optimization mode planes must be normal to an active grid axis."
                )
            position = float(device.center[axis]) / sim.resolution
            indices = coordinates[axis]
            if np.any(np.abs(indices - position) <= 2):
                raise ValueError(
                    "Keep the design at least two cells away from source and mode-monitor planes."
                )
        if grid.pml_data is not None:
            pml_mask = (
                np.asarray(grid.pml_data["mask"])
                if "mask" in grid.pml_data
                else (np.asarray(grid.pml_data["sigma_x"]) > 0)
                | (np.asarray(grid.pml_data["sigma_y"]) > 0)
                | (np.asarray(grid.pml_data["sigma_z"]) > 0)
            )
            if pml_mask.shape != spec.region_mask.shape:
                raise ValueError("Unsupported absorber mask for topology optimization.")
            if np.any(pml_mask & spec.region_mask):
                raise ValueError("The design region must not overlap the absorber.")

    def _physical_density(self, density, beta):
        return transform_density(
            jnp.where(self._mask, density, 0.0),
            beta=beta,
            **self.topology._transform_options,
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
        check_density(density, self.topology.region_mask.shape)
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
    def _beta(beta: float | None):
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
                self._density(result.final_params), self._beta(result.beta)
            )
        )

    def _check_result(self, result):
        if result.fingerprint != self.fingerprint:
            raise ValueError("Result belongs to a different optimization problem.")

    def run(
        self,
        num_steps=None,
        *,
        steps=None,
        initial_density=None,
        resume=None,
        stop_after=None,
        checkpoint=None,
        checkpoint_every=1,
        callback=None,
    ):
        """Optimize an arbitrary mask using the shared result/checkpoint API.

        steps defines the fixed beta horizon; stop_after pauses at an absolute
        completed-step limit. Retain that horizon when resuming. num_steps is
        accepted for existing callers. Rectangular regions can instead use
        InverseDesign and AdamOptimizer with an independent projection schedule.
        """
        from .optimizer import _optimize

        if steps is not None and num_steps is not None:
            raise ValueError("Supply steps or num_steps, not both.")
        num_steps = step_count(
            num_steps if steps is None else steps, "num_steps", minimum=1
        )
        limit = (
            num_steps if stop_after is None else step_count(stop_after, "stop_after")
        )
        if limit > num_steps:
            raise ValueError(
                "stop_after must be an integer between zero and num_steps."
            )
        if resume is not None:
            if initial_density is not None:
                raise ValueError("Use initial_density or resume, not both.")
            result = resume if isinstance(resume, TopologyResult) else self.load(resume)
            self._check_result(result)
            if result.total_steps != num_steps or limit < result.completed_steps:
                raise ValueError(
                    "Resume must preserve the total beta schedule and completed steps."
                )
        else:
            state = self.topology.initial_state(initial_density)
            density = self._density(state.density)
            state = replace(
                state,
                density=np.asarray(density),
                optimizer_state=self.topology._optimizer().init(density),
            )
            beta = self.topology.beta(0, num_steps)
            result = TopologyResult(
                self,
                (state.density,),
                state.optimizer_state,
                initial_objective=self.value(state.density, beta=beta),
                fingerprint=self.fingerprint,
                total_steps=num_steps,
                _initial_beta=beta,
            )

        def evaluate(params, settings):
            value, gradient = self.value_and_grad(params, **settings)
            return (value, (value, 0.0)), gradient

        def update(params, state, gradient):
            state, change = self.topology.apply_parameter_gradient(
                TopologyState(params, state), gradient
            )
            return state.density, state.optimizer_state, change

        return _optimize(
            result,
            limit,
            evaluate=evaluate,
            optimizer=self.topology._optimizer(),
            settings=lambda i: dict(beta=self.topology.beta(i, num_steps)),
            update=update,
            evaluate_after=lambda params, settings: self.value(params, **settings),
            checkpoint=checkpoint,
            checkpoint_every=checkpoint_every,
            callback=callback,
        )

    def load(self, path):
        return load_result(
            path,
            design=self,
            fingerprint=self.fingerprint,
            optimizer=self.topology._optimizer(),
            shape=self.topology.region_mask.shape,
        )

    def material_simulation(self, density, *, beta=1.0):
        """Create an ordinary simulation with the identical density material map."""
        eps = self._permittivity(self._density(density), self._beta(beta))
        materials = topology_material_grid(self.program, eps, self._mask)
        return self.simulation.updated_copy(material_grid=materials)

    to_simulation = material_simulation
    to_simulation_data = simulation_data

    def export_design(self, result, *, threshold=0.5):
        """Return fixed geometry plus thresholded topology polygons, for re-solving.

        The base design must contain only fixed structures in the design region's
        background. A low-index region polygon overwrites any initial geometry in
        that region before adding solid topology contours.
        """
        if self.program.config.is_3d:
            raise ValueError(
                "3D polygon export is not supported by TopologyProblem; use the extruded design-region export workflow."
            )
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
