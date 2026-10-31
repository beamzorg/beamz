"""User-defined objectives over differentiable simulation data."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

from beamz.devices.monitors import ModeMonitor
from beamz.simulation.api import Simulation

from .objectives import ModalObjective
from .problem import CompiledDesignProblem
from .region import MaskedDesignRegion, Specification, TopologyDesignRegion


@dataclass(frozen=True)
class InverseDesign(Specification):
    """One fixed-grid physical problem for rectangular or arbitrary-mask regions.

    A bound modal objective enables value/gradient/spectrum inspection. Alternatively
    supply a differentiable callback to Optimizer. Both gradient backends retain
    fixed geometry, source/basis, grid, boundaries, and observation clocks.
    """

    simulation: Simulation
    design_region: TopologyDesignRegion | MaskedDesignRegion
    objective: ModalObjective | None = None
    output_monitor_names: tuple[str, ...] | None = None
    gradient_backend: str = "adjoint"
    adjoint_decay_tolerance: float = 1e-4
    checkpoint_interval: int | None = 32
    adjoint_source_grouping: str = field(
        default="frequency", metadata={"beamz_cache": False}
    )
    _problem: CompiledDesignProblem = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        sim, region = self.simulation, self.design_region
        if not isinstance(region, (TopologyDesignRegion, MaskedDesignRegion)):
            raise TypeError(
                "design_region must be a TopologyDesignRegion or MaskedDesignRegion."
            )
        program = sim.compile(backend="jax")
        density_map = region.bind(sim, program.grid)
        names = self.output_monitor_names
        names = (
            tuple(names)
            if names is not None
            else tuple(m.name for m in sim.monitors if isinstance(m, ModeMonitor))
        )
        available = {m.name for m in sim.monitors if isinstance(m, ModeMonitor)}
        if (
            not names
            or not all(isinstance(n, str) for n in names)
            or len(set(names)) != len(names)
            or not set(names) <= available
        ):
            raise ValueError(
                "output_monitor_names must select unique existing ModeMonitor names."
            )
        problem = CompiledDesignProblem(
            sim,
            density_map,
            self.objective,
            program=program,
            output_monitor_names=names,
            gradient_backend=self.gradient_backend,
            adjoint_decay_tolerance=self.adjoint_decay_tolerance,
            checkpoint_interval=self.checkpoint_interval,
            adjoint_source_grouping=self.adjoint_source_grouping,
        )
        object.__setattr__(self, "output_monitor_names", names)
        object.__setattr__(self, "_problem", problem)

    @property
    def program(self):
        return self._problem.program

    @property
    def fingerprint(self):
        return self._problem.fingerprint

    def _beta(self, beta):
        return (
            1.0
            if beta is None and isinstance(self.design_region, MaskedDesignRegion)
            else beta
        )

    def value(self, params, *, beta=None):
        if self.objective is None:
            raise ValueError(
                "Supply a modal objective on InverseDesign, or use make_objective_fn for a callback."
            )
        return self._problem.value(self._density(params), beta=self._beta(beta))

    def value_and_grad(self, params, *, beta=None):
        if self.objective is None:
            raise ValueError(
                "Supply a modal objective on InverseDesign, or use make_objective_fn for a callback."
            )
        return self._problem.value_and_grad(
            self._density(params), beta=self._beta(beta)
        )

    def spectra(self, params, *, beta=None):
        if self.objective is None:
            raise ValueError(
                "Supply a modal objective on InverseDesign, or use make_objective_fn for a callback."
            )
        return self._problem.spectra(self._density(params), beta=self._beta(beta))

    def physical_density(self, params, *, beta=None):
        """Return placed physical density using the same map as material binding."""
        return np.asarray(
            self._problem._physical_density(self._density(params), self._beta(beta))
        )

    def _density(self, params, *, beta=None):
        if not isinstance(params, Tracer):
            self.design_region.check_params(np.asarray(params))
        return jnp.asarray(params)

    @property
    def initial_simulation(self):
        return self.to_simulation(self.design_region.initial_parameters)

    def to_simulation(self, params, *, beta=None):
        """Return an ordinary BeamZ simulation for inspection or verification."""
        return self._problem.material_simulation(
            self._density(params), beta=self._beta(beta)
        )

    def export_design(
        self, params, *, threshold=0.5, beta=None, mode="contour", sample_spacing=None
    ):
        """Threshold density and export extruded polygons (one layer per z voxel).

        The returned geometry can be rasterized at a different verification mesh.
        ``mode`` is "contour" or "pixel_exact"; ``sample_spacing`` optionally
        refines bilinear contour sampling in metres. Fixed cells remain exact.
        """
        self.design_region.check_params(params)
        return self._problem._density_map.export(
            self.simulation.design,
            params,
            beta=self._beta(beta),
            threshold=threshold,
            mode=mode,
            sample_spacing=sample_spacing,
        )

    def to_simulation_data(self, params, *, beta=None):
        """Run either backend and return trace-preserving modal amplitudes."""
        return self._problem.simulation_data(
            self._density(params),
            beta=self._beta(beta),
            monitor_names=self.output_monitor_names,
        )

    def _evaluate(
        self,
        params,
        post_process_fn,
        maximize,
        *,
        beta=None,
        penalty_weight=1.0,
        post_process_kwargs=None,
    ):
        if post_process_fn is self.objective and not post_process_kwargs:
            # Reuse the objective's already-bound reduction and its exact compiled arithmetic.
            # User callbacks consume the same cached modal amplitude plans below.
            post = self._problem._evaluate(
                self._density(params), self._problem._beta(self._beta(beta))
            )
        else:
            data = self.to_simulation_data(params, beta=beta)
            post = jnp.asarray(post_process_fn(data, **(post_process_kwargs or {})))
        penalty = penalty_weight * jnp.asarray(
            self.design_region.penalty_value(
                params, beta=self._beta(beta), density_map=self._problem._density_map
            )
        )
        if (
            post.shape != ()
            or jnp.iscomplexobj(post)
            or penalty.shape != ()
            or jnp.iscomplexobj(penalty)
        ):
            raise ValueError(
                "post_process_fn and combined penalties must return real scalars."
            )
        score = (post if maximize else -post) - penalty
        return score, (post, penalty)

    def _objective_gradient(self, objective, maximize):
        """Select the already-bound modal reduction or a user callback VJP.

        Both strategies return the same score/penalty/gradient contract. Reusing
        the bound reduction retains its compiled floating-point arithmetic and
        avoids projecting unused modes for a built-in objective.
        """
        callback = jax.value_and_grad(
            lambda params, settings: self._evaluate(
                params, objective, maximize, **settings
            ),
            has_aux=True,
        )
        if objective is not self.objective:
            return callback

        def evaluate(params, settings):
            if settings.get("post_process_kwargs"):
                return callback(params, settings)
            beta = self._problem._beta(self._beta(settings.get("beta")))
            post, gradient = self._problem._value_and_grad(params, beta)
            penalty, penalty_gradient = jax.value_and_grad(
                lambda values: (
                    settings.get("penalty_weight", 1.0)
                    * jnp.asarray(
                        self.design_region.penalty_value(
                            values, beta=beta, density_map=self._problem._density_map
                        )
                    )
                )
            )(params)
            sign = 1 if maximize else -1
            return (
                sign * post - penalty,
                (post, penalty),
            ), sign * gradient - penalty_gradient

        return evaluate

    def make_objective_fn(
        self,
        post_process_fn,
        maximize=True,
        *,
        beta=None,
        penalty_weight=1.0,
        post_process_kwargs=None,
    ):
        """Return a scalar function usable with jax.value_and_grad."""
        if not callable(post_process_fn):
            raise TypeError("post_process_fn must be callable.")
        return lambda params: self._evaluate(
            params,
            post_process_fn,
            maximize,
            beta=beta,
            penalty_weight=penalty_weight,
            post_process_kwargs=post_process_kwargs,
        )[0]

    @property
    def gradient_diagnostics(self):
        return self._problem.gradient_diagnostics
