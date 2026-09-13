"""User-defined objectives over differentiable simulation data."""

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

from beamz.devices.monitors import ModeMonitor
from beamz.optimization import (
    ModePower,
    TopologyProblem,
    TopologySpec,
    WeightedObjective,
)
from beamz.simulation.api import Simulation

from .region import Specification, TopologyDesignRegion


@dataclass(frozen=True)
class InverseDesign(Specification):
    simulation: Simulation
    design_region: TopologyDesignRegion
    output_monitor_names: tuple[str, ...] | None = None
    task_name: str = "beamz-inverse-design"
    gradient_backend: str = "adjoint"
    adjoint_decay_tolerance: float = 1e-4
    checkpoint_interval: int | None = 32
    _problem: TopologyProblem = field(init=False, repr=False, compare=False)
    _slices: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        sim, region = self.simulation, self.design_region
        if not np.isclose(region.pixel_size, sim.resolution, rtol=1e-10, atol=0):
            raise ValueError(
                "This workflow requires pixel_size == simulation.resolution."
            )
        # Convert the public (possibly centered) frame exactly once.
        center = np.asarray(region.center) + np.asarray(sim.coordinate_offset)
        low = (center[:2] - np.asarray(region.size[:2]) / 2) / sim.resolution
        if not np.allclose(low, np.round(low), rtol=0, atol=1e-6):
            raise ValueError("Design-region bounds must align to the simulation grid.")
        x0, y0 = np.round(low).astype(int)
        ny, nx = region.params_shape
        shape = sim.compile(backend="jax").grid.material_grid.shape
        if (
            len(shape) != 2
            or x0 < 0
            or y0 < 0
            or x0 + nx > shape[1]
            or y0 + ny > shape[0]
        ):
            raise ValueError("The 2D design region must lie inside the simulation.")
        slices = (slice(y0, y0 + ny), slice(x0, x0 + nx))
        mask = np.zeros(shape, dtype=bool)
        mask[slices] = True
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
        topology = TopologySpec(
            design=sim.design,
            region_mask=mask,
            resolution=sim.resolution,
            eps_min=region.eps_bounds[0],
            eps_max=region.eps_bounds[1],
            projection_type="identity",
        )
        # Plan every requested frequency. The actual user objective supplies the
        # VJP at simulation_data(); no modal-power restriction is imposed there.
        problem = TopologyProblem(
            sim,
            topology,
            WeightedObjective(
                tuple(ModePower(str(name)) for name in names), tuple(1.0 for _ in names)
            ),
            gradient_backend=self.gradient_backend,
            adjoint_decay_tolerance=self.adjoint_decay_tolerance,
            checkpoint_interval=self.checkpoint_interval,
        )
        object.__setattr__(self, "output_monitor_names", names)
        object.__setattr__(self, "_problem", problem)
        object.__setattr__(self, "_slices", slices)

    def _density(self, params):
        if not isinstance(params, Tracer):
            self.design_region.check_params(np.asarray(params))
        physical = self.design_region.material_density(params)
        return (
            jnp.zeros(self._problem.topology.region_mask.shape, dtype=physical.dtype)
            .at[self._slices]
            .set(physical)
        )

    @property
    def initial_simulation(self):
        return self.to_simulation(self.design_region.initial_parameters)

    def to_simulation(self, params):
        """Return an ordinary BeamZ simulation for inspection or verification."""
        return self._problem.material_simulation(self._density(params))

    def to_simulation_data(self, params):
        """Run either backend and return trace-preserving modal amplitudes."""
        return self._problem.simulation_data(
            self._density(params), monitor_names=self.output_monitor_names
        )

    def _evaluate(self, params, post_process_fn, maximize):
        data = self.to_simulation_data(params)
        post = jnp.asarray(post_process_fn(data))
        penalty = jnp.asarray(self.design_region.penalty_value(params))
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

    def make_objective_fn(self, post_process_fn, maximize=True):
        """Return a scalar function usable with jax.value_and_grad."""
        if not callable(post_process_fn):
            raise TypeError("post_process_fn must be callable.")
        return lambda params: self._evaluate(params, post_process_fn, maximize)[0]

    @property
    def gradient_diagnostics(self):
        return self._problem.gradient_diagnostics
