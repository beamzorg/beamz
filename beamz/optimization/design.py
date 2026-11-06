"""User-defined objectives over differentiable simulation data."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

from beamz.devices.monitors import FieldMonitor, ModeMonitor
from beamz.simulation.api import Simulation

from ._validation import execution_backend
from .problem import TopologyProblem
from .region import Specification, TopologyDesignRegion
from .topology import TopologySpec


@dataclass(frozen=True)
class InverseDesign(Specification):
    """Inverse design with modal/radiation outputs and selectable execution.

    ``backend`` selects the FDTD engine; ``gradient_backend`` selects the
    derivative method. Native ``backend='cuda'`` requires
    ``gradient_backend='autodiff'`` and the existing FP32 single-GPU runtime.
    """

    simulation: Simulation
    design_region: TopologyDesignRegion
    output_monitor_names: tuple[str, ...] | None = None
    gradient_backend: str = "adjoint"
    adjoint_decay_tolerance: float = 1e-4
    checkpoint_interval: int | None = 32
    adjoint_source_grouping: str = field(
        default="frequency", metadata={"beamz_cache": False}
    )
    backend: str = "jax"
    _problem: TopologyProblem = field(init=False, repr=False, compare=False)
    _slices: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        execution_backend(self.backend, self.gradient_backend)
        sim, region = self.simulation, self.design_region
        center = np.asarray(region.center) + np.asarray(sim.coordinate_offset)
        shape = sim.compile(backend=self.backend).grid.material_grid.shape
        ndim = len(shape)
        if region.parameterization == "volumetric" and ndim != 3:
            raise ValueError("Volumetric design regions require a 3D simulation.")
        sizes = np.asarray(region.size[:ndim])
        if ndim == 3 and (not np.isfinite(sizes[2]) or sizes[2] <= 0):
            raise ValueError("A 3D design region requires finite positive thickness.")
        low = (center[:ndim] - sizes / 2) / sim.resolution
        counts = sizes / sim.resolution
        if not (
            np.allclose(low, np.round(low), rtol=0, atol=1e-6)
            and np.allclose(counts, np.round(counts), rtol=0, atol=1e-6)
        ):
            raise ValueError("Design-region bounds must align to the simulation grid.")
        starts, counts = (
            np.round(low).astype(int)[::-1],
            np.round(counts).astype(int)[::-1],
        )
        if np.any(starts < 0) or np.any(starts + counts > shape):
            raise ValueError("The design region must lie inside the simulation.")
        slices = tuple(
            slice(start, start + count)
            for start, count in zip(starts, counts, strict=True)
        )
        mask = np.zeros(shape, dtype=bool)
        mask[slices] = True
        names = self.output_monitor_names
        names = (
            tuple(names)
            if names is not None
            else tuple(
                m.name
                for m in sim.monitors
                if isinstance(m, (ModeMonitor, FieldMonitor))
            )
        )
        available = {
            m.name for m in sim.monitors if isinstance(m, (ModeMonitor, FieldMonitor))
        }
        if (
            not names
            or not all(isinstance(n, str) for n in names)
            or len(set(names)) != len(names)
            or not set(names) <= available
        ):
            raise ValueError(
                "output_monitor_names must select unique existing ModeMonitor/FieldMonitor names."
            )
        topology = TopologySpec(
            design=sim.design,
            region_mask=mask,
            resolution=sim.resolution,
            eps_min=region.eps_bounds[0],
            eps_max=region.eps_bounds[1],
            projection_type="identity",
        )
        # Request monitor data directly; the user objective supplies its VJP.
        problem = TopologyProblem(
            sim,
            topology,
            output_monitor_names=names,
            gradient_backend=self.gradient_backend,
            adjoint_decay_tolerance=self.adjoint_decay_tolerance,
            checkpoint_interval=self.checkpoint_interval,
            adjoint_source_grouping=self.adjoint_source_grouping,
            backend=self.backend,
        )
        object.__setattr__(self, "output_monitor_names", names)
        object.__setattr__(self, "_problem", problem)
        object.__setattr__(self, "_slices", slices)

    def _density(self, params, *, beta=None):
        if not isinstance(params, Tracer):
            self.design_region.check_params(np.asarray(params))
        physical = self.design_region.material_density(params, beta=beta)
        target_shape = tuple(s.stop - s.start for s in self._slices)
        if self.design_region.parameterization == "planar":
            target_shape = target_shape[-2:]
        if physical.shape != target_shape:
            physical = jax.image.resize(
                physical, target_shape, method="linear", antialias=False
            )
        return (
            jnp.zeros(self._problem.topology.region_mask.shape, dtype=physical.dtype)
            .at[self._slices]
            .set(physical)
        )

    @property
    def initial_simulation(self):
        return self.to_simulation(self.design_region.initial_parameters)

    def to_simulation(self, params, *, beta=None):
        """Return an ordinary BeamZ simulation for inspection or verification."""
        return self._problem.material_simulation(self._density(params, beta=beta))

    def export_design(
        self, params, *, threshold=0.5, beta=None, mode="contour", sample_spacing=None
    ):
        """Threshold density and export extruded polygons (one layer per z voxel).

        The returned geometry can be rasterized at a different verification mesh.
        ``mode`` is "contour" or "pixel_exact"; ``sample_spacing`` optionally
        refines bilinear contour sampling in metres. Fixed cells remain exact.
        """
        from beamz.design.materials import Material
        from beamz.design.structures import Polygon, Rectangle

        from .polygonize import density_to_polygons

        self.design_region.check_params(params)
        if not 0 < threshold < 1:
            raise ValueError("threshold must lie strictly between zero and one.")
        region = self.design_region
        center = np.asarray(region.center) + np.asarray(
            self.simulation.coordinate_offset
        )
        low = center - np.asarray(region.size) / 2
        # Use the already-validated integer grid placement. Center-size arithmetic
        # can leave a one-ulp seam against guides on the same cell edge.
        low[:2] = [
            self._slices[-1].start * self.simulation.resolution,
            self._slices[-2].start * self.simulation.resolution,
        ]
        is_3d = self._problem.program.config.is_3d
        z, depth = (float(low[2]), region.size[2]) if is_3d else (0.0, 0.0)
        design = self.simulation.design
        design += Rectangle(
            position=(low[0], low[1], z),
            width=region.size[0],
            height=region.size[1],
            depth=depth,
            material=Material(region.eps_bounds[0]),
        )
        density = np.asarray(region.material_density(params, beta=beta))
        layers = density if region.parameterization == "volumetric" else density[None]
        layer_depth = (
            region.pixel_size if region.parameterization == "volumetric" else depth
        )
        for index, layer in enumerate(layers):
            layer_z = z + index * layer_depth
            solid = region.fixed_solid
            void = region.fixed_void
            if region.parameterization == "volumetric":
                solid = None if solid is None else solid[index]
                void = None if void is None else void[index]
            for polygon in density_to_polygons(
                layer,
                level=threshold,
                x0=low[0],
                y0=low[1],
                dx=region.pixel_size,
                material=Material(region.eps_bounds[1]),
                mode=mode,
                sample_spacing=sample_spacing,
                fixed_solid=solid,
                fixed_void=void,
            ):
                design += Polygon(
                    vertices=[(x, y, layer_z) for x, y, _ in polygon.vertices],
                    interiors=[
                        [(x, y, layer_z) for x, y, _ in ring]
                        for ring in polygon.interiors
                    ],
                    material=polygon.material,
                    depth=layer_depth,
                    z=layer_z,
                )
        return design

    def to_simulation_data(self, params, *, beta=None):
        """Run either backend and return differentiable modal or surface fields."""
        return self._problem.simulation_data(
            self._density(params, beta=beta), monitor_names=self.output_monitor_names
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
        data = self.to_simulation_data(params, beta=beta)
        post = jnp.asarray(post_process_fn(data, **(post_process_kwargs or {})))
        penalty = penalty_weight * jnp.asarray(
            self.design_region.penalty_value(params, beta=beta)
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
