"""Rectangular design regions and density parameterization."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import numpy as np

from beamz.optimization.design.initialization import (
    FilterProject,
    Specification,
    UniformInitializationSpec,
)
from beamz.optimization.design.validation import check_density, fixed_density_masks
from beamz.optimization.topology.transforms.density import apply_fixed_density


@dataclass(frozen=True)
class TopologyDesignRegion(Specification):
    """Planar or volumetric density region in public simulation coordinates.

    All lengths are metres. ``parameterization="planar"`` (default) uses (y, x)
    parameters extruded through z. ``"volumetric"`` uses independent (z, y, x)
    voxels. Bounds must align to the simulation grid. The density grid may have
    a different pixel_size; physical density is linearly sampled at FDTD cell
    centers, and gradients use the exact transpose of that interpolation.
    Optional boolean ``fixed_solid`` / ``fixed_void`` masks have ``params_shape`` and pin physical
    density to 1 / 0 before and after every transform, with zero parameter
    gradients at those cells. Export preserves their exact cell footprints.
    """

    size: tuple[float, float, float]
    center: tuple[float, float, float]
    eps_bounds: tuple[float, float]
    pixel_size: float
    transformations: tuple = ()
    penalties: tuple = ()
    initialization_spec: Any = UniformInitializationSpec()
    penalty_input: str = field(default="density", metadata={"beamz_cache": False})
    fixed_solid: Any = field(default=None, repr=False, compare=False)
    fixed_void: Any = field(default=None, repr=False, compare=False)
    parameterization: str = "planar"

    def __post_init__(self):
        if self.parameterization not in ("planar", "volumetric"):
            raise ValueError("parameterization must be planar or volumetric.")
        if self.penalty_input not in ("density", "parameters"):
            raise ValueError("penalty_input must be density or parameters.")
        for name in ("size", "center", "eps_bounds", "transformations", "penalties"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            len(self.size) != 3
            or len(self.center) != 3
            or not np.isfinite(self.center).all()
        ):
            raise ValueError("size and center must be xyz triples, with finite center.")
        if (
            not np.isfinite(self.size[:2]).all()
            or min(self.size[:2]) <= 0
            or np.isnan(self.size[2])
            or self.size[2] < 0
        ):
            raise ValueError(
                "Require finite positive x/y sizes and nonnegative z size."
            )
        if not np.isfinite(self.pixel_size) or self.pixel_size <= 0:
            raise ValueError("pixel_size must be finite and positive.")
        if (
            len(self.eps_bounds) != 2
            or not np.isfinite(self.eps_bounds).all()
            or not 0 < self.eps_bounds[0] < self.eps_bounds[1]
        ):
            raise ValueError("Require finite 0 < eps_bounds[0] < eps_bounds[1].")
        if self.parameterization == "volumetric" and (
            not np.isfinite(self.size[2]) or self.size[2] <= 0
        ):
            raise ValueError("Volumetric regions require finite positive z size.")
        sizes = self.size if self.parameterization == "volumetric" else self.size[:2]
        cells = np.asarray(sizes) / self.pixel_size
        if np.any(np.round(cells) < 1) or not np.allclose(
            cells, np.round(cells), rtol=0, atol=1e-6
        ):
            raise ValueError(
                "Design-region sizes must be integer multiples of pixel_size."
            )
        if not hasattr(self.initialization_spec, "create_parameters"):
            raise TypeError(
                "initialization_spec must provide create_parameters(shape)."
            )
        if any(not callable(t) for t in self.transformations + self.penalties):
            raise TypeError("Transformations and penalties must be callable.")

        solid, void = fixed_density_masks(
            self.fixed_solid, self.fixed_void, self.params_shape
        )
        object.__setattr__(self, "fixed_solid", solid)
        object.__setattr__(self, "fixed_void", void)

    @property
    def params_shape(self):
        sizes = self.size if self.parameterization == "volumetric" else self.size[:2]
        return tuple(int(round(size / self.pixel_size)) for size in sizes[::-1])

    @property
    def initial_parameters(self):
        params = np.asarray(
            self.initialization_spec.create_parameters(self.params_shape),
            dtype=np.float32,
        )
        self.check_params(params)
        return params

    def check_params(self, params):
        check_density(params, self.params_shape, "Parameters")

    def material_density(self, params, *, beta=None):
        params = jnp.asarray(params)
        if params.shape != self.params_shape:
            raise ValueError("Parameters must match design_region.params_shape.")
        if beta is not None and not any(
            isinstance(t, FilterProject) for t in self.transformations
        ):
            raise ValueError("A beta override requires a FilterProject transformation.")
        params = apply_fixed_density(params, self.fixed_solid, self.fixed_void)
        for transform in self.transformations:
            params = (
                transform.evaluate(params, self.pixel_size, beta=beta)
                if isinstance(transform, FilterProject)
                else transform(params, self.pixel_size)
            )
            params = apply_fixed_density(params, self.fixed_solid, self.fixed_void)
        return params

    def bind(self, simulation, grid):
        """Resolve a region once onto the simulation's canonical material lattice."""
        center = np.asarray(self.center) + np.asarray(simulation.coordinate_offset)
        shape, dl = grid.material_grid.shape, simulation.resolution
        ndim = len(shape)
        if self.parameterization == "volumetric" and ndim != 3:
            raise ValueError("Volumetric design regions require a 3D simulation.")
        sizes = np.asarray(self.size[:ndim])
        if ndim == 3 and (not np.isfinite(sizes[2]) or sizes[2] <= 0):
            raise ValueError("A 3D design region requires finite positive thickness.")
        low, counts = (center[:ndim] - sizes / 2) / dl, sizes / dl
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
        slices = tuple(slice(a, a + n) for a, n in zip(starts, counts, strict=True))
        mask = np.zeros(shape, dtype=bool)
        mask[slices] = True
        mask.setflags(write=False)
        # Preserve exact snapped cell edges, avoiding one-ulp seams at fixed guides.
        origin = [
            slices[-1].start * dl,
            slices[-2].start * dl,
            float(center[2] - sizes[2] / 2) if ndim == 3 else 0.0,
        ]
        depth = self.size[2] if ndim == 3 else 0.0
        return DensityParameterization(
            self.params_shape,
            self.pixel_size,
            self.eps_bounds,
            self.material_density,
            self.fixed_solid,
            self.fixed_void,
            mask,
            slices,
            configuration=self,
            geometry=dict(
                origin=tuple(origin),
                depth=depth,
                layer_depth=self.pixel_size
                if self.parameterization == "volumetric"
                else depth,
            ),
        )

    def eps_values(self, params, *, beta=None):
        low, high = self.eps_bounds
        return low + (high - low) * self.material_density(params, beta=beta)

    def penalty_value(self, params, *, beta=None, density_map=None):
        if not self.penalties:
            return 0.0
        density = (
            jnp.asarray(params)
            if self.penalty_input == "parameters"
            else self.material_density(params, beta=beta)
        )
        return sum(penalty(density, self.pixel_size) for penalty in self.penalties)


@dataclass(frozen=True)
class DensityParameterization:
    """Latent-to-material map shared by arbitrary masks and rectangular regions.

    Fixed constraints apply in parameter space around each transform. Linear
    cell-center interpolation follows transformation, then planar extrusion or
    volumetric placement. The outside raster remains fixed during binding.
    """

    shape: tuple
    pixel_size: float
    eps_bounds: tuple
    transform: Any
    fixed_solid: Any = None
    fixed_void: Any = None
    mask: Any = None
    slices: tuple | None = None
    configuration: Any = None
    context: Any = None
    geometry: Any = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "geometry", MappingProxyType(dict(self.geometry)))

    def canonical_spec(self):
        return self.configuration, self.mask, self.slices, self.context

    def density(self, params, *, beta=None):
        return self.transform(params, beta=beta)

    def place(self, physical):
        if self.slices is None:
            return physical
        import jax

        target = tuple(s.stop - s.start for s in self.slices)[-physical.ndim :]
        if physical.shape != target:
            physical = jax.image.resize(
                physical, target, method="linear", antialias=False
            )
        return (
            jnp.zeros(self.mask.shape, dtype=physical.dtype)
            .at[self.slices]
            .set(physical)
        )

    def physical(self, params, beta):
        if self.slices is None:
            params = jnp.where(self.mask, params, 0.0)
        return self.place(self.density(params, beta=beta))

    def permittivity(self, params, beta, base):
        low, high = self.eps_bounds
        physical = self.physical(params, beta)
        return jnp.where(self.mask, low + (high - low) * physical, base)

    def export(self, design, params, *, beta=None, context=None, **geometry):
        from beamz.optimization.design.geometry.export import _export_density

        if self.slices is None and np.ndim(self.mask) == 3:
            raise ValueError(
                "3D mask polygon export is unsupported; use a volumetric TopologyDesignRegion."
            )
        density = np.asarray(self.density(params, beta=beta))
        if self.slices is not None:
            geometry.setdefault("background", np.ones(self.shape[-2:]))
        context = self.context if context is None else context
        if context is not None:
            density = np.where(self.mask, self.physical(params, beta), context)
        return _export_density(
            design,
            density,
            eps_bounds=self.eps_bounds,
            dx=self.pixel_size,
            fixed_solid=self.fixed_solid,
            fixed_void=self.fixed_void,
            **{**self.geometry, **geometry},
        )
