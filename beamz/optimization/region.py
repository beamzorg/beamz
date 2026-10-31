"""Topology design regions, initialization, and transforms in metre units."""

from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import numpy as np

from beamz.devices._immutable import Replaceable

from ._validation import check_density, fixed_density_masks
from .autodiff import (
    _conic_filter,
    _conic_kernel,
    apply_fixed_density,
    transform_density,
)
from .projections import smoothed_heaviside, validate_projection_options


@dataclass(frozen=True)
class Specification(Replaceable):
    """Frozen optimization configuration with validated copying."""


@dataclass(frozen=True)
class UniformInitializationSpec(Specification):
    value: float = 0.5

    def __post_init__(self):
        if not np.isfinite(self.value) or not 0 <= self.value <= 1:
            raise ValueError("Initialization value must be finite and in [0, 1].")

    def create_parameters(self, shape):
        return np.full(shape, self.value, dtype=np.float32)


@dataclass(frozen=True)
class RandomInitializationSpec(Specification):
    min_value: float = 0.0
    max_value: float = 1.0
    seed: int | None = None

    def __post_init__(self):
        if not 0 <= self.min_value <= self.max_value <= 1:
            raise ValueError("Require 0 <= min_value <= max_value <= 1.")
        if self.seed is None:
            object.__setattr__(
                self, "seed", int(np.random.SeedSequence().generate_state(1)[0])
            )

    def create_parameters(self, shape):
        return (
            np.random.default_rng(self.seed)
            .uniform(self.min_value, self.max_value, shape)
            .astype(np.float32)
        )


@dataclass(frozen=True)
class FilterProject(Specification):
    """Conic filter with a radius rounded up to cells, then tanh projection.

    Radius is in metres. Reflection padding is local to the design region. This is a
    smooth transformation, not a minimum-feature-size guarantee.
    """

    radius: float
    beta: float = 1.0
    eta: float = 0.5
    padding: str = "reflect"

    def __post_init__(self):
        if (
            not np.isfinite([self.radius, self.beta, self.eta]).all()
            or self.radius < 0
            or self.beta <= 0
            or not 0 < self.eta < 1
        ):
            raise ValueError(
                "Require radius >= 0, beta > 0, and 0 < eta < 1, all finite."
            )
        if self.padding not in ("reflect", "edge"):
            raise ValueError("padding must be 'reflect' or 'edge'.")

    def evaluate(self, spatial_data, design_region_dl, *, beta=None):
        radius = self.radius / design_region_dl
        if radius > 0:
            pad = int(np.ceil(radius))
            if spatial_data.ndim not in (2, 3):
                raise ValueError("FilterProject requires a 2D or 3D density array.")
            spatial_data = _conic_filter(
                spatial_data,
                pad,
                self.padding,
                jnp.asarray(_conic_kernel(pad, spatial_data.ndim, np)),
            )
        # Convolution/projection can overshoot by one float32 ulp at a constant
        # zero/one region. Preserve the transformation's declared density range.
        return jnp.clip(
            smoothed_heaviside(
                spatial_data, self.beta if beta is None else beta, self.eta
            ),
            0,
            1,
        )

    __call__ = evaluate


@dataclass(frozen=True, eq=False)
class CustomInitializationSpec(Specification):
    params: Any

    def __post_init__(self):
        params = np.array(self.params, dtype=np.float32, copy=True)
        if not np.isfinite(params).all() or np.any((params < 0) | (params > 1)):
            raise ValueError("Initialization parameters must be finite in [0, 1].")
        params.setflags(write=False)
        object.__setattr__(self, "params", params)

    def create_parameters(self, shape):
        if self.params.shape != shape:
            raise ValueError("Custom initialization must match params_shape.")
        return self.params.copy()


@dataclass(frozen=True)
class ErosionDilationPenalty(Specification):
    """Weighted RMS difference of filtered closing and opening.

    Lengths are metres. The norm has 1e-12 smoothing at zero to give a finite
    zero subgradient. This is a soft penalty, not a fabrication constraint.
    """

    length_scale: float
    weight: float = 1.0
    beta: float = 100.0
    delta_eta: float = 0.01

    def __post_init__(self):
        if (
            not np.isfinite(
                [self.length_scale, self.weight, self.beta, self.delta_eta]
            ).all()
            or self.length_scale <= 0
            or self.weight < 0
            or self.beta <= 0
            or not 0 < self.delta_eta < 0.5
        ):
            raise ValueError(
                "Invalid erosion/dilation length, weight, beta, or threshold."
            )

    def evaluate(self, density, pixel_size):
        dilate = FilterProject(self.length_scale, self.beta, self.delta_eta)
        erode = FilterProject(self.length_scale, self.beta, 1 - self.delta_eta)
        opened = dilate(erode(density, pixel_size), pixel_size)
        closed = erode(dilate(density, pixel_size), pixel_size)
        return self.weight * (
            jnp.sqrt(jnp.mean((closed - opened) ** 2) + 1e-24) - 1e-12
        )

    __call__ = evaluate


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
class MaskedDesignRegion(Specification):
    """Arbitrary optimizable cells on the simulation's material grid.

    The full-grid parameters retain their outside-mask values during updates.
    Conic/morphology filters round radius to cells and use edge padding with
    fixed high-index geometry as boundary context. This differs deliberately
    from a rectangular FilterProject's reflect padding and rounded-up radius.
    Fixed solid/void cells must lie inside the mask; export clips to its exact
    cell union. Pixel size must equal the FDTD resolution for this region kind.
    """

    region_mask: Any
    pixel_size: float
    eps_bounds: tuple[float, float] = (1.0, 12.0)
    filter_radius: float = 0.0
    projection_eta: float = 0.5
    projection_type: str = "heaviside"
    ssp_smoothing_radius: float = 0.55
    filter_type: str = "conic"
    morphology_operation: str = "openclose"
    morphology_smooth_tau: float = 0.01
    fixed_solid: Any = field(default=None, repr=False, compare=False)
    fixed_void: Any = field(default=None, repr=False, compare=False)
    initialization_spec: Any = UniformInitializationSpec()
    penalties: tuple = ()
    penalty_input: str = field(default="density", metadata={"beamz_cache": False})

    def __post_init__(self):
        from beamz.devices._immutable import readonly_array

        mask = readonly_array(self.region_mask, dtype=bool)
        if mask.ndim not in (2, 3) or not np.any(mask):
            raise ValueError(
                "Topology mask must be nonempty and two- or three-dimensional."
            )
        if not np.isfinite(self.pixel_size) or self.pixel_size <= 0:
            raise ValueError("pixel_size must be finite and positive.")
        if (
            len(self.eps_bounds) != 2
            or not np.isfinite(self.eps_bounds).all()
            or not 0 < self.eps_bounds[0] < self.eps_bounds[1]
        ):
            raise ValueError("Require finite 0 < eps_bounds[0] < eps_bounds[1].")
        if not np.isfinite(self.filter_radius) or self.filter_radius < 0:
            raise ValueError("filter_radius must be finite and nonnegative.")
        if not 0 < self.projection_eta < 1:
            raise ValueError("projection_eta must lie strictly between zero and one.")
        validate_projection_options(self.projection_type, self.ssp_smoothing_radius)
        if not hasattr(self.initialization_spec, "create_parameters"):
            raise TypeError(
                "initialization_spec must provide create_parameters(shape)."
            )
        if self.penalty_input not in ("density", "parameters") or any(
            not callable(p) for p in self.penalties
        ):
            raise ValueError(
                "Use callable penalties and penalty_input=density or parameters."
            )
        solid, void = fixed_density_masks(self.fixed_solid, self.fixed_void, mask.shape)
        if any(p is not None and np.any(p & ~mask) for p in (solid, void)):
            raise ValueError("Fixed density cells must lie inside region_mask.")
        object.__setattr__(self, "region_mask", mask)
        object.__setattr__(self, "fixed_solid", solid)
        object.__setattr__(self, "fixed_void", void)
        object.__setattr__(self, "eps_bounds", tuple(self.eps_bounds))
        object.__setattr__(self, "penalties", tuple(self.penalties))

    @property
    def params_shape(self):
        return self.region_mask.shape

    @property
    def initial_parameters(self):
        values = self.initialization_spec.create_parameters(self.params_shape)
        values = np.where(self.region_mask, values, 0).astype(np.float32)
        self.check_params(values)
        return values

    def check_params(self, params):
        check_density(params, self.params_shape, "Parameters")

    def bind(self, simulation, grid):
        if self.region_mask.shape != grid.material_grid.shape:
            raise ValueError("Topology mask must match the simulation material grid.")
        if not np.isclose(self.pixel_size, simulation.resolution, rtol=1e-10, atol=0):
            raise ValueError("Topology and simulation resolution must agree.")
        if self.region_mask.ndim == 3 and (
            self.filter_radius or self.projection_type == "ssp"
        ):
            raise ValueError(
                "3D masked regions require zero filter radius and no SSP; use TopologyDesignRegion for 3D transforms."
            )
        low, high = self.eps_bounds
        context = np.asarray(grid.material_grid.permittivity) >= low + 0.9 * (
            high - low
        )
        context &= ~self.region_mask
        context.setflags(write=False)
        transform = partial(
            transform_density,
            mask=self.region_mask,
            eta=self.projection_eta,
            radius=int(round(self.filter_radius / self.pixel_size)),
            filter_type=self.filter_type,
            morphology_operation=self.morphology_operation,
            morphology_tau=self.morphology_smooth_tau,
            fixed_structure_mask=context,
            projection_type=self.projection_type,
            ssp_smoothing_radius=self.ssp_smoothing_radius,
            fixed_solid=self.fixed_solid,
            fixed_void=self.fixed_void,
        )
        return DensityParameterization(
            self.params_shape,
            self.pixel_size,
            self.eps_bounds,
            transform,
            self.fixed_solid,
            self.fixed_void,
            self.region_mask,
            configuration=self,
            context=context,
            geometry=dict(
                origin=grid.material_grid.origin,
                background=self.region_mask,
                region_mask=self.region_mask,
            ),
        )

    def penalty_value(self, params, *, beta=None, density_map=None):
        if not self.penalties:
            return 0.0
        assert density_map is not None
        values = (
            params
            if self.penalty_input == "parameters"
            else density_map.physical(params, beta)
        )
        return sum(p(values, self.pixel_size) for p in self.penalties)


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
        from .polygonize import _export_density

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
