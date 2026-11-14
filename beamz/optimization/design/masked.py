"""Topology design over an explicit material mask."""

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np

from beamz.optimization.design.initialization import (
    Specification,
    UniformInitializationSpec,
)
from beamz.optimization.design.region import DensityParameterization
from beamz.optimization.design.validation import check_density, fixed_density_masks
from beamz.optimization.topology.projections import validate_projection_options
from beamz.optimization.topology.transforms.density import transform_density


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
