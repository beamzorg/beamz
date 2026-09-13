"""Topology design regions, initialization, and transforms in metre units."""

from dataclasses import dataclass, field, replace
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import convolve2d

from ._validation import check_density
from .projections import smoothed_heaviside


@dataclass(frozen=True)
class Specification:
    def updated_copy(self, **changes):
        return replace(self, **changes)


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
            y, x = np.mgrid[-pad : pad + 1, -pad : pad + 1]
            kernel = np.maximum(0, 1 - np.hypot(x, y) / pad)
            kernel /= kernel.sum()
            spatial_data = convolve2d(
                jnp.pad(spatial_data, pad, mode=self.padding),
                jnp.asarray(kernel),
                mode="valid",
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
    """Planar density region in the simulation's public coordinates.

    All lengths are metres. Parameters have BeamZ's (y, x) order and are
    uniform along z through the specified thickness in 3D simulations. Bounds
    must align to the grid, with pixel_size == resolution.
    """

    size: tuple[float, float, float]
    center: tuple[float, float, float]
    eps_bounds: tuple[float, float]
    pixel_size: float
    transformations: tuple = ()
    penalties: tuple = ()
    initialization_spec: Any = UniformInitializationSpec()
    penalty_input: str = field(default="density", metadata={"beamz_cache": False})

    def __post_init__(self):
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
        cells = np.asarray(self.size[:2]) / self.pixel_size
        if not np.allclose(cells, np.round(cells), rtol=0, atol=1e-6):
            raise ValueError(
                "Design-region sizes must be integer multiples of pixel_size."
            )
        if not hasattr(self.initialization_spec, "create_parameters"):
            raise TypeError(
                "initialization_spec must provide create_parameters(shape)."
            )
        if any(not callable(t) for t in self.transformations + self.penalties):
            raise TypeError("Transformations and penalties must be callable.")

    @property
    def params_shape(self):
        return tuple(int(round(size / self.pixel_size)) for size in self.size[1::-1])

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
        for transform in self.transformations:
            params = (
                transform.evaluate(params, self.pixel_size, beta=beta)
                if isinstance(transform, FilterProject)
                else transform(params, self.pixel_size)
            )
        return params

    def eps_values(self, params, *, beta=None):
        low, high = self.eps_bounds
        return low + (high - low) * self.material_density(params, beta=beta)

    def penalty_value(self, params, *, beta=None):
        if not self.penalties:
            return 0.0
        density = (
            jnp.asarray(params)
            if self.penalty_input == "parameters"
            else self.material_density(params, beta=beta)
        )
        return sum(penalty(density, self.pixel_size) for penalty in self.penalties)
