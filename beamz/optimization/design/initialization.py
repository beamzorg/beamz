"""Design-region initialization and filter specifications."""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from beamz.devices._immutable import Replaceable
from beamz.optimization.topology.projections import smoothed_heaviside
from beamz.optimization.topology.transforms.conic import _conic_filter, _conic_kernel


@dataclass(frozen=True)
class Specification(Replaceable):
    """Frozen optimization configuration with validated copying."""


@dataclass(frozen=True)
class UniformInitializationSpec(Specification):
    """Initialize every design parameter to the same density."""

    value: float = 0.5

    def __post_init__(self):
        if not np.isfinite(self.value) or not 0 <= self.value <= 1:
            raise ValueError("Initialization value must be finite and in [0, 1].")

    def create_parameters(self, shape):
        return np.full(shape, self.value, dtype=np.float32)


@dataclass(frozen=True)
class RandomInitializationSpec(Specification):
    """Initialize design parameters uniformly within a seeded interval."""

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
    """Initialize design parameters from an immutable density array."""

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
