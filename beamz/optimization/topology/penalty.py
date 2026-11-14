"""Erosion and dilation design penalties."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from beamz.optimization.design.initialization import FilterProject, Specification


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
