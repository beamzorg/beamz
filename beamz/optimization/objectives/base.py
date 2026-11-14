"""Composable modal objective interface."""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from beamz.optimization.objectives.mode import ModePower


class ModalObjective:
    """Composable scalar modal objective; all optimization objectives are maximized.

    Combine objectives with ``+``, ``-``, and multiplication by finite real weights.
    For example, ``transmission - 0.2 * reflection`` rewards transmission while
    penalizing reflection. A weighted combination uses one forward simulation.
    Its derivative uses one autodiff reverse pass or frequency-grouped adjoint
    simulations, according to the selected gradient backend.
    """

    def __add__(self, other):
        if not isinstance(other, ModalObjective):
            return NotImplemented
        from beamz.optimization.objectives.composite import WeightedObjective

        return WeightedObjective(
            (*self.terms, *other.terms), (*self.weights, *other.weights)
        )

    def __radd__(self, other):
        if isinstance(other, Real) and other == 0:
            return self
        return NotImplemented

    def __mul__(self, weight):
        if not isinstance(weight, Real):
            return NotImplemented
        if not np.isfinite(float(weight)):
            raise ValueError("Objective weights must be finite.")
        from beamz.optimization.objectives.composite import WeightedObjective

        return WeightedObjective(
            self.terms, tuple(float(weight) * w for w in self.weights)
        )

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        if not isinstance(other, ModalObjective):
            return NotImplemented
        return self + (-other)

    @property
    def terms(self) -> tuple[ModePower, ...]:
        raise NotImplementedError

    @property
    def weights(self) -> tuple[float, ...]:
        raise NotImplementedError

    def bind(self, results, program, cache=None):
        raise NotImplementedError
