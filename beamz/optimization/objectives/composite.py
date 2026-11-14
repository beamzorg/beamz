"""Weighted and worst-case modal objectives."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from beamz.optimization.objectives.base import ModalObjective
from beamz.optimization.objectives.mode import ModePower


@dataclass(frozen=True)
class SoftMinModePower(ModePower):
    """Smooth lower bound on the weakest selected modal transmission.

    Reduce the normalized spectrum ``p`` as
    ``-temperature * logsumexp(-p / temperature)``. For N frequencies this
    lies between ``min(p) - temperature * log(N)`` and ``min(p)``. There is
    no division by N inside the logarithm. A smaller temperature focuses the
    derivative more strongly on the weakest frequencies. ``spectra`` still
    returns the unreduced physical powers.
    """

    temperature: float = 0.03

    def __post_init__(self):
        super().__post_init__()
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, Real)
            or not np.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("temperature must be a finite positive scalar.")
        object.__setattr__(self, "temperature", float(self.temperature))

    def _reduce(self, values):
        return -self.temperature * logsumexp(-values / self.temperature)


@dataclass(frozen=True, init=False)
class WeightedObjective(ModalObjective):
    """A weighted sum of reduced modal powers, evaluated in one FDTD solve.

    Negative weights penalize unwanted powers. Weights are not automatically
    normalized, and no power values are clipped. Usually construct this through
    objective arithmetic, e.g. ``0.5 * (short_to_top + long_to_bottom) - leakage``.
    """

    _terms: tuple[ModePower, ...]
    _weights: tuple[float, ...]

    def __init__(self, terms, weights):
        terms, weights = tuple(terms), tuple(float(w) for w in weights)
        if not terms or not all(isinstance(t, ModePower) for t in terms):
            raise ValueError("WeightedObjective requires nonempty ModePower terms.")
        if len(terms) != len(weights) or not np.isfinite(weights).all():
            raise ValueError("Objective weights must be finite and match the terms.")
        object.__setattr__(self, "_terms", terms)
        object.__setattr__(self, "_weights", weights)

    @property
    def terms(self):
        return self._terms

    @property
    def weights(self):
        return self._weights

    def __call__(self, data):
        return jnp.sum(
            jnp.asarray(self.weights) * jnp.stack([term(data) for term in self.terms])
        )

    def canonical_spec(self):
        return self.terms, self.weights

    def bind(self, results, program, cache=None):
        cache = {} if cache is None else cache
        objectives = [term.bind(results, program, cache) for term in self.terms]
        weights = jnp.asarray(self.weights)
        return lambda state: jnp.sum(
            weights * jnp.stack([objective(state) for objective in objectives])
        )
