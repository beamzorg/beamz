"""Trace-preserving modal data for objective callbacks."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ModeAmplitudes:
    """Represent differentiable modal amplitudes with named coordinates."""

    values: Any
    dims: tuple[str, ...]
    coords: Mapping[str, Any]

    @property
    def data(self):
        return self.values

    @property
    def shape(self):
        return self.values.shape

    def sel(self, **selectors):
        values, dims, coords = self.values, list(self.dims), dict(self.coords)
        for name, requested in selectors.items():
            if name not in dims:
                raise KeyError(f"Unknown or already selected dimension {name!r}.")
            axis = dims.index(name)
            coordinates = np.asarray(coords[name])
            scalar = np.ndim(requested) == 0
            indices = []
            for value in np.atleast_1d(requested):
                matches = np.flatnonzero(
                    np.isclose(coordinates, value, rtol=1e-10, atol=0)
                    if name == "f"
                    else coordinates == value
                )
                if len(matches) != 1:
                    raise KeyError(
                        f"Coordinate {name}={value!r} is not recorded exactly once."
                    )
                indices.append(int(matches[0]))
            values = jnp.take(
                values, indices[0] if scalar else jnp.asarray(indices), axis=axis
            )
            if scalar:
                dims.pop(axis)
                coords.pop(name)
            else:
                coords[name] = coordinates[indices]
        return ModeAmplitudes(values, tuple(dims), MappingProxyType(coords))


@dataclass(frozen=True)
class ModeData:
    """Pair modal amplitudes with their source power."""

    amps: ModeAmplitudes
    source_power: float = 1.0
