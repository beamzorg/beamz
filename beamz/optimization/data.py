"""Small, trace-preserving modal-data interface for user objective functions.

This deliberately implements named selection, not the complete xarray API.
Selecting static coordinates never converts the differentiable values to NumPy.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from beamz.devices.monitors import ModeMonitor

from .objectives import ModePower


@dataclass(frozen=True)
class ModeAmplitudes:
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
    amps: ModeAmplitudes


def bind_modal_data(results, program, names, cache=None):
    """Bind named complex amplitudes; every mode/direction stays differentiable."""
    cache = {} if cache is None else cache
    plans = []
    for name in names:
        monitor = results.monitors[name].monitor
        if not isinstance(monitor, ModeMonitor):
            raise ValueError(
                "Differentiable simulation data currently supports ModeMonitor only."
            )
        frequencies = tuple(float(f) for f in monitor.freqs)
        amplitudes = tuple(
            tuple(
                ModePower(name, mode_index=m, direction=d)._bind_amplitude(
                    results, program, frequencies, cache
                )
                for m in range(monitor.mode_spec.num_modes)
            )
            for d in ("+", "-")
        )
        coords = MappingProxyType(
            {
                "direction": ("+", "-"),
                "f": frequencies,
                "mode_index": tuple(range(monitor.mode_spec.num_modes)),
            }
        )
        plans.append((name, amplitudes, coords))

    def data(state):
        return MappingProxyType(
            {
                name: ModeData(
                    ModeAmplitudes(
                        jnp.stack(
                            [
                                jnp.stack(
                                    [amplitude(state) for amplitude in direction],
                                    axis=-1,
                                )
                                for direction in amplitudes
                            ]
                        ),
                        ("direction", "f", "mode_index"),
                        coords,
                    )
                )
                for name, amplitudes, coords in plans
            }
        )

    return data
