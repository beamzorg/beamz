"""Sparse constituent data for causal diagonal Farjadpour interfaces."""

from dataclasses import dataclass

import numpy as np

from beamz.design.dispersion import PoleResidue
from beamz.devices._immutable import readonly_array


@dataclass(frozen=True)
class DispersiveInterface:
    component: str
    indices: np.ndarray
    fractions: np.ndarray
    normal_squared: np.ndarray
    materials: tuple

    def __post_init__(self):
        indices = np.asarray(self.indices, dtype=np.int64)
        fractions = np.asarray(self.fractions, dtype=float)
        normal = np.asarray(self.normal_squared, dtype=float)
        if self.component not in {"Ex", "Ey", "Ez"}:
            raise ValueError("Interface component must be electric.")
        if (
            indices.ndim != 1
            or np.any(indices < 0)
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError("Interface indices must be unique nonnegative integers.")
        if (
            fractions.shape != (len(self.materials), len(indices))
            or normal.shape != indices.shape
        ):
            raise ValueError("Interface arrays have incompatible shapes.")
        if (
            not np.isfinite(fractions).all()
            or np.any(fractions < 0)
            or not np.allclose(fractions.sum(axis=0), 1, atol=1e-8)
        ):
            raise ValueError("Interface fractions must sum to one.")
        if not np.isfinite(normal).all() or np.any((normal < 0) | (normal > 1)):
            raise ValueError("Interface normal weights must be in [0,1].")
        for m in self.materials:
            if (
                np.ndim(m.permittivity)
                or np.ndim(m.conductivity)
                or m.conductivity != 0
            ):
                raise ValueError(
                    "Polarized dispersive interfaces require scalar media without separate Ohmic conductivity; represent loss with poles."
                )
        if not any(isinstance(m, PoleResidue) for m in self.materials):
            raise ValueError(
                "A dispersive interface requires at least one pole-residue medium."
            )
        for name, value in (
            ("indices", indices),
            ("fractions", fractions),
            ("normal_squared", normal),
        ):
            object.__setattr__(self, name, readonly_array(value))
        object.__setattr__(self, "materials", tuple(self.materials))

    def cache_spec(self):
        return (
            self.component,
            self.indices,
            self.fractions,
            self.normal_squared,
            tuple(m.cache_spec() for m in self.materials),
        )
