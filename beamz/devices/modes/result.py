"""Array-native mode family with labeled views at the public result boundary."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import xarray as xr

from .models import Grid


@dataclass(frozen=True)
class Result:
    """Solved modes and their exact supports.

    Fields use (frequency, mode, tangent, tangent) order, in the frame given
    by ``grid``. Solver, preview and Yee preparation consume these arrays
    directly. Labeled xarray views are constructed only when requested.
    """

    neffs: np.ndarray
    fields: dict[str, np.ndarray]
    grid: Grid
    frequencies: np.ndarray
    solver_info: dict[str, Any] | None = None

    @property
    def axes(self):
        return tuple(a for i, a in enumerate("xyz") if i != self.grid.normal_axis)

    @property
    def coordinates(self):
        return {
            axis: (edges[:-1] + edges[1:]) / 2
            for axis, edges in zip(
                self.axes,
                map(np.asarray, (self.grid.x_edges, self.grid.y_edges)),
                strict=True,
            )
        }

    def planes(self, axes=None):
        """Return component-major modes in the requested transverse axis order."""
        order = self.axes if axes is None else axes
        permutation = (0, *(1 + self.axes.index(a) for a in order))
        return {
            name: values[0].transpose(permutation)
            for name, values in self.fields.items()
        }

    @cached_property
    def n_complex(self) -> xr.DataArray:
        return xr.DataArray(
            self.neffs,
            dims=("f", "mode_index"),
            coords={
                "f": self.frequencies,
                "mode_index": np.arange(self.neffs.shape[1]),
            },
        )

    @cached_property
    def field_components(self) -> dict[str, xr.DataArray]:
        normal = "xyz"[self.grid.normal_axis]
        coords = {
            **self.coordinates,
            normal: [self.grid.normal_coordinate],
            "f": self.frequencies,
            "mode_index": np.arange(self.neffs.shape[1]),
            normal + "_width": (normal, np.ones(1)),
        }
        for axis, edges in zip(
            self.axes, (self.grid.x_edges, self.grid.y_edges), strict=True
        ):
            coords[axis + "_width"] = (axis, np.diff(edges))
        return {
            name: xr.DataArray(
                values.transpose(2, 3, 0, 1)[:, :, None],
                dims=(*self.axes, normal, "f", "mode_index"),
                coords=coords,
                attrs={"normal_dim": normal},
            )
            for name, values in self.fields.items()
        }

    @property
    def n_eff(self) -> xr.DataArray:
        return self.n_complex.real

    @property
    def k_eff(self) -> xr.DataArray:
        return self.n_complex.imag
