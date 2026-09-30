"""Uniform primal and dual metrics for the initial FIT backend."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class UniformFITMesh:
    """Complete uniform grid with PEC walls and origin at zero.

    ``shape`` counts cells in (y, x) or (z, y, x) order; ``spacing`` is
    metres. In 2D, integrated quantities use a one-metre extrusion along z,
    and energies are numerically energy per unit length in J/m.
    """

    shape: tuple[int, ...]
    spacing: float
    polarization: str = "tm"

    def __post_init__(self):
        shape = tuple(self.shape)
        if len(shape) not in (2, 3) or any(
            isinstance(n, bool) or not isinstance(n, Integral) or n < 2 for n in shape
        ):
            raise ValueError("shape must contain two or three integer cell counts >= 2")
        if not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("spacing must be finite and positive")
        if self.polarization not in ("te", "tm"):
            raise ValueError("polarization must be 'te' or 'tm'")
        object.__setattr__(self, "shape", tuple(int(n) for n in shape))
        object.__setattr__(self, "spacing", float(self.spacing))

    @property
    def dimension(self):
        return len(self.shape)

    @property
    def electric_shapes(self):
        if self.dimension == 3:
            nz, ny, nx = self.shape
            return {
                "Ex": (nz + 1, ny + 1, nx),
                "Ey": (nz + 1, ny, nx + 1),
                "Ez": (nz, ny + 1, nx + 1),
            }
        ny, nx = self.shape
        if self.polarization == "tm":
            return {"Ez": (ny + 1, nx + 1)}
        return {"Ex": (ny + 1, nx), "Ey": (ny, nx + 1)}

    @property
    def magnetic_shapes(self):
        if self.dimension == 3:
            nz, ny, nx = self.shape
            return {
                "Bx": (nz, ny, nx + 1),
                "By": (nz, ny + 1, nx),
                "Bz": (nz + 1, ny, nx),
            }
        ny, nx = self.shape
        if self.polarization == "tm":
            return {"Bx": (ny, nx + 1), "By": (ny + 1, nx)}
        return {"Bz": (ny, nx)}

    def _axis(self, component):
        return {"x": self.dimension - 1, "y": self.dimension - 2, "z": 0}[
            component[-1].lower()
        ]

    def edge_length(self, component):
        return 1.0 if self.dimension == 2 and component == "Ez" else self.spacing

    def face_area(self, component):
        return (
            self.spacing
            if self.dimension == 2 and component != "Bz"
            else self.spacing**2
        )

    def dual_lengths(self, n):
        result = np.full(n + 1, self.spacing)
        result[[0, -1]] *= 0.5
        return result

    def dual_area(self, component):
        result = np.ones(self.electric_shapes[component])
        axis = self._axis(component)
        for i, n in enumerate(self.shape):
            if self.dimension == 3 or component != "Ez":
                if i == axis:
                    continue
            broadcast = [1] * self.dimension
            broadcast[i] = n + 1
            result *= self.dual_lengths(n).reshape(broadcast)
        return result

    def dual_edge_length(self, component):
        if self.dimension == 2 and component == "Bz":
            return np.ones(self.magnetic_shapes[component])
        axis = self._axis(component)
        broadcast = [1] * self.dimension
        broadcast[axis] = self.shape[axis] + 1
        return np.broadcast_to(
            self.dual_lengths(self.shape[axis]).reshape(broadcast),
            self.magnetic_shapes[component],
        )

    def pec_mask(self, component):
        """True on tangential electric edges constrained by PEC."""
        mask = np.zeros(self.electric_shapes[component], dtype=bool)
        axis = self._axis(component)
        for i in range(self.dimension):
            if component != "Ez" or self.dimension == 3:
                if i == axis:
                    continue
            low, high = [slice(None)] * self.dimension, [slice(None)] * self.dimension
            low[i], high[i] = 0, -1
            mask[tuple(low)] = True
            mask[tuple(high)] = True
        return mask

    def coordinates(self, component):
        """Native field coordinates in metres, returned in storage axis order."""
        magnetic = component.startswith(("B", "H"))
        key = "B" + component[-1] if magnetic else component
        shapes = self.magnetic_shapes if magnetic else self.electric_shapes
        if key not in shapes:
            raise ValueError(f"Inactive field component: {component}")
        axis = self._axis(key)
        coords = []
        for i, n in enumerate(shapes[key]):
            if self.dimension == 2 and key[-1] == "z":
                offset = 0.5 if magnetic else 0.0
            else:
                offset = 0.5 if (i != axis if magnetic else i == axis) else 0.0
            coords.append((np.arange(n) + offset) * self.spacing)
        return tuple(coords)
