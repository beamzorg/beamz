"""Build exact polygon geometry from density pixels."""

from __future__ import annotations

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union


def _pixel_geometry(mask, *, x0, y0, dx, dy):
    """Union horizontal runs instead of allocating a polygon for every cell."""
    rectangles = []
    for row, values in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.pad(values.astype(np.int8), 1)))
        for start, stop in edges.reshape(-1, 2):
            rectangles.append(
                box(x0 + start * dx, y0 + row * dy, x0 + stop * dx, y0 + (row + 1) * dy)
            )
    return unary_union(rectangles)
