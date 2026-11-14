"""Canonical rectilinear grid and axis geometry."""

from .grid import (
    Axis,
    AxisGridQuality,
    Grid,
    GridAxis,
    GridQualityReport,
    RectilinearGrid,
)
from .grid import _axis_name as _axis_name
from .grid import _positive_counts as _positive_counts
from .grid import _readonly_edges as _readonly_edges
from .grid import _widths_are_equal as _widths_are_equal

__all__ = [
    "Axis",
    "AxisGridQuality",
    "Grid",
    "GridQualityReport",
    "RectilinearGrid",
]
