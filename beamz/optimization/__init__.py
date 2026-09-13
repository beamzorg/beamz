"""Adjoint-based optimization helpers for BEAMZ."""

from . import adjoint_memmap, topology
from .objectives import ModePower, SoftMinModePower, WeightedObjective
from .polygonize import (
    density_to_polygons,
    density_to_shapely_geometry,
    shapely_geometry_to_polygons,
)
from .problem import TopologyProblem
from .projections import smoothed_heaviside, subpixel_smoothed_projection
from .result import TopologyResult
from .topology import TopologySpec, TopologyState

__all__ = [
    "ModePower",
    "SoftMinModePower",
    "WeightedObjective",
    "TopologyProblem",
    "TopologyResult",
    "topology",
    "adjoint_memmap",
    "smoothed_heaviside",
    "subpixel_smoothed_projection",
    "density_to_shapely_geometry",
    "shapely_geometry_to_polygons",
    "density_to_polygons",
    "TopologySpec",
    "TopologyState",
]
