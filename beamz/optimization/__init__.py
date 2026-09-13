"""Inverse design with BeamZ objects, metre units, and selectable gradients."""

from . import adjoint_memmap, topology, utils
from .design import InverseDesign
from .objectives import ModePower, SoftMinModePower, WeightedObjective
from .optimizer import AdamOptimizer
from .polygonize import (
    density_to_polygons,
    density_to_shapely_geometry,
    shapely_geometry_to_polygons,
)
from .problem import TopologyProblem
from .projections import smoothed_heaviside, subpixel_smoothed_projection
from .region import (
    CustomInitializationSpec,
    ErosionDilationPenalty,
    FilterProject,
    RandomInitializationSpec,
    TopologyDesignRegion,
    UniformInitializationSpec,
)
from .result import InverseDesignResult, OptimizationStep, TopologyResult
from .schedules import LinearSchedule, OptimizationSchedule, StepSchedule
from .topology import TopologySpec, TopologyState

__all__ = [
    "LinearSchedule",
    "OptimizationSchedule",
    "StepSchedule",
    "AdamOptimizer",
    "CustomInitializationSpec",
    "ErosionDilationPenalty",
    "FilterProject",
    "InverseDesign",
    "InverseDesignResult",
    "OptimizationStep",
    "RandomInitializationSpec",
    "TopologyDesignRegion",
    "UniformInitializationSpec",
    "utils",
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
