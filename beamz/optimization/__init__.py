"""Inverse design with BeamZ objects, metre units, and selectable gradients."""

from . import adjoint_memmap, topology, utils
from .design import InverseDesign
from .objectives import ModePower, SoftMinModePower, WeightedObjective
from .optimizer import Optimizer
from .polygonize import (
    density_to_polygons,
    density_to_shapely_geometry,
    shapely_geometry_to_polygons,
)
from .projections import smoothed_heaviside, subpixel_smoothed_projection
from .region import (
    CustomInitializationSpec,
    ErosionDilationPenalty,
    FilterProject,
    MaskedDesignRegion,
    RandomInitializationSpec,
    TopologyDesignRegion,
    UniformInitializationSpec,
)
from .result import InverseDesignResult, OptimizationStep
from .schedules import LinearSchedule, OptimizationSchedule, StepSchedule

__all__ = [
    "LinearSchedule",
    "OptimizationSchedule",
    "StepSchedule",
    "Optimizer",
    "CustomInitializationSpec",
    "ErosionDilationPenalty",
    "FilterProject",
    "InverseDesign",
    "InverseDesignResult",
    "OptimizationStep",
    "RandomInitializationSpec",
    "TopologyDesignRegion",
    "MaskedDesignRegion",
    "UniformInitializationSpec",
    "utils",
    "ModePower",
    "SoftMinModePower",
    "WeightedObjective",
    "topology",
    "adjoint_memmap",
    "smoothed_heaviside",
    "subpixel_smoothed_projection",
    "density_to_shapely_geometry",
    "shapely_geometry_to_polygons",
    "density_to_polygons",
]
