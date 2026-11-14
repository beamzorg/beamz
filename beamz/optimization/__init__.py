"""Inverse design with BeamZ objects, metre units, and selectable gradients."""

from . import adjoint_memmap, topology, utils
from .design.geometry.density import density_to_shapely_geometry
from .design.geometry.polygons import density_to_polygons, shapely_geometry_to_polygons
from .design.initialization import (
    CustomInitializationSpec,
    FilterProject,
    RandomInitializationSpec,
    UniformInitializationSpec,
)
from .design.masked import MaskedDesignRegion
from .design.region import TopologyDesignRegion
from .design.spec import InverseDesign
from .objectives.composite import SoftMinModePower, WeightedObjective
from .objectives.mode import ModePower
from .optimizer.optimizer import Optimizer
from .optimizer.result import InverseDesignResult, OptimizationStep
from .optimizer.schedules import LinearSchedule, OptimizationSchedule, StepSchedule
from .topology.penalty import ErosionDilationPenalty
from .topology.projections import smoothed_heaviside, subpixel_smoothed_projection

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

# Checkpoints and pickled specifications can contain their original module names.
# Alias those names to their new homes so existing serialized objects still resolve.
import sys as _sys
from types import ModuleType as _ModuleType

from . import design as _region
from . import topology as _autodiff
from .design import geometry as _polygonize
from .design import validation as _validation
from .objectives import data as _data
from .objectives import projection as _modes
from .optimizer import checkpoint as _checkpoint
from .optimizer import problem as _problem
from .optimizer import result as _result
from .optimizer import schedules as _schedules
from .optimizer.runtime import load as _load
from .topology import projections as _projections

_data.bind_modal_data = _modes.bind_modal_data  # type: ignore[attr-defined]
_result.load_result = _load.load_result  # type: ignore[attr-defined]

for _name, _module in {
    "region": _region,
    "autodiff": _autodiff,
    "polygonize": _polygonize,
    "projections": _projections,
    "_validation": _validation,
    "_checkpoint": _checkpoint,
    "data": _data,
    "modes": _modes,
    "problem": _problem,
    "result": _result,
    "schedules": _schedules,
}.items():
    _legacy_name = f"{__name__}.{_name}"
    _legacy = _ModuleType(_legacy_name, _module.__doc__)
    _legacy.__dict__.update(
        {key: value for key, value in vars(_module).items() if not key.startswith("__")}
    )
    _sys.modules.setdefault(_legacy_name, _legacy)
