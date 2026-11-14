"""Physical inverse-design specifications and parameter regions."""

from ..topology.penalty import ErosionDilationPenalty
from .initialization import (
    CustomInitializationSpec,
    FilterProject,
    RandomInitializationSpec,
    Specification,
    UniformInitializationSpec,
)
from .masked import MaskedDesignRegion
from .region import DensityParameterization, TopologyDesignRegion
from .spec import InverseDesign
