"""
Design module for BEAMZ - Contains components for designing photonic structures.
"""

from beamz.design.core import Design
from beamz.design.discretization import MaterialGrid, build_material_grid
from beamz.design.dispersion import PoleResidue, fit_nk
from beamz.design.gds import ImportedComponent, export_gds, import_component, import_gds
from beamz.design.gdsfactory import (
    ComponentSimulationResults,
    PortMetadata,
    PreparedComponent,
)
from beamz.design.gdsfactory import (
    Settings as GDSFactorySettings,
)
from beamz.design.gdsfactory import (
    prepare as prepare_gdsfactory,
)
from beamz.design.grid import (
    AxisGridQuality,
    Grid,
    GridQualityReport,
    RectilinearGrid,
)
from beamz.design.grid_spec import GridSpec, MeshOverride
from beamz.design.materials import Material
from beamz.design.structures import (
    Box,
    Circle,
    CircularBend,
    Polygon,
    Rectangle,
    Ring,
    Taper,
)

__all__ = [
    "Material",
    "PoleResidue",
    "fit_nk",
    "Design",
    "MaterialGrid",
    "build_material_grid",
    "Box",
    "Rectangle",
    "Circle",
    "Ring",
    "CircularBend",
    "Polygon",
    "Taper",
    "GridSpec",
    "MeshOverride",
    "Grid",
    "RectilinearGrid",
    "AxisGridQuality",
    "GridQualityReport",
    "ImportedComponent",
    "import_component",
    "import_gds",
    "export_gds",
    "GDSFactorySettings",
    "ComponentSimulationResults",
    "PortMetadata",
    "PreparedComponent",
    "prepare_gdsfactory",
]
