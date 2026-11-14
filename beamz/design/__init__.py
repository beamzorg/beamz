"""
Design module for BEAMZ - Contains components for designing photonic structures.
"""

import sys as _sys
from collections.abc import Mapping as _Mapping
from importlib import import_module as _import_module
from types import MappingProxyType as _MappingProxyType
from types import ModuleType as _ModuleType
from typing import Any as _Any


class _LegacyModule(_ModuleType):
    def __init__(
        self,
        name: str = "",
        default: str = "",
        symbols: _Mapping[str, str] = _MappingProxyType({}),
    ) -> None:
        super().__init__(name)
        self.__dict__["_default"] = default
        self.__dict__["_symbols"] = symbols

    def __getattr__(self, name: str = "") -> _Any:
        if name == "__all__":
            explicit = {
                "beamz.design.gds": [
                    "ImportedComponent",
                    "export_gds",
                    "import_component",
                    "import_gds",
                ],
                "beamz.design.grid_spec": ["GridSpec", "MeshOverride"],
            }
            if self.__name__ in explicit:
                return explicit[self.__name__]
            target = _import_module(self.__dict__["_default"])
            if hasattr(target, "__all__"):
                return target.__all__
            return [n for n in self.__dir__() if not n.startswith("_")]
        if name.startswith("__"):
            raise AttributeError(name)
        target = self.__dict__["_symbols"].get(name, self.__dict__["_default"])
        return getattr(_import_module(target), name)

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("__") or name in {"_default", "_symbols"}:
            super().__setattr__(name, value)
            return
        target = self.__dict__["_symbols"].get(name, self.__dict__["_default"])
        setattr(_import_module(target), name, value)

    def __delattr__(self, name: str) -> None:
        if name.startswith("__") or name in {"_default", "_symbols"}:
            super().__delattr__(name)
            return
        target = self.__dict__["_symbols"].get(name, self.__dict__["_default"])
        delattr(_import_module(target), name)

    def __dir__(self) -> list[str]:
        target = _import_module(self.__dict__["_default"])
        return sorted(
            set(super().__dir__()) | set(dir(target)) | set(self.__dict__["_symbols"])
        )


_LEGACY_MODULES = {
    "_gds_decode": ("beamz.design.io.decode", {}),
    "core": (
        "beamz.design.geometry.core",
        {
            "Design": "beamz.design.geometry.core",
            "_material_key": "beamz.design.geometry.union",
            "_merge_group_key": "beamz.design.geometry.union",
            "_polygons_from_shapely": "beamz.design.geometry.union",
            "_union_replacements": "beamz.design.geometry.union",
            "_DEFAULT_DOMAIN_SIZE": "beamz.design.geometry.core",
        },
    ),
    "structures": (
        "beamz.design.geometry.shapes",
        {
            "_position": "beamz.design.geometry.shapes",
            "_vertices": "beamz.design.geometry.shapes",
            "_positive": "beamz.design.geometry.shapes",
            "_arc": "beamz.design.geometry.shapes",
            "Shape": "beamz.design.geometry.shapes",
            "Polygon": "beamz.design.geometry.primitives",
            "Rectangle": "beamz.design.geometry.primitives",
            "Circle": "beamz.design.geometry.primitives",
            "Ring": "beamz.design.geometry.primitives",
            "CircularBend": "beamz.design.geometry.primitives",
            "Taper": "beamz.design.geometry.primitives",
            "Box": "beamz.design.geometry.primitives",
            "Sphere": "beamz.design.geometry.primitives",
            "DEFAULT_COLOR": "beamz.design.geometry.shapes",
            "ShapeKind": "beamz.design.geometry.shapes",
            "Structure": "beamz.design.geometry.shapes",
        },
    ),
    "_geometry": ("beamz.design.geometry.bounds", {}),
    "_material_layout": ("beamz.design.material.layout", {}),
    "discretization": ("beamz.design.grid.discretization", {}),
    "dispersion": ("beamz.design.material.dispersion", {}),
    "dispersive_interfaces": ("beamz.design.material.interfaces", {}),
    "gds": (
        "beamz.design.io.gds",
        {
            "ImportedComponent": "beamz.design.io.gds",
            "_gdsfactory": "beamz.design.io.gds",
            "_resolve_component": "beamz.design.io.gds",
            "_polygons_on_layer": "beamz.design.io.gds",
            "_stack_layers": "beamz.design.io.gds",
            "_vertical_profile": "beamz.design.io.gds",
            "import_component": "beamz.design.io.gds",
            "import_gds": "beamz.design.io.gds",
            "_StackLayer": "beamz.design.io.ports",
            "_VerticalProfile": "beamz.design.io.ports",
            "_component_ports": "beamz.design.io.ports",
            "_extend_ports": "beamz.design.io.ports",
            "_beamz_ports": "beamz.design.io.ports",
            "_material_key": "beamz.design.io.gds_export",
            "_add_polygon": "beamz.design.io.gds_export",
            "export_gds": "beamz.design.io.gds_export",
            "Layer": "beamz.design.io.gds",
        },
    ),
    "gdsfactory": ("beamz.design.io.gdsfactory", {}),
    "grid_spec": (
        "beamz.design.grid.spec",
        {
            "_xyz": "beamz.design.grid.spec",
            "MeshOverride": "beamz.design.grid.spec",
            "GridSpec": "beamz.design.grid.spec",
            "_validate_grid_budget": "beamz.design.grid.spec",
            "_validate_grid_shape_budget": "beamz.design.grid.spec",
            "_Region": "beamz.design.grid.features.constraints",
            "_MeshFeatureRegion": "beamz.design.grid.features.constraints",
            "_OpposingFeature": "beamz.design.grid.features.constraints",
            "_clamp_spacing": "beamz.design.grid.features.constraints",
            "_unique_coordinates": "beamz.design.grid.features.constraints",
            "_axis_constraints": "beamz.design.grid.features.constraints",
            "_polygon_opposing_features": "beamz.design.grid.features.polygons",
            "_merge_mesh_feature_regions": "beamz.design.grid.features.polygons",
            "_polygon_mesh_feature_regions": "beamz.design.grid.features.polygons",
            "_material_index": "beamz.design.grid.features.regions",
            "_design_extents": "beamz.design.grid.features.regions",
            "_feature_sizes": "beamz.design.grid.features.regions",
            "_overlap": "beamz.design.grid.features.regions",
            "_clip_region_bounds": "beamz.design.grid.features.regions",
            "_geometry_regions": "beamz.design.grid.features.regions",
            "_realize_uniform_grid": "beamz.design.grid.realize",
            "_design_coordinate_offset": "beamz.design.grid.realize",
            "_realize_graded_grid": "beamz.design.grid.realize",
            "AxisValues": "beamz.design.grid.spec",
            "OptionalAxisValues": "beamz.design.grid.spec",
            "_AXIS_NORMAL_THRESHOLD": "beamz.design.grid.features.polygons",
            "_EQUIVALENT_ENVELOPE_RATIO": "beamz.design.grid.features.polygons",
        },
    ),
    "mesher": ("beamz.design.grid.mesher", {}),
    "vector_fit": (
        "beamz.design.material.vector_fit",
        {
            "fit_nk_vector": "beamz.design.material.vector_fit",
            "_basis": "beamz.design.material.fit_core",
            "_expand": "beamz.design.material.fit_core",
            "_canonical_poles": "beamz.design.material.fit_core",
            "_least_squares": "beamz.design.material.fit_core",
            "_relocate": "beamz.design.material.fit_core",
            "_trial": "beamz.design.material.fit_core",
            "_audit_grid": "beamz.design.material.fit_core",
            "_local_minima": "beamz.design.material.fit_core",
            "_refined_minima": "beamz.design.material.fit_core",
            "_passive_residues": "beamz.design.material.fit_core",
        },
    ),
    "raster._regions": ("beamz.design.raster.internal.regions", {}),
    "raster.integration": ("beamz.design.raster.internal.integration", {}),
}
for _name, (_target, _symbols) in _LEGACY_MODULES.items():
    _legacy_name = f"{__name__}.{_name}"
    _legacy = _sys.modules.setdefault(
        _legacy_name, _LegacyModule(_legacy_name, _target, _symbols)
    )
    if "." not in _name:
        globals()[_name] = _legacy

from beamz.design.geometry.core import Design  # noqa: E402
from beamz.design.geometry.primitives import (  # noqa: E402
    Box,  # noqa: E402
    Circle,  # noqa: E402
    CircularBend,  # noqa: E402
    Polygon,  # noqa: E402
    Rectangle,  # noqa: E402
    Ring,  # noqa: E402
    Taper,  # noqa: E402
)
from beamz.design.geometry.shapes import Shape  # noqa: E402
from beamz.design.grid import (  # noqa: E402
    AxisGridQuality,
    Grid,
    GridQualityReport,
    RectilinearGrid,
)
from beamz.design.grid.discretization import (  # noqa: E402
    MaterialGrid,  # noqa: E402
    build_material_grid,  # noqa: E402
)
from beamz.design.grid.spec import (  # noqa: E402
    GridSpec,  # noqa: E402
    MeshOverride,  # noqa: E402
)
from beamz.design.io.gds import (  # noqa: E402
    ImportedComponent,  # noqa: E402
    import_component,  # noqa: E402
    import_gds,  # noqa: E402
)
from beamz.design.io.gds_export import export_gds  # noqa: E402
from beamz.design.io.gdsfactory import (  # noqa: E402
    ComponentSimulationResults,  # noqa: E402
    PortMetadata,  # noqa: E402
    PreparedComponent,  # noqa: E402
)
from beamz.design.io.gdsfactory import Settings as GDSFactorySettings  # noqa: E402
from beamz.design.io.gdsfactory import prepare as prepare_gdsfactory  # noqa: E402
from beamz.design.material.dispersion import (  # noqa: E402
    PoleResidue,  # noqa: E402
    fit_nk,  # noqa: E402
)
from beamz.design.material.vector_fit import fit_nk_vector  # noqa: E402
from beamz.design.materials import Material  # noqa: E402

__all__ = [
    "Material",
    "PoleResidue",
    "fit_nk",
    "fit_nk_vector",
    "Design",
    "MaterialGrid",
    "build_material_grid",
    "Box",
    "Rectangle",
    "Circle",
    "Ring",
    "Shape",
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
