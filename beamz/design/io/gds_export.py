"""Design implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from beamz.const import µm
from beamz.design._gds_decode import material as _material  # noqa: F401
from beamz.design._geometry import polygon_rings
from beamz.design.core import Design
from beamz.design.io.gds import Layer, _gdsfactory
from beamz.design.materials import Material


def _material_key(structure: Any) -> tuple[float, float, float] | None:
    material = getattr(structure, "material", None)
    if material is None:
        return None
    return (
        float(getattr(material, "permittivity", 1.0)),
        float(getattr(material, "permeability", 1.0)),
        float(getattr(material, "conductivity", 0.0)),
    )


def _add_polygon(cell, structure, layer_index, *, kdb, precision):
    rings = polygon_rings(structure)
    if not rings[0]:
        raise TypeError(f"GDS export does not support {type(structure).__name__}.")
    points = []
    for ring in rings:
        xy = np.asarray(ring, dtype=float)[:, :2]
        if not np.isfinite(xy).all():
            raise ValueError("GDS coordinates must be finite.")
        if np.any(np.abs(xy) > precision * (np.iinfo(np.int32).max - 1)):
            raise ValueError("GDS precision is too fine for the exported coordinates.")
        points.append([kdb.DPoint(float(x / µm), float(y / µm)) for x, y in xy])
    polygon = kdb.DPolygon(points[0])
    for hole in points[1:]:
        polygon.insert_hole(hole)
    cell.shapes(layer_index).insert(polygon)


def export_gds(
    design: Design,
    path: str | Path,
    *,
    cell_name: str = "beamz_design",
    precision: float = 1e-9,
    layer_map: Mapping[Material, Layer] | None = None,
) -> Path:
    """Export planar structures to GDS with an explicit coordinate precision.

    ``precision`` is the database grid in metres (default 1 nm). A private
    KLayout database avoids quantization through gdsfactory's global grid.
    Coordinates remain in GDS micrometres. Interior rings remain holes.

    By default, equal materials share automatically assigned layers. An explicit
    ``layer_map={material: (layer, datatype)}`` selects only those materials,
    e.g. the core mask without low-index region-clearing polygons. Volumetric
    structures without a planar projection are rejected explicitly.
    """
    if not np.isfinite(precision) or precision <= 0:
        raise ValueError("precision must be finite and positive (metres).")
    if layer_map is not None:
        for layer in layer_map.values():
            if len(layer) != 2 or any(
                isinstance(v, bool)
                or not isinstance(v, (int, np.integer))
                or not 0 <= v <= 65535
                for v in layer
            ):
                raise ValueError("GDS layer/datatype must be integers in [0, 65535].")
    gf = _gdsfactory(activate_pdk=False)
    layout = gf.kdb.Layout()
    layout.dbu = precision / µm
    cell = layout.create_cell(cell_name)
    material_layers: dict[tuple[float, float, float], Layer] = {}
    count = 0
    for structure in design.unified_polygons().structures:
        if getattr(structure, "is_pml", False):
            continue
        if (
            layer_map is not None
            and getattr(structure, "material", None) not in layer_map
        ):
            continue
        if structure.shape_kind == "box":
            structure = structure.to_rectangle()
        material_key = _material_key(structure)
        if material_key is None:
            continue
        layer = (
            layer_map[structure.material]
            if layer_map is not None
            else material_layers.setdefault(material_key, (len(material_layers), 0))
        )
        _add_polygon(
            cell, structure, layout.layer(*layer), kdb=gf.kdb, precision=precision
        )
        count += 1
    if not count:
        raise ValueError("The design has no planar material structures to export.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    layout.write(str(destination))
    return destination
