"""Design implementation."""

from itertools import groupby

from shapely.ops import unary_union

from beamz.design._geometry import planar_geometry
from beamz.design.structures import (
    Polygon,
)


def _material_key(material):
    if hasattr(material, "cache_spec"):
        return material.cache_spec()
    return (
        getattr(material, "permittivity", None),
        getattr(material, "permeability", None),
        getattr(material, "conductivity", None),
    )


def _merge_group_key(structure):
    """Group only shapes whose union cannot change extrusion or material."""
    return (
        _material_key(getattr(structure, "material", None)),
        getattr(structure, "z", None),
        getattr(structure, "depth", None),
        getattr(structure, "sidewall_angle", None),
        getattr(structure, "width_to_z", None),
    )


def _polygons_from_shapely(merged, material, first):
    if merged.geom_type not in {"Polygon", "MultiPolygon"}:
        return None
    depth = getattr(first, "depth", 0)
    z = getattr(first, "z", 0)
    geoms = (merged,) if merged.geom_type == "Polygon" else merged.geoms
    polygons = []
    for geom in geoms:
        shell = list(geom.exterior.coords[:-1])
        if len(shell) < 3:
            return None
        polygons.append(
            Polygon(
                vertices=shell,
                interiors=[list(ring.coords[:-1]) for ring in geom.interiors],
                material=material,
                depth=depth,
                z=z,
                sidewall_angle=getattr(first, "sidewall_angle", 0.0),
                width_to_z=getattr(first, "width_to_z", 0.0),
            )
        )
    return polygons


def _union_replacements(structures):
    """Map indexes of merged polygons to their replacements.

    Rings are left in place so a neighbor cannot fill their holes. Groups of
    one, and unions Shapely cannot turn back into polygons, stay unchanged.
    """
    replacements = {}
    # Merge only within one painter-order run of compatible materials/extrusions.
    for _, group in groupby(
        enumerate(structures), key=lambda item: _merge_group_key(item[1])
    ):
        mergeable = []
        for index, structure in group:
            if (
                not getattr(structure, "material", None)
                or structure.shape_kind == "ring"
            ):
                continue
            polygon = planar_geometry(structure)
            if polygon is not None and polygon.is_valid:
                mergeable.append((index, structure, polygon))
        if len(mergeable) <= 1:
            continue
        merged = _polygons_from_shapely(
            unary_union([item[2] for item in mergeable]),
            mergeable[0][1].material,
            mergeable[0][1],
        )
        if merged is None:
            continue
        replacements[mergeable[0][0]] = merged
        for index, _structure, _polygon in mergeable[1:]:
            replacements[index] = ()
    return replacements
