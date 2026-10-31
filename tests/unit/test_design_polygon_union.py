"""Polygon unification preserves rasterized geometry and painter's order."""

import numpy as np
import pytest

import beamz as bz


@pytest.mark.parametrize(
    "case", ["overlap", "disjoint", "hole", "painter", "extrusion"]
)
def test_union_preserves_material_geometry(case):
    material = bz.Material(permittivity=4.0)
    first = bz.Polygon(
        vertices=((0.1, 0.1), (0.6, 0.1), (0.6, 0.6), (0.1, 0.6)), material=material
    )
    shift = 0.3 if case != "disjoint" else 0.65
    second = bz.Polygon(
        vertices=((shift, 0.3), (0.9, 0.3), (0.9, 0.9), (shift, 0.9)), material=material
    )
    shapes = [first, second]
    depth = 0.0
    if case == "hole":
        first = first.updated_copy(
            interiors=(((0.2, 0.2), (0.25, 0.2), (0.25, 0.25), (0.2, 0.25)),)
        )
        shapes = [first, second]
    elif case == "painter":
        blocker = bz.Rectangle(
            position=(0.4, 0.4),
            width=0.3,
            height=0.3,
            material=bz.Material(permittivity=9.0),
        )
        shapes.insert(1, blocker)
    elif case == "extrusion":
        depth = 1.0
        shapes = [
            first.updated_copy(z=0.1, depth=0.4),
            second.updated_copy(z=0.3, depth=0.5),
        ]
    design = bz.Design(width=1.0, height=1.0, depth=depth, structures=shapes)
    unified = design.unified_polygons()
    np.testing.assert_array_equal(
        design.rasterize(0.05).permittivity, unified.rasterize(0.05).permittivity
    )
    if case == "overlap":
        assert len(unified.structures) == 1
