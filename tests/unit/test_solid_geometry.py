"""One geometry representation must preserve shape-specific physical operations."""

import dataclasses

import numpy as np
import pytest

from beamz.design import raster
from beamz.design.structures import (
    Box,
    Circle,
    CircularBend,
    Polygon,
    Rectangle,
    Ring,
    Solid,
    Sphere,
    Taper,
)


@pytest.mark.parametrize(
    "constructor", [Box, Sphere, Circle, CircularBend, Polygon, Rectangle, Ring, Taper]
)
def test_named_constructors_return_immutable_canonical_geometry(constructor):
    shape = constructor()
    assert type(shape) is Solid
    with pytest.raises(dataclasses.FrozenInstanceError):
        shape.depth = 2


@pytest.mark.parametrize("constructor", [Rectangle, Ring])
def test_position_z_is_preserved_when_no_override_is_requested(constructor):
    shape = constructor(position=(1, 2, 3), depth=0.4)
    assert shape.z == 3
    assert shape.shift(0, 0, 0.7).z == 3.7
    assert shape.updated_copy(z=5).position == (1, 2, 5)


def test_radial_scaling_preserves_extrusion_and_annular_mesh_feature():
    circle = Circle(radius=2, depth=0.7).scale(3)
    ring = Ring(inner_radius=1, outer_radius=2, depth=0.7).scale(3)
    assert circle.radius == 6
    assert circle.depth == ring.depth == 0.7
    assert ring._mesh_feature_sizes() == (3, 3, 0.7)
    assert circle.scale(1, 2).shape_kind == "polygon"


def test_rectangle_scaling_keeps_lower_corner_but_polygon_scales_about_its_center():
    rectangle = Rectangle(position=(1, 2, 3), width=2, height=4, depth=0.5)
    scaled = rectangle.scale(2)
    assert scaled.position == rectangle.position
    assert (scaled.width, scaled.height, scaled.depth) == (4, 8, 1)
    polygon = Polygon(rectangle.vertices, z=rectangle.z, depth=rectangle.depth)
    np.testing.assert_allclose(
        np.mean(polygon.scale(2).vertices, axis=0), np.mean(polygon.vertices, axis=0)
    )


@pytest.mark.parametrize(
    "axis,expected", [("x", (1, -3, 2)), ("y", (3, 2, -1)), ("z", (-2, 1, 3))]
)
def test_rotation_uses_right_handed_xyz_axes(axis, expected):
    triangle = Polygon(((1, 2, 3), (2, 2, 3), (1, 3, 3)))
    actual = triangle.rotate(90, axis=axis, point=(0, 0, 0))
    np.testing.assert_allclose(actual.vertices[0], expected, atol=1e-14)


def test_native_box_uses_literal_bounds_without_center_size_rounding():
    lower, upper = (-0.2, -1, 0.111), (0.2, -0.2, 0.113)
    box = raster.Box(lower, upper)
    assert type(box) is Solid
    assert box.to_dict()["bounds"] == {"min": lower, "max": upper}
    shifted = box.shift(1, 0, 0)
    np.testing.assert_allclose(shifted.lower, (0.8, -1, 0.111))
    assert box.lower == lower


def test_native_extrusion_uses_the_same_solid_and_preserves_holes_and_taper():
    footprint = raster.Polygon(
        ((0, 0), (2, 0), (2, 2), (0, 2)),
        (((0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)),),
    )
    shape = raster.TaperedExtrudedPolygon(footprint, 0.1, 0.3, 7, 0.25)
    assert type(shape) is type(footprint) is Solid
    payload = shape.to_dict()
    assert payload["type"] == "tapered_extruded_polygon"
    assert payload["polygon"]["holes"] == footprint.holes
    assert payload["sidewall_angle_degrees"] == 7
    assert payload["width_to_z"] == 0.25
