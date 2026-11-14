"""Construct canonical photonic geometry primitives."""

from __future__ import annotations

from beamz.design.geometry.shapes import Shape


def Polygon(vertices=(), **attributes):
    return Shape("polygon", vertices=vertices, **attributes)


def Rectangle(position=(0.0, 0.0, 0.0), width=1.0, height=1.0, depth=1.0, **attributes):
    return Shape(
        "rectangle",
        position=position,
        width=width,
        height=height,
        depth=depth,
        z=attributes.pop("z", None),
        **attributes,
    )


def Circle(position=(0.0, 0.0), radius=1.0, points=32, **attributes):
    return Shape(
        "circle", position=position, radius=radius, points=points, **attributes
    )


def Ring(
    position=(0.0, 0.0), inner_radius=1.0, outer_radius=2.0, points=256, **attributes
):
    return Shape(
        "ring",
        position=position,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        points=points,
        z=attributes.pop("z", None),
        **attributes,
    )


def CircularBend(
    position=(0.0, 0.0),
    inner_radius=1.0,
    outer_radius=2.0,
    angle=90.0,
    rotation=0.0,
    points=64,
    **attributes,
):
    return Shape(
        "bend",
        position=position,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        angle=angle,
        rotation=rotation,
        points=points,
        **attributes,
    )


def Taper(
    position=(0.0, 0.0), input_width=1.0, output_width=0.5, length=1.0, **attributes
):
    return Shape(
        "taper",
        position=position,
        input_width=input_width,
        output_width=output_width,
        length=length,
        **attributes,
    )


def Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), material=None):
    return Shape("box", material=material, center=center, size=size)


def Sphere(position=(0.0, 0.0, 0.0), radius=1.0, **attributes):
    return Shape("sphere", position=position, radius=radius, **attributes)
