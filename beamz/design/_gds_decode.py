"""Layout decoding shared by the Design facade and native scene importer.

Import policies (ports, extensions, stack priority and bias) belong to callers.
"""

import numpy as np

from beamz.design.materials import Material


def polygons_by_layer(component):
    try:
        return component.get_polygons_points(by="tuple")
    except TypeError:
        return component.get_polygons_points(by_spec=True)


def polygon_points(points, scale=1.0, offset=(0.0, 0.0)):
    return (np.asarray(points, dtype=float)[:, :2] - offset) * scale


def material(value):
    return value if isinstance(value, Material) else Material(float(value))


def material_lookup(defaults, overrides):
    return dict(defaults) | {
        str(name).strip().lower(): material(value)
        for name, value in (overrides or {}).items()
    }


def vertical_bounds(level, scale):
    lower = float(level.zmin) * scale
    return lower, lower + float(level.thickness) * scale
