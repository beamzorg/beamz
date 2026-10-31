from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from beamz.design.materials import MaterialProtocol

DEFAULT_COLOR = "#6699cc"
ShapeKind = Literal[
    "polygon",
    "rectangle",
    "circle",
    "ring",
    "bend",
    "taper",
    "box",
    "sphere",
    "cylinder",
]


def _position(value, z=None):
    values = tuple(value)
    if len(values) == 2:
        values = (*values, 0.0)
    if len(values) != 3:
        raise ValueError("Position must be (x,y) or (x,y,z)")
    result = tuple(float(v) for v in values)
    if z is not None:
        result = (*result[:2], float(z))
    if not all(np.isfinite(result)):
        raise ValueError(f"Position must be finite, got {value!r}")
    return result


def _vertices(values, z=0.0, *, ccw=True):
    result = tuple(
        _position(vertex, z if len(vertex) == 2 else None) for vertex in values or ()
    )
    if ccw and len(result) >= 3:
        xy = np.asarray(result)[:, :2]
        area = np.sum(
            xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]
        )
        if area < 0:
            result = result[::-1]
    return result


def _positive(name, value, *, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}, got {value}")
    return value


def _arc(position, radius, angles):
    return tuple(
        (
            position[0] + radius * np.cos(t),
            position[1] + radius * np.sin(t),
            position[2],
        )
        for t in angles
    )


@dataclass(frozen=True, slots=True, eq=False)
class Solid:
    """One immutable geometry value shared by design and native raster scenes.

    Named constructors retain shape parameters for functional transforms and
    semantic mesh features. Exact primitives remain exact; polygon footprints
    retain holes, extrusion and sidewall reference independently of tessellation.
    Public coordinates use metres and XYZ order.
    """

    shape_kind: ShapeKind
    material: Any = None
    color: str = field(default=DEFAULT_COLOR, metadata={"beamz_cache": False})
    optimize: bool = False
    position: Any = (0.0, 0.0, 0.0)
    vertices: Any = ()
    interiors: Any = ()
    depth: float = 0.0
    z: Any = 0.0
    sidewall_angle: float = 0.0
    width_to_z: float = 0.0
    width: float = 0.0
    height: float = 0.0
    radius: float = 0.0
    points: int = 32
    inner_radius: float = 0.0
    outer_radius: float = 0.0
    angle: float = 90.0
    rotation: float = 0.0
    input_width: float = 0.0
    output_width: float = 0.0
    length: float = 0.0
    center: Any = (0.0, 0.0, 0.0)
    size: Any = (0.0, 0.0, 0.0)
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    is_pml: bool = False
    facecolor: str | None = None

    def __post_init__(self):
        assign = object.__setattr__
        if self.material is not None and not isinstance(
            self.material, MaterialProtocol
        ):
            raise TypeError("Solid.material must satisfy MaterialProtocol.")
        color = self.facecolor if self.shape_kind == "bend" else self.color
        assign(self, "color", DEFAULT_COLOR if color is None else str(color))
        for name in ("sidewall_angle", "width_to_z"):
            assign(self, name, float(getattr(self, name) or 0.0))
        assign(self, "optimize", bool(self.optimize))
        assign(self, "is_pml", bool(self.is_pml))
        kind = self.shape_kind
        if kind == "box":
            if self.bounds is not None:
                lower, upper = tuple(_position(point) for point in self.bounds)
                assign(self, "bounds", (lower, upper))
                center = tuple(
                    (lo + hi) / 2 for lo, hi in zip(lower, upper, strict=True)
                )
                size = tuple(hi - lo for lo, hi in zip(lower, upper, strict=True))
            else:
                center = _position(self.center)
                size = tuple(float(v) for v in self.size)
            if len(size) == 2:
                size = (*size, 0.0)
            if len(size) != 3 or any(np.isnan(v) or v < 0 for v in size):
                raise ValueError(f"Box sizes must be non-negative, got {size!r}.")
            assign(self, "center", center)
            assign(self, "size", size)
            assign(self, "position", self.lower)
            for name, value in zip(("width", "height", "depth"), size, strict=True):
                assign(self, name, value)
            assign(self, "z", self.lower[2])
            return
        if kind == "sphere":
            assign(self, "position", _position(self.position))
            assign(self, "center", self.position)
            assign(self, "radius", _positive("radius", self.radius))
            assign(self, "depth", 2 * self.radius)
            assign(self, "z", self.position[2] - self.radius)
            return
        requested_z = self.z
        z = float(self.z or 0.0)
        assign(self, "z", z)
        assign(self, "depth", _positive("depth", self.depth or 0.0, allow_zero=True))
        if kind == "polygon":
            assign(self, "vertices", _vertices(self.vertices, z))
            assign(
                self,
                "interiors",
                tuple(_vertices(path, z, ccw=False) for path in self.interiors or ()),
            )
            return
        position = _position(self.position, requested_z)
        assign(self, "z", position[2])
        assign(self, "position", position)
        assign(self, "center", position)
        dimensions = {
            "rectangle": ("width", "height"),
            "circle": ("radius",),
            "cylinder": ("radius",),
            "ring": ("inner_radius", "outer_radius"),
            "bend": ("inner_radius", "outer_radius"),
            "taper": ("input_width", "output_width", "length"),
        }
        if kind not in dimensions:
            raise ValueError(f"Unknown shape kind {kind!r}.")
        for name in dimensions[kind]:
            assign(self, name, _positive(name, getattr(self, name)))
        x, y, z = position
        if kind == "rectangle":
            vertices = (
                (x, y, z),
                (x + self.width, y, z),
                (x + self.width, y + self.height, z),
                (x, y + self.height, z),
            )
        elif kind == "taper":
            vertices = (
                (x, y - self.input_width / 2, z),
                (x + self.length, y - self.output_width / 2, z),
                (x + self.length, y + self.output_width / 2, z),
                (x, y + self.input_width / 2, z),
            )
        elif kind == "cylinder":
            return
        else:
            points = int(self.points)
            minimum = 2 if kind == "bend" else 3
            if points < minimum:
                raise ValueError(f"points must be at least {minimum}, got {points}")
            assign(self, "points", points)
            angles = np.linspace(
                0,
                np.radians(self.angle) if kind == "bend" else 2 * np.pi,
                points,
                endpoint=kind == "bend",
            )
            if kind == "bend":
                angles += np.radians(self.rotation)
            if kind == "circle":
                vertices = _arc(position, self.radius, angles)
            else:
                if self.outer_radius <= self.inner_radius:
                    raise ValueError(
                        f"outer_radius ({self.outer_radius}) must be greater than inner_radius ({self.inner_radius})"
                    )
                vertices = _arc(position, self.outer_radius, angles)
                inner = _arc(position, self.inner_radius, angles[::-1])
                if kind == "ring":
                    assign(self, "interiors", (inner,))
                else:
                    vertices += inner
        assign(self, "vertices", vertices)

    @property
    def lower(self):
        if self.bounds is not None:
            return self.bounds[0]
        return tuple(c - s / 2 for c, s in zip(self.center, self.size, strict=True))

    @property
    def upper(self):
        if self.bounds is not None:
            return self.bounds[1]
        return tuple(c + s / 2 for c, s in zip(self.center, self.size, strict=True))

    minimum = property(lambda self: self.lower)
    maximum = property(lambda self: self.upper)
    is_3d = property(lambda self: bool(self.depth))
    exterior = property(lambda self: tuple((x, y) for x, y, _ in self.vertices))
    holes = property(
        lambda self: tuple(tuple((x, y) for x, y, _ in path) for path in self.interiors)
    )
    polygon = property(lambda self: self)
    z_min = property(lambda self: self.z)
    z_max = property(lambda self: self.z + self.depth)
    sidewall_angle_degrees = property(lambda self: self.sidewall_angle)

    def updated_copy(self, **changes):
        if {"center", "size"} & changes.keys():
            changes["bounds"] = None
        if self.shape_kind == "bend" and "color" in changes:
            changes["facecolor"] = changes.pop("color")
        return replace(self, **changes)

    def with_material(self, material):
        return self.updated_copy(material=material)

    def copy(self):
        return self

    def shift(self, x, y, z=0):
        offset = (float(x), float(y), float(z))
        if self.shape_kind == "box":
            return self.updated_copy(
                center=tuple(c + d for c, d in zip(self.center, offset, strict=True))
            )
        if self.shape_kind != "polygon":
            position = tuple(c + d for c, d in zip(self.position, offset, strict=True))
            return self.updated_copy(position=position, z=position[2])

        def shifted(path):
            return tuple(
                tuple(c + d for c, d in zip(point, offset, strict=True))
                for point in path
            )

        return self.updated_copy(
            vertices=shifted(self.vertices),
            interiors=tuple(shifted(path) for path in self.interiors),
            z=self.z + offset[2],
        )

    def shifted(self, offset):
        return self.shift(*offset)

    def _as_polygon(self, **changes):
        values = dict(
            vertices=self.vertices,
            interiors=self.interiors,
            material=self.material,
            color=self.color,
            optimize=self.optimize,
            depth=self.depth,
            z=self.z,
            sidewall_angle=self.sidewall_angle,
            width_to_z=self.width_to_z,
        )
        values.update(changes)
        return Polygon(**values)

    def scale(self, s_x, s_y=None, s_z=None):
        kind = self.shape_kind
        if kind == "sphere":
            if any(
                v is not None and not np.isclose(float(s_x), float(v))
                for v in (s_y, s_z)
            ):
                raise ValueError("Sphere scaling must be uniform.")
            return self.updated_copy(radius=self.radius * float(s_x))
        if kind in {"circle", "ring", "bend"} and (
            s_y is None or np.isclose(float(s_x), float(s_y))
        ):
            names = (
                ("radius",) if kind == "circle" else ("inner_radius", "outer_radius")
            )
            return self.updated_copy(
                **{name: getattr(self, name) * float(s_x) for name in names}
            )
        s_x = float(s_x)
        s_y = s_x if s_y is None else float(s_y)
        s_z = (1.0 if s_y != s_x else s_x) if s_z is None else float(s_z)
        if kind == "rectangle":
            return self.updated_copy(
                width=self.width * s_x, height=self.height * s_y, depth=self.depth * s_z
            )
        if not self.vertices:
            return self
        center = np.mean(self.vertices, axis=0)

        def scaled(path):
            return tuple(
                map(tuple, center + (np.asarray(path) - center) * (s_x, s_y, s_z))
            )

        return self._as_polygon(
            vertices=scaled(self.vertices),
            interiors=tuple(scaled(path) for path in self.interiors),
            depth=self.depth * s_z,
        )

    def rotate(self, angle, axis="z", point=None):
        if (
            self.shape_kind == "bend"
            and axis == "z"
            and (point is None or _position(point) == self.position)
        ):
            return self.updated_copy(rotation=(self.rotation + float(angle)) % 360)
        if not self.vertices:
            return self
        if axis not in "xyz" or len(axis) != 1:
            raise ValueError(
                f"Invalid rotation axis {axis!r}; expected 'x', 'y', or 'z'."
            )
        center = (
            np.mean(self.vertices, axis=0)
            if point is None
            else np.asarray(_position(point))
        )
        first, second = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
        cosine, sine = (
            np.cos(np.radians(float(angle))),
            np.sin(np.radians(float(angle))),
        )

        def rotated(path):
            result = np.asarray(path, dtype=float).copy()
            a, b = result[:, first] - center[first], result[:, second] - center[second]
            result[:, first] = center[first] + a * cosine - b * sine
            result[:, second] = center[second] + a * sine + b * cosine
            return tuple(map(tuple, result))

        return self._as_polygon(
            vertices=rotated(self.vertices),
            interiors=tuple(rotated(path) for path in self.interiors),
        )

    def _mesh_feature_sizes(self):
        """Return optional axis-specific physical sizes for automatic meshing."""
        if self.shape_kind in {"box", "rectangle"}:
            return self.width or None, self.height or None, self.depth or None
        if self.shape_kind in {"circle", "sphere", "cylinder"}:
            return 2 * self.radius, 2 * self.radius, self.depth or None
        if self.shape_kind in {"ring", "bend"}:
            wall = self.outer_radius - self.inner_radius
            return wall, wall, self.depth or None
        if self.shape_kind == "taper":
            return (
                self.length,
                min(self.input_width, self.output_width),
                self.depth or None,
            )
        return None

    def to_rectangle(self, offset=(0.0, 0.0, 0.0), material=None):
        shifted = self.shifted(offset)
        if not np.isfinite((*shifted.lower, *shifted.size)).all():
            raise ValueError(
                "Infinite Box sizes must be clipped by Simulation(domain=...) before rasterization."
            )
        return Rectangle(
            position=shifted.lower,
            width=max(shifted.width, np.finfo(float).eps),
            height=max(shifted.height, np.finfo(float).eps),
            depth=max(shifted.depth, 0),
            material=self.material if material is None else material,
        )

    def to_dict(self):
        """Serialize exact geometry to the native scene ABI; material lives on Object."""
        if self.shape_kind == "box":
            return {"type": "box", "bounds": {"min": self.lower, "max": self.upper}}
        if self.shape_kind == "sphere":
            return {"type": "sphere", "center": self.center, "radius": self.radius}
        if self.shape_kind == "cylinder":
            return {
                "type": "cylinder",
                "center": self.center[:2],
                "radius": self.radius,
                "z_min": self.z,
                "z_max": self.z_max,
            }
        polygon = {"exterior": self.exterior, "holes": self.holes}
        if self.sidewall_angle:
            return dict(
                type="tapered_extruded_polygon",
                polygon=polygon,
                z_min=self.z,
                z_max=self.z_max,
                sidewall_angle_degrees=self.sidewall_angle,
                width_to_z=self.width_to_z,
            )
        return dict(
            type="extruded_polygon", polygon=polygon, z_min=self.z, z_max=self.z_max
        )


# Named constructors keep procedural parameters in the same immutable geometry.
Structure = Solid


def Polygon(vertices=(), **attributes):
    return Solid("polygon", vertices=vertices, **attributes)


def Rectangle(position=(0.0, 0.0, 0.0), width=1.0, height=1.0, depth=1.0, **attributes):
    return Solid(
        "rectangle",
        position=position,
        width=width,
        height=height,
        depth=depth,
        z=attributes.pop("z", None),
        **attributes,
    )


def Circle(position=(0.0, 0.0), radius=1.0, points=32, **attributes):
    return Solid(
        "circle", position=position, radius=radius, points=points, **attributes
    )


def Ring(
    position=(0.0, 0.0), inner_radius=1.0, outer_radius=2.0, points=256, **attributes
):
    return Solid(
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
    return Solid(
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
    return Solid(
        "taper",
        position=position,
        input_width=input_width,
        output_width=output_width,
        length=length,
        **attributes,
    )


def Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0), material=None):
    return Solid("box", material=material, center=center, size=size)


def Sphere(position=(0.0, 0.0, 0.0), radius=1.0, **attributes):
    return Solid("sphere", position=position, radius=radius, **attributes)
