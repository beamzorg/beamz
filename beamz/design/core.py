from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from beamz._cache_tokens import cache_token
from beamz.const import µm
from beamz.design.materials import Material, MaterialProtocol
from beamz.design.structures import (
    Polygon,
    Ring,
)

_DEFAULT_DOMAIN_SIZE = object()


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


def _to_shapely(structure):
    vertices = getattr(structure, "vertices", ())
    if not vertices:
        return None
    holes = [list(path) for path in getattr(structure, "interiors", ()) if path]
    polygon = (
        ShapelyPolygon(shell=vertices, holes=holes)
        if holes
        else ShapelyPolygon(shell=vertices)
    )
    return polygon if polygon.is_valid else None


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
    groups = {}
    for index, structure in enumerate(structures):
        if not getattr(structure, "material", None):
            continue
        polygon = _to_shapely(structure)
        if polygon is None:
            continue
        groups.setdefault(_merge_group_key(structure), []).append(
            (index, structure, polygon)
        )
    replacements = {}
    for group in groups.values():
        mergeable = [item for item in group if not isinstance(item[1], Ring)]
        if len(group) <= 1 or len(mergeable) <= 1:
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


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Design:
    """Immutable geometry in metres, ``(x, y, z)``, painter's order.

    ``background`` fills uncovered cells. ``material`` is the same argument;
    passing both with different values is an error. ``design += structure``
    rebinds to the design returned by :meth:`with_structure`. Omitting the
    domain size and passing only ``background`` marks the design as centered.
    """

    width: float
    height: float
    depth: float
    background: Any
    structures: tuple[Any, ...]
    _centered_coordinates: bool = field(
        default=False, repr=False, compare=False, metadata={"beamz_cache": False}
    )

    def __init__(
        self,
        width: float | object = _DEFAULT_DOMAIN_SIZE,
        height: float | object = _DEFAULT_DOMAIN_SIZE,
        depth: float | object = _DEFAULT_DOMAIN_SIZE,
        material: Any = None,
        background: Any = None,
        structures=(),
    ):
        """Create a design domain with specified dimensions and background material."""
        width_is_default = width is _DEFAULT_DOMAIN_SIZE
        height_is_default = height is _DEFAULT_DOMAIN_SIZE
        depth_is_default = depth is _DEFAULT_DOMAIN_SIZE
        background_arg = background
        if width_is_default:
            width = 4 * µm
        if height_is_default:
            height = 4 * µm
        if depth_is_default:
            depth = 0
        explicit_size = not (
            width_is_default and height_is_default and depth_is_default
        )
        if (
            background_arg is not None
            and material is not None
            and background_arg is not material
        ):
            raise ValueError("Pass only one of background=... or material=....")
        if background_arg is not None:
            material = background_arg
        if material is None:
            material = Material(permittivity=1.0, permeability=1.0, conductivity=0.0)
        if not isinstance(material, MaterialProtocol):
            raise TypeError("Design background must satisfy MaterialProtocol.")

        width = float(cast(float, width))
        height = float(cast(float, height))
        depth = float(cast(float, depth))
        if not np.isfinite(width) or width <= 0:
            raise ValueError("Design width must be finite and positive.")
        if not np.isfinite(height) or height <= 0:
            raise ValueError("Design height must be finite and positive.")
        if not np.isfinite(depth) or depth < 0:
            raise ValueError("Design depth must be finite and non-negative.")

        normalized_structures = tuple(structures)
        for structure in normalized_structures:
            self._validate_structure(structure)

        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "background", material)
        object.__setattr__(self, "structures", normalized_structures)
        object.__setattr__(
            self,
            "_centered_coordinates",
            bool(background_arg is not None and not explicit_size),
        )

    @staticmethod
    def _validate_structure(structure):
        from beamz.design.structures import Structure

        if not isinstance(structure, Structure):
            raise TypeError(
                "Design only accepts immutable geometry specs; pass sources and "
                "monitors to Simulation instead."
            )

    @property
    def is_3d(self):
        """Return whether the design has a positive z extent."""
        return self.depth > 0

    def __str__(self):
        return f"Design with {len(self.structures)} structures ({'3D' if self.is_3d else '2D'})"

    def canonical_spec(self):
        """Return the immutable values defining physical cache identity."""
        return self.width, self.height, self.depth, self.background, self.structures

    def __eq__(self, other):
        if not isinstance(other, Design):
            return NotImplemented
        return cache_token(self.canonical_spec()) == cache_token(other.canonical_spec())

    def __hash__(self):
        return hash(cache_token(self.canonical_spec()))

    def __iadd__(self, structure):
        return self.with_structure(structure)

    def with_structure(self, structure):
        """Return a design with one geometry appended in painter's order."""
        self._validate_structure(structure)
        return self.updated_copy(structures=(*self.structures, structure))

    def updated_copy(self, **changes):
        """Return a design with domain, background, or structures replaced."""
        unknown = set(changes) - {
            "width",
            "height",
            "depth",
            "background",
            "structures",
        }
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"Unknown Design field(s): {names}.")
        values = {
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "background": self.background,
            "structures": self.structures,
        }
        values.update(changes)
        result = type(self)(**values)
        if not ({"width", "height", "depth"} & changes.keys()):
            object.__setattr__(
                result, "_centered_coordinates", self._centered_coordinates
            )
        return result

    def unified_polygons(self):
        """Return a design with compatible overlapping polygons merged."""
        replacements = _union_replacements(self.structures)
        structures = []
        for index, structure in enumerate(self.structures):
            if index in replacements:
                structures.extend(replacements[index])
            else:
                structures.append(structure)
        return self.updated_copy(structures=tuple(structures))

    def rasterize(
        self,
        resolution,
        grid_type: str = "auto",
        force_recompute: bool = False,
        progress: bool = False,
        **kwargs,
    ):
        """Rasterize this design on a scalar spacing or realized grid."""
        from beamz.design.raster.integration import _rasterize_design

        return _rasterize_design(
            self,
            resolution,
            grid_type=grid_type,
            force_recompute=force_recompute,
            progress=progress,
            **kwargs,
        )

    def plot(self, **kwargs):
        """Plot the layout. ``show`` defaults to false."""
        from beamz.analysis.plotting import plot_design

        kwargs.setdefault("show", False)
        return plot_design(self, **kwargs)

    def show(self, **kwargs):
        """Plot the layout. ``show`` defaults to true."""
        kwargs.setdefault("show", True)
        return self.plot(**kwargs)

    def copy(self):
        """Return this immutable specification."""
        return self

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        del memo
        return self
