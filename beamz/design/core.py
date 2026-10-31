from dataclasses import dataclass, field
from itertools import groupby
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
    replacements = {}
    # Merge only within one painter-order run of compatible materials/extrusions.
    for _, group in groupby(
        enumerate(structures), key=lambda item: _merge_group_key(item[1])
    ):
        mergeable = []
        for index, structure in group:
            if not getattr(structure, "material", None) or isinstance(structure, Ring):
                continue
            polygon = _to_shapely(structure)
            if polygon is not None:
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


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Design:
    """Immutable geometry specification with a functional convenience facade.

    ``background`` is the sole background material. ``structures`` contains only
    user geometry, in painter's order. Python's augmented assignment still gives
    a friendly workflow because ``design += structure`` rebinds ``design`` to the
    new value returned by :meth:`with_structure`.

    Parameters
    ----------
    width : float, default=4 um
        Domain extent along x in metres; must be finite and positive.
    height : float, default=4 um
        Domain extent along y in metres; must be finite and positive.
    depth : float, default=0
        Domain extent along z in metres. Zero selects a 2D design.
    material : Material, optional
        Backward-compatible name for the background material.
    background : Material, optional
        Material filling cells not covered by a structure. Do not pass together
        with a different ``material`` value.
    structures : iterable, optional
        Geometry primitives in painter's order; later structures overwrite
        earlier structures during rasterization.

    Raises
    ------
    ValueError
        If an extent is invalid or both background aliases disagree.
    TypeError
        If ``structures`` contains unsupported mutable geometry.

    Notes
    -----
    Public geometry coordinates use ``(x, y, z)`` order and SI units. A design
    is immutable: methods such as ``with_structure`` return a new value.

    Examples
    --------
    >>> import beamz as bz
    >>> design = bz.Design(
    ...     width=6 * bz.um,
    ...     height=3 * bz.um,
    ...     background=bz.Material(permittivity=1.44**2),
    ... )
    >>> design += bz.Rectangle(
    ...     position=(1 * bz.um, 1.25 * bz.um),
    ...     width=4 * bz.um,
    ...     height=0.5 * bz.um,
    ...     material=bz.Material(permittivity=3.48**2),
    ... )
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
        """Return a design with one geometry appended in painter's order.

        Parameters
        ----------
        structure : Polygon or Box
            Immutable geometry to place above existing structures.
        """
        self._validate_structure(structure)
        return self.updated_copy(structures=(*self.structures, structure))

    def updated_copy(self, **changes):
        """Return a new design with the requested fields replaced.

        Parameters
        ----------
        **changes
            Any of ``width``, ``height``, ``depth``, ``background``, or
            ``structures`` and their replacement values.
        """
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

    def discretize(self, resolution, **kwargs):
        """Return design-owned material grids at the requested resolution.

        Parameters
        ----------
        resolution : float
            In-plane cell spacing in metres.
        **kwargs
            Additional options forwarded to :func:`build_material_grid`.
        """
        from beamz.design.discretization import build_material_grid

        return build_material_grid(self, resolution, **kwargs)

    def plot(self, **kwargs):
        """Plot the design layout using the matplotlib backend.

        Parameters
        ----------
        **kwargs
            Plotting options forwarded to the design plotting backend. ``show``
            defaults to false.
        """
        from beamz.analysis.plotting import plot_design

        kwargs.setdefault("show", False)
        return plot_design(self, **kwargs)

    def show(self, **kwargs):
        """Display the design layout using the matplotlib backend.

        Parameters
        ----------
        **kwargs
            Plotting options forwarded to :meth:`plot`. ``show`` defaults to
            true.
        """
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
