"""Immutable boundary-condition specifications consumed by simulations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

from beamz.const import µm

BoundaryEdges = tuple[str, ...] | str
BoundaryAxes = tuple[str, ...] | str
_PLANAR_EDGES = ("left", "right", "top", "bottom")
_VOLUME_EDGES = (*_PLANAR_EDGES, "front", "back")
_AXES = ("x", "y", "z")


def normalize_edges(edges: BoundaryEdges) -> BoundaryEdges:
    """Validate and freeze an edge selection while retaining the ``"all"`` sentinel."""
    if isinstance(edges, str) and edges.lower() == "all":
        return "all"
    values = sorted(edges) if isinstance(edges, set) else edges
    values = values if isinstance(values, (list, tuple)) else (values,)
    normalized = tuple(str(edge).lower() for edge in values)
    invalid = set(normalized).difference(_VOLUME_EDGES)
    if invalid:
        raise ValueError(f"Unsupported boundary edges: {sorted(invalid)}")
    return normalized


def edges_for_dimension(edges: BoundaryEdges, is_3d: bool) -> tuple[str, ...]:
    """Resolve an edge selection after the compiler knows the domain dimension."""
    resolved = _VOLUME_EDGES if edges == "all" and is_3d else _PLANAR_EDGES
    if edges != "all":
        resolved = tuple(edges)
    invalid = set(resolved).difference(_VOLUME_EDGES if is_3d else _PLANAR_EDGES)
    if invalid:
        raise ValueError(
            f"Boundary edges {sorted(invalid)} are only available in 3D simulations."
        )
    return resolved


@dataclass(frozen=True, slots=True)
class PEC:
    """Request perfect-electric-conductor behavior on domain edges.

    Parameters
    ----------
    edges : "all", str, or sequence of str, default="all"
        Boundary edges to constrain, such as ``"left"`` or ``("top", "bottom")``.
    thickness : float, default=0.0
        Compatibility field. PEC boundaries are enforced at zero thickness.

    Examples
    --------
    >>> boundary = PEC(edges=("top", "bottom"))

    Notes
    -----
    PEC forces tangential electric fields to zero on the selected domain faces and
    is reflective. Use ``PML`` or ``Absorber`` for open-domain truncation.
    """

    edges: BoundaryEdges = "all"
    thickness: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", normalize_edges(self.edges))
        object.__setattr__(self, "thickness", 0.0)

    def updated_copy(self, **changes):
        """Return a validated boundary with selected fields replaced.

        Parameters
        ----------
        **changes : object
            Dataclass fields to replace.

        Returns
        -------
        PEC
            New immutable boundary specification.

        Raises
        ------
        TypeError or ValueError
            If a field is unknown or a replacement is invalid.
        """
        return replace(self, **changes)

    def _get_edges_for_dimensionality(self, is_3d):
        """Compatibility adapter for older low-level callers."""
        return list(edges_for_dimension(self.edges, bool(is_3d)))


@dataclass(frozen=True, slots=True)
class PML:
    """Request a graded absorbing layer on selected domain edges.

    Parameters
    ----------
    edges : "all", str, or sequence of str, default="all"
        Domain faces covered by the layer.
    thickness : float, optional
        Physical layer thickness in metres. When omitted, the compiler uses
        exactly 12 grid cells on every selected face.
    sigma_max : float, optional
        Explicit maximum conductivity. When omitted, BeamZ derives it from
        ``target_reflection`` and the layer thickness.
    m : int, default=3
        Polynomial grading order.
    formulation : {"sponge", "cpml"}, default="sponge"
        Absorber formulation. CPML adds convolutional recurrence state and is
        generally preferred for broadband or oblique incidence.
    kappa_max : float, default=2.0
        Maximum CPML coordinate-stretching factor.
    alpha_max : float, optional
        Maximum CPML complex-frequency-shift coefficient.
    target_reflection : float, default=1e-6
        Reflection target used to derive automatic conductivity.

    Examples
    --------
    >>> boundary = PML(formulation="cpml")  # 12 cells per selected face

    Notes
    -----
    Geometry should normally be kept clear of the absorber or extruded through it
    along the boundary normal to avoid material discontinuities inside the layer.
    """

    _DEFAULT_CPML_ALPHA_NORMALIZED: ClassVar[float] = 0.1
    _DEFAULT_3D_CPML_ALPHA_NORMALIZED: ClassVar[float] = 0.05
    DEFAULT_CELLS: ClassVar[int] = 12

    edges: BoundaryEdges = "all"
    thickness: float | None = None
    sigma_max: float | None = None
    m: int = 3
    formulation: str = "sponge"
    kappa_max: float = 2.0
    alpha_max: float | None = None
    target_reflection: float = 1e-6

    def __post_init__(self) -> None:
        thickness = None if self.thickness is None else float(self.thickness)
        if thickness is not None and (not np.isfinite(thickness) or thickness < 0.0):
            raise ValueError("PML thickness must be a non-negative finite value.")
        formulation = str(self.formulation).lower()
        if formulation not in {"sponge", "cpml"}:
            raise ValueError(
                f"Unsupported boundary formulation {self.formulation!r}. "
                "Expected one of: 'sponge', 'cpml'."
            )
        object.__setattr__(self, "edges", normalize_edges(self.edges))
        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "sigma_max", None if self.sigma_max is None else float(self.sigma_max))  # fmt: skip
        object.__setattr__(self, "m", int(self.m))
        object.__setattr__(self, "formulation", formulation)
        object.__setattr__(self, "kappa_max", float(self.kappa_max))
        object.__setattr__(self, "alpha_max", None if self.alpha_max is None else float(self.alpha_max))  # fmt: skip
        object.__setattr__(self, "target_reflection", float(self.target_reflection))

    def updated_copy(self, **changes):
        """Return a validated boundary with selected fields replaced.

        Parameters
        ----------
        **changes : object
            Dataclass fields to replace.

        Returns
        -------
        PML
            New immutable absorbing-boundary specification.

        Raises
        ------
        TypeError or ValueError
            If a field is unknown or a replacement is invalid.
        """
        return replace(self, **changes)

    def _get_edges_for_dimensionality(self, is_3d):
        """Compatibility adapter for older low-level callers."""
        return list(edges_for_dimension(self.edges, bool(is_3d)))


@dataclass(frozen=True, slots=True)
class Absorber:
    """Request a graded-conductivity sponge on selected domain edges.

    Parameters
    ----------
    edges : "all", str, or sequence of str, default="all"
        Domain faces covered by the sponge.
    thickness : float, default=1 um
        Physical sponge thickness in metres.
    sigma_max : float, optional
        Explicit maximum conductivity. When omitted, BeamZ derives it from the
        reflection target.
    m : int, default=3
        Polynomial conductivity-grading order.
    target_reflection : float, default=1e-6
        Reflection target used to derive automatic conductivity.

    Examples
    --------
    >>> boundary = Absorber(edges=("left", "right"), thickness=1e-6)

    Notes
    -----
    ``Absorber`` is the explicit sponge-only counterpart of
    ``PML(formulation="sponge")``. Use CPML when convolutional matching is needed.
    """

    formulation: ClassVar[str] = "sponge"
    # Neutral CPML values let the shared profile compiler remain data-driven; they are
    # never used by the sponge branch.
    kappa_max: ClassVar[float] = 1.0
    alpha_max: ClassVar[None] = None
    _DEFAULT_CPML_ALPHA_NORMALIZED: ClassVar[float] = 0.0
    _DEFAULT_3D_CPML_ALPHA_NORMALIZED: ClassVar[float] = 0.0

    edges: BoundaryEdges = "all"
    thickness: float = 1 * µm
    sigma_max: float | None = None
    m: int = 3
    target_reflection: float = 1e-6

    def __post_init__(self) -> None:
        thickness = float(self.thickness)
        if not np.isfinite(thickness) or thickness < 0.0:
            raise ValueError("Absorber thickness must be a non-negative finite value.")
        object.__setattr__(self, "edges", normalize_edges(self.edges))
        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "sigma_max", None if self.sigma_max is None else float(self.sigma_max))  # fmt: skip
        object.__setattr__(self, "m", int(self.m))
        object.__setattr__(self, "target_reflection", float(self.target_reflection))

    def updated_copy(self, **changes):
        """Return a validated boundary with selected fields replaced.

        Parameters
        ----------
        **changes : object
            Dataclass fields to replace.

        Returns
        -------
        Absorber
            New immutable sponge specification.

        Raises
        ------
        TypeError or ValueError
            If a field is unknown or a replacement is invalid.
        """
        return replace(self, **changes)

    def _get_edges_for_dimensionality(self, is_3d):
        """Compatibility adapter for older low-level callers."""
        return list(edges_for_dimension(self.edges, bool(is_3d)))


@dataclass(frozen=True, slots=True)
class Periodic:
    """Identify axes whose opposite domain faces are periodically coupled.

    Parameters
    ----------
    axes : "all", str, or sequence of str, default="all"
        Physical Cartesian axes to wrap. A periodic axis always owns both of its
        domain faces, so mismatched one-sided periodic configurations cannot be
        represented.

    Examples
    --------
    >>> boundary = Periodic(axes=("x", "y"))

    Notes
    -----
    This boundary implements zero-phase periodicity. Bloch phase shifts are not
    accepted by this type; they require a complex-field execution path and will
    be introduced separately. Periodic boundaries currently execute through the
    JAX backend. Automatic backend selection chooses JAX, while an explicit CUDA
    backend request raises an unsupported-backend error.
    """

    axes: BoundaryAxes = "all"

    def __post_init__(self) -> None:
        axes = self.axes
        if isinstance(axes, str) and axes.lower() == "all":
            object.__setattr__(self, "axes", "all")
            return
        values = sorted(axes) if isinstance(axes, set) else axes
        values = values if isinstance(values, (list, tuple)) else (values,)
        normalized = tuple(str(axis).lower() for axis in values)
        invalid = set(normalized).difference(_AXES)
        if invalid:
            raise ValueError(f"Unsupported periodic axes: {sorted(invalid)}")
        if not normalized:
            raise ValueError("Periodic axes cannot be empty.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Periodic axes must be unique.")
        object.__setattr__(self, "axes", normalized)

    def updated_copy(self, **changes):
        """Return a validated periodic boundary with selected fields replaced.

        Parameters
        ----------
        **changes : object
            Dataclass fields to replace.

        Returns
        -------
        Periodic
            New immutable periodic-boundary specification.

        Raises
        ------
        TypeError or ValueError
            If a field is unknown or a replacement is invalid.
        """
        return replace(self, **changes)


Boundary = PEC | PML | Absorber | Periodic


def active_physical_axes(is_3d: bool, plane_2d: str = "xy") -> tuple[str, ...]:
    """Return public Cartesian axes represented by a simulation lattice."""
    if is_3d:
        return _AXES
    try:
        return {
            "xy": ("x", "y"),
            "xz": ("x", "z"),
            "yz": ("y", "z"),
        }[str(plane_2d).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported 2D plane {plane_2d!r}.") from exc


def periodic_storage_axes(
    boundaries, *, is_3d: bool, plane_2d: str = "xy"
) -> frozenset[int]:
    """Resolve periodic physical axes to canonical array-axis indices."""
    physical = active_physical_axes(bool(is_3d), plane_2d)
    requested: set[str] = set()
    for boundary in normalize_boundaries(boundaries):
        if not isinstance(boundary, Periodic):
            continue
        requested.update(physical if boundary.axes == "all" else boundary.axes)
    inactive = requested.difference(physical)
    if inactive:
        raise ValueError(
            f"Periodic axes {sorted(inactive)} are not active in a {plane_2d} simulation."
        )
    storage = ("z", "y", "x") if is_3d else tuple(reversed(physical))
    return frozenset(storage.index(axis) for axis in requested)


def validate_boundary_compatibility(
    boundaries, *, is_3d: bool, plane_2d: str = "xy"
) -> None:
    """Reject overlapping periodic and wall/absorber ownership before compilation."""
    normalized = normalize_boundaries(boundaries)
    periodic = periodic_storage_axes(normalized, is_3d=bool(is_3d), plane_2d=plane_2d)
    if not periodic:
        return
    edge_pairs = (
        (("front", "back"), ("bottom", "top"), ("left", "right"))
        if is_3d
        else (("bottom", "top"), ("left", "right"))
    )
    periodic_edges = {edge for axis in periodic for edge in edge_pairs[axis]}
    conflicts: set[str] = set()
    for boundary in normalized:
        if isinstance(boundary, Periodic):
            continue
        conflicts.update(
            periodic_edges.intersection(edges_for_dimension(boundary.edges, is_3d))
        )
    if conflicts:
        raise ValueError(
            "Periodic axes cannot also use PEC, PML, or Absorber faces; "
            f"conflicting edges: {sorted(conflicts)}."
        )


def normalize_boundaries(boundaries) -> tuple[Boundary, ...]:
    """Freeze boundary input and preserve Beamz's historical all-PEC default."""
    resolved = tuple(boundaries) if boundaries else (PEC(),)
    if not all(
        isinstance(boundary, (PEC, PML, Absorber, Periodic)) for boundary in resolved
    ):
        raise TypeError(
            "boundaries must contain only PEC, PML, Absorber, or Periodic specifications"
        )
    return resolved


__all__ = [
    "Absorber",
    "Boundary",
    "PEC",
    "PML",
    "Periodic",
    "active_physical_axes",
    "periodic_storage_axes",
    "validate_boundary_compatibility",
]
