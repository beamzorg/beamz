"""Immutable boundary-condition specifications consumed by simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from beamz.const import µm
from beamz.devices._immutable import _Replaceable

BoundaryEdges = tuple[str, ...] | str
BoundaryAxes = tuple[str, ...] | str
_PLANAR_EDGES = ("left", "right", "top", "bottom")
_VOLUME_EDGES = (*_PLANAR_EDGES, "front", "back")
_AXES = ("x", "y", "z")


def _validate_absorber_parameters(spec) -> None:
    for name in ("sigma_max", "alpha_max"):
        value = getattr(spec, name)
        if value is not None and (not np.isfinite(float(value)) or float(value) < 0):
            raise ValueError(f"{name} must be non-negative and finite.")
    if not np.isfinite(float(spec.kappa_max)) or float(spec.kappa_max) < 1:
        raise ValueError("kappa_max must be finite and at least 1.")
    order = float(spec.m)
    if not np.isfinite(order) or order < 1 or not order.is_integer():
        raise ValueError("m must be a positive integer.")
    if not 0 < float(spec.target_reflection) < 1:
        raise ValueError("target_reflection must lie strictly between 0 and 1.")


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


class _EdgeBoundary(_Replaceable):
    """Edge selection shared by reflective and absorbing boundaries."""

    __slots__ = ()

    def _get_edges_for_dimensionality(self, is_3d):
        """Compatibility adapter for older low-level callers."""
        return list(edges_for_dimension(self.edges, bool(is_3d)))


@dataclass(frozen=True, slots=True)
class PEC(_EdgeBoundary):
    """Perfect electric conductor on selected domain edges.

    Tangential E is zero, so the boundary is reflective. ``thickness`` is kept
    for compatibility and is always stored as zero. Use ``PML`` or ``Absorber``
    to truncate an open domain.
    """

    edges: BoundaryEdges = "all"
    thickness: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", normalize_edges(self.edges))
        object.__setattr__(self, "thickness", 0.0)


@dataclass(frozen=True, slots=True)
class PML(_EdgeBoundary):
    """Graded absorbing layer on selected domain edges.

    Omitted ``thickness`` means 12 cells. Omitted ``sigma_max`` is derived from
    ``target_reflection``; that value is an analytical amplitude target, not the
    measured reflection of the discrete layer. ``formulation`` is ``"sponge"`` or
    ``"cpml"``. Omitted ``alpha_max`` is ``0.1 * EPS_0 * LIGHT_SPEED / thickness``
    in both 2D and 3D. Keep geometry out of the layer or extrude it along the
    boundary normal.
    """

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
        _validate_absorber_parameters(self)
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


@dataclass(frozen=True, slots=True)
class Absorber(_EdgeBoundary):
    """Graded-conductivity sponge, the explicit form of ``PML(formulation="sponge")``.

    Omitted ``sigma_max`` is derived from ``target_reflection``. Use CPML when
    convolutional matching is needed.
    """

    formulation: ClassVar[str] = "sponge"
    # Neutral CPML values let the shared profile compiler remain data-driven; they are
    # never used by the sponge branch.
    kappa_max: ClassVar[float] = 1.0
    alpha_max: ClassVar[None] = None

    edges: BoundaryEdges = "all"
    thickness: float = 1 * µm
    sigma_max: float | None = None
    m: int = 3
    target_reflection: float = 1e-6

    def __post_init__(self) -> None:
        _validate_absorber_parameters(self)
        thickness = float(self.thickness)
        if not np.isfinite(thickness) or thickness < 0.0:
            raise ValueError("Absorber thickness must be a non-negative finite value.")
        object.__setattr__(self, "edges", normalize_edges(self.edges))
        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "sigma_max", None if self.sigma_max is None else float(self.sigma_max))  # fmt: skip
        object.__setattr__(self, "m", int(self.m))
        object.__setattr__(self, "target_reflection", float(self.target_reflection))


@dataclass(frozen=True, slots=True)
class Periodic(_Replaceable):
    """Zero-phase periodic coupling of both faces on each selected axis.

    One-sided periodicity is not representable. Nonzero phase belongs on
    ``Bloch``. Execution is JAX-only; an explicit CUDA request is rejected.
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


@dataclass(frozen=True, slots=True)
class Bloch(Periodic):
    """Phase-periodic faces with a physical Cartesian wavevector in rad/m.

    ``F(r + L_i e_i) = exp(1j * wavevector[i] * L_i) * F(r)``.
    Components outside the selected axes must be zero. Nonzero wavevectors
    require complex fields and single-device JAX execution.
    """

    wavevector: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        Periodic.__post_init__(self)
        vector = tuple(float(k) for k in self.wavevector)
        if len(vector) != 3 or not all(np.isfinite(k) for k in vector):
            raise ValueError(
                "Bloch wavevector must contain three finite values in rad/m."
            )
        selected = _AXES if self.axes == "all" else self.axes
        if any(
            k != 0
            for axis, k in zip(_AXES, vector, strict=True)
            if axis not in selected
        ):
            raise ValueError("Bloch wavevector must be zero outside its selected axes.")
        object.__setattr__(self, "wavevector", vector)


Boundary = PEC | PML | Absorber | Periodic | Bloch


def bloch_wavevector(boundaries, *, is_3d: bool, plane_2d: str = "xy"):
    """Resolve ownership and return the physical wavevector without aliasing."""
    active = active_physical_axes(is_3d, plane_2d)
    owners = {}
    for boundary in normalize_boundaries(boundaries):
        if not isinstance(boundary, Periodic):
            continue
        vector = boundary.wavevector if isinstance(boundary, Bloch) else (0.0,) * 3
        for axis in active if boundary.axes == "all" else boundary.axes:
            k = vector[_AXES.index(axis)]
            if axis in owners and owners[axis] != k:
                raise ValueError(
                    f"Conflicting periodic/Bloch wavevectors on axis {axis}."
                )
            owners[axis] = k
        if any(vector[_AXES.index(axis)] != 0 for axis in _AXES if axis not in active):
            raise ValueError("Bloch wavevector must be zero on inactive physical axes.")
    return tuple(owners.get(axis, 0.0) for axis in _AXES)


def bloch_storage_phases(boundaries, geometry, *, is_3d, plane_2d="xy"):
    """Compile forward seam phases in canonical storage-axis order."""
    vector = bloch_wavevector(boundaries, is_3d=is_3d, plane_2d=plane_2d)
    axes = _AXES if is_3d else active_physical_axes(False, plane_2d)
    grid_axes = _AXES if is_3d else ("x", "y")
    return tuple(
        complex(
            np.exp(
                1j * vector[_AXES.index(axis)] * np.ptp(geometry.axis_edges(grid_axis))
            )
        )
        if vector[_AXES.index(axis)] != 0
        else 1.0
        for axis, grid_axis in reversed(tuple(zip(axes, grid_axes, strict=True)))
    )


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
    bloch_wavevector(normalized, is_3d=is_3d, plane_2d=plane_2d)
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
    "Bloch",
    "PEC",
    "PML",
    "Periodic",
    "active_physical_axes",
    "bloch_wavevector",
    "bloch_storage_phases",
    "periodic_storage_axes",
    "validate_boundary_compatibility",
]
