"""BEAMZ-facing discrete mode contract.

This module is intentionally small and data-oriented. BeamZ owns geometry and
Yee-grid placement; the native mode package solves and converts fields into
component planes that the simulation can inject without another interpretation
layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from typing import Any, Literal, TypedDict, cast

import numpy as np

from beamz._cache_tokens import cache_token
from beamz.design.grid import RectilinearGrid
from beamz.devices._immutable import immutable_snapshot, readonly_array

from ._cache import _SolveCache
from ._symmetry import material_symmetric, mirror_closed, project_mode_symmetry
from ._yee import refine_x_mode_at_fixed_beta, validate_x_mode_refinement
from .constitutive import require_uniform_normal_spacing
from .fields import (
    _axis_coordinate,
    _axis_index,
    _modal_power,
    _normalize_profiles,
    _numeric_wave_number,
    _phase_delay,
)
from .result import Result
from .solver import solve_grid

AxisName = Literal["x", "y", "z"]
DirectionName = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
PolarizationName = Literal["te", "tm"]
ComponentIndex = tuple[slice | int, slice | int, slice | int]

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
DISCRETE_MODE_CONTRACT = "beamz.devices.modes.DiscreteMode/v1"
_AXIS_INDEX: dict[AxisName, Literal[0, 1, 2]] = {"x": 0, "y": 1, "z": 2}
_AXIS_NAMES: tuple[AxisName, AxisName, AxisName] = ("x", "y", "z")
_YEE_OFFSETS_3D = {
    "Ex": {"z": 0.0, "y": 0.0, "x": 0.5},
    "Ey": {"z": 0.0, "y": 0.5, "x": 0.0},
    "Ez": {"z": 0.5, "y": 0.0, "x": 0.0},
    "Hx": {"z": 0.5, "y": 0.5, "x": 0.0},
    "Hy": {"z": 0.5, "y": 0.0, "x": 0.5},
    "Hz": {"z": 0.0, "y": 0.5, "x": 0.5},
}


class _ModeCandidate(TypedDict):
    neff: complex
    fields: dict[str, np.ndarray]


@dataclass(frozen=True)
class ModePlaneSpec:
    """Exact mode-plane metadata passed to the BeamZ-native solver.

    ``scalar_permittivity`` uses ``transverse_axes`` order, not the solver's
    internal local-axis order. For example, an x-normal BeamZ plane is usually
    stored as ``("z", "y")``.
    """

    scalar_permittivity: np.ndarray
    frequency: float
    resolution: float
    dt: float | None
    axis: AxisName
    direction: DirectionName
    transverse_axes: tuple[AxisName, AxisName]
    grid_shape: tuple[int, int, int]
    center: tuple[float, float, float]
    width: float
    height: float
    plane_index: int
    offset_index: int
    mode_index: int = 0
    polarization: PolarizationName | None = None
    target_neff: float | None = None
    num_modes: int | None = None
    solver_direction: DirectionName | None = None
    aperture_pad_cells: int = 2
    aperture_window_alpha: float = 0.2
    component_permittivity: Mapping[str, np.ndarray] = field(default_factory=dict)
    component_permeability: Mapping[str, np.ndarray] = field(default_factory=dict)
    diagonal_permittivity: Mapping[str, np.ndarray] = field(default_factory=dict)
    diagonal_permeability: Mapping[str, np.ndarray] = field(default_factory=dict)
    yee_tensors: Mapping[str, np.ndarray] = field(default_factory=dict)
    offdiagonal_permittivity: Mapping[str, np.ndarray] = field(default_factory=dict)
    # Guarded by joint field, power, energy, and discrete-Maxwell validation.
    yee_refinement: bool = True
    grid: RectilinearGrid | None = None
    force_symmetry: bool = False
    min_aperture_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.min_aperture_fraction is not None and (
            not np.isfinite(self.min_aperture_fraction)
            or not 0 <= self.min_aperture_fraction <= 1
        ):
            raise ValueError(
                "min_aperture_fraction must be between zero and one or None"
            )
        if not isinstance(self.force_symmetry, bool):
            raise TypeError("force_symmetry must be a bool")
        eps = readonly_array(self.scalar_permittivity, dtype=np.complex128)
        if eps.ndim != 2:
            raise ValueError("scalar_permittivity must be a 2D transverse plane")
        object.__setattr__(self, "scalar_permittivity", eps)

        axis = str(self.axis).lower()
        if axis not in _AXIS_INDEX:
            raise ValueError("axis must be one of 'x', 'y', or 'z'")
        object.__setattr__(self, "axis", axis)

        direction = str(self.direction).lower()
        if direction not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
            raise ValueError(
                "direction must be one of '+x', '-x', '+y', '-y', '+z', '-z'"
            )
        if direction[1] != axis:
            raise ValueError("direction axis must match axis")
        object.__setattr__(self, "direction", direction)

        solver_direction = (
            self.direction
            if self.solver_direction is None
            else str(self.solver_direction).lower()
        )
        if solver_direction not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
            raise ValueError(
                "solver_direction must be one of '+x', '-x', '+y', '-y', '+z', '-z'"
            )
        if solver_direction[1] != axis:
            raise ValueError("solver_direction axis must match axis")
        object.__setattr__(self, "solver_direction", solver_direction)

        transverse_axes = tuple(str(value).lower() for value in self.transverse_axes)
        expected_axes = tuple(value for value in _AXIS_NAMES if value != axis)
        if set(transverse_axes) != set(expected_axes) or len(transverse_axes) != 2:
            raise ValueError(
                f"transverse_axes must be a permutation of {expected_axes!r}"
            )
        object.__setattr__(self, "transverse_axes", transverse_axes)

        grid_shape = tuple(int(v) for v in self.grid_shape)
        if len(grid_shape) != 3 or any(v <= 1 for v in grid_shape):
            raise ValueError("grid_shape must contain three dimensions larger than one")
        object.__setattr__(self, "grid_shape", grid_shape)
        if self.grid is not None:
            if not isinstance(self.grid, RectilinearGrid):
                raise TypeError("grid must be a RectilinearGrid or None")
            if self.grid.shape_zyx != grid_shape:
                raise ValueError(
                    f"grid shape {self.grid.shape_zyx} does not match {grid_shape}"
                )
            expected_profile_shape = tuple(
                self.grid.shape[{"x": 0, "y": 1, "z": 2}[name]]
                for name in transverse_axes
            )
            if eps.shape != expected_profile_shape:
                raise ValueError(
                    "scalar_permittivity shape must match the grid's transverse cells"
                )

        for name, value in (
            ("component_permittivity", self.component_permittivity),
            ("component_permeability", self.component_permeability),
            ("diagonal_permittivity", self.diagonal_permittivity),
            ("diagonal_permeability", self.diagonal_permeability),
            ("yee_tensors", self.yee_tensors),
            ("offdiagonal_permittivity", self.offdiagonal_permittivity),
        ):
            if name.startswith("diagonal_"):
                unknown = set(value) - {"xx", "yy", "zz"}
                if unknown:
                    raise ValueError(
                        f"{name} has unknown components: {', '.join(sorted(unknown))}"
                    )
                for component, array in value.items():
                    if np.asarray(array).shape != eps.shape:
                        raise ValueError(
                            f"{name}[{component!r}] must have shape {eps.shape}"
                        )
            object.__setattr__(self, name, immutable_snapshot(value))

        if self.yee_tensors:
            from beamz.lattice import component_shape_3d

            required = {"eps_x", "eps_y", "eps_z", "eps_node"}
            if set(self.yee_tensors) != required:
                raise ValueError(
                    "Full-tensor modes require all electric-support and shared-node tensors"
                )
            if self.dt is None or not np.isfinite(self.dt) or self.dt <= 0:
                raise ValueError(
                    "Full-tensor modes require a positive propagation timestep"
                )
            require_uniform_normal_spacing(self.grid, axis)
            if self.polarization is not None or self.force_symmetry:
                raise ValueError(
                    "Full-tensor modes require polarization=None and force_symmetry=False"
                )
            if not self.component_permeability or any(
                not np.all(np.asarray(v) == 1)
                for v in self.component_permeability.values()
            ):
                raise ValueError(
                    "Full-tensor modes currently require nonmagnetic materials (mu=1)"
                )
            for name, values in self.yee_tensors.items():
                expected = (
                    tuple(n + 1 for n in self.grid_shape)
                    if name == "eps_node"
                    else component_shape_3d("E" + name[-1], self.grid_shape)
                )
                if np.asarray(values).shape != (6, *expected):
                    raise ValueError(
                        f"{name} must cover the complete packed Yee support {(6, *expected)}"
                    )

                if not np.isfinite(values).all() or np.any(np.imag(values)):
                    raise ValueError(
                        "Full-tensor modes require finite real dielectric tensors"
                    )
                normal = {"z": 1, "y": 2, "x": 3}[self.axis]
                if not np.allclose(
                    values, np.take(values, [0], axis=normal), rtol=1e-6, atol=1e-12
                ):
                    raise ValueError(
                        "Full-tensor modes require longitudinally invariant constitutive support"
                    )
                xx, yy, zz, xy, xz, yz = np.real(values)
                determinant = (
                    xx * yy * zz
                    + 2 * xy * xz * yz
                    - xx * yz**2
                    - yy * xz**2
                    - zz * xy**2
                )
                if (
                    np.any(xx <= 0)
                    or np.any(xx * yy - xy**2 <= 0)
                    or np.any(determinant <= 0)
                ):
                    raise ValueError(
                        "Full-tensor modes require positive-definite dielectric tensors"
                    )

        if float(self.frequency) <= 0.0 or not np.isfinite(float(self.frequency)):
            raise ValueError("frequency must be finite and positive")
        if float(self.resolution) <= 0.0 or not np.isfinite(float(self.resolution)):
            raise ValueError("resolution must be finite and positive")
        if self.dt is not None and (
            float(self.dt) <= 0.0 or not np.isfinite(float(self.dt))
        ):
            raise ValueError("dt must be finite and positive when provided")
        object.__setattr__(self, "frequency", float(self.frequency))
        object.__setattr__(self, "resolution", float(self.resolution))
        object.__setattr__(self, "dt", None if self.dt is None else float(self.dt))
        object.__setattr__(self, "center", tuple(float(v) for v in self.center))
        object.__setattr__(self, "width", float(self.width))
        object.__setattr__(self, "height", float(self.height))
        object.__setattr__(self, "plane_index", int(self.plane_index))
        object.__setattr__(self, "offset_index", int(self.offset_index))
        object.__setattr__(self, "mode_index", int(self.mode_index))
        object.__setattr__(
            self, "num_modes", None if self.num_modes is None else int(self.num_modes)
        )
        object.__setattr__(self, "aperture_pad_cells", int(self.aperture_pad_cells))
        object.__setattr__(
            self, "aperture_window_alpha", float(self.aperture_window_alpha)
        )
        object.__setattr__(self, "yee_refinement", bool(self.yee_refinement))

        if self.polarization is not None:
            pol = str(self.polarization).lower()
            if pol not in {"te", "tm"}:
                raise ValueError("polarization must be 'te', 'tm', or None")
            object.__setattr__(self, "polarization", pol)


@dataclass(frozen=True)
class DiscreteMode:
    """Mode solved and shaped directly for BEAMZ component lattices."""

    neff: complex
    profiles: dict[str, np.ndarray]
    backward_profiles: dict[str, np.ndarray]
    component_indices: dict[str, ComponentIndex]
    axis: AxisName
    direction: DirectionName
    transverse_axes: tuple[AxisName, AxisName]
    phase_reference_component: str
    phase_reference_coord: float
    phase_plane_coord: float
    k_num_axis: float
    power_scale: float
    diagnostics: dict[str, object]
    integration_weights: dict[str, np.ndarray] = field(default_factory=dict)

    def component(self, name: str) -> np.ndarray:
        """Return one component profile."""
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise ValueError(f"Unknown component {name!r}") from exc


def solve_beamz_mode(spec: ModePlaneSpec) -> DiscreteMode:
    """Solve and return a BEAMZ-shaped discrete mode."""
    return _solve_beamz_mode(spec)


# Only analysis opts into reuse; keys retain content tokens, not simulations.
_CANDIDATE_CACHE: _SolveCache[Result] = _SolveCache(
    max_entries=32, max_bytes=64 * 1024 * 1024
)
_MODE_CACHE: _SolveCache[DiscreteMode] = _SolveCache(
    max_entries=128, max_bytes=64 * 1024 * 1024
)


def _solve_cached_beamz_mode(spec: ModePlaneSpec) -> DiscreteMode:
    """Cache by all spec inputs, including materials, grid and solver settings."""
    if not isinstance(spec, ModePlaneSpec):
        raise TypeError("spec must be a ModePlaneSpec")
    return _MODE_CACHE.get_or_compute(
        (solve_grid, cache_token(spec)),
        lambda: _solve_beamz_mode(spec, candidate_cache=_CANDIDATE_CACHE),
    )


def _solve_candidate_pool(cache: _SolveCache[Result] | None, **kwargs) -> Result:
    if cache is None:
        return solve_grid(**kwargs)
    return cache.get_or_compute(
        (solve_grid, cache_token(kwargs)), lambda: solve_grid(**kwargs)
    )


def _solve_beamz_mode(
    spec: ModePlaneSpec,
    *,
    candidate_cache: _SolveCache[Result] | None = None,
    basis_cache: dict | None = None,
) -> DiscreteMode:
    if not isinstance(spec, ModePlaneSpec):
        raise TypeError("spec must be a ModePlaneSpec")

    # Use the same discrete Maxwell/refinement contract for x/y ports.
    # The legacy y conversion skipped Yee refinement, biasing transmission
    # when input and output ports used different axes on the same uniform grid.
    if (
        (spec.axis == "y" or (spec.axis == "z" and spec.yee_tensors))
        and spec.component_permittivity
        and (
            spec.yee_tensors
            or spec.grid is None
            or spec.grid.metric_kind == "isotropic_uniform"
        )
    ):
        return _solve_mode_in_x_frame(
            spec, candidate_cache=candidate_cache, basis_cache=basis_cache
        )

    requested = max(spec.num_modes or 1, spec.mode_index + 1)
    # One family has one final pool, independent of which member is prepared.
    # This cache is request-local (one preview frequency), never process-global.
    key = (
        (
            solve_grid,
            requested,
            cache_token(
                {
                    f.name: getattr(spec, f.name)
                    for f in dataclass_fields(spec)
                    if f.name != "mode_index"
                }
            ),
        )
        if basis_cache is not None
        else None
    )
    if basis_cache is not None and key in basis_cache:
        basis = basis_cache[key]
    else:
        basis = _resolve_mode_basis(spec, candidate_cache=candidate_cache)
        if basis_cache is not None:
            basis_cache[key] = basis
    return _prepare_discrete_mode(spec, basis)


@dataclass(frozen=True)
class _ResolvedModeBasis:
    result: Result
    candidates: tuple[_ModeCandidate, ...]
    candidate_count: int
    qualifying_count: int | None


def _resolve_mode_basis(
    spec: ModePlaneSpec, *, candidate_cache=None
) -> _ResolvedModeBasis:
    """Resolve the shared requested family before assigning any output indices."""
    solver_axes = _solver_axes_for_axis(spec.axis)
    eps_solver = _transpose_between_axes(
        spec.scalar_permittivity, spec.transverse_axes, solver_axes
    )
    if spec.grid is None:
        dx_um = spec.resolution / 1e-6
        solver_edges = {
            axis: np.arange(spec.grid_shape[{"z": 0, "y": 1, "x": 2}[axis]] + 1) * dx_um
            for axis in spec.transverse_axes
        }
    else:
        solver_edges = {
            axis: np.asarray(spec.grid.axis_edges(axis), dtype=float) / 1e-6
            for axis in spec.transverse_axes
        }
    x_edges = tuple(float(v) for v in solver_edges[solver_axes[0]])
    y_edges = tuple(float(v) for v in solver_edges[solver_axes[1]])
    requested = max(spec.num_modes or 1, spec.mode_index + 1)
    mode_count = (
        spec.num_modes if spec.num_modes is not None else 2 * (spec.mode_index + 1) + 5
    )

    mode_count = max(mode_count, requested)

    # A single eigenpair cannot be reordered by polarization. Search a shared
    # candidate pool for all requested modes before selecting the TE/TM family.
    if spec.polarization is not None:
        mode_count = max(mode_count, 2 * requested + 5)

    diagonal_eps = {
        name: _transpose_between_axes(values, spec.transverse_axes, solver_axes)
        for name, values in spec.diagonal_permittivity.items()
    }
    cross_eps = {
        name: _transpose_between_axes(values, spec.transverse_axes, solver_axes)
        for name, values in spec.offdiagonal_permittivity.items()
    }
    diagonal_mu = {
        name: _transpose_between_axes(values, spec.transverse_axes, solver_axes)
        for name, values in spec.diagonal_permeability.items()
    }
    # Material components, like their sample axes above, must be in the local
    # (tangent, tangent, normal) frame expected by solve_grid.
    u, v, n = (*solver_axes, spec.axis)
    fallback_eps = diagonal_eps.get("xx", eps_solver)
    candidate_kwargs = dict(
        eps_xx=diagonal_eps.get(u * 2, fallback_eps),
        eps_yy=diagonal_eps.get(v * 2, fallback_eps),
        eps_zz=diagonal_eps.get(n * 2, fallback_eps),
        eps_xy=cross_eps.get("".join(sorted(u + v))),
        eps_yx=cross_eps.get("".join(sorted(u + v))),
        eps_xz=cross_eps.get("".join(sorted(u + n))),
        eps_zx=cross_eps.get("".join(sorted(u + n))),
        eps_yz=cross_eps.get("".join(sorted(v + n))),
        eps_zy=cross_eps.get("".join(sorted(v + n))),
        mu_xx=diagonal_mu.get(u * 2),
        mu_yy=diagonal_mu.get(v * 2),
        mu_zz=diagonal_mu.get(n * 2),
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[spec.frequency],
        direction="+" if str(spec.solver_direction).startswith("+") else "-",
        num_modes=mode_count,
        target_neff=spec.target_neff,
        normal_axis=_AXIS_INDEX[spec.axis],
    )

    def solve_candidates(count):
        return _solve_candidate_pool(
            candidate_cache, **{**candidate_kwargs, "num_modes": count}
        )

    result = solve_candidates(mode_count)
    candidates = _candidate_modes(result, spec)
    # A dense substrate spectrum can fill the initial search pool without any
    # mode of the requested port. Expand only for heterogeneous cross sections.
    search_limit = min(max(64, mode_count), max(1, 2 * eps_solver.size - 3))

    def confined_count():
        minimum = spec.min_aperture_fraction
        if minimum is None:
            return 0
        return sum(
            _aperture_fraction(c, spec) >= minimum
            and (
                spec.polarization is None
                or _polarization_fraction(c["fields"], spec.axis, spec.polarization)
                >= 0.5
            )
            for c in candidates
        )

    qualifying = None
    if spec.min_aperture_fraction is not None and np.ptp(np.real(eps_solver)) > 1e-12:
        while confined_count() < requested and mode_count < search_limit:
            mode_count = min(2 * mode_count, search_limit)
            result = solve_candidates(mode_count)
            candidates = _candidate_modes(result, spec)
        qualifying = confined_count()
    candidates = _sort_modes(candidates, spec)
    return _ResolvedModeBasis(result, tuple(candidates), mode_count, qualifying)


def _prepare_discrete_mode(
    spec: ModePlaneSpec, basis: _ResolvedModeBasis
) -> DiscreteMode:
    """Prepare one member without modifying the shared candidate fields."""
    result, candidates, mode_count = (
        basis.result,
        basis.candidates,
        basis.candidate_count,
    )
    solver_axes = _solver_axes_for_axis(spec.axis)
    if basis.qualifying_count is not None and spec.mode_index >= basis.qualifying_count:
        raise ValueError(
            "No requested mode meets min_aperture_fraction in the candidate pool; enlarge the aperture, adjust target_neff or num_modes, or explicitly disable aperture preference with min_aperture_fraction=None"
        )
    if spec.mode_index >= len(candidates):
        raise ValueError(
            f"Requested mode_index={spec.mode_index}, but only {len(candidates)} modes are available"
        )
    selected = candidates[spec.mode_index]
    phase_component = _select_phase_reference_component(
        spec.axis, spec.polarization, selected["fields"]
    )
    phase_ref = _dominant_phase(selected["fields"][phase_component])
    aligned = {
        name: value * np.exp(-1j * phase_ref)
        for name, value in selected["fields"].items()
    }

    symmetry_edges = tuple(_axis_edges(spec, a) for a in spec.transverse_axes)
    symmetry_supports = {
        name: tuple(np.asarray(data.coords[a]) * 1e-6 for a in spec.transverse_axes)
        for name, data in getattr(result, "field_components", {}).items()
        if all(a in data.coords for a in spec.transverse_axes)
    }
    rejected_axes = []
    if spec.force_symmetry:
        for a in spec.transverse_axes:
            midpoint = spec.center[_AXIS_INDEX[a]]
            for name, values in {
                **spec.component_permittivity,
                **spec.component_permeability,
            }.items():
                edges_a = _axis_edges(spec, a)
                coords = (
                    (edges_a[:-1] + edges_a[1:]) / 2
                    if _YEE_OFFSETS_3D[name][a] == 0.5
                    else edges_a
                )
                storage_axis = {"z": 0, "y": 1, "x": 2}[a]
                if (
                    values.ndim != 3
                    or values.shape[storage_axis] != len(coords)
                    or not mirror_closed(coords, midpoint)
                    or not material_symmetric(values, storage_axis)
                ):
                    rejected_axes.append(a)
                    break
    aligned, symmetry_diagnostics = project_mode_symmetry(
        aligned,
        requested=spec.force_symmetry,
        edges=symmetry_edges,
        axes=spec.transverse_axes,
        center=tuple(spec.center[_AXIS_INDEX[a]] for a in spec.transverse_axes),
        materials=[
            spec.scalar_permittivity,
            *spec.diagonal_permittivity.values(),
            *spec.diagonal_permeability.values(),
        ],
        supports=symmetry_supports,
        rejected_axes=rejected_axes,
    )
    profiles, indices, extra = _build_profiles(aligned, spec)

    phase_ref_coord = _axis_coordinate(
        phase_component,
        _axis_index(indices.get(phase_component), spec.axis),
        spec.axis,
        spec.resolution,
        spec.grid,
    )
    omega = 2.0 * np.pi * spec.frequency
    normal_spacing = (
        float(spec.grid.cell_widths(spec.axis)[spec.plane_index])
        if spec.grid is not None
        else spec.resolution
    )
    transverse_cell_widths = None
    if spec.grid is not None:
        transverse_cell_widths = tuple(
            _component_axis_measure(
                spec,
                "Ex",
                transverse_axis,
                indices["Ex"][{"z": 0, "y": 1, "x": 2}[transverse_axis]],
            )
            for transverse_axis in ("z", "y")
        )
    k_num = _numeric_wave_number(omega, spec.dt, normal_spacing, selected["neff"])
    boundary_neff = _boundary_refractive_index(spec.scalar_permittivity)
    yee_refinement_eligible = (
        spec.axis == "x"
        and bool(spec.component_permittivity)
        and float(np.real(selected["neff"])) > boundary_neff
    )
    yee_refinement_requested = bool(spec.yee_refinement)
    yee_refinement_attempted = yee_refinement_requested and yee_refinement_eligible
    yee_refinement_accepted = False
    yee_refinement_rejection_reason = ""
    yee_validation: dict[str, Any] = {}
    yee_residual = np.nan
    yee_frequency_ratio = np.nan
    yee_initial_frequency_ratio = np.nan
    if yee_refinement_requested and not yee_refinement_eligible:
        yee_refinement_rejection_reason = (
            "mode is not eligible for x-normal guided-mode refinement"
        )
    tensor_diagnostics = {}
    if spec.yee_tensors:
        if float(np.real(selected["neff"])) <= boundary_neff:
            raise ValueError(
                "Full-tensor ports currently require a confined guided mode; homogeneous/radiation ports are unsupported"
            )
        from ._tensor_yee import refine_tensor_mode

        profiles, k_num, tensor_diagnostics = refine_tensor_mode(
            profiles,
            indices,
            spec.yee_tensors,
            omega=omega,
            dt=spec.dt,
            spacing=normal_spacing,
            beta=k_num,
            direction_sign=_direction_sign(spec.direction),
            grid=spec.grid
            if spec.grid is not None and spec.grid.metric_kind != "isotropic_uniform"
            else None,
        )
        selected = _ModeCandidate(
            neff=complex(k_num * 299792458.0 / omega), fields=selected["fields"]
        )
        yee_refinement_attempted = False
    if yee_refinement_attempted:
        seed_profiles = {
            name: np.asarray(value, dtype=np.complex128).copy()
            for name, value in profiles.items()
        }
        seed_k_num = float(k_num)
        try:
            (
                candidate,
                yee_residual,
                yee_frequency_ratio,
                candidate_k_num,
                yee_initial_frequency_ratio,
            ) = refine_x_mode_at_fixed_beta(
                seed_profiles,
                indices,
                component_permittivity=spec.component_permittivity,
                component_permeability=spec.component_permeability,
                omega=omega,
                dt=spec.dt,
                resolution=normal_spacing,
                k_num=seed_k_num,
                direction_sign=_direction_sign(spec.direction),
                transverse_cell_widths=transverse_cell_widths,
            )
            yee_refinement_accepted, yee_validation = validate_x_mode_refinement(
                seed_profiles,
                candidate,
                indices,
                component_permittivity=spec.component_permittivity,
                component_permeability=spec.component_permeability,
                omega=omega,
                dt=spec.dt,
                resolution=normal_spacing,
                k_num=candidate_k_num,
                direction_sign=_direction_sign(spec.direction),
                transverse_cell_widths=transverse_cell_widths,
                integration_weights=_profile_integration_weights(
                    spec, candidate, indices
                ),
            )
            yee_refinement_rejection_reason = str(
                yee_validation.get("rejection_reason", "")
            )
            if yee_refinement_accepted:
                profiles = candidate
                k_num = float(candidate_k_num)
            else:
                profiles = seed_profiles
                k_num = seed_k_num
        except Exception as exc:
            profiles = seed_profiles
            k_num = seed_k_num
            yee_refinement_rejection_reason = f"{type(exc).__name__}: {exc}"
    integration_weights = _profile_integration_weights(spec, profiles, indices)
    profiles, power_scale, signed_power = _normalize_profiles_by_phase_referenced_flux(
        profiles,
        indices,
        axis=spec.axis,
        d_area=integration_weights,
        direction_sign=_direction_sign(spec.direction),
        omega=omega,
        k_num=k_num,
        ref_coord=phase_ref_coord,
        resolution=spec.resolution,
        grid=spec.grid,
    )
    profiles = _runtime_oriented_profiles(
        profiles, spec.axis, _direction_sign(spec.direction)
    )
    if signed_power < 0.0:
        profiles = _backward_mode_from_forward(profiles)
        signed_power = -signed_power
    backward_profiles = (
        {
            name: (1 if name.startswith("E") else -1) * np.conjugate(value)
            for name, value in profiles.items()
        }
        if spec.yee_tensors
        else _backward_mode_from_forward(profiles)
    )

    diagnostics = {
        **symmetry_diagnostics,
        **tensor_diagnostics,
        "full_tensor_operator": bool(spec.yee_tensors),
        "canonical_axis_rotation": bool(spec.yee_tensors),
        "contract": DISCRETE_MODE_CONTRACT,
        "normal_axis": spec.axis,
        "transverse_axes": spec.transverse_axes,
        "solver_axes": solver_axes,
        "solver_direction": spec.solver_direction,
        "mode_index": spec.mode_index,
        "selected_neff": complex(selected["neff"]),
        "candidate_count": mode_count,
        "aperture_fraction": _aperture_fraction(selected, spec),
        "min_aperture_fraction": spec.min_aperture_fraction,
        "phase_reference": "dominant_h_real_positive",
        "time_convention": (
            "exp(+i omega t); E at integer steps, H at half steps"
            if spec.yee_tensors
            else "exp(-i omega t); E at integer steps, H at half steps"
        ),
        "aperture_window_alpha": spec.aperture_window_alpha,
        "yee_refinement": yee_refinement_accepted,
        "yee_refinement_requested": yee_refinement_requested,
        "yee_refinement_eligible": yee_refinement_eligible,
        "yee_refinement_attempted": yee_refinement_attempted,
        "yee_refinement_accepted": yee_refinement_accepted,
        "yee_refinement_rejection_reason": yee_refinement_rejection_reason,
        "yee_refinement_validation": yee_validation,
        "boundary_neff": float(boundary_neff),
        "yee_residual": float(yee_residual),
        "yee_frequency_ratio": float(yee_frequency_ratio),
        "yee_initial_frequency_ratio": float(yee_initial_frequency_ratio),
        "power_before_phase_reference": float(extra.get("initial_power", np.nan)),
        "power_after_phase_reference": float(signed_power),
        "solver_info": result.solver_info or {},
    }
    return DiscreteMode(
        neff=complex(selected["neff"]),
        profiles=profiles,
        backward_profiles=backward_profiles,
        component_indices=indices,
        axis=spec.axis,
        direction=spec.direction,
        transverse_axes=spec.transverse_axes,
        phase_reference_component=phase_component,
        phase_reference_coord=float(phase_ref_coord),
        phase_plane_coord=float(
            _axis_coordinate(
                "Ey" if spec.axis == "x" else "Ex",
                spec.plane_index,
                spec.axis,
                spec.resolution,
                spec.grid,
            )
        ),
        k_num_axis=float(k_num),
        power_scale=float(power_scale),
        diagnostics=diagnostics,
        integration_weights=integration_weights,
    )


def _solve_mode_in_x_frame(
    spec: ModePlaneSpec,
    *,
    candidate_cache: _SolveCache[Result] | None = None,
    basis_cache: dict | None = None,
) -> DiscreteMode:
    """Solve a y/z port in an x-normal frame, then transform it back.

    Electric fields are polar vectors; magnetic fields are axial vectors. The
    x/y exchange is a reflection; the z/x/y cycle is a proper rotation.
    Staggered component indices and their material arrays rotate together.
    """
    # Each entry gives the old axis represented by a new (x, y, z) axis.
    old_axes = ("y", "x", "z") if spec.axis == "y" else ("z", "x", "y")
    inverse: dict[str, str] = dict(zip(old_axes, ("x", "y", "z"), strict=True))
    old_for_new: dict[str, str] = dict(zip(("x", "y", "z"), old_axes, strict=True))
    storage_order = tuple({"z": 0, "y": 1, "x": 2}[a] for a in old_axes[::-1])

    def rotate_materials(materials):
        return {
            prefix + new: (
                np.asarray(materials[prefix + old]).transpose(storage_order)
                if np.asarray(materials[prefix + old]).ndim == 3
                else materials[prefix + old]
            )
            for prefix in ("E", "H")
            for new, old in old_for_new.items()
            if prefix + old in materials
        }

    def rotate_diagonal(materials):
        return {
            new * 2: materials[old * 2]
            for new, old in old_for_new.items()
            if old * 2 in materials
        }

    tensor_components = ("xx", "yy", "zz", "xy", "xz", "yz")
    old_components = [
        "".join(sorted(old_for_new[a] for a in pair)) for pair in tensor_components
    ]
    component_order = [tensor_components.index(pair) for pair in old_components]
    rotated_tensors = {
        ("eps_node" if name == "eps_node" else "eps_" + inverse[name[-1]]): np.asarray(
            values
        )[component_order].transpose((0, *(i + 1 for i in storage_order)))
        for name, values in spec.yee_tensors.items()
    }
    rotated_cross = {
        new: spec.offdiagonal_permittivity[old]
        for new, old in zip(tensor_components[3:], old_components[3:], strict=True)
        if old in spec.offdiagonal_permittivity
    }
    canonical_grid = None
    if spec.grid is not None:
        canonical_grid = RectilinearGrid(
            *(spec.grid.axis_edges(old) for old in old_axes)
        )
    canonical = replace(
        spec,
        axis="x",
        direction=spec.direction[0] + "x",
        solver_direction=spec.direction[0] + "x",
        transverse_axes=tuple(inverse[a] for a in spec.transverse_axes),
        grid_shape=tuple(spec.grid_shape[i] for i in storage_order),
        center=tuple(spec.center[{"x": 0, "y": 1, "z": 2}[a]] for a in old_axes),
        component_permittivity=rotate_materials(spec.component_permittivity),
        component_permeability=rotate_materials(spec.component_permeability),
        diagonal_permittivity=rotate_diagonal(spec.diagonal_permittivity),
        diagonal_permeability=rotate_diagonal(spec.diagonal_permeability),
        yee_tensors=rotated_tensors,
        offdiagonal_permittivity=rotated_cross,
        grid=canonical_grid,
    )
    mode = _solve_beamz_mode(
        canonical, candidate_cache=candidate_cache, basis_cache=basis_cache
    )
    determinant = -1 if spec.axis == "y" else 1
    phase_component = (
        mode.phase_reference_component[0]
        + old_for_new[mode.phase_reference_component[1]]
    )
    # Restore the documented dominant-H-positive gauge after reflection.
    gauge = determinant

    def restore_profiles(profiles):
        return {
            name[0] + old_for_new[name[1]]: (
                gauge * (determinant if name.startswith("H") else 1) * values
            )
            for name, values in profiles.items()
        }

    inverse_storage = np.argsort(storage_order)
    indices = {
        name[0] + old_for_new[name[1]]: tuple(index[i] for i in inverse_storage)
        for name, index in mode.component_indices.items()
    }
    diagnostics = dict(mode.diagnostics)
    diagnostics.update(
        normal_axis=spec.axis,
        transverse_axes=spec.transverse_axes,
        solver_axes=_solver_axes_for_axis(spec.axis),
        solver_direction=spec.direction,
        canonical_axis_rotation=True,
        force_symmetry_axes=tuple(
            old_for_new[a]
            for a in cast(tuple[str, ...], diagnostics["force_symmetry_axes"])
        ),
        force_symmetry_skipped={
            old_for_new[a]: reason
            for a, reason in cast(
                Mapping[str, str], diagnostics["force_symmetry_skipped"]
            ).items()
        },
    )
    return replace(
        mode,
        axis=spec.axis,
        direction=spec.direction,
        transverse_axes=spec.transverse_axes,
        phase_reference_component=phase_component,
        profiles=restore_profiles(mode.profiles),
        backward_profiles=restore_profiles(mode.backward_profiles),
        component_indices=indices,
        integration_weights={
            name[0] + old_for_new[name[1]]: values
            for name, values in mode.integration_weights.items()
        },
        diagnostics=diagnostics,
    )


def _boundary_refractive_index(permittivity: np.ndarray) -> float:
    """Return the largest refractive index touching the mode-plane boundary."""
    eps = np.asarray(permittivity, dtype=np.complex128)
    if eps.ndim != 2 or min(eps.shape) == 0:
        return 0.0
    boundary = np.concatenate((eps[0], eps[-1], eps[1:-1, 0], eps[1:-1, -1]))
    return float(np.sqrt(max(float(np.max(np.real(boundary))), 0.0)))


def _candidate_modes(result, spec: ModePlaneSpec) -> list[_ModeCandidate]:
    candidates = []
    count = int(result.n_complex.shape[1])
    for mode_index in range(count):
        fields = {
            component: _field_plane(
                result.field_components[component],
                spec.axis,
                spec.transverse_axes,
                mode_index,
            )
            for component in _COMPONENTS
        }
        candidates.append(
            {"neff": complex(result.n_complex.values[0, mode_index]), "fields": fields}
        )
    return candidates


def _field_plane(
    data_array,
    axis: AxisName,
    transverse_axes: tuple[AxisName, AxisName],
    mode_index: int,
) -> np.ndarray:
    selected = data_array.isel(f=0, mode_index=mode_index)
    normal_dim = axis
    if normal_dim in selected.dims:
        selected = selected.isel({normal_dim: 0})
    selected = selected.transpose(*transverse_axes)
    return np.asarray(selected.values, dtype=np.complex128)


def _sort_modes(
    candidates: list[_ModeCandidate], spec: ModePlaneSpec
) -> list[_ModeCandidate]:
    def rank(item):
        polarization = (
            spec.polarization is None
            or _polarization_fraction(item["fields"], spec.axis, spec.polarization)
            >= 0.5
        )
        confined = (
            spec.min_aperture_fraction is None
            or _aperture_fraction(item, spec) >= spec.min_aperture_fraction
        )
        return polarization, confined, float(np.real(item["neff"]))

    return sorted(candidates, key=rank, reverse=True)


def _aperture_fraction(candidate: _ModeCandidate, spec: ModePlaneSpec) -> float:
    """Area-weighted transverse electric intensity inside the injection aperture.

    This is a selection diagnostic, not electromagnetic energy or modal power.
    Grouping by a threshold preserves neff ordering among confined modes.
    """
    axes = spec.transverse_axes
    extents = {a: v for a, v in zip(axes, (spec.height, spec.width), strict=True)}
    edges = [_axis_edges(spec, a) for a in axes]
    centers = [(e[:-1] + e[1:]) / 2 for e in edges]
    masks = [
        abs(c - spec.center[_AXIS_INDEX[a]]) <= 0.5 * extents[a]
        for a, c in zip(axes, centers, strict=True)
    ]
    area = np.multiply.outer(*(np.diff(e) for e in edges))
    intensity = sum(abs(candidate["fields"]["E" + a]) ** 2 for a in axes)
    density = intensity * area
    return float(
        np.sum(density * np.logical_and.outer(*masks))
        / max(float(np.sum(density)), np.finfo(float).tiny)
    )


def _polarization_fraction(
    fields: dict[str, np.ndarray], axis: AxisName, polarization: PolarizationName | None
) -> float:
    if polarization is None:
        return 1.0
    tangential_axes = tuple(idx for idx in range(3) if idx != _AXIS_INDEX[axis])
    first = fields[f"E{_AXIS_NAMES[tangential_axes[0]]}"]
    second = fields[f"E{_AXIS_NAMES[tangential_axes[1]]}"]
    numerator = np.sum(
        np.abs(first) ** 2 if polarization == "te" else np.abs(second) ** 2
    )
    denominator = np.sum(np.abs(first) ** 2 + np.abs(second) ** 2) + 1e-18
    return float(np.real(numerator / denominator))


def _select_phase_reference_component(
    axis: AxisName,
    polarization: PolarizationName | None,
    fields: dict[str, np.ndarray],
) -> str:
    preferred = {
        ("x", "tm"): "Hy",
        ("x", "te"): "Hz",
        ("y", "tm"): "Hx",
        ("y", "te"): "Hz",
        ("z", "tm"): "Hx",
        ("z", "te"): "Hy",
    }
    if polarization is not None and (axis, polarization) in preferred:
        candidate = preferred[(axis, polarization)]
        if np.max(np.abs(fields[candidate])) >= 1e-9:
            return candidate
    tangential_h = {"x": ("Hy", "Hz"), "y": ("Hx", "Hz"), "z": ("Hx", "Hy")}[axis]
    strengths = [float(np.max(np.abs(fields[name]))) for name in tangential_h]
    return tangential_h[int(np.argmax(strengths))]


def _dominant_phase(field: np.ndarray) -> float:
    flat = np.asarray(field, dtype=np.complex128).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(np.angle(flat[int(np.argmax(np.abs(flat)))]))


def _build_profiles(
    fields: dict[str, np.ndarray],
    spec: ModePlaneSpec,
) -> tuple[dict[str, np.ndarray], dict[str, ComponentIndex], dict[str, float]]:
    transverse_axes = _solver_axes_for_axis(spec.axis)[::-1]
    bounds = [
        _padded_bounds(
            spec.center[_AXIS_INDEX[axis]],
            extent,
            spec.resolution,
            spec.grid_shape[2 - _AXIS_INDEX[axis]],
            spec.aperture_pad_cells,
            edges=_axis_edges(spec, axis),
        )
        for axis, extent in zip(transverse_axes, (spec.height, spec.width), strict=True)
    ]
    row_start, row_stop = bounds[0]
    col_start, col_stop = bounds[1]
    normal_dimension = 2 - _AXIS_INDEX[spec.axis]
    staggered, indices = {}, {}
    for component in _COMPONENTS:
        offsets = _YEE_OFFSETS_3D[component]
        values = fields[component]
        if component == "H" + spec.axis:
            values = _stagger_both(values, spec=spec)
        else:
            for dimension in (1, 0):
                axis = transverse_axes[dimension]
                if offsets[axis] == 0.5:
                    values = _stagger_half(
                        values, dimension, coordinates=_axis_sampling(spec, axis)
                    )
        staggered[component] = values
        normal_index = (
            spec.offset_index if offsets[spec.axis] == 0.5 else spec.plane_index
        )
        if spec.axis == "z":
            # z-normal electric/magnetic planes use family-specific indices and
            # clip to each component's native normal support.
            normal_index = int(
                np.clip(
                    spec.plane_index
                    if component.startswith("E")
                    else spec.offset_index,
                    0,
                    spec.grid_shape[0] - 1 - int(offsets["z"] == 0.5),
                )
            )
        support: list[slice | int] = list(
            _support_slices(
                component,
                spec.axis,
                row_start,
                row_stop,
                col_start,
                col_stop,
                values.shape,
            )
        )
        support.insert(normal_dimension, normal_index)
        indices[component] = cast(ComponentIndex, tuple(support))
    direction_sign = _direction_sign(spec.direction)
    profiles = _crop_window_all(
        staggered, row_start, row_stop, col_start, col_stop, direction_sign, spec
    )
    if spec.axis == "y" and direction_sign < 0.0:
        for component in ("Ex", "Ey", "Ez"):
            profiles[component] = -profiles[component]
    initial_power = _normalize_profiles(
        profiles,
        axis=spec.axis,
        measure=_profile_integration_weights(spec, profiles, indices),
        direction_sign=direction_sign,
    )
    return profiles, indices, {"initial_power": initial_power}


def _transpose_between_axes(
    values: np.ndarray,
    src_axes: tuple[AxisName, AxisName],
    dst_axes: tuple[AxisName, AxisName],
) -> np.ndarray:
    return np.transpose(
        np.asarray(values, dtype=np.complex128),
        [src_axes.index(axis) for axis in dst_axes],
    )


def _solver_axes_for_axis(axis: AxisName) -> tuple[AxisName, AxisName]:
    return cast(
        tuple[AxisName, AxisName],
        tuple(value for value in _AXIS_NAMES if value != axis),
    )


def _axis_edges(spec: ModePlaneSpec, axis: AxisName) -> np.ndarray:
    count = spec.grid_shape[{"z": 0, "y": 1, "x": 2}[axis]]
    if spec.grid is None:
        return np.arange(count + 1, dtype=float) * float(spec.resolution)
    return np.asarray(spec.grid.axis_edges(axis), dtype=float)


def _axis_sampling(
    spec: ModePlaneSpec, axis: AxisName
) -> tuple[np.ndarray, np.ndarray]:
    edges = _axis_edges(spec, axis)
    return 0.5 * (edges[:-1] + edges[1:]), edges[1:-1]


def _stagger_half(
    field: np.ndarray,
    axis: int,
    coordinates: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    if field.shape[axis] <= 1:
        return field
    low = np.take(field, np.arange(field.shape[axis] - 1), axis=axis)
    high = np.take(field, np.arange(1, field.shape[axis]), axis=axis)
    if coordinates is None:
        alpha = np.full(field.shape[axis] - 1, 0.5, dtype=float)
    else:
        source, target = coordinates
        source = np.asarray(source, dtype=float)[: field.shape[axis]]
        target = np.asarray(target, dtype=float)[: field.shape[axis] - 1]
        alpha = (target - source[:-1]) / np.maximum(
            source[1:] - source[:-1], np.finfo(float).tiny
        )
    shape = [1] * field.ndim
    shape[axis] = alpha.size
    weights = alpha.reshape(shape)
    return (1.0 - weights) * low + weights * high


def _stagger_both(
    field: np.ndarray, *, spec: ModePlaneSpec | None = None
) -> np.ndarray:
    out = field
    sampling1 = None if spec is None else _axis_sampling(spec, spec.transverse_axes[1])
    sampling0 = None if spec is None else _axis_sampling(spec, spec.transverse_axes[0])
    if out.shape[1] > 1:
        out = _stagger_half(out, 1, sampling1)
    if out.shape[0] > 1:
        out = _stagger_half(out, 0, sampling0)
    return out


def _padded_bounds(
    center_value: float,
    extent: float,
    resolution: float,
    limit: int,
    pad_cells: int,
    *,
    edges: np.ndarray | None = None,
) -> tuple[int, int]:
    if edges is not None:
        edge_array = np.asarray(edges, dtype=float)
        lower = float(center_value) - 0.5 * float(extent)
        upper = float(center_value) + 0.5 * float(extent)
        start = int(np.searchsorted(edge_array, lower, side="right") - 1)
        stop = int(np.searchsorted(edge_array, upper, side="left"))
        start = max(0, start - max(0, int(pad_cells)))
        stop = min(int(limit), stop + max(0, int(pad_cells)))
        if stop - start < 2:
            center_idx = int(
                np.argmin(
                    np.abs(0.5 * (edge_array[:-1] + edge_array[1:]) - center_value)
                )
            )
            start = max(0, center_idx - 1)
            stop = min(int(limit), start + 2)
        return start, stop
    padded = float(extent) + 2.0 * max(0, int(pad_cells)) * float(resolution)
    center_idx = round(float(center_value) / float(resolution))
    half = max(1, round(0.5 * padded / float(resolution)))
    return max(0, center_idx - half), min(int(limit), center_idx + half)


def _support_slices(
    component: str,
    axis: AxisName,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
    field_shape: tuple[int, int],
) -> tuple[slice, slice]:
    row_axis, col_axis = {"x": ("z", "y"), "y": ("z", "x"), "z": ("y", "x")}[axis]
    row_stop = _support_stop_for_offset(
        row_start, row_stop, _YEE_OFFSETS_3D[component][row_axis]
    )
    col_stop = _support_stop_for_offset(
        col_start, col_stop, _YEE_OFFSETS_3D[component][col_axis]
    )
    return (
        slice(row_start, min(row_stop, int(field_shape[0]))),
        slice(col_start, min(col_stop, int(field_shape[1]))),
    )


def _support_stop_for_offset(start: int, stop: int, offset: float) -> int:
    if float(offset) == 0.5 and int(stop) - int(start) > 1:
        return int(stop) - 1
    return int(stop)


def _crop_window_all(
    staggered: dict[str, np.ndarray],
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
    direction_sign: float,
    spec: ModePlaneSpec,
) -> dict[str, np.ndarray]:
    profiles = {}
    row_axis, col_axis = {"x": ("z", "y"), "y": ("z", "x"), "z": ("y", "x")}[spec.axis]
    for component, values in staggered.items():
        comp_row_stop = _support_stop_for_offset(
            row_start, row_stop, _YEE_OFFSETS_3D[component][row_axis]
        )
        comp_col_stop = _support_stop_for_offset(
            col_start, col_stop, _YEE_OFFSETS_3D[component][col_axis]
        )
        row_end = min(comp_row_stop, values.shape[0])
        col_end = min(comp_col_stop, values.shape[1])
        cropped = values[row_start:row_end, col_start:col_end]
        window = _tukey2d(
            cast(tuple[int, int], cropped.shape), alpha=spec.aperture_window_alpha
        )
        profiles[component] = direction_sign * cropped * window
    return profiles


def _component_axis_measure(
    spec: ModePlaneSpec,
    component: str,
    axis: AxisName,
    selector: slice | int,
) -> np.ndarray:
    widths = np.diff(_axis_edges(spec, axis))
    if _YEE_OFFSETS_3D[component][axis] == 0.5:
        values = widths
    else:
        values = np.empty(widths.size + 1, dtype=float)
        values[0], values[-1] = 0.5 * widths[0], 0.5 * widths[-1]
        if widths.size > 1:
            values[1:-1] = 0.5 * (widths[:-1] + widths[1:])
    if isinstance(selector, slice):
        return np.asarray(values[selector], dtype=float)
    return np.asarray([values[int(selector)]], dtype=float)


def _profile_integration_weights(
    spec: ModePlaneSpec,
    profiles: Mapping[str, np.ndarray],
    indices: Mapping[str, ComponentIndex],
) -> dict[str, np.ndarray]:
    """Return component-staggered transverse area weights for modal flux."""
    weights = {}
    transverse_axes: dict[AxisName, tuple[AxisName, AxisName]] = {
        "x": ("z", "y"),
        "y": ("z", "x"),
        "z": ("y", "x"),
    }
    row_axis, col_axis = transverse_axes[spec.axis]
    axis_pos = {"z": 0, "y": 1, "x": 2}
    for component in _COMPONENTS:
        if component not in profiles or component not in indices:
            continue
        index = indices[component]
        row = _component_axis_measure(
            spec, component, row_axis, index[axis_pos[row_axis]]
        )
        col = _component_axis_measure(
            spec, component, col_axis, index[axis_pos[col_axis]]
        )
        shape = np.atleast_2d(np.asarray(profiles[component])).shape
        weights[component] = row[: shape[0], None] * col[None, : shape[1]]
    return weights


def _tukey2d(shape: tuple[int, int], alpha: float) -> np.ndarray:
    rows, cols = shape
    return _tukey(rows, alpha)[:, None] * _tukey(cols, alpha)[None, :]


def _tukey(count: int, alpha: float) -> np.ndarray:
    if count <= 0:
        return np.ones((0,), dtype=np.float64)
    if count == 1 or count <= 2:
        return np.ones((count,), dtype=np.float64)
    n = np.arange(count, dtype=np.float64)
    width = max(float(alpha) * (count - 1) / 2.0, np.finfo(float).eps)
    left = 0.5 * (1.0 + np.cos(np.pi * (n / width - 1.0)))
    right = 0.5 * (1.0 + np.cos(np.pi * ((n - (count - 1 - width)) / width)))
    return np.where(n < width, left, np.where(n > (count - 1) - width, right, 1.0))


def _normalize_profiles_by_phase_referenced_flux(
    profiles: dict[str, np.ndarray],
    indices: dict[str, ComponentIndex],
    *,
    axis: AxisName,
    d_area,
    direction_sign: float,
    omega: float,
    k_num: float,
    ref_coord: float,
    resolution: float,
    grid=None,
) -> tuple[dict[str, np.ndarray], float, float]:
    referenced = _phase_reference_profiles(
        profiles,
        indices,
        axis=axis,
        omega=omega,
        k_num=k_num,
        ref_coord=ref_coord,
        resolution=resolution,
        grid=grid,
    )
    flux = _modal_power(
        referenced,
        axis=axis,
        measure=d_area,
        direction_sign=direction_sign,
    )
    if (not np.isfinite(flux)) or abs(flux) <= np.finfo(float).tiny:
        return profiles, 1.0, float(flux)
    scale = float(np.sqrt(1.0 / abs(flux)))
    return (
        {
            key: np.asarray(value, dtype=np.complex128) * scale
            for key, value in profiles.items()
        },
        scale,
        float(flux) * scale**2,
    )


def _phase_reference_profiles(
    profiles: dict[str, np.ndarray],
    indices: dict[str, ComponentIndex],
    *,
    axis: AxisName,
    omega: float,
    k_num: float,
    ref_coord: float,
    resolution: float,
    grid=None,
) -> dict[str, np.ndarray]:
    out = {}
    for component, value in profiles.items():
        axis_idx = _axis_index(indices.get(component), axis)
        coord = _axis_coordinate(component, axis_idx, axis, resolution, grid)
        delay = _phase_delay(omega, k_num, coord - ref_coord)
        out[component] = np.asarray(value, dtype=np.complex128) * np.exp(
            -1j * omega * delay
        )
    return out


def _runtime_oriented_profiles(
    profiles: dict[str, np.ndarray],
    axis: AxisName,
    direction_sign: float,
) -> dict[str, np.ndarray]:
    out = {
        key: np.asarray(value, dtype=np.complex128) for key, value in profiles.items()
    }
    if axis != "y":
        return out
    if direction_sign > 0.0:
        out["Ex"] = -out["Ex"]
        out["Hz"] = -out["Hz"]
    else:
        out["Ez"] = -out["Ez"]
        out["Hx"] = -out["Hx"]
    return out


def _backward_mode_from_forward(
    profiles: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        key: (-value if key.startswith("H") else value.copy())
        for key, value in profiles.items()
    }


def _direction_sign(direction: str) -> float:
    return 1.0 if str(direction).startswith("+") else -1.0
