"""Rasterized mode-solver API for Beamz-style grid inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np

from ._bend import bend_materials
from ._scipy import solve_diagonal_scipy_reference, solve_tensorial_scipy_reference
from .constants import C_0
from .models import BoundaryCondition, BoundarySpec, Materials, PmlSpec
from .result import Result

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def solve_grid(
    *,
    material_grid: Materials,
    freqs: Sequence[float] | None = None,
    wavelength: float | Sequence[float] | None = None,
    num_modes: int = 1,
    target_neff: float | None = None,
    pml: PmlSpec | tuple[int, int] | None = None,
    boundary: BoundarySpec | tuple[str, str] | None = None,
    direction: Literal["+", "-"] = "+",
    components: Sequence[str] | None = None,
    krylov_dim: int | None = None,
    bend_radius: float | None = None,
    bend_axis: Literal["x", "y"] = "x",
) -> Result:
    """Solve modes for an already-rasterized material tensor grid.

    BeamZ owns geometry and material rasterization. Construct a Materials value
    once, then solve and prepare its fields without another tensor/axis adapter.
    Edges, wavelength and bend_radius are in microns; frequencies are in Hz.
    Finite signed bend radii apply Maxwell's tensor pullback in the selected
    local radial coordinate. Reference arc length is measured at that coordinate's
    zero: positive radius puts the curvature center at coordinate -R, and negative
    radius reverses the bend. The complete domain must satisfy 1 + coordinate/R > 0.
    Material tensors use the local frame and must follow the bend. Returned fields
    are physical components at the reference cross-section.
    """
    # Validate the owned material grid, solve each frequency, and preserve the
    # canonical array layout through preview and launch preparation. Labeled
    # coordinate views belong only to Result's explicit public boundary.
    if not isinstance(material_grid, Materials):
        raise TypeError("material_grid must be a Materials")

    # Validate edge arrays against the tensor shape and resolve the frequency
    # input before any expensive sparse work starts.
    shape = material_grid.shape
    x_edges_arr = _validate_edges("x_edges", material_grid.grid.x_edges, shape[0])
    y_edges_arr = _validate_edges("y_edges", material_grid.grid.y_edges, shape[1])
    solve_freqs = _resolve_freqs(freqs=freqs, wavelength=wavelength)

    # Normalize scalar options and compact model objects at the API boundary.
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")
    if direction not in {"+", "-"}:
        raise ValueError("direction must be '+' or '-'")
    pml_spec = _resolve_pml_spec(pml)
    boundary_spec = _resolve_boundary_spec(boundary)
    # Component filtering happens after solving so field reconstruction still has
    # every component needed for normalization and coordinate conversion.
    requested_components = tuple(components or _COMPONENTS)
    unknown = set(requested_components).difference(_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown field component(s): {', '.join(sorted(unknown))}")

    bend_metric = None
    if bend_radius is not None:
        material_grid, bend_metric = bend_materials(
            material_grid, bend_radius, bend_axis
        )

    # Keep frequency rows as raw arrays for component filtering and stacking.
    # No labeled field objects are allocated by the numerical solve.
    n_rows = []
    solver_runs = []
    fields_by_component: dict[str, list[np.ndarray]] = {
        component: [] for component in requested_components
    }
    for freq in solve_freqs:
        # The sparse solver is frequency-local; multi-frequency solves are a
        # deterministic loop plus a final frequency/mode/plane array stack.
        n_complex, fields, solver_info = _solve_one_frequency(
            x_edges=x_edges_arr,
            y_edges=y_edges_arr,
            freq=float(freq),
            num_modes=num_modes,
            target_neff=target_neff,
            pml_spec=pml_spec,
            direction=direction,
            krylov_dim=krylov_dim,
            boundary_spec=boundary_spec,
            material_grid=material_grid,
        )
        if bend_metric is not None:
            # Coordinate fields obey E'_z=h E_z and H'_z=h H_z. Transverse
            # fields (and hence cross-section power normalization) are unchanged.
            fields["Ez"] = fields["Ez"] / bend_metric
            fields["Hz"] = fields["Hz"] / bend_metric
        # The solver uses local coordinates where local z is the propagation
        # normal. Convert field labels back to the global x/y/z axes requested
        # by the material grid before exposing them.
        fields = _local_fields_to_global(
            fields, normal_axis=material_grid.grid.normal_axis
        )
        n_rows.append(n_complex)
        solver_runs.append(solver_info)
        for component, values in fields.items():
            if component in fields_by_component:
                fields_by_component[component].append(values)

    return Result(
        neffs=np.asarray(n_rows, dtype=np.complex128),
        fields={name: np.stack(rows) for name, rows in fields_by_component.items()},
        grid=material_grid.grid,
        frequencies=np.asarray(solve_freqs),
        solver_info={
            "backend": solver_runs[0].get("backend") if solver_runs else None,
            "runs": solver_runs,
            "pml": pml_spec.as_dict(),
            "boundary": boundary_spec.as_dict(),
            "normal_axis": material_grid.grid.normal_axis,
            **(
                {"bend_radius": float(bend_radius), "bend_axis": bend_axis}
                if bend_radius is not None
                else {}
            ),
        },
    )


def _solve_one_frequency(
    *,
    material_grid: Materials,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    freq: float,
    num_modes: int,
    target_neff: float | None,
    pml_spec: PmlSpec,
    direction: str,
    krylov_dim: int | None,
    boundary_spec: BoundarySpec,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Select the sparse formulation and solve a single frequency."""
    # Forward and backward spacings represent the local Yee grid. The derivative
    # builders need both because E and H components are staggered.
    dlf = (np.diff(x_edges), np.diff(y_edges))
    dlb = (_dual_steps(dlf[0]), _dual_steps(dlf[1]))

    # Flatten tensors to the layout expected by scipy_reference while preserving
    # the original grid metadata for result coordinates.
    eps_tensor = material_grid.flat_eps_tensor()
    mu_tensor = material_grid.flat_mu_tensor()
    if target_neff is None:
        # Default to the highest local material index. Users can supply a target
        # explicitly when hunting for modes near another branch.
        target_neff = float(np.sqrt(np.max(np.abs(eps_tensor))))
    target_neff = _shift_target_neff(float(target_neff))

    if material_grid.is_diagonal:
        solve, unknowns, backend_kind = (
            solve_diagonal_scipy_reference,
            2,
            "diagonal_scipy_reference",
        )
    else:
        solve, unknowns, backend_kind = (
            solve_tensorial_scipy_reference,
            4,
            "tensorial_scipy_reference",
        )
    nx, ny = material_grid.shape
    actual_krylov_dim = 32 if krylov_dim is None else int(krylov_dim)

    # Convert frequency into the nondimensional derivative scale and angular
    # frequency expected by the sparse reference implementation.
    n_complex, fields, solver_info = solve(
        eps_tensor=eps_tensor,
        mu_tensor=mu_tensor,
        dlf=dlf,
        dlb=dlb,
        num_modes=num_modes,
        neff_guess=target_neff,
        direction=direction,
        derivative_scale=C_0 / (2 * np.pi * freq),
        omega=2 * np.pi * freq,
        num_pml=pml_spec.num_cells,
        pml_profile=pml_spec.profile_dict(),
        dmin_pml=boundary_spec.dmin_pml,
        dmin_pmc=boundary_spec.dmin_pmc,
        krylov_dim=actual_krylov_dim,
        initial_vector=_default_initial_vector(unknowns * nx * ny, shape=(nx, ny)),
    )

    # Solver fields are flattened by component; reshape them back to grid
    # tensors and attach dispatch metadata for diagnostics.
    return (
        n_complex,
        {
            name: values.reshape(-1, nx, ny)
            for name, values in zip(_COMPONENTS, fields, strict=True)
        },
        _solver_info_with_context(
            solver_info,
            backend_kind=backend_kind,
            shape=(nx, ny),
            krylov_dim=actual_krylov_dim,
        ),
    )


def _resolve_pml_spec(
    pml: PmlSpec | tuple[int, int] | None,
) -> PmlSpec:
    """Normalize user PML input into a PmlSpec."""
    # The public API accepts compact tuples, but the solver only sees validated
    # PmlSpec objects.
    if pml is None:
        return PmlSpec()
    if isinstance(pml, PmlSpec):
        return pml
    return PmlSpec.from_num_cells(pml)


def _resolve_boundary_spec(
    boundary: BoundarySpec | tuple[str, str] | None,
) -> BoundarySpec:
    """Normalize user boundary input into a BoundarySpec."""
    # BoundarySpec handles case normalization and allowed-value validation.
    if boundary is None:
        return BoundarySpec()
    if isinstance(boundary, BoundarySpec):
        return boundary
    return BoundarySpec(low=cast(tuple[BoundaryCondition, BoundaryCondition], boundary))


def _solver_info_with_context(
    solver_info: dict[str, object],
    *,
    backend_kind: str,
    shape: tuple[int, int],
    krylov_dim: int,
) -> dict[str, object]:
    """Attach Python-side dispatch metadata to solver diagnostics."""
    # Add enough Python-side context for saved Result files and benchmark
    # reports to be self-describing.
    out = dict(solver_info)
    out["backend_kind"] = backend_kind
    out["shape"] = shape
    out["krylov_dim"] = krylov_dim
    out["phase_convention"] = "dominant_e_real_positive"
    out["normalization"] = "lorentz_orthogonal_unit_transverse_power"
    return out


def _shift_target_neff(target_neff: float) -> float:
    """Nudge the shift target away from exact eigenvalues for ARPACK stability."""
    # Exact shifts can make shift-invert factorization singular; the tiny nudge
    # preserves the requested target while avoiding exact equality.
    target_shift = float(10 * np.finfo(np.float32).eps)
    if abs(target_shift) > abs(target_neff * target_shift):
        return target_neff + target_shift
    return target_neff * (1.0 + target_shift)


def _validate_edges(name: str, values: Sequence[float], cell_count: int) -> np.ndarray:
    """Validate mode-plane edge coordinates against a cell count."""
    # Edge validation is repeated here because callers may provide an existing
    # Materials object from outside the dataclass constructors.
    edges = np.asarray(values, dtype=float)
    if edges.shape != (cell_count + 1,):
        raise ValueError(f"{name} must have length {cell_count + 1}")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return edges


def _resolve_freqs(
    *,
    freqs: Sequence[float] | None,
    wavelength: float | Sequence[float] | None,
) -> tuple[float, ...]:
    """Resolve exactly one frequency or wavelength input into frequencies."""
    # Accept either frequencies or wavelengths, never both, so the solve axis is
    # unambiguous.
    if (freqs is None) == (wavelength is None):
        raise ValueError("provide exactly one of freqs or wavelength")
    if freqs is not None:
        values = tuple(float(freq) for freq in freqs)
    else:
        # Wavelengths are in microns because C_0 is stored in um/s.
        wavelengths = np.asarray(wavelength, dtype=float).reshape(-1)
        values = tuple(float(C_0 / value) for value in wavelengths)
    if not values or any(not np.isfinite(freq) or freq <= 0 for freq in values):
        raise ValueError("frequencies must be finite and positive")
    return values


def _dual_steps(primal_steps: np.ndarray) -> np.ndarray:
    """Return backward-grid step sizes from primal cell widths."""
    # Interior backward steps live halfway between adjacent primal cells.
    if len(primal_steps) == 1:
        return primal_steps.copy()
    return np.hstack((primal_steps[0], (primal_steps[:-1] + primal_steps[1:]) / 2))


def _local_fields_to_global(
    fields: dict[str, np.ndarray], *, normal_axis: int
) -> dict[str, np.ndarray]:
    """Map solver-local field components onto global x/y/z component names.

    The SciPy solver uses local coordinates where local z is the
    propagation-normal axis. For x- or y-normal planes, component labels must be
    permuted back to global coordinates before returning a Result.
    """

    axis_names = ("x", "y", "z")
    if normal_axis not in {0, 1, 2}:
        raise ValueError("normal_axis must be 0, 1, or 2")

    # Local solver axes are always tangential-x, tangential-y, normal-z.
    local_to_global = (*(axis for axis in range(3) if axis != normal_axis), normal_axis)
    out: dict[str, np.ndarray] = {}
    for prefix in ("E", "H"):
        for local_axis, global_axis in enumerate(local_to_global):
            local_name = f"{prefix}{axis_names[local_axis]}"
            global_name = f"{prefix}{axis_names[global_axis]}"
            if local_name in fields:
                # For y-normal planes the local (x, z, y) order is left-handed;
                # flipping the local-y tangential component restores physical +y flux.
                sign = -1.0 if normal_axis == 1 and local_axis == 1 else 1.0
                out[global_name] = sign * fields[local_name]
    return out


def _default_initial_vector(
    size: int, shape: tuple[int, int] | None = None
) -> np.ndarray:
    """Build a deterministic ARPACK seed vector for reproducible solves."""
    # When grid shape is known, build a seed with the same flattened cell order
    # as the solver state vector.
    if shape is not None and size % (shape[0] * shape[1]) == 0:
        nx, ny = shape
        multiplier = size // (nx * ny)
        rng = np.random.default_rng(0)
        vector = rng.random((nx, ny, multiplier)) + 1j * rng.random(
            (nx, ny, multiplier)
        )

        # Zero low-edge rows when possible so the initial guess is compatible
        # with symmetry-wall derivative stencils.
        if nx > 1:
            vector[0, :, :] = 0
        if ny > 1:
            vector[:, 0, :] = 0
        stacked = np.concatenate(tuple(vector[ix, :, :] for ix in range(nx)), axis=0)
        return stacked.flatten("F")

    # Fallback for callers that only know the operator size.
    index = np.arange(1, size + 1, dtype=float)
    return np.sin(0.37 * index) + 1j * np.cos(0.53 * index)
