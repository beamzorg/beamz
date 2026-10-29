"""SciPy/ARPACK reference eigensolvers for diagonal and tensorial modes.

The numerical implementation was imported from MicroMode ``scipy_reference.py``
(see ``UPSTREAM.md``). Formulas live in the private modules next to this file.
``_selected_eigenpairs`` is resolved here so tests can replace it.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from ._scipy_constants import ETA0
from ._scipy_derivatives import _create_derivative_matrices
from ._scipy_eigen import (
    _import_scipy,
    _real_arpack_problem_if_close,
    _selected_eigenpairs,
)
from ._scipy_fields import _lorentz_orthogonalize_and_normalize, _ModeFields
from ._scipy_operators import (
    _assemble_diagonal_operators,
    _assemble_tensorial_operator,
)

SparseSolveResult = tuple[np.ndarray, list[np.ndarray], dict[str, object]]


def _require_solver_inputs(
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    flattened_size: int,
    num_modes: int,
) -> None:
    """Reject tensor layouts that do not match the flattened transverse grid."""
    if (
        eps_tensor.shape != mu_tensor.shape
        or eps_tensor.shape[:2] != (3, 3)
        or eps_tensor.shape[-1] != flattened_size
    ):
        raise ValueError(
            "eps_tensor and mu_tensor must both have shape "
            "(3, 3, len(dlf[0]) * len(dlf[1]))"
        )
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")


def _normalized_eigen_residuals(operator, values, vectors) -> np.ndarray:
    """Scale each eigen-residual by the corresponding eigenvector norm."""
    residuals = np.asarray(
        [
            np.linalg.norm(
                operator @ vectors[:, index] - values[index] * vectors[:, index]
            )
            for index in range(len(values))
        ],
        dtype=float,
    )
    vector_norms = np.maximum(np.linalg.norm(vectors, axis=0), np.finfo(float).eps)
    return residuals / vector_norms


def _sort_by_descending_real_index(
    modes: list[tuple[complex, np.ndarray, float]],
) -> list[tuple[complex, np.ndarray, float]]:
    """Order modes by descending real effective index."""
    modes.sort(key=lambda item: item[0].real, reverse=True)
    return modes


def _transverse_cell_areas(
    dlf: tuple[np.ndarray, np.ndarray], nx: int, ny: int
) -> np.ndarray:
    """Cell areas in flattened x-major order."""
    return np.repeat(np.asarray(dlf[0], dtype=float), ny) * np.tile(
        np.asarray(dlf[1], dtype=float), nx
    )


def _apply_output_field_convention(
    hx: np.ndarray,
    hy: np.ndarray,
    hz: np.ndarray,
    ez: np.ndarray,
    direction: str,
) -> None:
    """Convert H to Result units and flip signs for backward propagation."""
    h_scale = -1j / ETA0
    hx *= h_scale
    hy *= h_scale
    hz *= h_scale
    if direction == "-":
        hx *= -1.0
        hy *= -1.0
        ez *= -1.0


def _package_reference_modes(
    n_complex: list[complex],
    mode_fields: list[_ModeFields],
    residuals: list[float],
    operator,
    cell_areas: np.ndarray,
) -> SparseSolveResult:
    """Lorentz-normalize the set and return component-major fields."""
    orthogonalization = _lorentz_orthogonalize_and_normalize(mode_fields, cell_areas)
    fields = [
        np.asarray(
            [getattr(mode, component) for mode in mode_fields], dtype=np.complex128
        )
        for component in ("ex", "ey", "ez", "hx", "hy", "hz")
    ]
    return (
        np.asarray(n_complex, dtype=np.complex128),
        fields,
        {
            "backend": "scipy_arpack_reference",
            "operator_size": int(operator.shape[0]),
            "operator_nnz": int(operator.nnz),
            "residuals": np.asarray(residuals, dtype=float),
            "power_norms": orthogonalization["power_norms"],
            "lorentz_norms": orthogonalization["lorentz_norms"],
            "lorentz_orthogonality_error": orthogonalization[
                "lorentz_orthogonality_error"
            ],
        },
    )


def _stretched_derivatives(
    sparse,
    *,
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    omega: float | None,
    num_pml: tuple[int, int],
    pml_profile: dict[str, float | int] | None,
    dmin_pml: tuple[bool, bool],
    dmin_pmc: tuple[bool, bool],
    derivative_scale: float,
):
    """Build the shared Yee derivatives, including optional PML stretching."""
    return _create_derivative_matrices(
        sparse,
        eps_tensor=eps_tensor,
        mu_tensor=mu_tensor,
        shape=(len(dlf[0]), len(dlf[1])),
        dlf=dlf,
        dlb=dlb,
        omega=omega,
        num_pml=num_pml,
        pml_profile=pml_profile,
        dmin_pml=dmin_pml,
        dmin_pmc=dmin_pmc,
        scale=float(derivative_scale),
    )


def solve_diagonal_scipy_reference(**kwargs) -> SparseSolveResult:
    """Solve the diagonal sparse eigenproblem with SciPy/ARPACK.

    This is the reduced ``[Ex, Ey]`` transverse eigenproblem for diagonal
    material tensors.
    """
    return _solve_scipy_reference(False, **kwargs)


def _reconstruct_diagonal_modes(
    modes: list[tuple[complex, np.ndarray, float]],
    *,
    n: int,
    operators,
    derivatives,
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    sparse,
    direction: str,
) -> tuple[list[complex], list[_ModeFields], list[float]]:
    """Recover six field components from the reduced ``[Ex, Ey]`` eigenvectors."""
    inv_eps_zz = sparse.diags(1.0 / eps_tensor[2, 2, :], format="csc")
    inv_mu_zz = sparse.diags(1.0 / mu_tensor[2, 2, :], format="csc")
    dxf, dxb, dyf, dyb = derivatives

    n_complex: list[complex] = []
    mode_fields: list[_ModeFields] = []
    sorted_residuals: list[float] = []
    for mode_n, vector, residual in modes:
        # The eigenvector stores Ex then Ey in flattened cell order.
        ex = vector[:n].copy()
        ey = vector[n:].copy()
        denom = complex(-mode_n.imag, mode_n.real)

        # Magnetic transverse components come from the Q operator divided by the
        # effective-index factor.
        h_field = operators["qmat"] @ vector
        hx = np.asarray(h_field[:n] / denom, dtype=np.complex128)
        hy = np.asarray(h_field[n:] / denom, dtype=np.complex128)

        # Longitudinal H follows from the curl of transverse E.
        hz_source = dxf @ ey - dyf @ ex
        hz = np.asarray(inv_mu_zz @ hz_source, dtype=np.complex128)

        # Longitudinal E follows from the curl of the material electric block.
        h_partial = np.asarray(
            (operators["q_ep"] @ vector) / denom, dtype=np.complex128
        )
        ez_source = dxb @ h_partial[n:] - dyb @ h_partial[:n]
        ez = np.asarray(inv_eps_zz @ ez_source, dtype=np.complex128)

        _apply_output_field_convention(hx, hy, hz, ez, direction)
        n_complex.append(mode_n)
        sorted_residuals.append(residual)
        mode_fields.append(_ModeFields(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz))
    return n_complex, mode_fields, sorted_residuals


def solve_tensorial_scipy_reference(**kwargs) -> SparseSolveResult:
    """Solve the first-order tensorial sparse eigenproblem with SciPy/ARPACK."""
    return _solve_scipy_reference(True, **kwargs)


def _solve_scipy_reference(
    tensorial: bool,
    *,
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    num_modes: int,
    neff_guess: float,
    direction: str,
    derivative_scale: float,
    omega: float | None = None,
    num_pml: tuple[int, int] = (0, 0),
    pml_profile: dict[str, float | int] | None = None,
    dmin_pml: tuple[bool, bool] = (True, True),
    dmin_pmc: tuple[bool, bool] = (False, False),
    krylov_dim: int = 32,
    initial_vector: np.ndarray | None = None,
) -> SparseSolveResult:
    """Run one ARPACK reference solve for either material formulation."""
    sparse, spla, scipy_linalg = _import_scipy()
    nx = len(dlf[0])
    ny = len(dlf[1])
    n = nx * ny
    _require_solver_inputs(eps_tensor, mu_tensor, n, num_modes)
    derivatives = _stretched_derivatives(
        sparse,
        eps_tensor=eps_tensor,
        mu_tensor=mu_tensor,
        dlf=dlf,
        dlb=dlb,
        omega=omega,
        num_pml=num_pml,
        pml_profile=pml_profile,
        dmin_pml=dmin_pml,
        dmin_pmc=dmin_pmc,
        derivative_scale=derivative_scale,
    )
    operators = None
    if tensorial:
        operator = _assemble_tensorial_operator(
            sparse, eps_tensor, mu_tensor, derivatives
        )
        sigma = complex(neff_guess, 0.0)
        arpack_initial = initial_vector
    else:
        operators = _assemble_diagonal_operators(
            sparse, eps_tensor, mu_tensor, derivatives
        )
        operator = cast(Any, operators["mat"])
        operator, arpack_initial, sigma = _real_arpack_problem_if_close(
            operator, initial_vector, complex(-(neff_guess * neff_guess), 0.0)
        )
    values, vectors = _selected_eigenpairs(
        operator,
        num_modes=num_modes,
        sigma=sigma,
        krylov_dim=krylov_dim,
        initial_vector=arpack_initial,
        spla=spla,
        scipy_linalg=scipy_linalg,
    )
    residuals = _normalized_eigen_residuals(operator, values, vectors)
    modes = _sort_by_descending_real_index(
        [
            (
                complex(value if tensorial else np.sqrt(-value + 0j)),
                np.asarray(vector, dtype=np.complex128),
                float(residual),
            )
            for value, vector, residual in zip(
                values, vectors.T, residuals, strict=True
            )
        ]
    )
    reconstruct = (
        _reconstruct_tensorial_modes if tensorial else _reconstruct_diagonal_modes
    )
    reconstruct_kwargs = {
        "n": n,
        "derivatives": derivatives,
        "eps_tensor": eps_tensor,
        "mu_tensor": mu_tensor,
        "sparse": sparse,
        "direction": direction,
    }
    if not tensorial:
        reconstruct_kwargs["operators"] = operators
    n_complex, mode_fields, sorted_residuals = reconstruct(modes, **reconstruct_kwargs)
    return _package_reference_modes(
        n_complex,
        mode_fields,
        sorted_residuals,
        operator,
        _transverse_cell_areas(dlf, nx, ny),
    )


def _reconstruct_tensorial_modes(
    modes: list[tuple[complex, np.ndarray, float]],
    *,
    n: int,
    derivatives,
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    sparse,
    direction: str,
) -> tuple[list[complex], list[_ModeFields], list[float]]:
    """Recover six field components from ``[Ex, Ey, Hx, Hy]`` eigenvectors."""
    inv_eps_zz = sparse.diags(1.0 / eps_tensor[2, 2, :], format="csc")
    inv_mu_zz = sparse.diags(1.0 / mu_tensor[2, 2, :], format="csc")
    dxf, dxb, dyf, dyb = derivatives

    n_complex: list[complex] = []
    mode_fields: list[_ModeFields] = []
    sorted_residuals: list[float] = []
    for mode_n, vector, residual in modes:
        # The first-order eigenvector stores transverse E followed by transverse H.
        ex = vector[:n].copy()
        ey = vector[n : 2 * n].copy()
        hx = vector[2 * n : 3 * n].copy()
        hy = vector[3 * n : 4 * n].copy()

        # Longitudinal fields include off-diagonal material coupling through the
        # z-row of the tensor.
        hz_source = (
            dxf @ ey - dyf @ ex - mu_tensor[2, 0, :] * hx - mu_tensor[2, 1, :] * hy
        )
        hz = np.asarray(inv_mu_zz @ hz_source, dtype=np.complex128)

        ez_source = (
            dxb @ hy - dyb @ hx - eps_tensor[2, 0, :] * ex - eps_tensor[2, 1, :] * ey
        )
        ez = np.asarray(inv_eps_zz @ ez_source, dtype=np.complex128)

        _apply_output_field_convention(hx, hy, hz, ez, direction)
        n_complex.append(mode_n)
        sorted_residuals.append(residual)
        mode_fields.append(_ModeFields(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz))
    return n_complex, mode_fields, sorted_residuals
