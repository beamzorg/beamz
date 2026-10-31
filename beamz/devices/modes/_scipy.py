"""Readable SciPy implementation for the mode solver.

This module assembles the sparse mode-solver paths in plain Python/SciPy. It
keeps the numerical contract inspectable by users who want to audit the
finite-difference operators against SciPy/ARPACK.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

ETA0 = 376.730_313_666_853_5
MU0 = 1.256_637_062_12e-12
C0 = 2.997_924_58e14
EPSILON0 = 1.0 / (MU0 * C0 * C0)


SparseSolveResult = tuple[np.ndarray, list[np.ndarray], dict[str, object]]


# The SciPy reference layer stays deliberately explicit so finite-difference
# assembly can be audited against the physics model.
@dataclass
class _ModeFields:
    """Mutable six-component field container used during normalization."""

    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray
    hx: np.ndarray
    hy: np.ndarray
    hz: np.ndarray

    def components(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return field components in electric-then-magnetic order."""
        return self.ex, self.ey, self.ez, self.hx, self.hy, self.hz

    def add_scaled(self, other: _ModeFields, scale: complex) -> None:
        """Add another mode into this mode with a complex scale factor."""
        # Orthogonalization operates in-place to avoid repeatedly allocating six
        # large component arrays per mode.
        for left, right in zip(self.components(), other.components(), strict=True):
            left += scale * right


def solve_diagonal_scipy_reference(**options) -> SparseSolveResult:
    """Solve the reduced [Ex, Ey] diagonal sparse eigenproblem with SciPy/ARPACK."""
    return _solve_scipy(diagonal=True, **options)


def solve_tensorial_scipy_reference(**options) -> SparseSolveResult:
    """Solve the first-order [Ex, Ey, Hx, Hy] tensorial eigenproblem with SciPy/ARPACK."""
    return _solve_scipy(diagonal=False, **options)


def _solve_scipy(
    *,
    diagonal: bool,
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
    """Assemble, solve and normalize either sparse constitutive formulation."""
    sparse, spla, scipy_linalg = _import_scipy()
    nx, ny = len(dlf[0]), len(dlf[1])
    n = nx * ny
    if (
        eps_tensor.shape != mu_tensor.shape
        or eps_tensor.shape[:2] != (3, 3)
        or eps_tensor.shape[-1] != n
    ):
        raise ValueError(
            "eps_tensor and mu_tensor must both have shape (3, 3, len(dlf[0]) * len(dlf[1]))"
        )
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")
    # Both formulations use the same staggered derivatives and PML stretching.
    derivatives = _create_derivative_matrices(
        sparse,
        eps_tensor=eps_tensor,
        mu_tensor=mu_tensor,
        shape=(nx, ny),
        dlf=dlf,
        dlb=dlb,
        omega=omega,
        num_pml=num_pml,
        pml_profile=pml_profile,
        dmin_pml=dmin_pml,
        dmin_pmc=dmin_pmc,
        scale=float(derivative_scale),
    )
    operators = (
        _assemble_diagonal_operators(sparse, eps_tensor, mu_tensor, derivatives)
        if diagonal
        else {
            "mat": _assemble_tensorial_operator(
                sparse, eps_tensor, mu_tensor, derivatives
            )
        }
    )
    operator = cast(Any, operators["mat"])
    guess = complex(-(neff_guess**2) if diagonal else neff_guess)
    # Only the reduced diagonal problem uses the real ARPACK optimization.
    if diagonal:
        operator, initial_vector, guess = _real_arpack_problem_if_close(
            operator, initial_vector, guess
        )
    values, vectors = _selected_eigenpairs(
        operator,
        num_modes=num_modes,
        sigma=guess,
        krylov_dim=krylov_dim,
        initial_vector=initial_vector,
        spla=spla,
        scipy_linalg=scipy_linalg,
    )
    residuals = np.asarray(
        [
            np.linalg.norm(operator @ vectors[:, i] - value * vectors[:, i])
            for i, value in enumerate(values)
        ]
    ) / np.maximum(np.linalg.norm(vectors, axis=0), np.finfo(float).eps)
    # The diagonal eigenvalues are -neff^2; the tensorial eigenvalues are neff.
    indices: np.ndarray = np.asarray(
        np.sqrt(-values + 0j) if diagonal else values, dtype=np.complex128
    )
    order = np.argsort(-np.real(indices), kind="stable")
    indices, residuals = indices[order], residuals[order]
    vectors = np.asarray(vectors[:, order], dtype=np.complex128)
    inv_eps_zz = sparse.diags(1.0 / eps_tensor[2, 2, :], format="csc")
    inv_mu_zz = sparse.diags(1.0 / mu_tensor[2, 2, :], format="csc")
    dxf, dxb, dyf, dyb = derivatives
    cell_areas = np.repeat(np.asarray(dlf[0], dtype=float), ny) * np.tile(
        np.asarray(dlf[1], dtype=float), nx
    )
    mode_fields = []
    for neff, vector in zip(indices, vectors.T, strict=True):
        ex, ey = vector[:n].copy(), vector[n : 2 * n].copy()
        if diagonal:
            # Q reconstructs transverse H. Q_ep supplies the constitutive
            # electric block needed for the longitudinal electric field.
            denom = complex(-neff.imag, neff.real)
            h = operators["qmat"] @ vector / denom
            hx, hy = h[:n], h[n:]
            hz_source = dxf @ ey - dyf @ ex
            partial = operators["q_ep"] @ vector / denom
            ez_source = dxb @ partial[n:] - dyb @ partial[:n]
        else:
            hx, hy = vector[2 * n : 3 * n].copy(), vector[3 * n : 4 * n].copy()
            # The z-row of each full tensor couples into longitudinal fields.
            hz_source = (
                dxf @ ey - dyf @ ex - mu_tensor[2, 0] * hx - mu_tensor[2, 1] * hy
            )
            ez_source = (
                dxb @ hy - dyb @ hx - eps_tensor[2, 0] * ex - eps_tensor[2, 1] * ey
            )
        hz = np.asarray(inv_mu_zz @ hz_source, dtype=np.complex128)
        ez = np.asarray(inv_eps_zz @ ez_source, dtype=np.complex128)
        # Match Result's physical magnetic convention. Backward propagation
        # flips transverse H and longitudinal E in either formulation.
        hx, hy, hz = (
            np.asarray(h * (-1j / ETA0), dtype=np.complex128) for h in (hx, hy, hz)
        )
        if direction == "-":
            hx, hy, ez = -hx, -hy, -ez
        mode_fields.append(_ModeFields(ex, ey, ez, hx, hy, hz))
    orthogonalization = _lorentz_orthogonalize_and_normalize(mode_fields, cell_areas)
    fields = [
        np.asarray([getattr(mode, c) for mode in mode_fields], dtype=np.complex128)
        for c in ("ex", "ey", "ez", "hx", "hy", "hz")
    ]
    return (
        np.asarray(indices, dtype=np.complex128),
        fields,
        {
            "backend": "scipy_arpack_reference",
            "operator_size": int(operator.shape[0]),
            "operator_nnz": int(operator.nnz),
            "residuals": np.asarray(residuals, dtype=float),
            **orthogonalization,
        },
    )


def _import_scipy():
    """Import SciPy modules lazily so package import stays lightweight."""
    try:
        import scipy.linalg as scipy_linalg
        import scipy.sparse as sparse
        import scipy.sparse.linalg as spla
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise ImportError(
            "the native mode solver requires BeamZ's SciPy runtime dependency"
        ) from exc
    return sparse, spla, scipy_linalg


def _create_derivative_matrices(
    sparse,
    *,
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    shape: tuple[int, int],
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    omega: float | None,
    num_pml: tuple[int, int],
    pml_profile: dict[str, float | int] | None,
    dmin_pml: tuple[bool, bool],
    dmin_pmc: tuple[bool, bool],
    scale: float,
):
    """Assemble scaled Yee derivative matrices with optional PML stretching."""
    # Build raw finite-difference stencils first; optional PML scaling is a
    # diagonal pre-multiplication applied afterward.
    matrices = tuple(
        _make_derivative(
            sparse,
            np.asarray(steps[axis], dtype=float),
            shape,
            axis=axis,
            forward=forward,
            pmc=dmin_pmc[axis],
        )
        for axis in (0, 1)
        for forward, steps in ((True, dlf), (False, dlb))
    )
    if num_pml[0] > 0 or num_pml[1] > 0:
        # Complex stretch factors require angular frequency to scale sigma.
        if omega is None:
            raise ValueError("omega is required when num_pml is nonzero")
        pml_values = _create_s_diagonal_values(
            shape=shape,
            num_pml=num_pml,
            dlf=dlf,
            dlb=dlb,
            eps_tensor=eps_tensor,
            mu_tensor=mu_tensor,
            dmin_pml=dmin_pml,
            omega=float(omega),
            profile=pml_profile,
        )
        matrices = tuple(
            sparse.diags(values, format="csc") @ matrix
            for values, matrix in zip(pml_values, matrices, strict=True)
        )

    # derivative_scale nondimensionalizes spatial derivatives by wavelength.
    return tuple(matrix * complex(scale, 0.0) for matrix in matrices)


def _make_derivative(sparse, dls, shape, *, axis: int, forward: bool, pmc: bool):
    """Expand a one-dimensional PEC/PMC Yee stencil in Cartesian storage order."""
    count = shape[axis]
    if count == 1:
        size = int(np.prod(shape))
        return sparse.csc_matrix((size, size), dtype=np.complex128)
    values = 1.0 / dls
    diagonal = -values if forward else values.copy()
    # PEC drops the low diagonal; PMC keeps forward or doubles backward.
    boundary_factor = -1.0 if forward else 2.0
    diagonal[0] = boundary_factor * values[0] if pmc else 0.0
    neighbor = values[:-1] if forward else -values[1:]
    stencil = sparse.diags(
        (diagonal, neighbor),
        (0, 1 if forward else -1),
        shape=(count, count),
        dtype=np.complex128,
        format="csc",
    )
    identity = sparse.eye(shape[1 - axis], format="csc")
    left, right = (stencil, identity) if axis == 0 else (identity, stencil)
    return sparse.kron(left, right, format="csc")


def _assemble_diagonal_operators(
    sparse, eps: np.ndarray, mu: np.ndarray, der_mats
) -> dict[str, object]:
    """Assemble the reduced diagonal-material eigen-operator blocks."""
    # All blocks operate on flattened cell vectors.
    n = eps.shape[-1]
    zero = sparse.csc_matrix((n, n), dtype=np.complex128)
    dxf, dxb, dyf, dyb = der_mats

    # Longitudinal component elimination requires inverse zz material terms.
    inv_eps_zz = sparse.diags(1.0 / eps[2, 2, :], format="csc")
    inv_mu_zz = sparse.diags(1.0 / mu[2, 2, :], format="csc")

    # The block names mirror docs/physics-model.md:
    # A_diag = P_mu Q + P_partial Q_epsilon, with Q = Q_epsilon + Q_partial.
    p_mu = sparse.bmat(
        [
            [zero, sparse.diags(mu[1, 1, :], format="csc")],
            [-sparse.diags(mu[0, 0, :], format="csc"), zero],
        ],
        format="csc",
    )

    # P_partial captures derivative coupling introduced by eliminating Ez.
    p_partial = sparse.bmat(
        [
            [-(dxf @ inv_eps_zz @ dyb), dxf @ inv_eps_zz @ dxb],
            [-(dyf @ inv_eps_zz @ dyb), dyf @ inv_eps_zz @ dxb],
        ],
        format="csc",
    )
    q_ep = sparse.bmat(
        [
            [zero, sparse.diags(eps[1, 1, :], format="csc")],
            [-sparse.diags(eps[0, 0, :], format="csc"), zero],
        ],
        format="csc",
    )

    # Q_partial captures derivative coupling introduced by eliminating Hz.
    q_partial = sparse.bmat(
        [
            [-(dxb @ inv_mu_zz @ dyf), dxb @ inv_mu_zz @ dxf],
            [-(dyb @ inv_mu_zz @ dyf), dyb @ inv_mu_zz @ dxf],
        ],
        format="csc",
    )
    qmat = q_ep + q_partial

    # The final reduced operator acts on [Ex, Ey].
    mat = p_mu @ qmat + p_partial @ q_ep
    return {
        "q_ep": _canonical_sparse(q_ep),
        "qmat": _canonical_sparse(qmat),
        "mat": _canonical_sparse(mat),
    }


def _assemble_tensorial_operator(sparse, eps: np.ndarray, mu: np.ndarray, der_mats):
    """Assemble the full tensorial first-order eigen-operator."""
    # The tensorial operator acts on [Ex, Ey, Hx, Hy] and leaves Ez/Hz implicit.
    dxf, dxb, dyf, dyb = der_mats
    inv_eps_22 = sparse.diags(1.0 / eps[2, 2, :], format="csc")
    inv_mu_22 = sparse.diags(1.0 / mu[2, 2, :], format="csc")

    def diag(values):
        """Return a sparse diagonal matrix for one flattened tensor component."""
        return sparse.diags(values, format="csc")

    # Precompute tensor ratios that appear repeatedly after eliminating z fields.
    eps_20_over_22 = eps[2, 0, :] / eps[2, 2, :]
    eps_21_over_22 = eps[2, 1, :] / eps[2, 2, :]
    eps_02_over_22 = eps[0, 2, :] / eps[2, 2, :]
    eps_12_over_22 = eps[1, 2, :] / eps[2, 2, :]
    mu_20_over_22 = mu[2, 0, :] / mu[2, 2, :]
    mu_21_over_22 = mu[2, 1, :] / mu[2, 2, :]
    mu_02_over_22 = mu[0, 2, :] / mu[2, 2, :]
    mu_12_over_22 = mu[1, 2, :] / mu[2, 2, :]

    # Schur-complement terms eliminate Ez and Hz before solving the first-order
    # transverse system [Ex, Ey, Hx, Hy]. Suffix "_s" means the component has
    # been corrected by its coupling through the local z component.
    mu_10_s = mu[1, 0, :] - mu[1, 2, :] * mu_20_over_22
    mu_11_s = mu[1, 1, :] - mu[1, 2, :] * mu_21_over_22
    mu_00_s = mu[0, 0, :] - mu[0, 2, :] * mu_20_over_22
    mu_01_s = mu[0, 1, :] - mu[0, 2, :] * mu_21_over_22
    eps_10_s = eps[1, 0, :] - eps[1, 2, :] * eps_20_over_22
    eps_11_s = eps[1, 1, :] - eps[1, 2, :] * eps_21_over_22
    eps_00_s = eps[0, 0, :] - eps[0, 2, :] * eps_20_over_22
    eps_01_s = eps[0, 1, :] - eps[0, 2, :] * eps_21_over_22

    # Block names encode output row and input column:
    # a = electric transverse rows, b = magnetic transverse rows, x/y = local
    # transverse components. For example, axby maps Hy into the Ex equation.
    # Electric rows include derivative terms and magnetic tensor corrections.
    axax = -(dxf @ diag(eps_20_over_22)) - diag(mu_12_over_22) @ dyf
    axay = -(dxf @ diag(eps_21_over_22)) + diag(mu_12_over_22) @ dxf
    axbx = -(dxf @ inv_eps_22 @ dyb) + diag(mu_10_s)
    axby = dxf @ inv_eps_22 @ dxb + diag(mu_11_s)

    ayax = -(dyf @ diag(eps_20_over_22)) + diag(mu_02_over_22) @ dyf
    ayay = -(dyf @ diag(eps_21_over_22)) - diag(mu_02_over_22) @ dxf
    aybx = -(dyf @ inv_eps_22 @ dyb) - diag(mu_00_s)
    ayby = dyf @ inv_eps_22 @ dxb - diag(mu_01_s)

    # Magnetic rows mirror the electric rows with eps/mu roles swapped.
    bxax = -(dxb @ inv_mu_22 @ dyf) + diag(eps_10_s)
    bxay = dxb @ inv_mu_22 @ dxf + diag(eps_11_s)
    bxbx = -(dxb @ diag(mu_20_over_22)) - diag(eps_12_over_22) @ dyb
    bxby = -(dxb @ diag(mu_21_over_22)) + diag(eps_12_over_22) @ dxb

    byax = -(dyb @ inv_mu_22 @ dyf) - diag(eps_00_s)
    byay = dyb @ inv_mu_22 @ dxf - diag(eps_01_s)
    bybx = -(dyb @ diag(mu_20_over_22)) + diag(eps_02_over_22) @ dyb
    byby = -(dyb @ diag(mu_21_over_22)) - diag(eps_02_over_22) @ dxb

    # The leading -i matches the chosen time-harmonic convention.
    return _canonical_sparse(
        -1j
        * sparse.bmat(
            [
                [axax, axay, axbx, axby],
                [ayax, ayay, aybx, ayby],
                [bxax, bxay, bxbx, bxby],
                [byax, byay, bybx, byby],
            ],
            format="csc",
        )
    )


def _canonical_sparse(matrix):
    """Return a cleaned CSC sparse matrix."""
    # CSC is the format used by ARPACK and sparse block algebra here.
    matrix = matrix.tocsc(copy=True)
    matrix.eliminate_zeros()
    return matrix


def _real_arpack_problem_if_close(
    matrix, initial_vector: np.ndarray | None, guess: complex
):
    """Use a real ARPACK problem when the operator and shift are effectively real."""
    # Empty operators cannot be inspected for imaginary scale.
    if matrix.nnz == 0:
        return matrix, initial_vector, guess
    matrix_imag = matrix.data.imag
    matrix_scale = max(float(np.max(np.abs(matrix.data))), 1.0)
    guess_is_real = abs(guess.imag) <= 1e-14 * max(abs(guess), 1.0)

    # If the complex part is numerical noise, cast to real to avoid SciPy's
    # ComplexWarning path and improve reproducibility.
    if np.max(np.abs(matrix_imag)) <= 1e-14 * matrix_scale and guess_is_real:
        real_vector = (
            None
            if initial_vector is None
            else np.asarray(np.real(initial_vector), dtype=float)
        )
        return matrix.real.astype(float), real_vector, float(guess.real)
    return matrix, initial_vector, guess


def _selected_eigenpairs(
    mat,
    *,
    num_modes: int,
    sigma: complex,
    krylov_dim: int,
    initial_vector: np.ndarray | None,
    spla,
    scipy_linalg,
) -> tuple[np.ndarray, np.ndarray]:
    """Select eigenpairs nearest the requested shift."""
    size = mat.shape[0]

    # Very small operators cannot use sparse eigs with k >= N - 1, so fall back
    # to dense eig and select the nearest modes manually.
    if num_modes >= size - 1:
        values, vectors = scipy_linalg.eig(mat.toarray())
        order = np.argsort(np.abs(values - sigma))[:num_modes]
        return values[order], vectors[:, order]

    # Keep ARPACK's Krylov subspace large enough for the requested number of
    # modes but bounded by the matrix size.
    ncv = min(size, max(int(krylov_dim), num_modes + 2))

    # Shift-invert mode returns eigenvalues closest to sigma.
    complex_warning = getattr(
        getattr(np, "exceptions", np), "ComplexWarning", RuntimeWarning
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=complex_warning, module="scipy")
        values, vectors = spla.eigs(
            mat,
            k=num_modes,
            sigma=sigma,
            which="LM",
            v0=None if initial_vector is None else np.asarray(initial_vector),
            ncv=ncv,
            tol=1e-10,
        )
    return np.asarray(values, dtype=np.complex128), np.asarray(
        vectors, dtype=np.complex128
    )


def _create_s_diagonal_values(
    *,
    shape: tuple[int, int],
    num_pml: tuple[int, int],
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    dmin_pml: tuple[bool, bool],
    omega: float,
    profile: dict[str, float | int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build inverse PML stretch vectors for every derivative stencil."""
    nx, ny = shape

    # Estimate boundary-relative speed so PML sigma scales with local material.
    avg_speed = _average_relative_speed(shape, num_pml, eps_tensor, mu_tensor)

    # Forward and backward derivative stencils sample stretch factors at
    # different Yee locations.
    vectors = []
    for axis, count in enumerate(shape):
        for direction, widths in (("f", dlf), ("b", dlb)):
            stretch = _create_sfactor(
                direction,
                omega,
                np.asarray(widths[axis], dtype=float),
                count,
                num_pml[axis],
                dmin_pml[axis],
                avg_speed[2 * axis : 2 * axis + 2],
                profile,
            )
            # Flattening is x-major: x values repeat along y, y values tile.
            expand = np.repeat if axis == 0 else np.tile
            vectors.append(expand(1.0 / stretch, shape[1 - axis]))
    return tuple(vectors)


def _average_relative_speed(
    shape: tuple[int, int],
    num_pml: tuple[int, int],
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
) -> np.ndarray:
    """Estimate relative wave speed near PML boundaries."""
    # Average diagonal epsilon and mu on each side, then convert to relative
    # speed for damping-strength scaling.
    eps_avg = _pml_average_all_sides(shape, num_pml, eps_tensor)
    mu_avg = _pml_average_all_sides(shape, num_pml, mu_tensor)
    return 1.0 / np.sqrt(eps_avg * mu_avg)


def _pml_average_all_sides(
    shape: tuple[int, int], num_pml: tuple[int, int], tensor: np.ndarray
) -> np.ndarray:
    """Average diagonal material components on each boundary side."""
    diagonal = np.asarray([tensor[i, i] for i in range(3)]).reshape(3, *shape)
    averages = []
    for axis, (count, cells) in enumerate(zip(shape, num_pml, strict=True), start=1):
        for side in (slice(0, cells), slice(max(count - cells, 0), count)):
            selectors = [slice(None)] * 3
            selectors[axis] = side
            values = diagonal[tuple(selectors)]
            averages.append(np.mean(values) if values.size else 1.0)
    return np.asarray(averages, dtype=np.complex128)


def _create_sfactor(direction, omega, dls, n, n_pml, dmin_pml, avg_speed, profile):
    """Build PML stretches with explicit forward/backward Yee offsets.

    The backward high boundary excludes its interface node. Low-side PML wins
    when layers overlap, matching the derivative stencil's boundary convention.
    """
    sfactor = np.ones(n, dtype=np.complex128)
    if n_pml == 0:
        return sfactor
    if direction not in {"f", "b"}:
        raise ValueError(f"direction value {direction} not recognized")
    offset = 0.5 if direction == "f" else 0.0
    for i in range(n):
        if i < n_pml and dmin_pml:
            side, depth = 0, n_pml - i - offset
        elif i + offset > n - n_pml:
            side, depth = 1, i - (n - n_pml) + offset
        else:
            continue
        sfactor[i] = _s_value(
            dls[0 if side == 0 else -1], depth / n_pml, omega, avg_speed[side], profile
        )
    return sfactor


def _s_value(
    dl: float,
    step: float,
    omega: float,
    avg_speed: complex,
    profile: dict[str, float | int] | None,
) -> complex:
    """Evaluate one polynomial PML stretch factor."""
    # Defaults match PmlSpec and are overridden by a validated profile dict.
    values = {
        "sigma_max": 2.0,
        "kappa_min": 1.0,
        "kappa_max": 3.0,
        "order": 3,
    }
    if profile is not None:
        values.update(profile)

    # Polynomial depth controls both attenuation sigma and coordinate kappa.
    step_power = step ** int(values["order"])
    kappa = (
        float(values["kappa_min"])
        + (float(values["kappa_max"]) - float(values["kappa_min"])) * step_power
    )
    sigma = avg_speed * (float(values["sigma_max"]) / (ETA0 * dl) * step_power)
    return complex(kappa, 0.0) + 1j * sigma / (omega * EPSILON0)


def _lorentz_orthogonalize_and_normalize(
    modes: list[_ModeFields], cell_areas: np.ndarray
) -> dict[str, object]:
    """Normalize modes and enforce Lorentz orthogonality."""
    # Begin with unit transverse power so projection coefficients are stable.
    for mode in modes:
        _normalize_to_unit_power(mode, cell_areas)

    # Modified Gram-Schmidt using the unconjugated Lorentz overlap.
    for mode_index, mode in enumerate(modes):
        for previous in modes[:mode_index]:
            denom = _lorentz_overlap(previous, previous, cell_areas)
            if abs(denom) <= np.finfo(float).eps:
                continue
            coeff = _lorentz_overlap(previous, mode, cell_areas) / denom
            mode.add_scaled(previous, -coeff)

        # Renormalize and pin phase after each projection step.
        _normalize_to_unit_power(mode, cell_areas)
        _apply_dominant_e_phase_convention(mode)

    # Diagnostics expose both power normalization and residual Lorentz coupling.
    power_norms = np.asarray(
        [abs(_transverse_power(mode, cell_areas)) for mode in modes], dtype=float
    )
    lorentz_norms = np.asarray(
        [_lorentz_overlap(mode, mode, cell_areas) for mode in modes],
        dtype=np.complex128,
    )
    error = 0.0
    for left_index, left in enumerate(modes):
        for right_index, right in enumerate(modes):
            if left_index == right_index:
                continue
            denom = float(
                np.sqrt(
                    abs(lorentz_norms[left_index]) * abs(lorentz_norms[right_index])
                )
            )
            if denom <= np.finfo(float).eps:
                continue
            # Track the largest normalized off-diagonal overlap.
            error = max(error, abs(_lorentz_overlap(left, right, cell_areas)) / denom)
    return {
        "power_norms": power_norms,
        "lorentz_norms": lorentz_norms,
        "lorentz_orthogonality_error": float(error),
    }


def _normalize_to_unit_power(mode: _ModeFields, cell_areas: np.ndarray) -> float:
    """Scale a mode to unit transverse power."""
    # Use magnitude so forward/backward sign conventions do not prevent scaling.
    norm = abs(_transverse_power(mode, cell_areas))
    if norm <= np.finfo(float).eps:
        return 0.0
    scale = 1.0 / np.sqrt(norm)

    # Apply one scalar to all six components to preserve field ratios.
    for component in mode.components():
        component *= scale
    return abs(_transverse_power(mode, cell_areas))


def _transverse_power(mode: _ModeFields, cell_areas: np.ndarray) -> complex:
    """Compute conjugated transverse power flux."""
    # For local z propagation, transverse flux is (E x H*) . z.
    return complex(
        np.sum((mode.ex * np.conj(mode.hy) - mode.ey * np.conj(mode.hx)) * cell_areas)
    )


def _lorentz_overlap(
    left: _ModeFields, right: _ModeFields, cell_areas: np.ndarray
) -> complex:
    """Compute the unconjugated reciprocal-product overlap."""
    # Symmetrize the unconjugated cross product to match reciprocal-mode theory.
    left_cross_right = np.sum((left.ex * right.hy - left.ey * right.hx) * cell_areas)
    right_cross_left = np.sum((right.ex * left.hy - right.ey * left.hx) * cell_areas)
    return complex(0.5 * (left_cross_right + right_cross_left))


def _apply_dominant_e_phase_convention(mode: _ModeFields) -> None:
    """Rotate a mode so its largest electric component is real-positive."""
    # Choose a robust phase anchor from the largest electric-field entry.
    electric = np.concatenate((mode.ex, mode.ey, mode.ez))
    if electric.size == 0:
        return
    anchor = electric[int(np.argmax(np.abs(electric) ** 2))]
    if abs(anchor) <= np.finfo(float).eps:
        return

    # Multiplying by conj(anchor)/|anchor| rotates the anchor onto the positive
    # real axis without changing norms.
    phase = np.conj(anchor) / abs(anchor)
    for component in mode.components():
        component *= phase
