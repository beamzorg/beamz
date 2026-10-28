"""ARPACK problem setup and eigenpair selection.

Moved unchanged from the MicroMode ``scipy_reference.py`` import
(``_scipy.py``; see ``UPSTREAM.md``).
"""

from __future__ import annotations

import warnings

import numpy as np


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
