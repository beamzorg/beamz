"""Diagonal and tensorial sparse eigen-operators.

Moved unchanged from the MicroMode ``scipy_reference.py`` import
(``_scipy.py``; see ``UPSTREAM.md``). Block names follow the reduced
transverse formulation: diagonal materials act on ``[Ex, Ey]``; the
tensorial operator acts on ``[Ex, Ey, Hx, Hy]``.
"""

from __future__ import annotations

import numpy as np


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
