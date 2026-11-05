"""Diagonal and tensorial sparse eigen-operators.

Formulas follow the MicroMode import (see ``UPSTREAM.md``). Diagonal materials
act on ``[Ex, Ey]``; the tensorial operator acts on ``[Ex, Ey, Hx, Hy]``.
"""

from __future__ import annotations

import numpy as np


def _diag(sparse, values):
    return sparse.diags(np.asarray(values), format="csc")


def _bmat(sparse, rows):
    return sparse.bmat(rows, format="csc")


def _skew(sparse, xx, yy):
    """``+yy`` at (0, 1) and ``-xx`` at (1, 0)."""
    n = np.asarray(xx).reshape(-1).size
    zero = sparse.csc_matrix((n, n), dtype=np.complex128)
    return _bmat(sparse, [[zero, _diag(sparse, yy)], [-_diag(sparse, xx), zero]])


def _z_elimination(sparse, out_d, in_d, inverse):
    """Column 0 is minus the other transverse derivative."""
    return _bmat(
        sparse,
        [
            [
                (-1.0 if col == 0 else 1.0) * out_d[row] @ inverse @ in_d[1 - col]
                for col in range(2)
            ]
            for row in range(2)
        ],
    )


def _family_blocks(sparse, out_d, in_d, ratios, cross):
    """Same-family blocks: material under the output derivative, then the cross term."""
    return [
        [
            -(out_d[i] @ _diag(sparse, ratios[j]))
            + (-1.0 if i == j else 1.0) * _diag(sparse, cross[i]) @ in_d[1 - j]
            for j in range(2)
        ]
        for i in range(2)
    ]


def _cross_blocks(sparse, out_d, in_d, inverse, schur):
    """Cross-family blocks plus the Schur correction through z."""
    return [
        [
            (-1.0 if j == 0 else 1.0) * out_d[i] @ inverse @ in_d[1 - j]
            + (1.0 if i == 0 else -1.0) * _diag(sparse, schur[i][j])
            for j in range(2)
        ]
        for i in range(2)
    ]


def _assemble_diagonal_operators(
    sparse, eps: np.ndarray, mu: np.ndarray, der_mats
) -> dict[str, object]:
    """Assemble ``P_mu Q + P_partial Q_epsilon`` on ``[Ex, Ey]``."""
    dxf, dxb, dyf, dyb = der_mats
    forward, backward = (dxf, dyf), (dxb, dyb)
    inv_eps = _diag(sparse, 1.0 / eps[2, 2, :])
    inv_mu = _diag(sparse, 1.0 / mu[2, 2, :])
    p_mu = _skew(sparse, mu[0, 0, :], mu[1, 1, :])
    q_ep = _skew(sparse, eps[0, 0, :], eps[1, 1, :])
    p_partial = _z_elimination(sparse, forward, backward, inv_eps)
    q_partial = _z_elimination(sparse, backward, forward, inv_mu)
    qmat = q_ep + q_partial
    return {
        "q_ep": _canonical_sparse(q_ep),
        "qmat": _canonical_sparse(qmat),
        "mat": _canonical_sparse(p_mu @ qmat + p_partial @ q_ep),
    }


def _zz_rows(tensor):
    ratios = tensor[2, :2, :] / tensor[2, 2, :]
    schur = [tensor[1 - i, :2, :] - tensor[1 - i, 2, :] * ratios for i in range(2)]
    return ratios, tensor[[1, 0], 2, :] / tensor[2, 2, :], schur


def _assemble_tensorial_operator(sparse, eps: np.ndarray, mu: np.ndarray, der_mats):
    """Assemble the first-order operator on ``[Ex, Ey, Hx, Hy]``."""
    dxf, dxb, dyf, dyb = der_mats
    forward, backward = (dxf, dyf), (dxb, dyb)
    eps_z, eps_cross, eps_schur = _zz_rows(eps)
    mu_z, mu_cross, mu_schur = _zz_rows(mu)
    electric = _family_blocks(sparse, forward, forward, eps_z, mu_cross)
    magnetic = _family_blocks(sparse, backward, backward, mu_z, eps_cross)
    from_h = _cross_blocks(
        sparse, forward, backward, _diag(sparse, 1.0 / eps[2, 2, :]), mu_schur
    )
    from_e = _cross_blocks(
        sparse, backward, forward, _diag(sparse, 1.0 / mu[2, 2, :]), eps_schur
    )
    rows = [
        [*electric[i], *from_h[i]] if i < 2 else [*from_e[i - 2], *magnetic[i - 2]]
        for i in range(4)
    ]
    # The leading -i matches the chosen time-harmonic convention.
    return _canonical_sparse(-1j * _bmat(sparse, rows))


def _canonical_sparse(matrix):
    matrix = matrix.tocsc(copy=True)
    matrix.eliminate_zeros()
    return matrix
