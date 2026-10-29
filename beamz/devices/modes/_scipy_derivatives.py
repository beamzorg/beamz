"""Yee derivative matrices for the SciPy reference mode solver."""

from __future__ import annotations

import numpy as np

from ._scipy_pml import _create_s_diagonal_values


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
    steps = (dlf[0], dlb[0], dlf[1], dlb[1])
    axes = (0, 0, 1, 1)
    forward = (True, False, True, False)
    matrices = tuple(
        _yee_derivative(
            sparse,
            np.asarray(step, dtype=float),
            shape,
            dmin_pmc[axis],
            axis=axis,
            forward=is_forward,
        )
        for step, axis, is_forward in zip(steps, axes, forward, strict=True)
    )
    if num_pml[0] > 0 or num_pml[1] > 0:
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
    return tuple(matrix * complex(scale, 0.0) for matrix in matrices)


def _yee_derivative(sparse, dls, shape, pmc, *, axis: int, forward: bool):
    """Forward or backward difference on one transverse axis of the local Yee grid."""
    nx, ny = shape
    count = shape[axis]
    if count == 1:
        other = shape[1 - axis]
        return sparse.csc_matrix((other, other), dtype=np.complex128)

    stride = ny if axis == 0 else 1
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    for ix in range(nx):
        for iy in range(ny):
            index = ix if axis == 0 else iy
            row = ix * ny + iy
            value = 1.0 / dls[index]
            if forward:
                # PEC drops the low-edge diagonal; PMC keeps the one-sided stencil.
                diagonal = 0.0 if index == 0 and not pmc else -value
                neighbor = row + stride if index + 1 < count else None
                coeff = value
            else:
                # PMC mirrors the low edge and doubles that diagonal contribution.
                diagonal = (
                    2.0 * value
                    if index == 0 and pmc
                    else (0.0 if index == 0 else value)
                )
                neighbor = None if index == 0 else row - stride
                coeff = -value
            if diagonal != 0.0:
                rows.append(row)
                cols.append(row)
                data.append(diagonal)
            if neighbor is not None:
                rows.append(row)
                cols.append(neighbor)
                data.append(coeff)
    return sparse.csc_matrix(
        (data, (rows, cols)), shape=(nx * ny, nx * ny), dtype=np.complex128
    )


def _make_dxf(sparse, dls: np.ndarray, shape: tuple[int, int], pmc: bool):
    return _yee_derivative(sparse, dls, shape, pmc, axis=0, forward=True)


def _make_dxb(sparse, dls: np.ndarray, shape: tuple[int, int], pmc: bool):
    return _yee_derivative(sparse, dls, shape, pmc, axis=0, forward=False)


def _make_dyf(sparse, dls: np.ndarray, shape: tuple[int, int], pmc: bool):
    return _yee_derivative(sparse, dls, shape, pmc, axis=1, forward=True)


def _make_dyb(sparse, dls: np.ndarray, shape: tuple[int, int], pmc: bool):
    return _yee_derivative(sparse, dls, shape, pmc, axis=1, forward=False)
