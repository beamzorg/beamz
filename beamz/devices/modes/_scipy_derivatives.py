"""Yee derivative matrices for the SciPy reference mode solver.

Moved unchanged from the MicroMode ``scipy_reference.py`` import
(``_scipy.py``; see ``UPSTREAM.md``). Forward and backward differences along
each transverse axis keep the original PEC/PMC edge rules.
"""

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
    # Build raw finite-difference stencils first; optional PML scaling is a
    # diagonal pre-multiplication applied afterward.
    matrices = (
        _make_dxf(sparse, np.asarray(dlf[0], dtype=float), shape, dmin_pmc[0]),
        _make_dxb(sparse, np.asarray(dlb[0], dtype=float), shape, dmin_pmc[0]),
        _make_dyf(sparse, np.asarray(dlf[1], dtype=float), shape, dmin_pmc[1]),
        _make_dyb(sparse, np.asarray(dlb[1], dtype=float), shape, dmin_pmc[1]),
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


def _make_dxf(sparse, dls: np.ndarray, shape: tuple[int, int], pmc: bool):
    """Build the forward x derivative on the local Yee grid."""
    nx, ny = shape
    if nx == 1:
        return sparse.csc_matrix((ny, ny), dtype=np.complex128)

    # Assemble in coordinate-list form, then convert to CSC for sparse algebra.
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    for ix in range(nx):
        for iy in range(ny):
            row = ix * ny + iy
            value = 1.0 / dls[ix]

            # At the low edge, PEC-style symmetry drops the diagonal term while
            # PMC keeps the one-sided stencil.
            diagonal = 0.0 if ix == 0 and not pmc else -value
            if diagonal != 0.0:
                rows.append(row)
                cols.append(row)
                data.append(diagonal)
            if ix + 1 < nx:
                # Interior forward difference points to the next x cell.
                rows.append(row)
                cols.append((ix + 1) * ny + iy)
                data.append(value)
    return sparse.csc_matrix(
        (data, (rows, cols)), shape=(nx * ny, nx * ny), dtype=np.complex128
    )


def _make_dxb(sparse, dls: np.ndarray, shape: tuple[int, int], pmc: bool):
    """Build the backward x derivative on the local Yee grid."""
    nx, ny = shape
    if nx == 1:
        return sparse.csc_matrix((ny, ny), dtype=np.complex128)

    # Backward stencils share the same flattened row order as forward stencils.
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    for ix in range(nx):
        for iy in range(ny):
            row = ix * ny + iy
            value = 1.0 / dls[ix]

            # PMC at the low boundary mirrors the field and doubles the diagonal
            # contribution; otherwise the first row has no backward neighbor.
            diagonal = 2.0 * value if ix == 0 and pmc else (0.0 if ix == 0 else value)
            if diagonal != 0.0:
                rows.append(row)
                cols.append(row)
                data.append(diagonal)
            if ix > 0:
                # Interior backward difference points to the previous x cell.
                rows.append(row)
                cols.append((ix - 1) * ny + iy)
                data.append(-value)
    return sparse.csc_matrix(
        (data, (rows, cols)), shape=(nx * ny, nx * ny), dtype=np.complex128
    )


def _make_dyf(sparse, dls: np.ndarray, shape: tuple[int, int], pmc: bool):
    """Build the forward y derivative on the local Yee grid."""
    nx, ny = shape
    if ny == 1:
        return sparse.csc_matrix((nx, nx), dtype=np.complex128)

    # y derivatives use the same flat index but move along the inner y axis.
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    for ix in range(nx):
        for iy in range(ny):
            row = ix * ny + iy
            value = 1.0 / dls[iy]

            # Low-edge y boundary uses the same PEC/PMC convention as x.
            diagonal = 0.0 if iy == 0 and not pmc else -value
            if diagonal != 0.0:
                rows.append(row)
                cols.append(row)
                data.append(diagonal)
            if iy + 1 < ny:
                # Interior forward y difference points to the next y cell.
                rows.append(row)
                cols.append(ix * ny + iy + 1)
                data.append(value)
    return sparse.csc_matrix(
        (data, (rows, cols)), shape=(nx * ny, nx * ny), dtype=np.complex128
    )


def _make_dyb(sparse, dls: np.ndarray, shape: tuple[int, int], pmc: bool):
    """Build the backward y derivative on the local Yee grid."""
    nx, ny = shape
    if ny == 1:
        return sparse.csc_matrix((nx, nx), dtype=np.complex128)

    # Backward y differences mirror _make_dxb on the inner axis.
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    for ix in range(nx):
        for iy in range(ny):
            row = ix * ny + iy
            value = 1.0 / dls[iy]

            # A PMC low edge doubles the reflected diagonal contribution.
            diagonal = 2.0 * value if iy == 0 and pmc else (0.0 if iy == 0 else value)
            if diagonal != 0.0:
                rows.append(row)
                cols.append(row)
                data.append(diagonal)
            if iy > 0:
                # Interior backward y difference points to the previous y cell.
                rows.append(row)
                cols.append(ix * ny + iy - 1)
                data.append(-value)
    return sparse.csc_matrix(
        (data, (rows, cols)), shape=(nx * ny, nx * ny), dtype=np.complex128
    )
