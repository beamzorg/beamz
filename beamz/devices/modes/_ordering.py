"""Sparse shift-invert factorization on a transverse Yee plane."""

import numpy as np


def _separator_order(shapes):
    """Eliminate the interior of each grid partition before its separator."""
    nz = max(rows for rows, _ in shapes)
    ny = max(columns for _, columns in shapes)
    sites = np.full((nz, ny, len(shapes)), -1, dtype=np.intp)
    start = 0
    for component, (rows, columns) in enumerate(shapes):
        stop = start + rows * columns
        sites[:rows, :columns, component] = np.arange(start, stop).reshape(
            rows, columns
        )
        start = stop

    def split(tile):
        height, width, _ = tile.shape
        if width * height <= 8:
            yield tile.ravel()
        else:
            if width > height:
                tile = tile.swapaxes(0, 1)
            middle = tile.shape[0] // 2
            yield from split(tile[:middle])
            yield from split(tile[middle + 1 :])
            yield tile[middle].ravel()

    order = np.concatenate(tuple(split(sites)))
    return order[order >= 0]


def shift_inverse(spla, operator, target, shapes, mass=None):
    """Supply ARPACK a lower-fill solve for large real transverse stencils."""
    if not shapes or operator.shape[0] < 16384 or np.iscomplexobj(operator.data):
        return {}
    if mass is None:
        shifted = operator.tocsc(copy=True)
        shifted.setdiag(shifted.diagonal() - target)
    else:
        shifted = (operator - target * mass).tocsc()
    order = _separator_order(shapes)
    try:
        factor = spla.splu(shifted[order, :][:, order], permc_spec="NATURAL")
    except RuntimeError:
        return {}
    unpermute = np.argsort(order)
    return {
        "OPinv": spla.LinearOperator(
            operator.shape,
            matvec=lambda x: factor.solve(x[order])[unpermute],
            dtype=shifted.dtype,
        )
    }
