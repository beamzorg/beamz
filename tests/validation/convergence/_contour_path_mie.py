"""TE cylinder Mie oracle versus the harmonic Yee spatial operator.

Exact exterior Dirichlet data isolates material discretization from source and
absorber errors. This is a frequency-domain spatial validation, not a TFSF run.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import h1vp, hankel1, jv, jvp

from beamz.design import raster


def mie_hz(x, y, *, radius=0.4, epsilon=3.0, wavelength=1.0):
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    k = 2 * np.pi / wavelength
    ki = k * np.sqrt(epsilon)
    outside = r >= radius
    result = np.zeros(r.shape, complex)
    for m in range(-32, 33):
        ji, jip = jv(m, ki * radius), jvp(m, ki * radius)
        jo, jop = jv(m, k * radius), jvp(m, k * radius)
        ho, hop = hankel1(m, k * radius), h1vp(m, k * radius)
        a = (ki / epsilon * jip * jo - k * jop * ji) / (
            k * hop * ji - ki / epsilon * jip * ho
        )
        b = (jo + a * ho) / ji
        radial = np.empty(r.shape, complex)
        radial[outside] = jv(m, k * r[outside]) + a * hankel1(m, k * r[outside])
        radial[~outside] = b * jv(m, ki * r[~outside])
        result += 1j**m * radial * np.exp(1j * m * theta)
    return result


def inverse(tensor):
    packed = np.asarray(tensor)[:, 0]
    if packed.shape[0] == 1:
        return (1 / packed[0], 1 / packed[0], np.zeros_like(packed[0]))
    xx, yy = packed[:2]
    xy = packed[3] if packed.shape[0] == 6 else np.zeros_like(xx)
    det = xx * yy - xy**2
    return yy / det, xx / det, -xy / det


def cylinder_error(cells, method, *, shift=0.0, vertices=2048):
    if method not in {"volume", "contour_path"}:
        raise ValueError("This test oracle supports scalar Yee coefficients only")
    length = 2.0
    h = length / cells
    center = np.array([shift * h, 0.37 * shift * h])
    theta = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
    polygon = np.column_stack((0.4 * np.cos(theta), 0.4 * np.sin(theta))) + center
    result = raster.rasterize(
        raster.Scene(
            (raster.Material(), raster.Material(3)),
            (raster.Object(raster.ExtrudedPolygon(raster.Polygon(polygon), -1, 1), 1),),
        ),
        raster.Grid.uniform((-1, -1, -0.5), (1, 1, 0.5), (cells, cells, 1)),
        options=raster.RasterOptions(smoothing=method, components="two_dimensional_te"),
    )
    n = cells
    ids = np.arange(n * n).reshape(n, n)

    def gradient(left, right):
        left, right = left.ravel(), right.ravel()
        rows = np.arange(len(left))
        return sparse.csr_matrix(
            (
                np.r_[-np.ones(len(left)) / h, np.ones(len(left)) / h],
                (np.r_[rows, rows], np.r_[left, right]),
            ),
            shape=(len(left), n * n),
        )

    gx = gradient(ids[:, :-1], ids[:, 1:])
    gy = gradient(ids[:-1], ids[1:])
    invxx, _, _ = inverse(result.yee_tensors["epsilon_ex"])
    _, invyy, _ = inverse(result.yee_tensors["epsilon_ey"])
    matrix = (
        gx.T @ sparse.diags(invyy[:, 1:-1].ravel()) @ gx
        + gy.T @ sparse.diags(invxx[1:-1].ravel()) @ gy
    )
    k = 2 * np.pi
    matrix = matrix - k * k * sparse.eye(n * n)
    coords = -1 + (np.arange(n) + 0.5) * h
    x, y = np.meshgrid(coords - center[0], coords - center[1])
    exact = mie_hz(x, y).ravel()
    boundary = np.ones((n, n), bool)
    boundary[1:-1, 1:-1] = False
    interior = np.flatnonzero(~boundary.ravel())
    edge = np.flatnonzero(boundary.ravel())
    numeric = exact.copy()
    numeric[interior] = spsolve(
        matrix[interior][:, interior].tocsc(), -matrix[interior][:, edge] @ exact[edge]
    )
    error = np.linalg.norm((numeric - exact)[interior]) / np.linalg.norm(
        exact[interior]
    )
    return float(error), dict(result.diagnostics)
