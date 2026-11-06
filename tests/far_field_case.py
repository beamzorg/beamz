"""Independent analytic electric-dipole fields and Gaussian surface quadrature."""

import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0


def sphere_quadrature(n_theta=10, n_phi=20):
    cosine, weights = np.polynomial.legendre.leggauss(n_theta)
    phi = np.arange(n_phi) * 2 * np.pi / n_phi
    cosine, phi = np.meshgrid(cosine, phi, indexing="ij")
    sine = np.sqrt(1 - cosine**2)
    directions = np.stack((sine * np.cos(phi), sine * np.sin(phi), cosine), -1)
    weights = np.broadcast_to(weights[:, None] * 2 * np.pi / n_phi, cosine.shape)
    return directions.reshape(-1, 3), weights.ravel()


def box_quadrature(half_size=0.4, center=(0, 0, 0), count=16):
    nodes, weights = np.polynomial.legendre.leggauss(count)
    xx, yy = np.meshgrid(nodes * half_size, nodes * half_size, indexing="ij")
    area = np.outer(weights, weights).ravel() * half_size**2
    points, normals, areas = [], [], []
    for axis in range(3):
        tangential = [i for i in range(3) if i != axis]
        for sign in (-1, 1):
            p = np.zeros((count**2, 3))
            p[:, axis] = sign * half_size
            p[:, tangential[0]], p[:, tangential[1]] = xx.ravel(), yy.ravel()
            points.append(p + center)
            normals.append(np.broadcast_to(np.eye(3)[axis] * sign, p.shape))
            areas.append(area)
    return np.concatenate(points), np.concatenate(normals), np.concatenate(areas)


def dipole_fields(points, dipole=(0, 0, 1), *, frequency=LIGHT_SPEED, index=1):
    """Exact radiating and reactive fields, scaled to unit far E amplitude.

    Independent closed-form Maxwell solution, including 1/r^2 and 1/r^3 terms.
    """
    r = np.linalg.norm(points, axis=-1)[:, None]
    s = points / r
    p = np.asarray(dipole)
    k = 2 * np.pi * frequency * index / LIGHT_SPEED
    radial = s * np.sum(s * p, axis=-1, keepdims=True)
    transverse = p - radial
    e = (
        transverse / r + (3 * radial - p) * (1 / (k * k * r**3) - 1j / (k * r**2))
    ) * np.exp(1j * k * r)
    impedance = np.sqrt(MU_0 / EPS_0) / index
    h = np.cross(s, p) * (1 / r + 1j / (k * r**2)) * np.exp(1j * k * r) / impedance
    return e, h
