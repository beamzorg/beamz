"""Region-wise Yee material preparation for partitioned three-dimensional grids."""

import numpy as np

from beamz._region_array import RegionArray
from beamz.lattice import electric_update_factors, magnetic_update_factors


def mapped_material(shape, function, *values):
    if all(value.ndim == 0 for value in values):
        return np.asarray(function(*values), dtype=np.float32)
    return RegionArray(
        tuple(shape),
        np.dtype(np.float32),
        lambda region: function(*(v if v.ndim == 0 else v[region] for v in values)),
    )


def _mapped_factors(shape, function, *values):
    return tuple(
        mapped_material(shape, lambda *v, i=i: function(*v)[i], *values)
        for i in range(2)
    )


def electric_coefficients(shape, sigma, epsilon, dt):
    decay, source = _mapped_factors(
        shape, lambda s, e: electric_update_factors(s, e, dt), sigma, epsilon
    )
    if sigma.ndim == 0 and float(sigma) == 0:
        decay = np.asarray(1, np.float32)
    return decay, source


def magnetic_coefficients(sigma, dt):
    return _mapped_factors(sigma.shape, lambda s: magnetic_update_factors(s, dt), sigma)
