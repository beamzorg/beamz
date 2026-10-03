"""Region-wise Yee material preparation for partitioned three-dimensional grids."""

import numpy as np

from beamz._region_array import RegionArray
from beamz.const import EPS_0, MU_0
from beamz.lattice import MaterialCoefficients, component_axis_offsets_3d


def sample_material(grid, component, shape):
    """Match lattice.py's axis-ordered interpolation using only local input cells."""
    if grid.ndim == 0:
        return np.asarray(grid, dtype=np.float32)
    offsets = component_axis_offsets_3d(component)

    def read(region):
        pairs, bounds = [], []
        for axis, key, n in zip("zyx", region, grid.shape, strict=True):
            hi = np.clip(np.arange(key.start, key.stop), 0, n - 1)
            lo = (
                np.clip(hi - 1, 0, n - 1)
                if component.startswith("E") and offsets[axis] == 0
                else hi
            )
            # At the high boundary both bracketing cells clamp to the last voxel.
            if component.startswith("E") and offsets[axis] == 0:
                lo = np.clip(np.arange(key.start, key.stop) - 1, 0, n - 1)
            if not hi.size:
                return np.empty(tuple(s.stop - s.start for s in region), np.float32)
            start = int(min(lo.min(), hi.min()))
            stop = int(max(lo.max(), hi.max())) + 1
            bounds.append(slice(start, stop))
            pairs.append((lo - start, hi - start))
        values = np.asarray(grid[tuple(bounds)], dtype=np.float32)
        for axis, ((lo, hi), name) in enumerate(zip(pairs, "zyx", strict=True)):
            if component.startswith("E") and offsets[name] == 0:
                values = np.float32(0.5) * (
                    np.take(values, lo, axis=axis) + np.take(values, hi, axis=axis)
                )
            else:
                values = np.take(values, hi, axis=axis)
        return values

    return RegionArray(tuple(shape), np.dtype(np.float32), read)


def mapped_material(shape, function, *values):
    if all(value.ndim == 0 for value in values):
        return np.asarray(function(*values), dtype=np.float32)
    return RegionArray(
        tuple(shape),
        np.dtype(np.float32),
        lambda region: function(*(v if v.ndim == 0 else v[region] for v in values)),
    )


def build_region_materials(fields):
    sigma, mu = fields.conductivity, fields.permeability
    # This path supports CPML (or no absorber), whose loss lives in psi.
    # Physical electric conductivity must not introduce magnetic damping.
    sigma_base = np.asarray(0, np.float32)
    data = {"total_conductivity": sigma}
    for axis in "xyz":
        e, h = "E" + axis, "H" + axis
        data["eps_" + axis] = sample_material(
            fields.permittivity, e, getattr(fields, e).shape
        )
        data["sig_" + axis] = sample_material(sigma, e, getattr(fields, e).shape)
        data["region_" + axis] = (slice(None),) * 3
        data["eps_e" + axis] = data["eps_" + axis]
        data["sigma_m_h" + axis] = sample_material(
            sigma_base, h, getattr(fields, h).shape
        )
        data["mu_h" + axis] = sample_material(mu, h, getattr(fields, h).shape)
    return MaterialCoefficients(**data)


def electric_coefficients(shape, sigma, epsilon, dt):
    # Match the eager FP32 coefficient algebra, including its rounding order.
    def alpha(s, e):
        return s * np.float32(dt) / (np.float32(2 * EPS_0) * e)

    decay = mapped_material(
        shape,
        lambda s, e: (np.float32(1) - alpha(s, e)) / (np.float32(1) + alpha(s, e)),
        sigma,
        epsilon,
    )
    if sigma.ndim == 0 and float(sigma) == 0:
        decay = np.asarray(1, np.float32)
    source = mapped_material(
        shape,
        lambda s, e: (
            (np.float32(dt) / (np.float32(EPS_0) * e)) / (np.float32(1) + alpha(s, e))
        ),
        sigma,
        epsilon,
    )
    return decay, source


def magnetic_coefficients(sigma, dt):
    def alpha(s):
        return s * np.float32(dt) / np.float32(2 * MU_0)

    return (
        mapped_material(
            sigma.shape,
            lambda s: (np.float32(1) - alpha(s)) / (np.float32(1) + alpha(s)),
            sigma,
        ),
        mapped_material(
            sigma.shape,
            lambda s: np.float32(dt / MU_0) / (np.float32(1) + alpha(s)),
            sigma,
        ),
    )
