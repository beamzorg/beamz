"""Region-wise Yee material preparation for partitioned three-dimensional grids."""

import numpy as np

from beamz._region_array import TILE_BYTES, RegionArray, broadcast_base
from beamz.const import EPS_0, MU_0
from beamz.lattice import MaterialCoefficients, component_axis_offsets_3d


def sample_material(grid, component, shape):
    """Match lattice.py's axis-ordered interpolation using only local input cells."""
    if grid.ndim == 0 or (isinstance(grid, np.ndarray) and not any(grid.strides)):
        return np.asarray(grid[(0,) * grid.ndim], dtype=np.float32)
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

    if isinstance(grid, np.ndarray) and 0 in grid.strides:
        # Translation along a repeated axis does not change Yee interpolation.
        # Compute the distinct positions once when the compact result fits a tile.
        base_shape = tuple(1 if stride == 0 else n for n, stride in zip(shape, grid.strides, strict=True))
        if np.prod(base_shape, dtype=np.int64) * np.dtype(np.float32).itemsize <= TILE_BYTES:
            values = read(tuple(slice(0, n) for n in base_shape))
            return np.broadcast_to(values, shape)
    return RegionArray(tuple(shape), np.dtype(np.float32), read)


def mapped_material(shape, function, *values):
    if all(value.ndim == 0 for value in values):
        return np.asarray(function(*values), dtype=np.float32)
    if all(value.ndim == 0 or isinstance(value, np.ndarray) and 0 in value.strides for value in values):
        bases = tuple(broadcast_base(np.asarray(value)) for value in values)
        physical = np.broadcast_shapes(*(base.shape for base in bases))
        if np.prod(physical, dtype=np.int64) * np.dtype(np.float32).itemsize <= TILE_BYTES:
            return np.broadcast_to(np.asarray(function(*bases), dtype=np.float32), shape)
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


def lossless_electric_source(epsilon, dt, eps0=EPS_0):
    """Apply the CPU Yee coefficient's two FP32 roundings to one material tile."""
    denominator = np.asarray(np.float32(eps0) * epsilon)
    return np.divide(np.float32(dt), denominator, out=denominator)


def electric_coefficients(shape, sigma, epsilon, dt):
    # Match the eager FP32 coefficient algebra, including its rounding order.
    if sigma.ndim == 0 and float(sigma) == 0:
        return np.asarray(1, np.float32), mapped_material(
            shape, lambda e: lossless_electric_source(e, dt), epsilon
        )

    def alpha(s, e):
        return s * np.float32(dt) / (np.float32(2 * EPS_0) * e)

    decay = mapped_material(
        shape,
        lambda s, e: (np.float32(1) - alpha(s, e)) / (np.float32(1) + alpha(s, e)),
        sigma,
        epsilon,
    )
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
