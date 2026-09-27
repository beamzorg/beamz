"""Internal immutable region recipes for bounded host preparation.

A recipe is not runtime state. Indexing evaluates only the requested bounding
region; explicit NumPy/JAX conversion materializes it for small-grid diagnostics.
"""

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np

# Each reader's intermediate working set is a small multiple of this tile size.
TILE_BYTES = 16 * 1024 * 1024


def tiles(shape, dtype, budget=None):
    """Yield rectangular tiles, bounding all dimensions (including large planes)."""
    budget = TILE_BYTES if budget is None else budget
    if budget <= 0:
        raise ValueError("Tile budget must be positive")
    if not all(shape):
        return
    extent = list(shape)
    while np.prod(extent, dtype=np.int64) * np.dtype(dtype).itemsize > budget:
        axis = int(np.argmax(extent))
        extent[axis] = (extent[axis] + 1) // 2
    for starts in product(
        *(range(0, n, k) for n, k in zip(shape, extent, strict=True))
    ):
        yield tuple(
            slice(i, min(i + k, n))
            for i, k, n in zip(starts, extent, shape, strict=True)
        )


@dataclass(frozen=True, eq=False)
class RegionArray:
    shape: tuple[int, ...]
    dtype: np.dtype
    reader: Callable

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape, dtype=np.int64))

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        if any(k is Ellipsis for k in key):
            at = next(i for i, k in enumerate(key) if k is Ellipsis)
            key = key[:at] + (slice(None),) * (self.ndim - len(key) + 1) + key[at + 1 :]
        key = key + (slice(None),) * (self.ndim - len(key))
        if len(key) != self.ndim or any(k is None for k in key):
            raise IndexError("Region arrays require one index per spatial axis")
        bounds, local = [], []
        for k, n in zip(key, self.shape, strict=True):
            if isinstance(k, slice):
                start, stop, step = k.indices(n)
                if step == 1:
                    bounds.append(slice(start, max(start, stop)))
                    local.append(slice(None))
                    continue
                indices = np.arange(start, stop, step)
            else:
                indices = np.asarray(k)
                if indices.dtype.kind not in "iu":
                    raise IndexError("Region array indices must be integers")
                indices = np.where(indices < 0, indices + n, indices)
                if np.any((indices < 0) | (indices >= n)):
                    raise IndexError("Region array index out of bounds")
            lo = int(indices.min()) if indices.size else 0
            hi = int(indices.max()) + 1 if indices.size else 0
            bounds.append(slice(lo, hi))
            if isinstance(k, slice):
                if not indices.size:
                    local.append(slice(0, 0))
                else:
                    end = int(indices[-1] - lo) + (1 if step > 0 else -1)
                    local.append(
                        slice(int(indices[0] - lo), None if end == -1 else end, step)
                    )
            else:
                local.append(indices - lo)
        value = np.asarray(self.reader(tuple(bounds)), dtype=self.dtype)
        return value[tuple(local)]

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("A region recipe cannot be materialized without a copy")
        out = np.empty(self.shape, dtype=self.dtype if dtype is None else dtype)
        for tile in tiles(self.shape, out.dtype):
            out[tile] = self[tile]
        return out

    def __jax_array__(self):
        import jax.numpy as jnp

        return jnp.asarray(np.asarray(self))

    def padded(self, shape, fill=0):
        shape = tuple(shape)
        if shape == self.shape:
            return self

        def read(index):
            out = np.full(tuple(s.stop - s.start for s in index), fill, self.dtype)
            clipped = tuple(
                slice(min(s.start, n), min(s.stop, n))
                for s, n in zip(index, self.shape, strict=True)
            )
            lengths = tuple(s.stop - s.start for s in clipped)
            if all(lengths):
                out[tuple(slice(0, n) for n in lengths)] = self[clipped]
            return out

        return RegionArray(shape, self.dtype, read)


def constant_array(shape, value, dtype=np.float32):
    return np.broadcast_to(np.asarray(value, dtype=dtype), shape)


def place_region_array(value, target):
    """Fill destination buffers with bounded tiles; never assemble a host shard.

    Donated local updates retain one destination buffer plus one input tile.
    Transfers finish before the next tile is read, bounding asynchronous staging.
    """
    import jax
    import jax.numpy as jnp

    arrays = []
    for device, index in target.addressable_devices_indices_map(value.shape).items():
        index = tuple(
            slice(0 if s.start is None else s.start, n if s.stop is None else s.stop)
            for s, n in zip(index, value.shape, strict=True)
        )
        shape = tuple(s.stop - s.start for s in index)
        placement = jax.sharding.SingleDeviceSharding(device)
        out = jax.jit(
            lambda shape=shape: jnp.zeros(shape, dtype=value.dtype),
            out_shardings=placement,
        )()
        update = jax.jit(
            jax.lax.dynamic_update_slice, donate_argnums=(0,), out_shardings=placement
        )
        for tile in tiles(shape, value.dtype):
            global_index = tuple(
                slice(s.start + t.start, s.start + t.stop)
                for s, t in zip(index, tile, strict=True)
            )
            data = jax.device_put(value[global_index], placement)
            starts = tuple(jax.device_put(np.int32(t.start), placement) for t in tile)
            out = update(out, data, starts)
            out.block_until_ready()
        arrays.append(out)
    return jax.make_array_from_single_device_arrays(value.shape, target, arrays)


class SeparableMask(RegionArray):
    """Boolean wall masks stored as short axis profiles, including padding."""

    def __init__(self, profiles):
        profiles = tuple(np.asarray(p, dtype=bool) for p in profiles)
        shape = tuple(p.size for p in profiles)
        for profile in profiles:
            profile.setflags(write=False)

        def read(region):
            result = np.zeros(tuple(s.stop - s.start for s in region), bool)
            for axis, (profile, part) in enumerate(zip(profiles, region, strict=True)):
                view = [1] * len(profiles)
                view[axis] = part.stop - part.start
                result |= profile[part].reshape(view)
            return result

        super().__init__(shape, np.bool_, read)
        object.__setattr__(self, "profiles", profiles)

    def padded(self, shape, fill=False):
        profiles = []
        for old, size in zip(self.profiles, shape, strict=True):
            new = np.full(size, fill, bool)
            new[: min(size, old.size)] = old[:size]
            profiles.append(new)
        return SeparableMask(profiles)
