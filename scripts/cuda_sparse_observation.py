"""CPU construction of an exact sparse raw-observation lookup.

Raw IDs remain the positions of the original sorted native component keys.
Only lookup changes: coarse[z,y//8,x//16] then leaf[128 cells,6 fields].
"""

from __future__ import annotations

import math

import numpy as np


def build_sparse_observation(field_shapes, keys):
    shapes = np.asarray(field_shapes, dtype=np.int64)
    keys = np.asarray(keys)
    if shapes.shape != (6, 3) or np.any(shapes < 1):
        raise ValueError("Expected six positive ZYX field shapes")
    sizes = [math.prod(map(int, shape)) for shape in shapes]
    if any(size > np.iinfo(np.int32).max for size in sizes):
        raise ValueError("Field offsets exceed int32")
    sizes = np.asarray(sizes, dtype=np.int64)
    prefix = np.r_[np.int64(0), np.cumsum(sizes)]
    if keys.ndim != 1 or keys.dtype != np.dtype(np.int64):
        raise ValueError("Raw keys must be one-dimensional int64")
    if keys.size > np.iinfo(np.int32).max or (
        keys.size
        and (keys[0] < 0 or keys[-1] >= prefix[-1] or np.any(keys[1:] <= keys[:-1]))
    ):
        raise ValueError("Raw keys must be valid, unique, and sorted")
    extent = shapes.max(axis=0)
    grid = (int(extent[0]), (int(extent[1]) + 7) // 8, (int(extent[2]) + 15) // 16)
    grid_cells = math.prod(grid)
    if grid_cells > np.iinfo(np.int32).max:
        raise ValueError("Coarse observation offsets exceed int32")
    field = np.searchsorted(prefix[1:], keys, side="right")
    local = keys - prefix[field]
    z = local // (shapes[field, 1] * shapes[field, 2])
    y = (local // shapes[field, 2]) % shapes[field, 1]
    x = local % shapes[field, 2]
    tile = (z * grid[1] + y // 8) * grid[2] + x // 16
    occupied, inverse = np.unique(tile, return_inverse=True)
    coarse = np.full(grid_cells, -1, dtype=np.int32)
    coarse[occupied] = np.arange(occupied.size, dtype=np.int32)
    leaf = np.full((occupied.size, 128, 6), -1, dtype=np.int32)
    leaf[inverse, (y % 8) * 16 + x % 16, field] = np.arange(keys.size, dtype=np.int32)
    return {
        "grid_zyx": grid,
        "coarse": coarse,
        "leaf": leaf,
        "raw_key_count": int(keys.size),
        "occupied_tiles": int(occupied.size),
        "extra_gpu_bytes": int(coarse.nbytes + leaf.nbytes),
    }


def lookup_sparse(plan, field, z, y, x):
    nz, ny, nx = plan["grid_zyx"]
    if (
        z < 0
        or y < 0
        or x < 0
        or z >= nz
        or y // 8 >= ny
        or x // 16 >= nx
        or field < 0
        or field >= 6
    ):
        return -1
    slot = int(plan["coarse"][(z * ny + y // 8) * nx + x // 16])
    return -1 if slot < 0 else int(plan["leaf"][slot, (y % 8) * 16 + x % 16, field])
