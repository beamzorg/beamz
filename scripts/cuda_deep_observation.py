"""Unique-cell temporal observation lowering over the real packed DFT plan.

Pure NumPy host setup. It preserves invalid-index skipping, every valid record
(including zero weights), and original gather/weight order. The staged table
can be consumed by the unchanged native DFT kernel through remapped indices.
"""

from __future__ import annotations

import numpy as np


def build_plan(field_shapes_native, packed_indices):
    shapes = tuple(tuple(map(int, s)) for s in field_shapes_native)
    if len(shapes) != 6 or any(len(s) != 3 or min(s) < 1 for s in shapes):
        raise ValueError("Expected six positive native Hx,Hy,Hz,Ex,Ey,Ez shapes")
    indices = np.asarray(packed_indices)
    if indices.dtype != np.int32 or indices.ndim != 4 or indices.shape[1] != 6:
        raise ValueError("Expected native packed int32 DFT indices[M,6,P,K]")
    sizes = np.asarray([np.prod(s, dtype=np.int64) for s in shapes], np.int64)
    bases = np.concatenate((np.zeros(1, np.int64), np.cumsum(sizes)[:-1]))
    keys = np.full(indices.shape, -1, np.int64)
    for c in range(6):
        native = (c + 3) % 6
        raw = indices[:, c]
        valid = (raw >= 0) & (raw < sizes[native])
        keys[:, c] = np.where(valid, bases[native] + raw, -1)
    unique = np.unique(keys[keys >= 0])
    if unique.size > np.iinfo(np.int32).max:
        raise ValueError("Unique observed-cell count exceeds native int32 indexing")
    remapped = np.full(indices.shape, -1, np.int32)
    valid = keys >= 0
    remapped[valid] = np.searchsorted(unique, keys[valid]).astype(np.int32)
    mask_shape = tuple(max(s[a] for s in shapes) for a in range(3))
    mask_bytes = np.zeros((int(np.prod(mask_shape)) + 3) & ~3, np.uint8)
    for c in range(6):
        local = unique[(unique >= bases[c]) & (unique < bases[c] + sizes[c])] - bases[c]
        z, y, x = np.unravel_index(local, shapes[c])
        flat = (z * mask_shape[1] + y) * mask_shape[2] + x
        mask_bytes[flat] |= np.uint8(1 << c)
    words = np.zeros(mask_bytes.size // 4, np.uint32)
    for lane in range(4):
        words |= mask_bytes[lane::4].astype(np.uint32) << (8 * lane)
    return {
        "keys": unique,
        "remapped_indices": remapped,
        "publication_words": words.view(np.int32),
        "field_base": bases,
        "mask_shape_zyx": np.asarray(mask_shape, np.int32),
        "key_count": int(unique.size),
        "staged_bytes_d4": int(unique.size) * 4 * 4,
        "original_record_count": int(indices.size),
        "mask_bytes": int(words.nbytes),
    }
