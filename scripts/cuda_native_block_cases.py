"""Lossless byte-to-mixed32 coefficient encoding for admitted-path validation."""

from __future__ import annotations

import numpy as np


def rank2_e_packet(inputs):
    """Re-encode all three E coefficients without changing any physical float bits.

    Positive block headers point to32 direct float slots. Invalid original byte
    codes decode to positive zero, exactly as PackedMaterialSource. Tail padding
    is initialized but never represents a physical Yee point.
    """
    changed = list(inputs)
    for component in range(3):
        table, codes = inputs[37 + component], inputs[40 + component]
        if (table.ndim != 1 or table.dtype != np.float32 or codes.ndim != 1
                or codes.dtype != np.int32 or len(table) > 256):
            raise ValueError("Rank2 transform requires canonical byte-packed F32 E")
        size = inputs[3 + component].size
        if codes.size != (size + 3) // 4:
            raise ValueError("Packed code count does not match the Yee component")
        lookup = np.zeros(256, np.float32)
        lookup[:len(table)] = table
        shifts = np.arange(4, dtype=np.uint32) * np.uint32(8)
        ids = ((codes.view(np.uint32)[:, None] >> shifts) & np.uint32(255)).ravel()[:size]
        blocks = (size + 31) // 32
        pool = np.zeros(blocks * 32, np.float32)
        pool[:size] = lookup[ids]
        headers = (np.arange(blocks, dtype=np.int32) * np.int32(32)).reshape(1, -1)
        changed[37 + component], changed[40 + component] = pool, headers
    return changed
