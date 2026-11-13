"""Exact mixed/constant block packets, including negative-header decoding."""

import numpy as np

from scripts.cuda_native_block_cases import rank2_e_packet


def constant_rank2_e_packet(inputs):
    direct = rank2_e_packet(inputs)
    changed = list(inputs)
    for c in range(3):
        values = direct[37 + c].reshape(-1, 32)
        bits = values.view(np.uint32)
        constant = np.all(bits == bits[:, :1], axis=1)
        lookup = np.zeros(256, np.float32)
        lookup[:inputs[37 + c].size] = inputs[37 + c]
        by_bits = {}
        for i, value in enumerate(lookup.view(np.uint32)):
            by_bits.setdefault(int(value), i)
        mixed = np.flatnonzero(~constant)
        pool = np.concatenate([lookup, values[mixed].ravel()])
        headers = np.empty(values.shape[0], np.int32)
        headers[mixed] = 256 + np.arange(mixed.size, dtype=np.int32) * 32
        for i in np.flatnonzero(constant):
            headers[i] = ~by_bits[int(bits[i, 0])]
        changed[37 + c], changed[40 + c] = pool, headers.reshape(1, -1)
    return changed
