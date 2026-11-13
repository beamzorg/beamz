"""Representation-only block packets preserve every physical coefficient bit."""

import numpy as np
import pytest

from scripts.cuda_native_block_cases import rank2_e_packet


def byte_packet():
    inputs = [np.zeros((), np.float32) for _ in range(134)]
    table = np.array([0., -0., .125, -3.5], np.float32)
    for c, size in enumerate((33, 65, 37)):
        inputs[3 + c] = np.zeros((1, 1, size), np.float32)
        ids = np.array(([0, 1, 2, 3, 255] * (size // 5 + 1))[:size], np.uint8)
        padded = list(map(int, ids)) + [255] * (-size % 4)
        words = [sum(padded[i + j] << (8 * j) for j in range(4))
                 for i in range(0, len(padded), 4)]
        inputs[37 + c] = table.copy()
        inputs[40 + c] = np.array(words, np.uint32).view(np.int32)
    return inputs


def test_rank2_decoding_matches_original_bits_including_signed_zero_and_tail():
    inputs = byte_packet()
    old = [x.copy() for x in inputs]
    changed = rank2_e_packet(inputs)
    for c in range(3):
        size = inputs[3 + c].size
        headers, pool = changed[40 + c], changed[37 + c]
        expected = np.array([inputs[37 + c][i % 5] if i % 5 < 4 else 0.
                             for i in range(size)], np.float32)
        decoded = np.array([pool[int(headers[0, i // 32]) + i % 32]
                            for i in range(size)], np.float32)
        assert np.array_equal(decoded.view(np.uint32), expected.view(np.uint32))
        assert np.all(pool[size:].view(np.uint32) == 0)
        assert headers.ndim == 2 and pool.size % 32 == 0
    assert all(a.tobytes() == b.tobytes()
               for a, b in zip(inputs, old, strict=True))


def test_rank2_conversion_rejects_already_changed_or_malformed_encoding():
    inputs = byte_packet()
    converted = rank2_e_packet(inputs)
    with pytest.raises(ValueError, match="byte-packed"):
        rank2_e_packet(converted)
    inputs[40] = inputs[40][:-1]
    with pytest.raises(ValueError, match="code count"):
        rank2_e_packet(inputs)


def test_negative_constant_headers_preserve_signed_zero_and_mixed_blocks():
    from scripts.cuda_native_const_block_cases import constant_rank2_e_packet
    inputs = byte_packet()
    for c in range(3):
        inputs[40+c].view(np.uint32)[:8] = np.uint32(0x01010101)
    direct = rank2_e_packet(inputs)
    compressed = constant_rank2_e_packet(inputs)
    seen_negative = seen_positive = False
    for c in range(3):
        pool, headers = compressed[37+c], compressed[40+c].ravel()
        decoded = []
        for i in range(inputs[3+c].size):
            h = int(headers[i//32])
            decoded.append(pool[~h] if h < 0 else pool[h+i%32])
        assert np.array_equal(np.array(decoded,np.float32).view(np.uint32),
                              direct[37+c][:inputs[3+c].size].view(np.uint32))
        seen_negative |= bool(np.any(headers < 0))
        seen_positive |= bool(np.any(headers >= 0))
    assert seen_negative and seen_positive
