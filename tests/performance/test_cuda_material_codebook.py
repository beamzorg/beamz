from __future__ import annotations

import numpy as np

from beamz.simulation.compile import (
    _pack_cuda_coefficient_ids,
    _pack_cuda_lossless_e_coefficients,
)


def _unpack(table, packed, size):
    words = np.asarray(packed).view(np.uint32)
    shifts = np.arange(4, dtype=np.uint32) * np.uint32(8)
    codes = ((words[:, None] >> shifts) & np.uint32(0xFF)).reshape(-1)[:size]
    return np.asarray(table)[codes]


def test_cuda_coefficient_ids_round_trip_exact_fp32_values():
    values = np.asarray([[[1.0, 2.0, 4.0], [2.0, 1.0, 8.0]]], dtype=np.float32)

    encoded = _pack_cuda_coefficient_ids(values)

    assert encoded is not None
    table, packed = encoded
    np.testing.assert_array_equal(_unpack(table, packed, values.size), values.ravel())
    assert packed.dtype == np.int32
    assert packed.size == 2


def test_cuda_lossless_codebook_requires_scalar_unit_decay():
    sources = tuple(np.asarray([[[1.0, 2.0, 1.0]]], dtype=np.float32) for _ in range(3))

    packed = _pack_cuda_lossless_e_coefficients(
        tuple(np.asarray(1.0, dtype=np.float32) for _ in range(3)), sources
    )

    assert packed is not None
    tables, ids = packed
    for table, words, source in zip(tables, ids, sources, strict=True):
        np.testing.assert_array_equal(
            _unpack(table, words, source.size), source.ravel()
        )

    assert (
        _pack_cuda_lossless_e_coefficients(
            (np.asarray(0.9, dtype=np.float32), *(np.asarray(1.0) for _ in range(2))),
            sources,
        )
        is None
    )


def test_codebook_blocks_match_dense_encoding_with_late_values_and_padding(monkeypatch):
    import importlib

    module = importlib.import_module("beamz.simulation.compile")
    monkeypatch.setattr(module, "_CUDA_CODEBOOK_CHUNK_CELLS", 8)
    values = np.array(
        [3, 1, 3, 2, 1, 3, 2, 1, -4, 3, 8, 2, 9], dtype=np.float32
    ).reshape(1, 1, -1)
    expected_table, inverse = np.unique(values, return_inverse=True)
    ids = np.zeros((values.size + 3) & ~3, dtype=np.uint32)
    ids[: values.size] = inverse.ravel()
    expected_words = ids[::4] | (ids[1::4] << 8) | (ids[2::4] << 16) | (ids[3::4] << 24)
    table, words = _pack_cuda_coefficient_ids(values)
    np.testing.assert_array_equal(table, expected_table)
    np.testing.assert_array_equal(np.asarray(words).view(np.uint32), expected_words)
    np.testing.assert_array_equal(_unpack(table, words, values.size), values.ravel())
    assert _pack_cuda_coefficient_ids(values, max_values=4) is None


def test_codebook_rejects_more_than_256_values_across_blocks(monkeypatch):
    import importlib

    module = importlib.import_module("beamz.simulation.compile")
    monkeypatch.setattr(module, "_CUDA_CODEBOOK_CHUNK_CELLS", 8)
    assert (
        _pack_cuda_coefficient_ids(np.arange(257, dtype=np.float32).reshape(1, 1, -1))
        is None
    )
