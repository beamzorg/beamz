"""Array hashing preserves cache keys without duplicating contiguous volumes."""

import hashlib

import numpy as np
import pytest

from beamz import _cache_tokens


@pytest.mark.parametrize(
    "array",
    [
        np.array(3.0),
        np.empty((0, 3)),
        np.arange(24).reshape(2, 3, 4),
        np.arange(24).reshape(4, 6).T,
        np.arange(24)[::-2],
        np.array([1 + 2j, 3 - 4j]),
        np.array([1, 2], dtype=">i4"),
    ],
)
def test_array_token_preserves_legacy_digest(array):
    expected = hashlib.blake2b(
        np.ascontiguousarray(array).tobytes(), digest_size=16
    ).hexdigest()
    assert _cache_tokens.array_cache_token(array) == (
        "array",
        array.shape,
        str(array.dtype),
        expected,
    )


def test_contiguous_array_hash_uses_original_buffer(monkeypatch):
    array = np.arange(128, dtype=np.float32)
    original = hashlib.blake2b

    def inspect_buffer(data, **kwargs):
        assert isinstance(data, memoryview)
        assert np.shares_memory(np.asarray(data), array)
        return original(data, **kwargs)

    monkeypatch.setattr(_cache_tokens.hashlib, "blake2b", inspect_buffer)
    _cache_tokens.array_cache_token(array)
