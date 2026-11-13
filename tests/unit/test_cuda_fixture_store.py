"""Exercise real mmap fixture roundtrips and bounded numerical gates."""

import numpy as np

from scripts.cuda_fixture_store import (
    ArrayReader,
    ArrayWriter,
    compare_arrays,
    digest_array,
)
from scripts.export_cuda_native_fixture import buffer_digest


def test_mmap_roundtrip_alias_scalar_empty_and_sparse_zeros(tmp_path):
    writer = ArrayWriter(tmp_path)
    source = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
    writer.put("field", source)
    writer.alias("initial_field", "field")
    writer.put("clock", np.asarray(7, np.int32))
    writer.put("empty", np.empty((0, 3), np.float32))
    writer.put("workspace", shape=(1024, 1024), dtype=np.float32, zero=True)
    writer.finish()
    reader = ArrayReader(tmp_path)
    assert isinstance(reader["field"], np.memmap)
    assert not reader["field"].flags.writeable
    assert reader["field"] is reader["initial_field"]
    np.testing.assert_array_equal(reader["field"], source)
    assert reader["clock"].shape == () and reader["clock"] == 7
    assert reader["empty"].shape == (0, 3)
    assert not np.any(reader["workspace"])
    for key in reader.metadata:
        reader.verify(key)
        assert digest_array(reader[key]) == buffer_digest(reader[key])


def test_chunked_gate_finds_last_chunk_error_and_signed_zero():
    expected = np.arange(100, dtype=np.float32)
    actual = expected.copy()
    actual[-1] += 1
    result = compare_arrays(actual, expected, chunk_bytes=8)
    assert not result["passed"] and result["max_abs_error"] == 1
    zero = compare_arrays(np.array([0.0], np.float32), np.array([-0.0], np.float32))
    assert zero["passed"] and zero["exact"] and not zero["bitwise_exact"]


def test_gate_rejects_nonfinite_and_dtype_or_shape_changes():
    assert not compare_arrays(
        np.array([np.nan], np.float32), np.array([np.nan], np.float32)
    )["passed"]
    assert not compare_arrays(np.ones(2, np.float32), np.ones(2, np.int32))["passed"]
    assert not compare_arrays(np.ones((1, 2), np.float32), np.ones(2, np.float32))[
        "passed"
    ]


def test_window_infinity_sentinel_roundtrips_but_state_requires_finite(tmp_path):
    writer = ArrayWriter(tmp_path)
    window = np.array([-np.inf, np.inf, 1], np.float32)
    info = writer.put("window", window)
    assert not info["finite"] and info["max_abs"] == 1
    writer.finish()
    reader = ArrayReader(tmp_path)
    np.testing.assert_array_equal(reader["window"], window)
    reader.verify("window")
    try:
        writer.put("invalid_state", window, require_finite=True)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite persistent state must be rejected")
