"""Lossless coefficient export checks, independent of either GPU engine."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.beamz_release_adapter import parameter_difference
from scripts.export_fdtdx_comparison import array_digest, decode_e_source


def test_byte_code_export_preserves_every_lane_and_zero_boundary_coefficients():
    indices = np.array([0, 1, 2, 3, 4, 0, 3, 2, 1], np.uint32)
    padded = np.pad(indices, (0, 3))
    packed = np.zeros(3, np.uint32)
    for lane in range(4):
        packed |= padded[lane::4] << np.uint32(8 * lane)
    pool = np.array([0, 0.25, 0.5, 1, 2], np.float32)
    coefficients = SimpleNamespace(e_source_x=packed.view(np.int32), e_decay_x=pool)
    decoded = decode_e_source(coefficients, "Ex", (1, 3, 3))
    np.testing.assert_array_equal(decoded.reshape(-1), pool[indices])


def test_dense_scalar_export_broadcasts_to_the_complete_component_support():
    coefficients = SimpleNamespace(e_source_y=np.float32(0.125))
    decoded = decode_e_source(coefficients, "Ey", (2, 3, 4))
    np.testing.assert_array_equal(decoded, np.full((2, 3, 4), 0.125, np.float32))


def test_contract_array_hash_covers_shape_dtype_and_all_bits():
    values = np.arange(12, dtype=np.float32)
    assert array_digest(values) != array_digest(values.reshape(3, 4))
    assert array_digest(values) != array_digest(values.view(np.uint32))
    modified = values.copy()
    modified[-1] = np.nextafter(modified[-1], np.float32(np.inf))
    assert array_digest(values) != array_digest(modified)
    assert array_digest(values) == array_digest(values.copy())


def test_parameter_pin_accepts_reported_rounding_and_rejects_physics_changes():
    expected = np.array([-0.1, -0.01, 0, 1], np.float32)
    rounded = np.nextafter(expected, np.float32(np.inf))
    report = parameter_difference(rounded, expected)
    assert report["max_ulp"] == 1
    assert report["changed_values"] == 4
    with pytest.raises(ValueError, match="8 ULP"):
        parameter_difference(expected * np.float32(1.01), expected)
    with pytest.raises(ValueError, match="shape"):
        parameter_difference(expected.reshape(2, 2), expected)
