"""Guard against accepting unit, amplitude, and phasor clock mismatches."""

import numpy as np

from scripts.validate_fdtdx_comparison import compare


def test_gate_rejects_wrong_h_units_and_amplitude():
    signal = np.linspace(-1, 1, 192, dtype=np.float32).reshape(3, 4, 4, 4)
    assert compare(signal, signal)["relative_l2"] == 0
    assert compare(signal, signal * 377)["relative_l2"] > 0.99
    assert compare(signal, signal * 1.02)["relative_l2"] > 0.01


def test_gate_rejects_unaccounted_phasor_rotation():
    phase = np.exp(1j * np.linspace(0, 2, 90)).reshape(3, 30)
    assert compare(phase, phase * np.exp(0.1j))["relative_l2"] > 0.09


def test_gate_never_discards_an_unexpected_imaginary_error():
    real = np.ones((3, 8), np.float32)
    assert compare(real, real + 1j)["relative_l2"] > 0.7


def test_gate_excludes_only_declared_outer_field_cells():
    a = np.ones((3, 32, 32, 32), np.float32)
    b = a.copy()
    b[:, :14] = 100
    assert compare(a, b, interior=14)["relative_l2"] == 0
    b[:, 14] = 2
    assert compare(a, b, interior=14)["relative_l2"] > 0.01
