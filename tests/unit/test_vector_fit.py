"""Independent analytical checks of vector fitting, units and passivity."""

import numpy as np
import pytest

import beamz as bz
from beamz.design.vector_fit import _basis, _passive_residues


@pytest.mark.parametrize("kind", ["lorentz", "debye", "two_lorentz"])
def test_recover_known_response_on_unseen_frequencies(kind):
    wavelength = np.linspace(350e-9, 1000e-9, 101)
    frequency = bz.LIGHT_SPEED / wavelength
    omega = 2 * np.pi * frequency

    def response(w):
        if kind == "debye":
            return 2.0 + 1.7 / (1 - 1j * w * 3e-16)
        result = 2.0 + 1.4 * (3.4e15) ** 2 / ((3.4e15) ** 2 - w * w - 1j * 2e14 * w)
        if kind == "two_lorentz":
            result += 0.7 * (5.1e15) ** 2 / ((5.1e15) ** 2 - w * w - 1j * 6e14 * w)
        return result

    nk = np.sqrt(response(omega))
    model, report = bz.fit_nk_vector(
        wavelength,
        nk.real,
        nk.imag,
        epsilon_inf=2,
        max_poles=2 if kind == "two_lorentz" else 1,
        tolerance_rms=1e-7,
        num_iters=40,
    )
    unseen = np.geomspace(frequency.min() * 0.8, frequency.max() * 1.2, 217)
    np.testing.assert_allclose(
        model.eps_model(unseen), response(2 * np.pi * unseen), rtol=1e-7, atol=1e-7
    )
    assert report["tolerance_met"]
    assert all(a.real <= 0 for a, _ in model.poles)


def test_residue_projection_removes_out_of_band_gain():
    omega = np.linspace(0.7, 1.3, 100)
    poles = np.array([-0.1 - 1j, -0.1 - 3j])
    coefficients = np.array([0.0, 0.6, 0.0, -0.04])
    target = 1 + _basis(omega, poles) @ coefficients
    assert target.imag.min() > 0
    audit = np.geomspace(1e-6, 1e6, 10003)
    assert (_basis(audit, poles) @ coefficients).imag.min() < -0.1
    passive = _passive_residues(omega, target, poles, coefficients, 1, np.ones(2))
    assert passive is not None
    fitted, _ = passive
    assert (_basis(audit, poles) @ fitted).imag.min() >= -1e-10


def test_reports_missed_tolerance_and_weighted_error():
    wavelength = np.linspace(400e-9, 700e-9, 61)
    n = 1.5 + 0.08 * np.sin(np.linspace(0, 6 * np.pi, len(wavelength)))
    k = np.full_like(n, 0.1)
    medium, report = bz.fit_nk_vector(
        wavelength,
        n,
        k,
        max_poles=1,
        num_iters=20,
        weights=(0.1, 1.9),
        tolerance_rms=1e-10,
    )
    delta = medium.eps_model(bz.LIGHT_SPEED / wavelength) - (n + 1j * k) ** 2
    expected = np.sqrt(np.mean((0.1 * delta.real) ** 2 + (1.9 * delta.imag) ** 2))
    assert report["weighted_rms_epsilon"] == pytest.approx(expected)
    assert not report["tolerance_met"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights": (np.nan, 1)},
        {"weights": (0, 1)},
        {"num_iters": 0},
        {"max_poles": 1.5},
        {"epsilon_inf": 0.9},
        {"tolerance_rms": np.inf},
    ],
)
def test_invalid_options_fail(kwargs):
    with pytest.raises(ValueError):
        bz.fit_nk_vector(
            np.linspace(400e-9, 700e-9, 5), np.ones(5), np.zeros(5), **kwargs
        )


def test_repeated_wavelengths_fail():
    with pytest.raises(ValueError, match="unique"):
        bz.fit_nk_vector(np.full(5, 500e-9), np.ones(5), np.zeros(5))
