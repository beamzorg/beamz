"""Independent discrete-wave and power-balance controls for detector diagnostics."""

import numpy as np
import pytest

import beamz as bz


@pytest.mark.parametrize("index", [1.0, 2.0, 3.5 + 0.1j])
def test_outgoing_yee_reference_matches_exact_uniform_wave(index):
    from tests.validation.analytical.discrete_power_reference import solve_yee

    z = np.linspace(-500e-9, 500e-9, 101)
    frequency, dt = bz.LIGHT_SPEED / 450e-9, 1e-17
    omega = 2 * np.sin(np.pi * frequency * dt) / dt
    spacing = z[1] - z[0]
    k = 2 * np.arcsin(omega * spacing * index / (2 * bz.LIGHT_SPEED)) / spacing
    e, h, _, _, residual = solve_yee(
        z, np.full(z.shape, index**2, complex), frequency, dt
    )
    expected_e = np.exp(-1j * k * (z - z[-1]))
    expected_h = (
        -index
        / np.sqrt(bz.MU_0 / bz.EPS_0)
        * np.exp(-1j * k * ((z[:-1] + z[1:]) / 2 - z[-1]))
    )
    np.testing.assert_allclose(e, expected_e, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(h, expected_h, rtol=1e-11, atol=1e-13)
    assert np.max(abs(residual)) < 1e-14


def test_graded_absorbing_interface_obeys_discrete_power_balance():
    from tests.validation.analytical.discrete_power_reference import solve_yee

    z = np.r_[np.linspace(-1e-6, 0, 90), np.linspace(0, 1e-6, 41)[1:]]
    eps = np.where(z < 0, (3.5 + 0.2j) ** 2, 1).astype(complex)
    e, h, flux, loss, residual = solve_yee(z, eps, bz.LIGHT_SPEED / 450e-9, 1e-17)
    assert np.isfinite(e).all() and np.isfinite(h).all()
    assert np.min(loss) >= 0
    assert np.max(abs(residual)) / np.max(abs(flux)) < 1e-11
    np.testing.assert_allclose(flux[0] - flux[-1], loss.sum(), rtol=1e-11)


def test_three_dimensional_closed_box_accounts_for_lateral_power():
    from beamz.lattice import component_axis_offsets_3d
    from tests.validation.analytical.discrete_power_reference import (
        closed_box_integrals,
    )

    # A superposition of two absorbing oblique discrete modes has lateral flow.
    d = 10e-9
    x = y = z = np.arange(41) * d
    indices = np.arange(40)
    f, dt = bz.LIGHT_SPEED / 450e-9, 1e-17
    omega = 2 * np.sin(np.pi * f * dt) / dt
    eps = (3.5 + 0.1j) ** 2
    fields = {}
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        offsets = component_axis_offsets_3d(name)
        zz, yy, xx = np.meshgrid(
            (indices + offsets["z"]) * d,
            (indices + offsets["y"]) * d,
            (indices + offsets["x"]) * d,
            indexing="ij",
        )
        values = np.zeros(xx.shape, complex)
        for mode, amplitude in ((0, 1), (1, 0.7j)):
            kx = mode * 2 * np.pi / (40 * d)
            ax = 2 * np.sin(kx * d / 2) / d
            az = np.sqrt(eps * (omega / bz.LIGHT_SPEED) ** 2 - ax**2)
            kz = 2 * np.arcsin(az * d / 2) / d
            factor = {"Ex": 1, "Ez": ax / az, "Hy": -omega * bz.EPS_0 * eps / az}.get(
                name, 0
            )
            values += amplitude * factor * np.exp(1j * kx * xx - 1j * kz * zz)
        fields[name] = values[:, None, :, :]
    top, bottom, lateral, intensity = closed_box_integrals(
        fields, x, y, z, indices, ((6, 24), (7, 25), (8, 28))
    )
    loss = 0.5 * omega * bz.EPS_0 * eps.imag * intensity
    assert abs(lateral[0]) > 0.01 * loss[0]
    np.testing.assert_allclose(top - bottom + lateral, -loss, rtol=1e-11, atol=1e-30)


def test_flux_phase_and_nonuniform_aperture_weights_match_phasor_power():
    from types import SimpleNamespace

    from beamz.simulation.observe import monitor_dft_flux

    f, dt = 6e14, 1.5e-17
    electric = np.array([[1 + 2j, 3 - 1j]])
    index = np.array([[2 + 0.2j, 3 + 0.1j]])
    impedance = np.sqrt(bz.MU_0 / bz.EPS_0)
    magnetic = -index / impedance * electric * np.exp(1j * np.pi * f * dt)
    area = np.array([1e-12, 4e-12])
    monitor = bz.FluxMonitor(center=(0, 0, 0), size=(1e-6, 5e-6, 0), freqs=[f])
    saved = SimpleNamespace(
        monitor=monitor,
        dft_fields={"Ex": electric, "Hy": magnetic},
        dft_frequencies=np.array([f]),
        dft_weight_sum=np.array([2.0]),
        dft_base_dt=dt,
        normal_axis=2,
        normal_sign=1.0,
        resolution=1e-6,
        power_scale=0.0,
        integration_weights=area,
    )
    expected = -0.5 * np.sum(index.real * abs(electric) ** 2 * area) / impedance
    np.testing.assert_allclose(monitor_dft_flux(saved), [expected], rtol=1e-14)
