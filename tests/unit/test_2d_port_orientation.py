"""A positive port basis must carry positive physical Poynting flux."""

from types import SimpleNamespace

import numpy as np
import pytest

from beamz import LIGHT_SPEED, ModeSpec, Port
from beamz.analysis.modal_projection.geometry import _mode_components_for_port
from beamz.analysis.mode_projection import _build_port_projection_2d


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("polarization", ["tm", "te"])
def test_forward_basis_has_positive_flux_and_backward_basis_negative(
    monkeypatch, axis, polarization
):
    # Use actual native slab eigenmodes, not synthetic profiles whose sign would
    # already assume the convention under test.
    dx = 50e-9
    eps = np.full(60, 1.44**2)
    eps[25:35] = 3.4**2
    monkeypatch.setattr(
        "beamz.analysis.mode_projection._monitor_profile_slice",
        lambda *_: (eps, np.arange(eps.size), dx, None),
    )
    port = Port(
        center=(1.5e-6, 1.5e-6, 0),
        size=(0, 3e-6, 1e-6) if axis == "x" else (3e-6, 0, 1e-6),
        mode_spec=ModeSpec(num_modes=1, polarization=polarization),
        name="port",
        direction="+",
    )
    parts = _mode_components_for_port(port)
    projection = _build_port_projection_2d(
        SimpleNamespace(resolution=dx, is_3d=False),
        spec=port,
        monitor=None,
        frequency=LIGHT_SPEED / 1.55e-6,
        parts=parts,
        mode_pad_cells=0,
    )
    for column, expected in [(0, 1.0), (1, -1.0)]:
        electric, magnetic = np.split(projection["mode_matrix"][:, column], 2)
        # Basis samples occupy different normal Yee planes. Evaluate physical
        # power after translating E to the H reference, with the wave's sign.
        electric = electric * np.exp(
            expected * 1j * 2 * np.pi / 1.55e-6 * projection["mode_neff"] * dx / 2
        )
        power = (
            0.5
            * dx
            * np.real(np.sum(parts["signed_flux_sign"] * electric * magnetic.conj()))
        )
        assert power == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("polarization", ["tm", "te"])
@pytest.mark.parametrize("direction", ["+", "-"])
@pytest.mark.parametrize("weighted", [False, True])
def test_staggered_plane_wave_superposition_recovers_both_amplitudes(
    monkeypatch, axis, polarization, direction, weighted
):
    """An analytic mixture must not acquire a reflected wave during extraction."""
    dx, wavelength, neff, count = 80e-9, 1.55e-6, 1.7, 5
    weights = dx * np.linspace(0.8, 1.2, count) if weighted else dx
    monkeypatch.setattr(
        "beamz.analysis.mode_projection._monitor_profile_slice",
        lambda *_: (np.full(count, neff**2), np.arange(count), weights, None),
    )
    port = Port(
        center=(1e-6, 1e-6, 0),
        size=(0, 1e-6, 1e-6) if axis == "x" else (1e-6, 0, 1e-6),
        mode_spec=ModeSpec(polarization=polarization),
        name="port",
        direction=direction,
    )
    parts = _mode_components_for_port(port)
    sign = parts["signed_flux_sign"]

    def solve(**_):
        e = np.zeros((1, 3, count), dtype=complex)
        h = np.zeros_like(e)
        e[0, parts["e_mode_index"]] = 1
        h[0, parts["h_mode_index"]] = sign
        return np.array([neff]), e, h, None

    monkeypatch.setattr("beamz.analysis.mode_projection.solve_modes", solve)
    projection = _build_port_projection_2d(
        SimpleNamespace(resolution=dx, is_3d=False),
        spec=port,
        monitor=None,
        frequency=LIGHT_SPEED / wavelength,
        parts=parts,
        mode_pad_cells=0,
    )
    # Independent traveling-wave data: H at x=0 and E at x=-dx/2,
    # with exp(+/- i*k*x) under the exp(-i*omega*t) convention.
    amplitude_scale = np.sqrt(2 / np.sum(np.broadcast_to(weights, (count,))))
    k = 2 * np.pi * neff / wavelength
    for plus, minus in [(1 + 0.2j, 0), (0, 0.7 - 0.4j), (1 + 0.2j, 0.7 - 0.4j)]:
        electric = amplitude_scale * (
            plus * np.exp(-1j * k * dx / 2) + minus * np.exp(1j * k * dx / 2)
        )
        magnetic = amplitude_scale * sign * (plus - minus)
        measured = np.r_[np.full(count, electric), np.full(count, magnetic)]
        recovered = projection["pinv"] @ (
            projection.get("projection_weights", 1) * measured
        )
        np.testing.assert_allclose(recovered, [plus, minus], rtol=1e-12, atol=1e-12)
    assert projection["modal_plane_delay_s"] == 0


@pytest.mark.parametrize("weighted", [False, True])
def test_modal_extraction_rejects_power_orthogonal_fields(monkeypatch, weighted):
    """Unrelated transverse fields must not pollute the guided-wave amplitudes."""
    dx = 80e-9
    measure = dx * np.array([1.0, 1.5, 2.0]) if weighted else dx
    monkeypatch.setattr(
        "beamz.analysis.mode_projection._monitor_profile_slice",
        lambda *_: (np.full(3, 2.0), np.arange(3), measure, None),
    )
    port = Port(
        center=(1e-6, 1e-6, 0),
        size=(0, 1e-6, 1e-6),
        mode_spec=ModeSpec(polarization="te"),
        name="port",
        direction="+",
    )

    def solve(**_):
        e, h = np.zeros((1, 3, 3), complex), np.zeros((1, 3, 3), complex)
        e[0, 1], h[0, 2] = [1, 2, 4], [1, 1, 1]
        return np.array([1.7]), e, h, None

    monkeypatch.setattr("beamz.analysis.mode_projection.solve_modes", solve)
    projection = _build_port_projection_2d(
        SimpleNamespace(resolution=dx, is_3d=False),
        spec=port,
        monitor=None,
        frequency=LIGHT_SPEED / 1.55e-6,
        parts=_mode_components_for_port(port),
        mode_pad_cells=0,
    )
    weights = np.broadcast_to(measure, (3,))
    # Zero Lorentz overlaps: sum(E_other * H_mode*) and
    # sum(E_mode* * H_other), including the physical integration measure.
    other_e = np.array([1, -2, 1]) / weights
    other_h = np.array([2, -1, 0]) / weights
    assert np.dot(weights, other_e) == pytest.approx(0)
    assert np.dot(weights * np.array([1, 2, 4]), other_h) == pytest.approx(0)
    expected = np.array([1 + 0.2j, 0.7 - 0.4j])
    field = projection["mode_matrix"] @ expected + np.r_[other_e, other_h] * 0.01j
    actual = projection["pinv"] @ (projection.get("projection_weights", 1) * field)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
