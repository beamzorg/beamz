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
        lambda *_: (eps, np.arange(eps.size), dx),
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
        power = (
            0.5
            * dx
            * np.real(np.sum(parts["signed_flux_sign"] * electric * magnetic.conj()))
        )
        assert power == pytest.approx(expected, abs=1e-12)
