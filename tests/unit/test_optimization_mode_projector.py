"""Frozen projections preserve interpolation and the complete monitor mode spec."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.analysis.mode_projection import _project_modal_coefficients_3d_group
from beamz.optimization import modes


@pytest.mark.parametrize("axis", ("x", "y", "z"))
@pytest.mark.parametrize("singular", (False, True))
def test_projector_interpolation_and_coupled_basis(monkeypatch, axis, singular):
    rng = np.random.default_rng(14)
    size = [1.0] * 3
    size["xyz".index(axis)] = 0
    monitor = bz.ModeMonitor(
        center=(0, 0, 0),
        size=tuple(size),
        name="port",
        freqs=[2.0],
        mode_spec=bz.ModeSpec(
            num_modes=2,
            polarization="te",
            target_neff=2.0,
            num_freqs=3,
            force_symmetry=True,
            min_aperture_fraction=0.7,
            solver_padding_cells=(2, 3),
        ),
    )
    src0, src1 = np.linspace(0, 1, 3), np.linspace(0, 1, 4)
    dst0, dst1 = np.linspace(0.1, 0.9, 2), np.linspace(0.1, 0.9, 3)
    components = {
        "x": ("Ey", "Ez", "Hz", "Hy"),
        "y": ("Ez", "Ex", "Hx", "Hz"),
        "z": ("Ex", "Ey", "Hy", "Hx"),
    }[axis]

    def fields():
        return {c: rng.normal(size=6) + 1j * rng.normal(size=6) for c in components}

    projections = []
    for _ in range(2):
        forward = fields()
        projections.append(
            dict(
                components=components,
                axis=axis,
                direction_sign=1.0,
                d_area=0.02,
                mode_components=forward,
                mode_components_bwd=forward if singular else fields(),
                analysis_coords0=dst0,
                analysis_coords1=dst1,
            )
        )
    data = SimpleNamespace(
        is_3d=True,
        monitor_geometry=monitor,
        frequencies=[2.0],
        field=lambda _: np.zeros((1, 12)),
    )
    monkeypatch.setattr(modes, "analysis_data", lambda *_: data)
    monkeypatch.setattr(modes, "_monitor_analysis_plane_3d", lambda *_: (src0, src1))
    seen = []

    def build(data, port, monitor, frequency, cache):
        seen.append(port.mode_spec)
        return projections[port.mode_spec.mode_index]

    monkeypatch.setattr("beamz.analysis.mode_projection._build_port_projection", build)
    project = modes.build_mode_projector(None, "port")
    for index, spec in enumerate(seen):
        assert spec == replace(monitor.mode_spec, mode_index=index)
    measured = rng.normal(size=(6, 1, 12)) + 1j * rng.normal(size=(6, 1, 12))
    sampled = {}
    for c in components:
        plane = measured[("Ex", "Ey", "Ez", "Hx", "Hy", "Hz").index(c), 0].reshape(3, 4)
        along0 = np.array([np.interp(dst0, src0, col) for col in plane.T]).T
        sampled[c] = np.array([np.interp(dst1, src1, row) for row in along0]).ravel()
    expected = np.asarray(_project_modal_coefficients_3d_group(sampled, projections)[0])
    if axis in {"x", "y"}:
        expected = expected[:, ::-1]
    np.testing.assert_allclose(project(measured)[0], expected.T, rtol=2e-5, atol=2e-6)

    def objective(scale):
        return jnp.sum(jnp.abs(project(jnp.asarray(measured) * scale)) ** 2)

    np.testing.assert_allclose(jax.grad(objective)(1.0), 2 * objective(1.0), rtol=2e-6)
