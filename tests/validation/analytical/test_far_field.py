"""Near-to-far fields against exact Maxwell dipoles, flux and surface movement."""

import jax
import numpy as np
import pytest

from beamz.analysis import near_to_far
from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from tests.far_field_case import box_quadrature, dipole_fields, sphere_quadrature


@pytest.mark.parametrize("dipole", [(0, 0, 1), (1, 0, 0), (1, 1j, 0)])
@pytest.mark.parametrize("index", [1.0, 1.5])
def test_dipole_complex_fields_polarization_and_absolute_power(dipole, index):
    points, normals, weights = box_quadrature()
    directions, angular_weights = sphere_quadrature()
    e, h = dipole_fields(points, dipole, index=index)
    far = jax.jit(near_to_far)(
        e[None],
        h[None],
        points,
        normals,
        weights,
        np.array([LIGHT_SPEED]),
        directions,
        refractive_index=index,
    )
    p = np.asarray(dipole)
    expected = p - directions * np.sum(directions * p, axis=-1, keepdims=True)
    np.testing.assert_allclose(far.electric[0], expected, rtol=2e-5, atol=3e-6)
    impedance = np.sqrt(MU_0 / EPS_0) / index
    np.testing.assert_allclose(
        far.magnetic[0],
        np.cross(directions, expected) / impedance,
        rtol=3e-5,
        atol=1e-8,
    )
    np.testing.assert_allclose(np.sum(far.electric * directions, axis=-1), 0, atol=3e-7)
    e_theta, e_phi = far.polarization
    np.testing.assert_allclose(
        abs(e_theta) ** 2 + abs(e_phi) ** 2,
        np.sum(abs(expected) ** 2, axis=-1)[None],
        atol=6e-6,
    )
    expected_power = (8 * np.pi / 3) * np.sum(abs(p) ** 2) / (2 * impedance)
    np.testing.assert_allclose(far.power(angular_weights), [expected_power], rtol=3e-6)
    flux = 0.5 * np.sum(np.real(np.cross(e, np.conj(h))) * normals * weights[:, None])
    np.testing.assert_allclose(far.power(angular_weights), flux, rtol=3e-6)


def test_enclosing_surface_translation_and_size_invariance():
    directions, _ = sphere_quadrature(5, 10)
    actual = []
    for radius, center in [(0.3, (0, 0, 0)), (0.55, (0.05, -0.08, 0.04))]:
        points, normals, weights = box_quadrature(radius, center, count=22)
        e, h = dipole_fields(points)
        actual.append(
            near_to_far(
                e[None],
                h[None],
                points,
                normals,
                weights,
                np.array([LIGHT_SPEED]),
                directions,
            ).electric
        )
    np.testing.assert_allclose(actual[0], actual[1], rtol=2e-5, atol=2e-6)


def test_translating_emitter_changes_angular_phase_not_power():
    points, normals, weights = box_quadrature(count=20)
    directions, angular_weights = sphere_quadrature(5, 10)
    shift = np.array([0.1, -0.03, 0.02])
    e, h = dipole_fields(points - shift)
    far = near_to_far(
        e[None], h[None], points, normals, weights, np.array([LIGHT_SPEED]), directions
    )
    expected = np.array([0, 0, 1]) - directions * directions[:, 2, None]
    expected = expected * np.exp(-2j * np.pi * (directions @ shift))[:, None]
    np.testing.assert_allclose(far.electric[0], expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        far.overlap_power(expected, angular_weights),
        far.power(angular_weights),
        rtol=3e-6,
    )
