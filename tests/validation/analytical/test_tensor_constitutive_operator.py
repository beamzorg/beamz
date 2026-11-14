"""Independent matrix algebra and Fourier checks of the coupled Yee operator."""

import jax.numpy as jnp
import numpy as np
import pytest

from beamz.const import EPS_0
from beamz.constitutive import (
    electric_tensor_coefficients,
    inverse_permittivity_components,
)
from beamz.devices.modes._tensor_yee import inverse_operator
from beamz.lattice import component_shape_3d
from beamz.simulation.compile.kernels import advance_e_centered_tensor
from tests.jax_compat import enable_x64


def packed(matrix):
    return matrix[..., (0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)].transpose(3, 0, 1, 2)


def supports(shape, matrix):
    return {
        name: np.broadcast_to(np.asarray(matrix), (*size, 3, 3)).copy()
        for name, size in [
            ("eps_" + a, component_shape_3d("E" + a, shape)) for a in "xyz"
        ]
        + [("eps_node", tuple(n + 1 for n in shape))]
    }


def test_inverse_coefficients_match_independent_dense_inverse():
    rng = np.random.default_rng(361)
    a = rng.normal(size=(5, 4, 3, 3, 3))
    matrix = a @ np.swapaxes(a, -1, -2) + np.eye(3)
    actual = inverse_permittivity_components(packed(matrix))
    expected = np.linalg.inv(matrix)[..., (0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("beta_dx", [0.1, 0.7, 1.8])
def test_fourier_constitutive_matrix_matches_production_timestep(beta_dx):
    rng = np.random.default_rng(361)
    shape = (7, 8, 10)
    tensors = supports(shape, [[3.0, 0.3, 0.2], [0.3, 2.0, 0.4], [0.2, 0.4, 1.5]])
    for name, matrix in tensors.items():
        # Vary in both transverse directions, constant in the propagation axis.
        matrix *= rng.uniform(0.8, 1.2, size=(*matrix.shape[:2], 1, 1, 1))
        tensors[name] = packed(matrix)
    names = ("Ex", "Ey", "Ez")
    profiles = {
        k: rng.normal(size=component_shape_3d(k, shape)[:2])
        + 1j * rng.normal(size=component_shape_3d(k, shape)[:2])
        for k in names
    }
    indices = {k: (slice(None), slice(None), 4) for k in names}
    vector = np.concatenate([profiles[k].reshape(-1) for k in names])
    expected = (
        inverse_operator(profiles, indices, tensors, beta=beta_dx, spacing=1.0) @ vector
    )
    phases = {
        k: np.exp(
            -1j
            * beta_dx
            * (np.arange(component_shape_3d(k, shape)[2]) + (0.5 if k == "Ex" else 0))
        )
        for k in names
    }
    curls = tuple(jnp.asarray(profiles[k][..., None] * phases[k]) for k in names)
    diagonal, cross = electric_tensor_coefficients(tensors)
    with enable_x64(True):
        curls = tuple(jnp.asarray(profiles[k][..., None] * phases[k]) for k in names)
        actual = advance_e_centered_tensor(
            tuple(jnp.zeros_like(v) for v in curls),
            curls,
            tuple(jnp.asarray(v) for v in diagonal),
            jnp.asarray(cross),
            names,
            EPS_0,
        )
        measured = np.concatenate(
            [
                (np.asarray(v)[..., 4] / phases[k][4]).reshape(-1)
                for k, v in zip(names, actual, strict=True)
            ]
        )
    np.testing.assert_allclose(measured, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_homogeneous_tensor_fresnel_eigenvalues(axis):
    # Independent continuum Fresnel reduction: eps_tt - eps_tn eps_nt / eps_nn.
    epsilon = np.array([[3.0, 0.3, 0.2], [0.3, 2.0, 0.4], [0.2, 0.4, 1.5]])
    transverse = [i for i in range(3) if i != axis]
    effective = (
        epsilon[np.ix_(transverse, transverse)]
        - np.outer(epsilon[transverse, axis], epsilon[axis, transverse])
        / epsilon[axis, axis]
    )
    n_squared = np.linalg.eigvalsh(effective)
    inverse = np.linalg.inv(epsilon)
    inverse_transverse = inverse[np.ix_(transverse, transverse)]
    np.testing.assert_allclose(
        np.sort(1 / np.linalg.eigvalsh(inverse_transverse)), n_squared, rtol=2e-14
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("band", [0, 1])
def test_analytic_fresnel_plane_wave_is_a_discrete_timestep_eigenfield(axis, band):
    from types import SimpleNamespace

    from beamz.const import LIGHT_SPEED, MU_0
    from beamz.devices.sources.planar_tfsf import (
        advance_incident_e_3d,
        advance_incident_h_3d,
    )
    from beamz.lattice import component_axis_offsets_3d

    epsilon = np.array([[3.0, 0.3, 0.2], [0.3, 2.0, 0.4], [0.2, 0.4, 1.5]])
    transverse = [i for i in range(3) if i != axis]
    effective = (
        epsilon[np.ix_(transverse, transverse)]
        - np.outer(epsilon[transverse, axis], epsilon[axis, transverse])
        / epsilon[axis, axis]
    )
    values, vectors = np.linalg.eigh(effective)
    n = np.sqrt(values[band])
    spacing = 0.08e-6
    dt = 0.4 * spacing / LIGHT_SPEED
    omega = 2 * np.pi * LIGHT_SPEED / 1.55e-6
    omega_discrete = 2 * np.sin(0.5 * omega * dt) / dt
    beta = 2 * np.arcsin(0.5 * spacing * omega_discrete * n / LIGHT_SPEED) / spacing
    electric = np.zeros(3)
    electric[transverse] = vectors[:, band]
    electric[axis] = (
        -np.cos(0.5 * beta * spacing)
        * np.dot(epsilon[axis, transverse], electric[transverse])
        / epsilon[axis, axis]
    )
    normal = np.eye(3)[axis]
    magnetic = n / (MU_0 * LIGHT_SPEED) * np.cross(normal, electric)
    shape = (8, 9, 10)
    tensors = {k: packed(v) for k, v in supports(shape, epsilon).items()}
    fields = SimpleNamespace(
        material_grid=SimpleNamespace(yee_tensors=tensors), boundaries=None
    )
    for a in "xyz":
        setattr(fields, "sig_" + a, 0.0)
        setattr(fields, "sigma_m_h" + a, 0.0)
    state = {}
    for prefix, vector in [("E", electric), ("H", magnetic)]:
        for i, a in enumerate("xyz"):
            component = prefix + a
            size = component_shape_3d(component, shape)
            offsets = component_axis_offsets_3d(component)
            storage_axis = 2 - axis
            coordinate = (
                np.arange(size[storage_axis]) + offsets["xyz"[axis]]
            ) * spacing
            reshape = [1, 1, 1]
            reshape[storage_axis] = len(coordinate)
            phase = np.exp(-1j * beta * coordinate).reshape(reshape)
            time_phase = np.exp(-0.5j * omega * dt) if prefix == "H" else 1.0
            state[component] = np.broadcast_to(
                vector[i] * phase * time_phase, size
            ).copy()
    with enable_x64(True):
        h = advance_incident_h_3d(fields, state, dt, resolution=spacing)
        e = advance_incident_e_3d(fields, state, h, dt, resolution=spacing)
    selection = (slice(2, -2),) * 3
    for names, update in [(("Ex", "Ey", "Ez"), e), (("Hx", "Hy", "Hz"), h)]:
        actual = np.concatenate([update[k][selection].ravel() for k in names])
        expected = np.concatenate(
            [state[k][selection].ravel() for k in names]
        ) * np.exp(1j * omega * dt)
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 2e-12
