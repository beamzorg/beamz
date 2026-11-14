"""Independent volume identities and Fourier checks on genuinely graded grids."""

import jax.numpy as jnp
import numpy as np
import pytest

from beamz.const import EPS_0
from beamz.constitutive import electric_tensor_coefficients
from beamz.design.grid import RectilinearGrid
from beamz.devices.modes._tensor_yee import inverse_operator
from beamz.lattice import component_axis_offsets_3d, component_shape_3d
from beamz.simulation.compile.kernels import (
    advance_e_centered_tensor,
    collocate_yee_component,
)
from tests.jax_compat import enable_x64
from tests.validation.analytical.test_tensor_constitutive_operator import (
    packed,
    supports,
)


def dual_widths(widths):
    return np.r_[widths[0] / 2, (widths[:-1] + widths[1:]) / 2, widths[-1] / 2]


def volume_weights(shape, component, widths):
    offsets = (
        component_axis_offsets_3d(component)
        if component != "Node"
        else dict.fromkeys("xyz", 0)
    )
    measures = [
        widths["xyz".index(a)]
        if offsets[a] == 0.5
        else dual_widths(widths["xyz".index(a)])
        for a in "zyx"
    ]
    result = (
        measures[0][:, None, None]
        * measures[1][None, :, None]
        * measures[2][None, None, :]
    )
    assert result.shape == shape
    return result


@pytest.mark.parametrize("component", ["Ex", "Ey", "Ez"])
def test_rectilinear_constitutive_averages_are_volume_adjoint(component):
    rng = np.random.default_rng(361)
    shape = (5, 6, 7)
    widths = tuple(rng.uniform(0.2, 2.0, n) for n in shape[::-1])
    electric_shape = component_shape_3d(component, shape)
    node_shape = tuple(n + 1 for n in shape)
    electric = rng.normal(size=electric_shape)
    node = rng.normal(size=node_shape)
    with enable_x64(True):
        to_node = np.asarray(
            collocate_yee_component(
                electric, component, "Node", node_shape, cell_widths=widths
            )
        )
        to_e = np.asarray(
            collocate_yee_component(
                node, "Node", component, electric_shape, cell_widths=widths
            )
        )
    # Independent control-volume inner products, including half-cells at outer nodes.
    left = np.sum(volume_weights(node_shape, "Node", widths) * to_node * node)
    right = np.sum(volume_weights(electric_shape, component, widths) * electric * to_e)
    np.testing.assert_allclose(left, right, rtol=2e-14, atol=2e-14)
    with enable_x64(True):
        constant = collocate_yee_component(
            np.ones(electric_shape), component, "Node", node_shape, cell_widths=widths
        )
    np.testing.assert_allclose(constant, 1, rtol=0, atol=2e-16)


@pytest.mark.parametrize("beta_dx", [0.1, 0.7, 1.8])
def test_rectilinear_fourier_tensor_matrix_matches_full_volume_kernel(beta_dx):
    rng = np.random.default_rng(361)
    shape = (7, 8, 10)
    widths = (
        np.ones(shape[2]),
        rng.uniform(0.5, 1.5, shape[1]),
        rng.uniform(0.5, 1.5, shape[0]),
    )
    grid = RectilinearGrid(*(np.r_[0, np.cumsum(w)] for w in widths))
    tensors = supports(shape, [[3.0, 0.3, 0.2], [0.3, 2.0, 0.4], [0.2, 0.4, 1.5]])
    for name, matrix in tensors.items():
        matrix *= rng.uniform(0.8, 1.2, size=(*matrix.shape[:2], 1, 1, 1))
        tensors[name] = packed(matrix)
    profiles = {
        k: rng.normal(size=component_shape_3d(k, shape)[:2])
        + 1j * rng.normal(size=component_shape_3d(k, shape)[:2])
        for k in ("Ex", "Ey", "Ez")
    }
    indices = {k: (slice(None), slice(None), 4) for k in profiles}
    vector = np.concatenate([v.reshape(-1) for v in profiles.values()])
    expected = (
        inverse_operator(
            profiles, indices, tensors, beta=beta_dx, spacing=1.0, grid=grid
        )
        @ vector
    )
    diagonal, cross = electric_tensor_coefficients(tensors)
    with enable_x64(True):
        curls = tuple(
            jnp.asarray(
                profiles[k][..., None]
                * np.exp(
                    -1j
                    * beta_dx
                    * (
                        np.arange(component_shape_3d(k, shape)[2])
                        + (0.5 if k == "Ex" else 0)
                    )
                )
            )
            for k in profiles
        )
        updated = advance_e_centered_tensor(
            tuple(jnp.zeros_like(v) for v in curls),
            curls,
            tuple(jnp.asarray(v) for v in diagonal),
            jnp.asarray(cross),
            tuple(profiles),
            EPS_0,
            cell_widths=widths,
        )
    actual = np.concatenate(
        [
            (
                np.asarray(v)[..., 4]
                / np.exp(-1j * beta_dx * (4 + (0.5 if k == "Ex" else 0)))
            ).reshape(-1)
            for k, v in zip(profiles, updated, strict=True)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_rectilinear_full_volume_tensor_update_is_reciprocal():
    """Reciprocity of the production kernel with grading in all three axes."""
    rng = np.random.default_rng(362)
    shape = (5, 6, 7)
    widths = tuple(rng.uniform(0.2, 2.0, n) for n in shape[::-1])
    tensors = supports(shape, [[3.0, 0.3, 0.2], [0.3, 2.0, 0.4], [0.2, 0.4, 1.5]])
    tensors = {
        name: packed(values * rng.uniform(0.8, 1.2, (*values.shape[:3], 1, 1)))
        for name, values in tensors.items()
    }
    diagonal, cross = electric_tensor_coefficients(tensors)
    names = ("Ex", "Ey", "Ez")
    u, v = (
        tuple(rng.normal(size=component_shape_3d(k, shape)) for k in names)
        for _ in range(2)
    )
    with enable_x64(True):

        def apply(values):
            return advance_e_centered_tensor(
                tuple(jnp.zeros_like(a) for a in values),
                tuple(jnp.asarray(a) for a in values),
                tuple(jnp.asarray(a) for a in diagonal),
                jnp.asarray(cross),
                names,
                EPS_0,
                cell_widths=widths,
            )

        bu, bv = tuple(map(np.asarray, apply(u))), tuple(map(np.asarray, apply(v)))
    weights = [
        volume_weights(a.shape, k, widths) for k, a in zip(names, u, strict=True)
    ]
    left = sum(np.sum(w * a * b) for w, a, b in zip(weights, u, bv, strict=True))
    right = sum(np.sum(w * a * b) for w, a, b in zip(weights, bu, v, strict=True))
    np.testing.assert_allclose(left, right, rtol=2e-14, atol=2e-14)
