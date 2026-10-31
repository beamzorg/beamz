"""Regression contracts for complex interpolation and projection diagnostics."""

import numpy as np
import pytest

from beamz.analysis.modal_projection.colocation import _interpolate_plane_matrix_2d
from beamz.analysis.modal_projection.diagnostics import (
    _modal_projection_reconstruction_diagnostics_from_matrix,
    _modal_projection_reconstruction_residual,
    _relative_residual,
)
from beamz.analysis.sparameters import _resample_complex_matrix


@pytest.mark.parametrize("src0", [[0.0, 0.3, 2.0], [0.3]])
def test_plane_interpolation_preserves_complex_bilinear_field_and_clamps(src0):
    src0 = np.asarray(src0)
    src1 = np.array([-1.0, 0.2, 3.0])
    dst0 = np.array([-2.0, 0.1, 0.7, 4.0])
    dst1 = np.array([-3.0, -0.4, 1.3, 5.0])

    def field(x, y):
        return (1 + 2j) + (3 - 4j) * x + (-2 + 5j) * y + (2 + 1j) * x * y

    values = field(src0[:, None], src1[None, :])
    actual = _interpolate_plane_matrix_2d(values, src0, src1, dst0, dst1)
    expected = field(
        np.clip(dst0, src0[0], src0[-1])[:, None],
        np.clip(dst1, src1[0], src1[-1])[None, :],
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert actual.dtype == np.complex128


@pytest.mark.parametrize(
    "destinations",
    [[0.0, 1.0, 1.7, 4.0, 5.0, 6.0], [1.7, 4.0], [], [2.5]],
)
def test_frequency_resampling_flattens_space_and_zero_fills_outside_band(destinations):
    frequencies = np.array([1.0, 2.5, 5.0])
    destinations = np.asarray(destinations)
    slopes = np.array([[1 + 2j, -3 + 4j], [2 - 1j, -1 - 3j]])
    intercepts = np.array([[2 - 3j, 5 + 1j], [-4 + 2j, 3 - 2j]])
    values = frequencies[:, None, None] * slopes + intercepts
    expected = (destinations[:, None, None] * slopes + intercepts).reshape(
        len(destinations), 4
    )
    expected[(destinations < frequencies[0]) | (destinations > frequencies[-1])] = 0
    actual = _resample_complex_matrix(frequencies, values, destinations)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert actual.dtype == np.complex128


def test_component_diagnostics_recover_independent_complex_scales():
    matrix = np.array([[1, 0], [2, 0], [0, 1j], [0, 2j]])
    coeff = np.ones(2)
    e_scale, h_scale = 2 + 1j, -1 + 0.5j
    field = np.array([e_scale, 2 * e_scale, 1j * h_scale, 2j * h_scale])
    result = _modal_projection_reconstruction_diagnostics_from_matrix(
        field, matrix, coeff, [("Ex", 0, 2), ("Hy", 2, 4)]
    )
    expected_residual = np.sqrt(
        (abs(e_scale - 1) ** 2 + abs(h_scale - 1) ** 2)
        / (abs(e_scale) ** 2 + abs(h_scale) ** 2)
    )
    assert result["residual"] == pytest.approx(expected_residual)
    assert result["residual_e"] == pytest.approx(abs(e_scale - 1) / abs(e_scale))
    assert result["residual_h"] == pytest.approx(abs(h_scale - 1) / abs(h_scale))
    assert result["e_scale"] == pytest.approx(e_scale)
    assert result["h_scale"] == pytest.approx(h_scale)
    for key in ("residual_balanced", "residual_e_scaled", "residual_h_scaled"):
        assert result[key] == pytest.approx(0, abs=1e-14)
    legacy = _modal_projection_reconstruction_residual(
        field, {"mode_matrix": matrix}, coeff
    )
    assert legacy == pytest.approx(expected_residual)


@pytest.mark.parametrize("target", [[], [0.0], [1e-31], [np.nan], [np.inf]])
def test_relative_residual_is_undefined_for_invalid_target_norm(target):
    target = np.asarray(target, dtype=np.complex128)
    assert np.isnan(_relative_residual(target, np.ones_like(target)))


@pytest.mark.parametrize("magnitude", [0.0, 1e-31])
def test_projection_diagnostics_handle_tiny_targets_and_missing_components(magnitude):
    field = np.full(2, magnitude, dtype=np.complex128)
    matrix = np.eye(2, dtype=np.complex128)
    coeff = np.ones(2)
    assert np.isnan(
        _modal_projection_reconstruction_residual(field, {"mode_matrix": matrix}, coeff)
    )
    result = _modal_projection_reconstruction_diagnostics_from_matrix(
        field, matrix, coeff
    )
    for key, value in result.items():
        assert np.isnan(value), key


@pytest.mark.parametrize(
    "axis,components",
    [
        ("x", ("Ey", "Ez", "Hy", "Hz")),
        ("y", ("Ex", "Ez", "Hx", "Hz")),
        ("z", ("Ex", "Ey", "Hx", "Hy")),
    ],
)
@pytest.mark.parametrize("ill_conditioned", [False, True])
def test_coupled_plan_matches_lorentz_overlaps_and_reconstruction(
    axis, components, ill_conditioned
):
    from beamz.analysis.mode_projection import (
        _project_modal_coefficients_3d_group,
        build_modal_projection_plan,
    )
    from beamz.devices.modes.fields import _modal_overlap

    rng = np.random.default_rng(841)
    basis = [
        {name: rng.normal(size=7) + 1j * rng.normal(size=7) for name in components}
        for _ in range(4)
    ]
    if ill_conditioned:
        basis[-1] = {
            name: basis[0][name] + 1e-10 * basis[-1][name] for name in components
        }
    weights = {name: np.linspace(0.1, 0.8, 7) for name in components}
    projections = [
        dict(
            components=components,
            axis=axis,
            integration_weights=weights,
            direction_sign=-1.0,
            d_area=1.0,
            mode_components=basis[index],
            mode_components_bwd=basis[index + 1],
        )
        for index in (0, 2)
    ]
    field = {name: rng.normal(size=7) + 1j * rng.normal(size=7) for name in components}
    overlap = np.array(
        [[_modal_overlap(a, b, axis, weights, -1.0) for b in basis] for a in basis]
    )
    rhs = np.array([_modal_overlap(field, mode, axis, weights, -1.0) for mode in basis])
    system = overlap.T
    expected = (
        np.linalg.solve(system, rhs)
        if np.linalg.cond(system) < 1e8
        else np.linalg.pinv(system) @ rhs
    )
    plan = build_modal_projection_plan(projections)
    actual, diagnostics = plan.project(field)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    vector = np.concatenate([field[name] for name in components])
    np.testing.assert_allclose(
        build_modal_projection_plan(projections).coefficient_rows @ vector,
        actual,
        rtol=2e-6,
        atol=2e-6,
    )
    grouped, residual, condition, legacy_diag = _project_modal_coefficients_3d_group(
        field, projections
    )
    np.testing.assert_allclose(np.ravel(grouped), actual)
    assert condition == plan.condition_number
    assert residual == diagnostics["residual"]
    expected_power = np.real(expected @ overlap @ expected.conj())
    assert legacy_diag["projected_signed_power"] == pytest.approx(
        expected_power, rel=2e-6
    )


def test_group_colocation_reuses_geometry_and_preserves_batch_shape(monkeypatch):
    from beamz.analysis.modal_projection import colocation

    src0, src1 = np.array([0.0, 1.0]), np.array([0.0, 0.5, 2.0])
    dst0, dst1 = np.array([0.3]), np.array([0.2, 0.8])
    calls = []

    def coordinates(*args):
        calls.append(args)
        return src0, src1

    monkeypatch.setattr(colocation, "_monitor_analysis_plane_3d", coordinates)
    plane = (1 + 2j) * src0[:, None] + (3 - 4j) * src1[None, :]
    components = {
        "Ey": plane.ravel(),
        "Hz": np.array([plane.ravel(), 2 * plane.ravel()]),
    }
    actual = colocation._colocate_field_components_to_projection_3d(
        None,
        None,
        components,
        {"axis": "x", "analysis_coords0": dst0, "analysis_coords1": dst1},
    )
    expected = ((1 + 2j) * dst0[:, None] + (3 - 4j) * dst1[None, :]).ravel()
    np.testing.assert_allclose(actual["Ey"], expected)
    np.testing.assert_allclose(actual["Hz"], [expected, 2 * expected])
    assert len(calls) == 1
