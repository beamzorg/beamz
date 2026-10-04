"""Contracts for optional circular-bend evaluation."""

import numpy as np
import pytest

from beamz.devices.modes import solve_grid
from beamz.devices.modes._bend import bend_materials
from beamz.devices.modes.models import Materials
from tests.unit.modes.test_api import _strip


def _inputs():
    eps, x, y = _strip()
    return dict(eps_xx=eps, x_edges=x, y_edges=y, wavelength=1.55, target_neff=2.5)


@pytest.mark.parametrize("tensor", ["scalar", "diagonal", "full"])
def test_none_preserves_straight_indices_fields_and_diagnostics(tensor):
    args = _inputs()
    if tensor != "scalar":
        args["eps_yy"] = args["eps_xx"] * 1.01
    if tensor == "full":
        args["eps_xy"] = args["eps_yx"] = np.full_like(args["eps_xx"], 0.01)
    straight = solve_grid(**args)
    explicit = solve_grid(**args, bend_radius=None)
    np.testing.assert_array_equal(straight.n_complex, explicit.n_complex)
    for name in straight.field_components:
        assert straight.field_components[name].identical(
            explicit.field_components[name]
        )
    assert straight.solver_info.keys() == explicit.solver_info.keys()
    assert "bend_radius" not in explicit.solver_info
    for key in straight.solver_info["runs"][0]:
        np.testing.assert_equal(
            straight.solver_info["runs"][0][key], explicit.solver_info["runs"][0][key]
        )


@pytest.mark.parametrize(
    "radius", [0, np.nan, np.inf, -np.inf, 1, -1, 0.5, -0.5, 1e-320]
)
def test_invalid_radius_or_axis_intersection(radius):
    with pytest.raises(ValueError, match="bend"):
        solve_grid(**_inputs(), bend_radius=radius)


def test_invalid_bend_axis():
    with pytest.raises(ValueError, match="bend_axis"):
        solve_grid(**_inputs(), bend_radius=10, bend_axis="z")


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("radius", [10, -10])
def test_material_pullback_includes_mu_and_preserves_inputs(axis, radius):
    args = _inputs()
    args.pop("wavelength")
    args.pop("target_neff")
    args.update(
        eps_xy=np.full((6, 5), 0.02),
        eps_xz=np.full((6, 5), 0.03),
        mu_xx=np.full((6, 5), 1.2),
        mu_zz=np.full((6, 5), 1.3),
    )
    original = Materials.from_components(**args)
    transformed, h = bend_materials(original, radius, axis)
    edges = np.asarray(args[f"{axis}_edges"])
    expected_h = 1 + (edges[:-1] + edges[1:]) / (2 * radius)
    expected_h = expected_h[:, None] if axis == "x" else expected_h[None, :]
    np.testing.assert_allclose(h, np.broadcast_to(expected_h, original.shape))
    for old, new in [
        (original.eps_tensor, transformed.eps_tensor),
        (original.mu_tensor, transformed.mu_tensor),
    ]:
        for i in range(3):
            for j in range(3):
                expected = old[i, j] * h / (h if i == 2 else 1) / (h if j == 2 else 1)
                np.testing.assert_allclose(new[i, j], expected)
    np.testing.assert_array_equal(original.eps_tensor[0, 0], args["eps_xx"])
    assert not original.eps_tensor.flags.writeable


@pytest.mark.parametrize("normal_axis", [0, 1, 2])
@pytest.mark.parametrize("direction", ["+", "-"])
def test_bent_fields_are_physical_and_keep_global_labels(normal_axis, direction):
    args = _inputs()
    args.update(normal_axis=normal_axis, direction=direction, wavelength=[1.5, 1.55])
    actual = solve_grid(**args, bend_radius=10, bend_axis="y")
    local = _inputs()
    local.pop("wavelength")
    local.pop("target_neff")
    transformed, h = bend_materials(Materials.from_components(**local), 10, "y")
    equivalent = dict(args)
    for i, axis in enumerate("xyz"):
        equivalent[f"eps_{axis}{axis}"] = transformed.eps_tensor[i, i]
        equivalent[f"mu_{axis}{axis}"] = transformed.mu_tensor[i, i]
    expected = solve_grid(**equivalent)
    np.testing.assert_allclose(actual.n_complex, expected.n_complex, atol=1e-12)
    for name in actual.field_components:
        values = expected.field_components[name].values
        if name[1] == "xyz"[normal_axis]:
            values = values / h[:, :, None, None, None]
        np.testing.assert_allclose(actual.field_components[name], values, atol=1e-12)
    assert actual.solver_info["bend_axis"] == "y"
    assert actual.solver_info["bend_radius"] == 10


def test_bend_supports_tensor_dispatch_pml_and_component_filter():
    args = _inputs()
    args["eps_xy"] = args["eps_yx"] = np.full_like(args["eps_xx"], 0.01)
    result = solve_grid(**args, bend_radius=20, pml=(1, 1), components=["Ey"])
    assert set(result.field_components) == {"Ey"}
    assert result.solver_info["runs"][0]["backend_kind"] == "tensorial_scipy_reference"


def test_explicit_isotropic_diagonal_inputs_match_scalar_bend():
    args = _inputs()
    scalar = solve_grid(**args, bend_radius=-20)
    diagonal = solve_grid(
        **args,
        eps_yy=args["eps_xx"],
        eps_zz=args["eps_xx"],
        mu_xx=np.ones_like(args["eps_xx"]),
        mu_yy=np.ones_like(args["eps_xx"]),
        mu_zz=np.ones_like(args["eps_xx"]),
        bend_radius=-20,
    )
    np.testing.assert_array_equal(scalar.n_complex, diagonal.n_complex)


def test_reference_coordinate_changes_index_but_not_angular_propagation():
    args = _inputs()
    original = solve_grid(**args, bend_radius=10)
    args["x_edges"] = np.asarray(args["x_edges"]) - 2
    args["target_neff"] *= 10 / 12
    shifted = solve_grid(**args, bend_radius=12)
    np.testing.assert_allclose(
        original.n_complex * 10, shifted.n_complex * 12, rtol=1e-12
    )
