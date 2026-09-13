"""Independent lattice and modal reduction contracts for vector topology."""

import numpy as np
import pytest

from beamz.analysis.mode_projection import (
    _modal_coefficient_rows_3d,
    _project_modal_coefficients_3d_group,
)
from beamz.devices.sources.planar_tfsf import component_slices_from_cell_bounds
from beamz.lattice import component_shape_3d
from beamz.simulation.differentiable import _cell_centers_to_yee


@pytest.mark.parametrize("component", ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
def test_compact_source_preserves_complete_yee_support(component):
    shape = (8, 10, 12)
    full = component_shape_3d(component, shape)
    bounds = ((0, 8), (0, 10), (0, 12))
    slices = component_slices_from_cell_bounds(component, bounds, full)
    assert tuple(s.stop - s.start for s in slices) == full
    bounds = ((1, 7), (2, 8), (3, 9))
    slices = component_slices_from_cell_bounds(component, bounds, full)
    assert tuple(s.stop - s.start for s in slices) == component_shape_3d(
        component, (6, 6, 6)
    )


@pytest.mark.parametrize("component", ["Ex", "Ey", "Ez"])
def test_vector_material_interpolation_preserves_reflections_and_bounds(component):
    values = np.random.default_rng(4).uniform(1, 4, (5, 7, 9)).astype(np.float32)
    result = np.asarray(_cell_centers_to_yee(values, component, None))
    assert result.shape == component_shape_3d(component, values.shape)
    assert result.min() >= 1 and result.max() <= 4
    for axis in range(3):
        reflected = np.asarray(
            _cell_centers_to_yee(np.flip(values, axis), component, None)
        )
        np.testing.assert_allclose(reflected, np.flip(result, axis), rtol=2e-7)


@pytest.mark.parametrize(
    "axis,components",
    [
        ("x", ("Ey", "Ez", "Hz", "Hy")),
        ("y", ("Ez", "Ex", "Hx", "Hz")),
        ("z", ("Ex", "Ey", "Hy", "Hx")),
    ],
)
def test_modal_rows_match_ordinary_coupled_projection(axis, components):
    rng = np.random.default_rng(8)

    def fields():
        return {c: rng.normal(size=12) + 1j * rng.normal(size=12) for c in components}

    projections = [
        dict(
            components=components,
            axis=axis,
            direction_sign=1.0,
            d_area=0.02,
            mode_components=fields(),
            mode_components_bwd=fields(),
        )
        for _ in range(2)
    ]
    field = fields()
    rows = _modal_coefficient_rows_3d(projections)
    expected = _project_modal_coefficients_3d_group(field, projections)[0]
    actual = rows @ np.concatenate([field[c] for c in components])
    np.testing.assert_allclose(
        actual, np.asarray(expected).ravel(), rtol=1e-12, atol=1e-12
    )


def test_mode_source_symmetry_depends_only_on_fixed_cross_section():
    from types import SimpleNamespace

    from beamz.devices.sources.mode_launch import _centered_transverse_symmetry_axes

    values = np.ones((12, 14, 18))
    source = SimpleNamespace(axis="x", center=(4.5, 7.0, 6.0))
    fields = SimpleNamespace(permittivity=values)
    expected = _centered_transverse_symmetry_axes(source, fields, None, 1.0)
    assert expected == (0, 1)
    values[2, 3, 12] = 4.0  # A distant design update must not change injection.
    assert _centered_transverse_symmetry_axes(source, fields, None, 1.0) == expected
    values[2, 3, 4] = 4.0  # A changed source cross-section really is asymmetric.
    assert _centered_transverse_symmetry_axes(source, fields, None, 1.0) == ()
