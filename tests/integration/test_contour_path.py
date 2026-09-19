"""The 3D update must consume contour-path Yee coefficients without resampling."""

import numpy as np
import pytest

import beamz as bz
from beamz.design import raster


@pytest.mark.parametrize("graded", [False, True])
def test_contour_path_3d_compile_and_update_preserve_yee_coefficients(graded):
    grid = (
        raster.Grid(
            np.array([0, 0.2, 0.55, 1]) * bz.um,
            np.array([0, 0.4, 0.6, 1]) * bz.um,
            np.array([0, 0.3, 0.7, 1]) * bz.um,
        )
        if graded
        else raster.Grid.uniform((0, 0, 0), (bz.um,) * 3, (3, 3, 3))
    )
    simulation = bz.Simulation(
        scene=raster.Scene(
            (raster.Material(), raster.Material(4)),
            (
                raster.Object(
                    raster.Box((-bz.um,) * 3, (0.43 * bz.um, 2 * bz.um, 2 * bz.um)), 1
                ),
            ),
        ),
        raster_grid=grid,
        raster_options=raster.RasterOptions(smoothing="contour_path"),
        time=np.arange(4) * 1e-17,
        sources=(),
        normalize_source=None,
    )
    material = simulation.to_request().materials
    program = simulation.compile(backend="jax")
    assert material.smoothing == "contour_path"
    assert material.uses_direct_yee_materials
    for name, coefficients in material.yee_materials.items():
        np.testing.assert_array_equal(getattr(program.grid, name), coefficients)
    assert program.coefficients.e_inverse_offdiagonal.size == 0
    state = simulation.initial_state()
    state = state._replace(
        hx=np.arange(state.hx.size, dtype=np.float32).reshape(state.hx.shape)
    )
    advanced = simulation.step(state)
    assert all(
        np.isfinite(getattr(advanced, name)).all()
        for name in ("ex", "ey", "ez", "hx", "hy", "hz")
    )
    assert np.max(np.abs(advanced.ey)) + np.max(np.abs(advanced.ez)) > 0
