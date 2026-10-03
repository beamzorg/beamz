"""Native periodic support ownership and simulation raster cache contracts."""

import numpy as np
import pytest

import beamz as bz
from beamz.design.raster import Box, Grid, Object, RasterOptions, Scene


def test_periodic_options_validate_and_canonicalize():
    assert RasterOptions(periodic_axes=("y", "x", "y")).periodic_axes == ("x", "y")
    with pytest.raises(ValueError, match="periodic_axes"):
        RasterOptions(periodic_axes=("u",))


def test_metal_seam_has_one_point_owner_for_epsilon_and_poles():
    metal = bz.PoleResidue.drude(
        2, plasma_frequency=1e16, damping=1e14, frequency_range=(3e14, 9e14)
    )
    scene = Scene(
        (bz.Material(4), metal), (Object(Box((0, 0, 0), (0.2e-6, 0.4e-6, 0.1e-6)), 1),)
    )
    sim = bz.Simulation(
        scene=scene,
        raster_grid=Grid.uniform((0, 0, 0), (0.4e-6, 0.4e-6, 0.1e-6), (8, 8, 2)),
        boundaries=[bz.Periodic(axes=("x", "y"))],
        time=np.arange(3) * 1e-17,
    )
    plan = sim.compile(backend="jax")
    np.testing.assert_allclose(np.asarray(plan.grid.eps_y)[..., [0, -1]], 2)
    region = next(r for r in plan.dispersion.regions if r.component == "ey")
    np.testing.assert_allclose(np.asarray(region.weights)[..., [0, -1]], 1)
    assert not plan.dispersion.interfaces


def test_design_raster_cache_distinguishes_periodic_axes():
    design = bz.Design(width=0.4e-6, height=0.4e-6, background=bz.Material(1))
    design += bz.Rectangle(
        position=(0, 0), width=0.4e-6, height=0.13e-6, material=bz.Material(3)
    )
    first = bz.Simulation(
        design=design, resolution=50e-9, polarization="te", time=np.arange(3) * 1e-17
    )
    second = first.updated_copy(boundaries=[bz.Periodic(axes="y")])
    assert first._material_grid_token() != second._material_grid_token()
    a, b = first._material_grid(), second._material_grid()
    np.testing.assert_allclose(a.yee_materials["eps_x"][0], 3)
    np.testing.assert_allclose(b.yee_materials["eps_x"][0], 2)
    np.testing.assert_allclose(
        b.yee_materials["eps_x"][0], b.yee_materials["eps_x"][-1]
    )


def test_native_disk_cache_distinguishes_periodic_supports(tmp_path):
    scene = Scene(
        (bz.Material(1), bz.Material(3)), (Object(Box((0, 0, 0), (1, 0.3, 1)), 1),)
    )
    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (4, 4, 1))
    plain = RasterOptions()
    periodic = RasterOptions(periodic_axes=("y",))
    assert not scene.rasterize(grid, options=plain, cache_directory=tmp_path).cache_hit
    assert not scene.rasterize(
        grid, options=periodic, cache_directory=tmp_path
    ).cache_hit
    assert scene.rasterize(grid, options=periodic, cache_directory=tmp_path).cache_hit
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_scene_freezes_periodic_boundary_iterable_before_rasterization():
    sim = bz.Simulation(
        scene=Scene((bz.Material(1),)),
        raster_grid=Grid.uniform((0, 0, 0), (1e-6, 1e-6, 1e-6), (4, 4, 4)),
        boundaries=iter([bz.Periodic(axes=("x", "y"))]),
        time=np.arange(2) * 1e-17,
    )
    assert sim.compile(backend="jax").boundary.periodic_axes == {1, 2}
