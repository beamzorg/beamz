"""Check opt-in region results against the normal full-grid raster."""

import numpy as np
import pytest

from beamz.design.discretization import MaterialGrid
from beamz.design.raster import (
    Box,
    Grid,
    Material,
    Object,
    RasterOptions,
    RasterResult,
    Scene,
)
from beamz.design.raster import rasterize as full_rasterize
from beamz.design.raster._regions import rasterize_regions


def _assert_material_fields_match(full, reduced):
    for group_name in ("tensors", "yee_tensors"):
        for name, expected in getattr(full, group_name).items():
            np.testing.assert_array_equal(expected, getattr(reduced, group_name)[name])


def test_tiled_x_matches_full_grid():
    scene = Scene(
        (Material(), Material(permittivity=4.0)),
        (Object(Box((0.2, 0.2, 0.2), (0.8, 0.8, 0.8)), material_id=1),),
    )
    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (12, 12, 12))
    options = RasterOptions()
    full = full_rasterize(scene, grid, options=options)
    reduced = rasterize_regions(scene, grid, options=options, tile_x_cells=3)
    assert isinstance(reduced, RasterResult)
    assert not reduced.cache_hit
    assert reduced.scene_hash == full.scene_hash
    _assert_material_fields_match(full, reduced)
    regular_grid = MaterialGrid.from_raster_result(full, dimensions=3)
    region_grid = MaterialGrid.from_raster_result(reduced, dimensions=3)
    np.testing.assert_array_equal(regular_grid.permittivity, region_grid.permittivity)


def test_split_substrate_matches_full_grid():
    substrate = Object(Box((0, 0, 0), (1, 1, 0.3)), material_id=2, priority=10)
    scene = Scene(
        (Material(), Material(permittivity=4.0), Material(permittivity=8.0)),
        (
            substrate,
            Object(Box((0.2, 0.2, 0.7), (0.8, 0.8, 0.8)), material_id=1, priority=20),
        ),
    )
    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (12, 12, 20))
    options = RasterOptions()
    full = full_rasterize(scene, grid, options=options)
    reduced = rasterize_regions(
        scene, grid, options=options, tile_x_cells=3, substrate_object=substrate
    )
    _assert_material_fields_match(full, reduced)


def test_region_rejects_periodic_and_anisotropic_background():
    obj = Object(Box((0.2, 0.2, 0.2), (0.8, 0.8, 0.8)), material_id=1)
    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (8, 8, 8))
    scene = Scene((Material(), Material(permittivity=4.0)), (obj,))
    with pytest.raises(ValueError, match="periodic"):
        rasterize_regions(scene, grid, options=RasterOptions(periodic_axes=("x",)))
    anisotropic = Scene(
        (Material(permittivity=(1.0, 1.1, 1.2)), Material(permittivity=4.0)),
        (obj,),
    )
    with pytest.raises(ValueError, match="isotropic"):
        rasterize_regions(anisotropic, grid)


def test_nonuniform_grid_and_nonunit_permeability_match_full_raster():
    scene = Scene(
        (
            Material(permeability=2.0),
            Material(permittivity=4.0, permeability=3.0),
        ),
        (Object(Box((0.3, 0.2, 0.1), (0.7, 0.8, 0.9)), material_id=1),),
    )
    grid = Grid(
        np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
        np.array([0.0, 0.2, 0.6, 0.8, 1.0]),
        np.array([0.0, 0.1, 0.4, 0.6, 0.9, 1.0]),
    )
    full = full_rasterize(scene, grid)
    reduced = rasterize_regions(scene, grid, tile_x_cells=2)
    _assert_material_fields_match(full, reduced)


@pytest.mark.parametrize(
    "smoothing", ["volume", "farjadpour_diagonal", "farjadpour_full"]
)
def test_polygon_strips_with_different_tensor_components_match_full(smoothing):
    from beamz.design.raster import ExtrudedPolygon, Polygon

    # The first strips are homogeneous; later strips cross a sloping interface.
    scene = Scene(
        (Material(permittivity=2.0), Material(permittivity=5.0)),
        (
            Object(Box((0.0, 0.0, 0.0), (0.3, 1.0, 1.0)), material_id=1),
            Object(
                ExtrudedPolygon(
                    Polygon(((0.6, 0.2), (0.95, 0.3), (0.8, 0.85))), 0.2, 0.8
                ),
                material_id=1,
            ),
        ),
    )
    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (20, 10, 10))
    options = RasterOptions(smoothing=smoothing)
    full = full_rasterize(scene, grid, options=options)
    reduced = rasterize_regions(scene, grid, options=options, tile_x_cells=2)
    _assert_material_fields_match(full, reduced)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"scene": None}, TypeError, "Scene and Grid"),
        ({"grid": None}, TypeError, "Scene and Grid"),
        ({"options": {}}, TypeError, "RasterOptions"),
        ({"tile_x_cells": 0}, ValueError, "tile_x_cells"),
        ({"tile_x_cells": 1.5}, TypeError, "tile_x_cells"),
        ({"guard_cells": 1}, ValueError, "guard_cells"),
    ],
)
def test_region_rejects_invalid_arguments(kwargs, error, message):
    arguments = {
        "scene": Scene((Material(),), (Object(Box((0.2,) * 3, (0.8,) * 3), 0),)),
        "grid": Grid.uniform((0, 0, 0), (1, 1, 1), (8, 8, 8)),
    }
    arguments.update(kwargs)
    with pytest.raises(error, match=message):
        rasterize_regions(**arguments)


def test_region_rejects_empty_and_unsupported_scenes():
    from beamz.design.raster import Sphere

    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (8, 8, 8))
    with pytest.raises(ValueError, match="at least one object"):
        rasterize_regions(Scene((Material(),)), grid)
    with pytest.raises(TypeError, match="Unsupported raster geometry"):
        rasterize_regions(
            Scene((Material(),), (Object(Sphere((0.5,) * 3, 0.2), 0),)), grid
        )
    with pytest.raises(ValueError, match="No occupied strips"):
        rasterize_regions(
            Scene((Material(),), (Object(Box((-5.0,) * 3, (-4.0,) * 3), 0),)), grid
        )


@pytest.mark.parametrize(
    "invalid", ["absent", "polygon", "narrow", "raised", "overlap"]
)
def test_split_substrate_rejects_invalid_geometry(invalid):
    from beamz.design.raster import ExtrudedPolygon, Polygon

    geometry = {
        "absent": Box((0, 0, 0), (1, 1, 0.3)),
        "polygon": ExtrudedPolygon(Polygon(((0, 0), (1, 0), (0, 1))), 0, 0.3),
        "narrow": Box((0, 0, 0), (0.9, 1, 0.3)),
        "raised": Box((0, 0, 0.1), (1, 1, 0.3)),
        "overlap": Box((0, 0, 0), (1, 1, 0.7)),
    }[invalid]
    substrate = Object(geometry, 1)
    core = Object(Box((0.2, 0.2, 0.7), (0.8, 0.8, 0.8)), 1)
    scene = Scene((Material(), Material(permittivity=4)), (substrate, core))
    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (12, 12, 20))
    selected = Object(geometry, 1) if invalid == "absent" else substrate
    with pytest.raises(
        ValueError,
        match={
            "absent": "object in the scene",
            "polygon": "must be a Box",
            "narrow": "entire x-y grid",
            "raised": "lower z grid edge",
            "overlap": "overlap",
        }[invalid],
    ):
        rasterize_regions(scene, grid, substrate_object=selected)


@pytest.mark.parametrize("smoothing", ["farjadpour_diagonal", "farjadpour_full"])
def test_sparse_mesh_matches_full_raster(smoothing):
    from beamz.design.raster import Mesh

    mesh = Mesh(
        np.array([[0.2, 0.2, 0.2], [0.8, 0.2, 0.2], [0.5, 0.8, 0.2], [0.5, 0.5, 0.8]]),
        np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]),
    )
    scene = Scene((Material(), Material(permittivity=4)), (Object(mesh, 1),))
    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (12, 12, 12))
    options = RasterOptions(smoothing=smoothing)
    _assert_material_fields_match(
        full_rasterize(scene, grid, options=options),
        rasterize_regions(scene, grid, options=options, tile_x_cells=2),
    )


def test_split_substrate_full_tensor_background_and_promotion():
    from beamz.design.raster import ExtrudedPolygon, Polygon

    substrate = Object(Box((0, 0, 0), (1, 1, 0.15)), 1)
    core = Object(
        ExtrudedPolygon(Polygon(((0.3, 0.3), (0.7, 0.4), (0.4, 0.7))), 0.75, 0.85), 2
    )
    scene = Scene(
        (
            Material(permittivity=2, permeability=2, conductivity=0.1),
            Material(permittivity=8, permeability=3, conductivity=0.2),
            Material(permittivity=4, permeability=4, conductivity=0.3),
        ),
        (substrate, core),
    )
    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (12, 12, 24))
    options = RasterOptions(smoothing="farjadpour_full")
    _assert_material_fields_match(
        full_rasterize(scene, grid, options=options),
        rasterize_regions(
            scene, grid, options=options, tile_x_cells=3, substrate_object=substrate
        ),
    )


def test_region_rejects_dispersive_material():
    from beamz.design.dispersion import PoleResidue

    scene = Scene(
        (Material(), PoleResidue(3, [(-1e15, 1e14)], frequency_range=(1e14, 2e15))),
        (Object(Box((0.2,) * 3, (0.8,) * 3), 1),),
    )
    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (8, 8, 8))
    with pytest.raises(ValueError, match="nondispersive"):
        rasterize_regions(scene, grid)
