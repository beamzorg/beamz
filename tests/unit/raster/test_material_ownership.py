"""Equivalent material distributions must produce the same solver coefficients."""

import numpy as np
import pytest

from beamz.design import raster

MATERIALS = (raster.Material(1.44402**2), raster.Material(3.47644**2))


def compare_results(left, right):
    for collection in ("tensors", "yee_tensors"):
        for name, expected in getattr(left, collection).items():
            np.testing.assert_allclose(
                getattr(right, collection)[name],
                expected,
                rtol=2e-6,
                atol=2e-6,
                err_msg=f"{collection}.{name}",
            )


@pytest.mark.parametrize(
    "smoothing", ["volume", "farjadpour_diagonal", "farjadpour_full"]
)
@pytest.mark.parametrize("scale", [1.0, 1e-6])
def test_duplicate_diagonal_interface_preserves_cell_and_yee_tensors(smoothing, scale):
    geometry = raster.ExtrudedPolygon(
        raster.Polygon(
            tuple((x * scale, y * scale) for x, y in ((-2, -2), (2, -2), (-2, 2)))
        ),
        -2 * scale,
        2 * scale,
    )
    grid = raster.Grid.uniform((-0.5 * scale,) * 3, (0.5 * scale,) * 3, (1, 1, 1))
    options = raster.RasterOptions(smoothing=smoothing)
    results = [
        raster.rasterize(
            raster.Scene(
                MATERIALS, tuple(raster.Object(geometry, 1, id=i) for i in range(n))
            ),
            grid,
            options=options,
        )
        for n in (1, 2, 3)
    ]
    arithmetic = sum(m.epsilon_r[0] for m in MATERIALS) / 2
    harmonic = 2 / sum(1 / m.epsilon_r[0] for m in MATERIALS)
    expected = {
        "volume": [arithmetic],
        "farjadpour_diagonal": [(arithmetic + harmonic) / 2] * 2 + [arithmetic],
        "farjadpour_full": [(arithmetic + harmonic) / 2] * 2
        + [arithmetic, (harmonic - arithmetic) / 2, 0, 0],
    }[smoothing]
    for result in results:
        np.testing.assert_allclose(
            result.tensors["epsilon"].ravel(), expected, rtol=2e-6
        )
        compare_results(results[0], result)
        assert result.diagnostics["adaptive_samples"] == 0
        assert result.diagnostics["fallback_multiple_objects"] == 0


@pytest.mark.parametrize(
    "smoothing", ["volume", "farjadpour_diagonal", "farjadpour_full"]
)
def test_stepped_core_slab_union_matches_disjoint_decomposition(smoothing):
    slab = raster.Box((-2, -2, -0.3), (2, 2, 0.15))
    core = raster.Box((-0.2, -2, -0.3), (0.2, 2, 0.22))
    upper_core = raster.Box((-0.2, -2, 0.15), (0.2, 2, 0.22))
    # Include unequal widths and supports crossing the hidden slab/core seam.
    grid = raster.Grid([-0.5, -0.1, 0.3, 0.5], [-0.5, 0.1, 0.5], [-0.1, 0.1, 0.19, 0.3])
    representations = (
        (slab, core),
        (slab, upper_core),
        (core, slab),
        (slab, core, core),
    )
    results = [
        raster.rasterize(
            raster.Scene(
                MATERIALS, tuple(raster.Object(g, 1, id=i) for i, g in enumerate(gs))
            ),
            grid,
            options=raster.RasterOptions(smoothing=smoothing),
        )
        for gs in representations
    ]
    for result in results:
        compare_results(results[0], result)
        assert result.diagnostics["adaptive_samples"] == 0
        assert result.diagnostics["fallback_multiple_objects"] == 0
    if smoothing != "volume":
        assert results[0].diagnostics["fallback_multiple_orientations"] > 0


@pytest.mark.parametrize(
    "smoothing", ["volume", "farjadpour_diagonal", "farjadpour_full", "contour_path"]
)
def test_hole_and_thin_layer_match_tiled_union(smoothing):
    outer = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    hole = ((-0.2, -0.2), (-0.2, 0.2), (0.2, 0.2), (0.2, -0.2))
    ring = raster.ExtrudedPolygon(raster.Polygon(outer, (hole,)), 0.111, 0.113)
    boxes = [
        raster.Box((-1, -1, 0.111), (-0.2, 1, 0.113)),
        raster.Box((0.2, -1, 0.111), (1, 1, 0.113)),
        raster.Box((-0.2, -1, 0.111), (0.2, -0.2, 0.113)),
        raster.Box((-0.2, 0.2, 0.111), (0.2, 1, 0.113)),
    ]
    grid = raster.Grid([-1, -0.3, 0.1, 0.5, 1], [-1, -0.1, 0.4, 1], [0, 0.15, 0.3])
    results = [
        raster.rasterize(
            raster.Scene(
                MATERIALS, tuple(raster.Object(g, 1, id=i) for i, g in enumerate(gs))
            ),
            grid,
            options=raster.RasterOptions(smoothing=smoothing),
        )
        for gs in [(ring,), boxes, (ring, *boxes)]
    ]
    for result in results[1:]:
        compare_results(results[0], result)
    volume = raster.rasterize(
        raster.Scene(MATERIALS, (raster.Object(ring, 1),)),
        grid,
        options=raster.RasterOptions(smoothing="volume"),
    )
    fractions = (volume.tensors["epsilon"][0] - MATERIALS[0].epsilon_r[0]) / (
        MATERIALS[1].epsilon_r[0] - MATERIALS[0].epsilon_r[0]
    )
    weights = (
        np.diff(grid.edges[2])[:, None, None]
        * np.diff(grid.edges[1])[None, :, None]
        * np.diff(grid.edges[0])[None, None, :]
    )
    assert np.sum(fractions * weights) == pytest.approx((4 - 0.16) * 0.002, abs=2e-8)


@pytest.mark.parametrize(
    "smoothing", ["volume", "farjadpour_diagonal", "farjadpour_full"]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_priority_and_background_occlusion_have_analytic_yee_coefficients(
    smoothing, reverse
):
    materials = (
        raster.Material(2, 1, 0),
        raster.Material(4, 2, 1),
        raster.Material(8, 3, 2),
    )
    objects = [
        raster.Object(raster.Box((-2, -2, -2), (0.4, 2, 2)), 1, priority=0, id=10),
        raster.Object(raster.Box((0.2, -2, -2), (2, 2, 2)), 2, priority=0, id=20),
        raster.Object(raster.Box((0.7, -2, -2), (0.71, 2, 2)), 0, priority=1, id=1),
        # Entirely hidden, including its non-laminar corners.
        raster.Object(
            raster.Box((0.3, 0.2, 0.2), (0.6, 0.4, 0.4)), 1, priority=-1, id=30
        ),
    ]
    if reverse:
        objects.reverse()
    grid = raster.Grid([0, 0.1, 0.5, 1], [0, 0.4, 1], [0, 0.6, 1])
    result = raster.rasterize(
        raster.Scene(materials, tuple(objects)),
        grid,
        options=raster.RasterOptions(smoothing=smoothing),
    )
    x = np.asarray(grid.edges[0])
    centers = (x[:-1] + x[1:]) / 2
    dual = np.r_[x[0], centers, x[-1]]
    segments = [(0, 0.2, 1), (0.2, 0.7, 2), (0.7, 0.71, 0), (0.71, 1, 2)]
    for name in ("epsilon", "epsilon_ex", "epsilon_ey", "epsilon_ez"):
        actual = result.tensors[name] if name == "epsilon" else result.yee_tensors[name]
        edges = x if name in ("epsilon", "epsilon_ex") else dual
        # Every support has planar x-normal layering, regardless of its y/z size.
        for i, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            fractions = np.zeros(3)
            for start, end, material in segments:
                fractions[material] += max(0, min(end, right) - max(start, left)) / (
                    right - left
                )
            epsilon = np.array([2, 4, 8])
            arithmetic = fractions @ epsilon
            harmonic = 1 / (fractions @ (1 / epsilon))
            expected = (
                [arithmetic]
                if smoothing == "volume"
                else [harmonic, arithmetic, arithmetic]
            )
            np.testing.assert_allclose(
                actual[..., i],
                np.broadcast_to(
                    np.array(expected)[:, None, None], actual[..., i].shape
                ),
                rtol=2e-6,
            )
    assert result.diagnostics["fallback_multiple_objects"] == 0
    assert result.diagnostics["fallback_multiple_orientations"] == 0
    assert result.diagnostics["adaptive_samples"] == 0


@pytest.mark.parametrize(
    "smoothing", ["volume", "farjadpour_diagonal", "farjadpour_full"]
)
def test_unrelated_curved_object_does_not_disable_exact_union(smoothing):
    grid = raster.Grid.uniform((-0.5,) * 3, (0.5,) * 3, (2, 2, 2))
    shapes = (
        raster.Box((-2, -2, -2), (0.1, 2, 0.1)),
        raster.Box((-2, -2, -2), (0.3, 2, -0.1)),
    )
    objects = tuple(raster.Object(g, 1, id=i) for i, g in enumerate(shapes))
    options = raster.RasterOptions(smoothing=smoothing)
    expected = raster.rasterize(raster.Scene(MATERIALS, objects), grid, options=options)
    actual = raster.rasterize(
        raster.Scene(
            MATERIALS,
            objects + (raster.Object(raster.Sphere((10, 10, 10), 1), 1, id=99),),
        ),
        grid,
        options=options,
    )
    compare_results(expected, actual)
    assert actual.diagnostics["adaptive_samples"] == 0


@pytest.mark.parametrize("quality", ["fast", "balanced", "reference"])
def test_curved_duplicate_preserves_adaptive_estimate(quality):
    sphere = raster.Sphere((0, 0, 0), 1)
    grid = raster.Grid.uniform((0.8, -0.05, -0.05), (1.1, 0.05, 0.05), (2, 1, 1))
    options = raster.RasterOptions(quality=quality)
    results = [
        raster.rasterize(
            raster.Scene(
                MATERIALS, tuple(raster.Object(sphere, 1, id=i) for i in range(n))
            ),
            grid,
            options=options,
        )
        for n in (1, 3)
    ]
    compare_results(*results)
    assert results[1].diagnostics["fallback_multiple_objects"] == 0


def test_tiled_diagonal_seams_and_material_aliases_are_hidden():
    material = raster.Material((4, 5, 6), (1, 2, 3), (0.1, 0.2, 0.3))
    materials = (raster.Material(), material, material)
    square = raster.Box((-1, -1, -1), (1, 1, 0.13))
    triangles = [
        raster.ExtrudedPolygon(raster.Polygon(points), -1, 0.13)
        for points in (((-1, -1), (1, -1), (1, 1)), ((-1, -1), (1, 1), (-1, 1)))
    ]
    grid = raster.Grid.uniform((-0.5,) * 3, (0.5,) * 3, (3, 3, 3))
    expected = raster.rasterize(
        raster.Scene(materials, (raster.Object(square, 1),)), grid
    )
    actual = raster.rasterize(
        raster.Scene(
            materials,
            tuple(raster.Object(g, i + 1, id=i) for i, g in enumerate(triangles)),
        ),
        grid,
    )
    compare_results(expected, actual)
    assert actual.diagnostics["fallback_multiple_orientations"] == 0


@pytest.mark.parametrize("scale", [1.0, 1e-6])
def test_uniform_polygon_interior_has_no_phantom_background(scale):
    polygon = raster.ExtrudedPolygon(
        raster.Polygon(
            tuple((scale * x, scale * y) for x, y in ((-3, -3), (3, -3), (-3, 3)))
        ),
        -scale,
        scale,
    )
    grid = raster.Grid.uniform(
        tuple(scale * v for v in (-1.13, -1.27, -0.3)),
        tuple(scale * v for v in (-0.31, -0.43, 0.4)),
        (7, 11, 2),
    )
    result = raster.rasterize(
        raster.Scene(
            MATERIALS,
            (raster.Object(polygon, 1, id=1), raster.Object(polygon, 1, id=2)),
        ),
        grid,
    )
    for name, value in result.tensors.items():
        key = {"epsilon": "epsilon_r", "mu": "mu_r", "conductivity": "conductivity"}[
            name
        ]
        expected = MATERIALS[1].to_dict()[key][0]
        np.testing.assert_array_equal(value, np.full_like(value, expected))
    assert result.diagnostics["ambiguous_interface_samples"] == 0


def test_rectangular_polygon_clipping_preserves_box_mirror_symmetry():
    # Non-binary meter coordinates made Boolean clipping introduce tiny material
    # variations along an otherwise invariant waveguide cross section.
    shape = raster.Box((0, 1.26e-6, 1.12e-6), (4.06e-6, 1.75e-6, 1.4e-6))
    grid = raster.Grid.uniform((0, 0, 0), (4.06e-6, 3.01e-6, 2.52e-6), (58, 43, 36))
    result = raster.rasterize(raster.Scene(MATERIALS, (raster.Object(shape, 1),)), grid)
    for name in ("epsilon_ex", "epsilon_ey", "epsilon_ez"):
        value = result.yee_tensors[name]
        for axis in (1, 2):
            np.testing.assert_array_equal(value, np.flip(value, axis=axis))
        np.testing.assert_array_equal(
            value, np.broadcast_to(value[..., :1], value.shape)
        )
