"""Independent path/surface oracles and ownership controls for issue #243."""

import numpy as np
import pytest

from beamz.design import raster


def run(geometry, grid, smoothing="contour_path", materials=None, copies=1):
    return raster.rasterize(
        raster.Scene(
            materials or (raster.Material(1.0), raster.Material(4.0)),
            tuple(raster.Object(geometry, 1, id=i) for i in range(copies)),
        ),
        grid,
        options=raster.RasterOptions(smoothing=smoothing),
    )


def value(result, component, index):
    tensor = result.yee_tensors[f"epsilon_e{component}"]
    axis = "xyz".index(component)
    return tensor[(0 if tensor.shape[0] == 1 else axis, *index)]


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("offset", [-0.37, -0.11, 0.19, 0.43])
def test_arithmetic_harmonic_limits(axis, offset):
    lo, hi = [-5.0] * 3, [5.0] * 3
    hi[axis] = offset
    grid = raster.Grid.uniform((-1.5,) * 3, (1.5,) * 3, (3, 3, 3))
    result = run(raster.Box(lo, hi), grid)
    # E_axis lies at the center of its primal interval [-.5,.5].
    f = offset + 0.5
    assert value(result, "xyz"[axis], (1, 1, 1)) == pytest.approx(
        1 / (f / 4 + (1 - f)), rel=2e-6
    )
    # The tangential component's dual interval on that axis is [-1,0].
    tangent = (axis + 1) % 3
    f = np.clip(offset + 1, 0, 1)
    assert value(result, "xyz"[tangent], (1, 1, 1)) == pytest.approx(
        f * 4 + 1 - f, rel=2e-6
    )


@pytest.mark.parametrize("angle", [7, 27, 45, 73, 87])
@pytest.mark.parametrize("offset", [-0.13, 0.08, 0.31])
@pytest.mark.parametrize("scale", [1, 1e-6])
def test_tilted_interface_against_independent_equation_7_8(angle, offset, scale):
    normal = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    tangent = np.array([-normal[1], normal[0]])
    center = np.array([0.0, 0.0])
    p = normal * offset
    polygon = [
        p - 10 * tangent,
        p + 10 * tangent,
        p + 10 * tangent - 10 * normal,
        p - 10 * tangent - 10 * normal,
    ]
    geometry = raster.ExtrudedPolygon(
        raster.Polygon(np.array(polygon) * scale), -5 * scale, 5 * scale
    )
    # Ex location (0,0,0), primal X [-.5,.5], dual Y [-.5,.5].
    grid = raster.Grid(
        np.array([-0.5, 0.5]) * scale,
        np.array([-1.0, 0.0, 1.0]) * scale,
        np.array([-1.0, 0.0, 1.0]) * scale,
    )
    result = run(geometry, grid)
    eps_center = 4 if normal @ center < offset else 1
    f_line = np.clip(offset / normal[0] + 0.5, 0, 1)
    f_flux = np.clip(offset / normal[1] + 0.5, 0, 1)
    q = normal[0] ** 2
    numerator = (1 - q) * (1 + 3 * f_flux) + q * eps_center
    denominator = (1 - q) + q * eps_center * ((1 - f_line) + f_line / 4)
    assert value(result, "x", (1, 1, 0)) == pytest.approx(
        numerator / denominator, rel=2e-6
    )
    assert result.smoothing == "contour_path"


def test_nonuniform_grid_uses_yee_location_not_support_midpoint():
    # Ex y=0, but the transverse support is [-.5,1.5], centered at .5.
    geometry = raster.ExtrudedPolygon(
        raster.Polygon([[-10, -10], [10, -10], [-10, 10.4]]), -5, 5
    )
    grid = raster.Grid([-0.5, 0.5], [-1, 0, 3], [-1, 0, 1])
    result = run(geometry, grid)
    n = np.array([20.4, 20.0])
    n /= np.linalg.norm(n)
    crossing_x = 0.2 / 1.02  # line y=0
    f_line = crossing_x + 0.5
    f_surface = (0.2 + 0.5) / 2
    q = n[0] ** 2
    expected = ((1 - q) * (1 + 3 * f_surface) + q * 4) / (
        (1 - q) + q * 4 * (1 - f_line + f_line / 4)
    )
    assert value(result, "x", (1, 1, 0)) == pytest.approx(expected, rel=2e-6)


def test_same_material_union_and_priority_do_not_change_coefficients():
    grid = raster.Grid([-1, -0.2, 0.3, 1], [-1, 0, 0.4, 1], [-1, 0.1, 0.2, 1])
    slab = raster.Box((-5, -5, -5), (5, 5, 0.15))
    core = raster.Box((-0.3, -5, -5), (0.3, 5, 0.22))
    top = raster.Box((-0.3, -5, 0.15), (0.3, 5, 0.22))
    outputs = []
    for objects in [(slab, core), (slab, top), (core, slab, core)]:
        outputs.append(
            raster.rasterize(
                raster.Scene(
                    (raster.Material(), raster.Material(4)),
                    tuple(raster.Object(g, 1, id=i) for i, g in enumerate(objects)),
                ),
                grid,
                options=raster.RasterOptions(smoothing="contour_path"),
            )
        )
    for result in outputs[1:]:
        for name, expected in outputs[0].yee_tensors.items():
            np.testing.assert_allclose(
                result.yee_tensors[name], expected, rtol=2e-6, atol=2e-6
            )
    assert outputs[0].diagnostics["fallback_multiple_orientations"] > 0


@pytest.mark.parametrize(
    "material",
    [
        raster.Material((2, 3, 4)),
        raster.Material(4, mu_r=2),
        raster.Material(4, conductivity=0.1),
    ],
)
def test_rejects_unsupported_materials(material):
    with pytest.raises(ValueError, match="contour_path requires"):
        run(
            raster.Box((-1,) * 3, (1,) * 3),
            raster.Grid.uniform((-1,) * 3, (1,) * 3, (2,) * 3),
            materials=(raster.Material(), material),
        )


def test_cache_roundtrip_retains_method(tmp_path):
    scene = raster.Scene(
        (raster.Material(), raster.Material(4)),
        (raster.Object(raster.Box((-1, -1, -1), (0.1, 1, 1)), 1),),
    )
    grid = raster.Grid.uniform((-1,) * 3, (1,) * 3, (3,) * 3)
    options = raster.RasterOptions(smoothing="contour_path")
    first = raster.rasterize(scene, grid, options=options, cache_directory=tmp_path)
    second = raster.rasterize(scene, grid, options=options, cache_directory=tmp_path)
    assert second.smoothing == "contour_path"
    for name, expected in first.yee_tensors.items():
        np.testing.assert_array_equal(second.yee_tensors[name], expected)


def test_separate_intersection_normals_are_not_volume_averaged():
    # Intersection of x+2y<.2 and 3x-y<.4; p=(0,0) is in epsilon=4.
    polygon = [(-5, -5), (-4.6 / 3, -5), (1 / 7, 1 / 35), (-5, 2.6)]
    geometry = raster.ExtrudedPolygon(raster.Polygon(polygon), -2, 2)
    grid = raster.Grid([-0.5, 0.5], [-1, 0, 1], [-1, 0, 1])
    result = run(geometry, grid)
    # Flux path: material 4 on y in [-.4,.1], different normals at both ends.
    flux = 0.5 * 4 + 0.1 * (0.1 + 4 * 0.9) + 0.4 * (0.8 + 4 * 0.2)
    # Electric line: material 4 on x in [-.5, .4/3], q=9/10 at the crossing.
    inside = 0.5 + 0.4 / 3
    circulation = inside + (1 - inside) * (0.1 + 0.9 * 4)
    assert value(result, "x", (1, 1, 0)) == pytest.approx(flux / circulation, rel=2e-6)
    volume = run(geometry, grid, smoothing="volume")
    assert abs(value(result, "x", (1, 1, 0)) - value(volume, "x", (1, 1, 0))) > 0.1


def test_thin_parallel_layer_is_resolved_by_paths():
    grid = raster.Grid([-0.5, 0.5], [-1, 0, 1], [-1, 0, 1])
    result = run(raster.Box((0.1, -5, -5), (0.101, 5, 5)), grid)
    assert value(result, "x", (1, 1, 0)) == pytest.approx(
        1 / (0.999 + 0.001 / 4), rel=2e-6
    )


def test_uniform_grid_simulation_retains_contour_yee_coefficients():
    from beamz.design.discretization import MaterialGrid

    grid = raster.Grid.uniform((-1,) * 3, (1,) * 3, (3,) * 3)
    result = run(raster.Box((-2,) * 3, (0.13, 2, 2)), grid)
    material_grid = MaterialGrid.from_raster_result(result)
    assert material_grid.uses_direct_yee_materials
    assert not material_grid.uses_full_permittivity


@pytest.mark.parametrize("reverse", [False, True])
def test_priority_background_holes_and_material_aliases(reverse):
    materials = (
        raster.Material(1),
        raster.Material(4),
        raster.Material(4),
        raster.Material(8),
    )
    objects = [
        raster.Object(raster.Box((-2, -2, -2), (0.4, 2, 2)), 1, id=1),
        raster.Object(raster.Box((-0.3, -2, -2), (0.4, 2, 2)), 2, id=2),
        raster.Object(raster.Box((0.2, -2, -2), (2, 2, 2)), 3, id=3),
        raster.Object(raster.Box((0.7, -2, -2), (0.71, 2, 2)), 0, id=4, priority=1),
    ]
    if reverse:
        objects.reverse()
    result = raster.rasterize(
        raster.Scene(materials, tuple(objects)),
        raster.Grid([0, 1], [-1, 0, 1], [-1, 0, 1]),
        options=raster.RasterOptions(smoothing="contour_path"),
    )
    assert value(result, "x", (1, 1, 0)) == pytest.approx(
        1 / (0.2 / 4 + 0.79 / 8 + 0.01), rel=2e-6
    )
    assert result.diagnostics["fallback_multiple_objects"] == 0


def test_hidden_height_seam_does_not_disable_curved_contour_paths():
    polygon = raster.Polygon([(-5, -5), (-4.6 / 3, -5), (1 / 7, 1 / 35), (-5, 2.6)])
    whole = raster.ExtrudedPolygon(polygon, -2, 2)
    pieces = (
        raster.ExtrudedPolygon(polygon, -2, 0.13),
        raster.ExtrudedPolygon(polygon, 0.13, 2),
    )
    grid = raster.Grid([-0.5, 0.5], [-1, 0, 1], [-1, 0, 1])
    results = [
        raster.rasterize(
            raster.Scene(
                (raster.Material(), raster.Material(4)),
                tuple(raster.Object(g, 1, id=i) for i, g in enumerate(gs)),
            ),
            grid,
            options=raster.RasterOptions(smoothing="contour_path"),
        )
        for gs in ((whole,), pieces)
    ]
    for name, expected in results[0].yee_tensors.items():
        np.testing.assert_allclose(
            results[1].yee_tensors[name], expected, rtol=2e-6, atol=2e-6
        )
