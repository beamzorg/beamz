"""Boundary normal classification must not depend on object count or origin."""

import numpy as np
import pytest

from beamz.design import raster as r

from .test_mesh import cube


def assert_same_coefficients(expected, actual):
    def expand(value):
        if value.shape[0] == 1:
            value = np.repeat(value, 3, axis=0)
        if value.shape[0] == 3:
            value = np.concatenate((value, np.zeros_like(value)), axis=0)
        return value

    for group in ("tensors", "yee_tensors"):
        first, second = getattr(expected, group), getattr(actual, group)
        assert first.keys() == second.keys()
        for name in first:
            np.testing.assert_allclose(
                expand(first[name]),
                expand(second[name]),
                rtol=2e-6,
                atol=2e-7,
                err_msg=f"{group}.{name}",
            )


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("offset", [-1e6, 0, 1e6])
@pytest.mark.parametrize("representation", ["box", "mesh"])
@pytest.mark.parametrize("smoothing", ["farjadpour_diagonal", "farjadpour_full"])
def test_boundary_roundoff_preserves_single_duplicate_and_split_geometry(
    axis, scale, offset, representation, smoothing
):
    lower = np.full(3, (offset + 0.125) * scale)
    upper = lower + scale
    material = r.Material(4, mu_r=2, conductivity=0.3)
    materials = (r.Material(), material)
    box_upper = upper.copy()
    box_upper[axis] = lower[axis] + 0.5 * scale
    box_lower = lower.copy()
    transverse = (axis + 1) % 3
    box_lower[transverse] = np.nextafter(lower[transverse], np.inf)

    def geometry(lo, hi):
        if representation == "box":
            return r.Box(tuple(lo), tuple(hi))
        vertices, triangles = cube()
        return r.Mesh(vertices * (hi - lo) + lo, triangles)

    shape = geometry(box_lower, box_upper)
    split = (box_lower[axis] + box_upper[axis]) / 2
    left_upper, right_lower = box_upper.copy(), box_lower.copy()
    left_upper[axis] = right_lower[axis] = split
    scenes = [
        r.Scene(materials, (r.Object(shape, 1),)),
        r.Scene(materials, (r.Object(shape, 1, id=1), r.Object(shape, 1, id=2))),
        r.Scene(
            materials,
            (
                r.Object(geometry(box_lower, left_upper), 1, id=1),
                r.Object(geometry(right_lower, box_upper), 1, id=2),
            ),
        ),
    ]
    grid = r.Grid.uniform(tuple(lower), tuple(upper), (1, 1, 1))
    options = r.RasterOptions(smoothing=smoothing, quality="reference")
    # An aligned half slab has an independent harmonic-normal/arithmetic-
    # tangential answer. The one-ULP gap changes fractions below this tolerance.
    baseline = r.rasterize(
        r.Scene(materials, (r.Object(r.Box(tuple(lower), tuple(box_upper)), 1),)),
        grid,
        options=options,
    )
    for scene in scenes:
        result = r.rasterize(scene, grid, options=options)
        assert_same_coefficients(baseline, result)
        assert result.tensors["epsilon"][axis, 0, 0, 0] == pytest.approx(1.6, rel=2e-6)
        assert result.tensors["mu"][axis, 0, 0, 0] == pytest.approx(4 / 3, rel=2e-6)
        assert result.diagnostics["adaptive_samples"] == 0


@pytest.mark.parametrize("copies", [1, 2])
@pytest.mark.parametrize("offset", [-1e6, 0, 1e6])
def test_resolved_corner_is_not_smoothed_away(copies, offset):
    lower = np.full(3, offset + 0.125)
    upper = lower + 1
    lo = lower.copy()
    lo[1] += 1e-5  # Far larger than coordinate roundoff at every tested origin.
    hi = upper.copy()
    hi[0] = lower[0] + 0.5
    scene = r.Scene(
        (r.Material(), r.Material(4)),
        tuple(
            r.Object(r.Box(tuple(lo), tuple(hi)), 1, id=i + 1) for i in range(copies)
        ),
    )
    result = r.rasterize(scene, r.Grid.uniform(tuple(lower), tuple(upper), (1, 1, 1)))
    expected = 1 + 3 * 0.5 * (upper[1] - lo[1])
    np.testing.assert_allclose(result.tensors["epsilon"], expected, rtol=2e-6)
