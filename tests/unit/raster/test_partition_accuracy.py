"""Regression oracles for stretched supports and adaptive sampling aliasing."""

import numpy as np
import pytest

from beamz.design import raster as r


@pytest.mark.parametrize("thickness", [1e-8, 1e-10])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e6])
def test_stretched_support_preserves_thin_material_interface(axis, scale, thickness):
    def point(value):
        return tuple(np.roll(np.asarray(value) * scale, axis))

    scene = r.Scene(
        (r.Material(), r.Material(1e8, conductivity=1e8), r.Material(2)),
        (
            r.Object(
                r.Box(point((0.4, 0, 0)), point((0.4 + thickness, 1e6, 1))),
                1,
                priority=0,
            ),
            r.Object(r.Box(point((0, 0, 0)), point((1, 5e5, 1))), 2, priority=1),
        ),
    )
    result = r.rasterize(
        scene,
        r.Grid.uniform((0, 0, 0), point((1, 1e6, 1)), (1, 1, 1)),
        options=r.RasterOptions(smoothing="volume", quality="reference"),
    )
    width = ((0.4 + thickness) * scale - 0.4 * scale) / scale
    assert result.tensors["epsilon"][0, 0, 0, 0] == pytest.approx(
        1.5 + 0.5 * width * (1e8 - 1), rel=2e-6
    )
    assert result.tensors["conductivity"][0, 0, 0, 0] == pytest.approx(
        0.5 * width * 1e8, rel=2e-6
    )
    assert result.diagnostics["adaptive_samples"] == 0


@pytest.mark.parametrize("quality,tolerance", [("balanced", 0.04), ("reference", 0.01)])
@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e6])
def test_extruded_circle_does_not_falsely_converge(quality, tolerance, scale):
    scene = r.Scene(
        (r.Material(), r.Material(4)),
        (
            r.Object(
                r.Cylinder((0.5 * scale, 0.5 * scale), 0.4 * scale, -scale, 2 * scale),
                1,
            ),
        ),
    )
    result = r.rasterize(
        scene,
        r.Grid.uniform((0, 0, 0), (scale,) * 3, (1, 1, 1)),
        options=r.RasterOptions(smoothing="volume", quality=quality),
    )
    expected = 1 + 3 * np.pi * 0.4**2
    assert abs(float(result.tensors["epsilon"][0, 0, 0, 0]) - expected) < tolerance
    assert result.diagnostics["maximum_estimated_fraction_error"] > 0
