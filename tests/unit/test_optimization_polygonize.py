"""Even/odd contour fill preserves nested holes and islands."""

import numpy as np
import pytest

from beamz.optimization.polygonize import density_to_shapely_geometry


@pytest.mark.parametrize("spacing", (None, 0.25))
def test_nested_contours_match_independent_boundary_composition(spacing):
    density = np.zeros((25, 25))
    boundaries = []
    for edge in (2, 5, 8, 11):
        rectangle = np.zeros_like(density)
        rectangle[edge:-edge, edge:-edge] = 1
        density = np.abs(density - rectangle)
        boundaries.append(
            density_to_shapely_geometry(rectangle, sample_spacing=spacing)
        )
    expected = boundaries[0]
    for boundary in boundaries[1:]:
        expected = expected.symmetric_difference(boundary)
    actual = density_to_shapely_geometry(density, sample_spacing=spacing)
    assert actual.symmetric_difference(expected).area < 1e-12
    assert len(actual.geoms) == 2
    assert sum(len(p.interiors) for p in actual.geoms) == 2
