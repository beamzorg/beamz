"""Periodic materials must agree with an independently rasterized interior tile."""

import numpy as np
import pytest

from tests.validation.analytical.periodic_seam_reference import compare


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("dispersive", [False, True])
@pytest.mark.parametrize("smoothing", ["volume", "farjadpour_diagonal"])
def test_tilted_dielectric_seam_matches_tiled_interior(graded, dispersive, smoothing):
    measured = compare(graded=graded, dispersive=dispersive, smoothing=smoothing)
    np.testing.assert_allclose(
        list(measured["epsilon_max_difference"].values()), 0, atol=1e-6
    )
    for errors in measured["field_max_difference"].values():
        np.testing.assert_allclose(list(errors.values()), 0, atol=1e-6)


def test_curved_seam_uses_full_support_curvature_and_converged_quadrature():
    measured = compare(
        geometry="curved",
        quality="reference",
        dispersive=True,
        smoothing="farjadpour_diagonal",
    )
    # Curved volumes use adaptive quadrature, unlike the exactly integrated
    # flat-stripe test. A half-support curvature gate gave O(0.1) epsilon errors.
    assert max(measured["epsilon_max_difference"].values()) < 0.005
    for errors in measured["field_max_difference"].values():
        assert max(errors.values()) < 1e-4
