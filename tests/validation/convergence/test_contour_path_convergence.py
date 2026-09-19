"""Analytical convergence gates for the restricted contour-path implementation."""

import numpy as np
import pytest

from tests.validation.convergence._contour_path_mie import cylinder_error, mie_hz
from tests.validation.convergence.test_full_tensor_fdtd_convergence import (
    _analytical_interface_error,
)


def test_mie_oracle_reduces_to_incident_wave_without_contrast():
    x, y = np.meshgrid(np.linspace(-0.8, 0.8, 15), np.linspace(-0.9, 0.9, 17))
    np.testing.assert_allclose(
        mie_hz(x, y, epsilon=1), np.exp(2j * np.pi * x), atol=2e-8
    )


@pytest.mark.parametrize("shift", [0.0, 0.5])
def test_contour_cylinder_converges_against_mie(shift):
    grids = (48, 96, 192)
    errors = [cylinder_error(n, "contour_path", shift=shift)[0] for n in grids]
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all(np.diff(errors) < 0), errors
    # The coarsest grid is pre-asymptotic; gate the finest observed order.
    assert 1.7 < orders[-1] < 2.4, (errors, orders)
    volume_error, _ = cylinder_error(grids[-1], "volume", shift=shift)
    assert errors[-1] < volume_error / 2


@pytest.mark.parametrize(
    "method", ["volume", "farjadpour_diagonal", "farjadpour_full", "contour_path"]
)
def test_identical_grid_fresnel_comparison_converges(method):
    grids = (192, 384, 768)
    errors = [
        _analytical_interface_error(
            n,
            anisotropic=False,
            smoothing=method,
            angle_degrees=27.0,
            interface_offset=-0.11e-6,
        )
        for n in grids
    ]
    order = np.polyfit(np.log(grids), np.log(errors), 1)[0] * -1
    assert 0.7 < order < 2.3, (errors, order)
    assert errors[0] > errors[1] > errors[2]
    if method == "contour_path":
        assert errors[-1] < 0.005
