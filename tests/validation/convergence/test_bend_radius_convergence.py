"""Increasing bend radius recovers existing BeamZ slab and strip cases."""

import numpy as np
import pytest

from beamz.devices.modes import solve_grid
from tests.unit.modes.test_api import _strip
from tests.utils import slab_waveguide_neff_te


@pytest.mark.parametrize("case", ["slab", "strip"])
def test_bend_radius_recovers_existing_straight_case(case, validation_metrics):
    if case == "slab":
        # Same material/width/wavelength as test_slab_waveguide_modes.py (25 nm).
        x = np.linspace(-2.5, 2.5, 201)
        eps = np.where(abs((x[:-1] + x[1:]) / 2) < 0.3, 2.04**2, 1.444**2)[:, None]
        y, target, count = [0.0, 1.0], 1.94, 2
    else:
        eps, x, y = _strip()
        target, count = 2.5, 1
    args = dict(
        eps_xx=eps,
        x_edges=x,
        y_edges=y,
        wavelength=1.55,
        target_neff=target,
        num_modes=count,
    )
    straight = float(solve_grid(**args).n_eff.values[0, 0])
    radii = [100, 200, 400]
    indices = [
        float(solve_grid(**args, bend_radius=r).n_eff.values[0, 0]) for r in radii
    ]
    errors = np.abs(np.asarray(indices) - straight)
    assert np.all(np.diff(errors) < 0)
    order = float(np.log2(errors[0] / errors[-1]) / 2)
    validation_metrics.check_lower(
        "radius convergence order",
        order,
        0.8,
        metadata={"radii_um": radii, "indices": indices, "straight": straight},
    )
    near_straight = float(solve_grid(**args, bend_radius=1e8).n_eff.values[0, 0])
    validation_metrics.check_upper(
        "large-radius index error", abs(near_straight - straight), 1e-8
    )
    if case == "slab":
        analytical = slab_waveguide_neff_te(2.04, 1.444, 0.6, 1.55)
        assert analytical is not None
        validation_metrics.check(
            "large-radius slab analytical index",
            near_straight,
            analytical,
            tolerance="waveguide_neff",
            resolution="25 nm",
        )
