"""A 2D field plane must agree with independent Yee-line acquisitions."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from beamz import FieldMonitor
from tests.topology_case import make_problem


@pytest.mark.parametrize(
    "polarization,components", [("tm", ("Ez", "Hx", "Hy")), ("te", ("Ex", "Ey", "Hz"))]
)
def test_field_plane_matches_lines_and_plot(polarization, components):
    problem = make_problem(polarization=polarization, target_mode=0, run_time=80e-15)
    sim = problem.simulation
    frequency = sim.monitors[0].freqs[0]
    common = dict(freqs=[frequency, frequency * 1.03], fields=components)
    plane = FieldMonitor(
        center=(3e-6, 2e-6, 0), size=(4e-6, 2e-6, 0), name="plane", **common
    )
    row = FieldMonitor(center=(3e-6, 2e-6, 0), size=(4e-6, 0, 0), name="row", **common)
    upper_row = FieldMonitor(
        center=(3e-6, 2.1e-6, 0), size=(4e-6, 0, 0), name="upper_row", **common
    )
    data = sim.updated_copy(monitors=[plane, row, upper_row]).run(
        backend="jax", performance=False
    )
    region = data["plane"].sample_region
    assert region.axis_interval("x").size == pytest.approx(4e-6)
    assert region.axis_interval("y").size == pytest.approx(2e-6)
    for component in components:
        field = data["plane"].get_dft_component(component).reshape(2, 20, 40)
        assert np.isfinite(field).all() and np.max(np.abs(field)) > 0
        lower = data["row"].get_dft_component(component)
        upper = data["upper_row"].get_dft_component(component)
        # Uniform legacy line acquisitions expose native Yee samples. Average
        # their staggered coordinates independently onto the plane's cell centers.
        expected = (lower + upper) / 2 if component in {"Ez", "Hy", "Ex"} else lower
        if component in {"Ez", "Hx", "Ey"}:
            expected = (expected[:, :-1] + expected[:, 1:]) / 2
        else:
            expected = expected[:, :-1]
        np.testing.assert_allclose(field[:, 10, :], expected, rtol=3e-5, atol=1e-6)
    fig, ax = data.plot_field("plane", components[0], val="abs^2", frequency=frequency)
    assert ax.images[0].get_array().shape == (20, 40)
    np.testing.assert_allclose(ax.images[0].get_extent(), [1, 5, 1, 3])
    plt.close(fig)
