"""Grid-aligned interfaces still need smoothed native Yee coefficients."""

import numpy as np

import beamz as bz
from beamz.design.raster import RasterOptions


def test_full_smoothing_keeps_tangential_interface_average():
    design = bz.Design(background=bz.Material(2.0))
    design += bz.Rectangle(
        position=(-0.8e-6, -0.2e-6),
        width=1.6e-6,
        height=0.4e-6,
        material=bz.Material(12.0),
    )
    program = bz.Simulation(
        domain=(1.6e-6,) * 2,
        design=design,
        grid_spec=bz.GridSpec.uniform(resolution=0.1e-6),
        raster_options=RasterOptions(smoothing="farjadpour_full"),
        run_time=1e-15,
        polarization="te",
    ).compile(backend="jax")
    assert not program.grid.material_grid.uses_full_permittivity
    # Tangential Yee samples straddle equal volumes of eps=2 and eps=12,
    # although every cell center lies entirely in one material.
    np.testing.assert_allclose(
        np.asarray(program.grid.eps_x)[[6, 10], 8], [7, 7], rtol=1e-6
    )
