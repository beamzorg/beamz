"""Rasterize dispersive material ownership on electric Yee supports.

A separate indicator raster for each medium preserves painter order and native
geometry integration. Both epsilon-infinity and residues are volume averaged;
static harmonic smoothing is not a valid broadband interface model.
"""

from dataclasses import replace

import numpy as np

from beamz.design.discretization import MaterialGrid
from beamz.design.dispersion import PoleResidue
from beamz.design.materials import Material

from .engine import RasterOptions, rasterize


def rasterize_dispersion(
    scene, grid, *, kind, polarization, quality, resolution, cache_directory
):
    regions = []
    for index, medium in enumerate(scene.materials):
        if not isinstance(medium, PoleResidue):
            continue
        indicators = tuple(
            Material(1.0, conductivity=float(i == index))
            for i in range(len(scene.materials))
        )
        result = rasterize(
            replace(scene, materials=indicators),
            grid,
            options=RasterOptions(
                quality=quality,
                smoothing="volume",
                components="all" if kind == "3d" else f"two_dimensional_{polarization}",
            ),
            cache_directory=cache_directory,
        )
        material_grid = MaterialGrid.from_raster_result(
            result,
            dimensions=3 if kind == "3d" else 2,
            polarization=polarization,
            resolution=resolution,
        )
        supports = {
            "E" + axis: np.clip(
                np.asarray(material_grid.yee_materials["sig_" + axis]), 0.0, 1.0
            )
            for axis in "xyz"
            if "sig_" + axis in material_grid.yee_materials
        }
        regions.append((medium, supports))
    return tuple(regions)
