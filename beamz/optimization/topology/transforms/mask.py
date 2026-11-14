"""Build the topology optimization mask."""

from __future__ import annotations

import numpy as np

from beamz.design.core import Design
from beamz.design.materials import Material
from beamz.design.structures import Structure


def create_optimization_mask(grid, region_structure):
    """
    Helper to create a boolean mask from a structure on a grid.
    Uses rasterization to ensure exact alignment with how structures are mapped to the grid.
    """
    # Create temp design to rasterize mask exactly as grid does
    temp_design = Design(
        width=grid.width, height=grid.height, material=Material(permittivity=1.0)
    )

    # Set to a distinct permittivity to detect it without mutating the input.
    marker = Material(permittivity=2.0)
    if not isinstance(region_structure, Structure):
        raise TypeError("Optimization regions must be canonical Structure objects.")
    temp_design += region_structure.with_material(marker)

    # Use the production Rust rasterizer so optimization masks follow the same
    # geometry classification as the solver material grid.
    temp_grid = temp_design.rasterize(resolution=grid.dx)

    # Mask is where permittivity > background
    # Use a safe threshold to include any partial fill
    mask = np.asarray(temp_grid.permittivity) > 1.001

    return mask
