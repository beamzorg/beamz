"""A mirror-symmetric material/grid must yield symmetric unprojected mode magnitudes."""

import numpy as np
import pytest

from beamz import LIGHT_SPEED, GaussianPulse, ModeSource, ModeSpec, Simulation
from beamz.design.discretization import MaterialGrid
from beamz.design.grid import RectilinearGrid
from beamz.lattice import (
    component_shape_3d,
    sample_voxel_grid_at_e_component_3d_centered,
)


def _symmetric_edges(half_core, half_extent, core_cells, clad_cells):
    return np.concatenate(
        (
            np.linspace(-half_extent, -half_core, clad_cells + 1)[:-1],
            np.linspace(-half_core, half_core, core_cells + 1),
            np.linspace(half_core, half_extent, clad_cells + 1)[1:],
        )
    )


@pytest.mark.parametrize("polarization", ["te", "tm"])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("direction", ["+", "-"])
def test_unforced_3d_mode_magnitudes_respect_physical_mirrors(
    axis, direction, polarization
):
    dl = 0.03e-6
    y = _symmetric_edges(0.215e-6, 0.7e-6, 15, 17) + 0.7e-6
    z = _symmetric_edges(0.11e-6, 0.6e-6, 8, 17) + 0.6e-6
    yc, zc = 0.5 * (y[:-1] + y[1:]), 0.5 * (z[:-1] + z[1:])
    plane = np.where(
        (abs(zc[:, None] - 0.6e-6) < 0.11e-6) & (abs(yc[None, :] - 0.7e-6) < 0.215e-6),
        3.48**2,
        1.444**2,
    )
    eps = np.repeat(plane[..., None], 4, axis=2).astype(np.float32)
    edges = (np.linspace(0, 4 * dl, 5), y, z)
    center, size = (0.06e-6, 0.7e-6, 0.6e-6), (0, 1.4e-6, 1.2e-6)
    if axis == "y":
        eps = eps.swapaxes(1, 2)
        edges = edges[1], edges[0], edges[2]
        center, size = (center[1], center[0], center[2]), (size[1], size[0], size[2])
    elif axis == "z":
        eps = eps.transpose(2, 0, 1)
        edges = edges[1], edges[2], edges[0]
        center, size = (center[1], center[2], center[0]), (size[1], size[2], size[0])
    material = MaterialGrid(
        eps,
        0.0,
        1.0,
        dl,
        eps.shape,
        grid=RectilinearGrid(*edges),
        yee_materials={
            "eps_" + axis: np.asarray(
                sample_voxel_grid_at_e_component_3d_centered(eps, "E" + axis)
            )
            for axis in "xyz"
        }
        | {
            "mu_h" + axis: np.ones(component_shape_3d("H" + axis, eps.shape))
            for axis in "xyz"
        },
    )
    frequency = LIGHT_SPEED / 1.55e-6
    source = ModeSource(
        center=center,
        size=size,
        direction=direction,
        mode_spec=ModeSpec(num_modes=1, target_neff=3.47, polarization=polarization),
        source_time=GaussianPulse(freq0=frequency, fwidth=frequency / 10),
    )
    sim = Simulation(material_grid=material, sources=[source], run_time=1e-15)
    modes = source.solve_modes(sim)
    assert not modes.diagnostics[0]["force_symmetry_requested"]
    for field in (*modes.e_fields[0, 0], *modes.h_fields[0, 0]):
        magnitude = abs(field)
        norm = np.linalg.norm(magnitude)
        if norm > 1e-10 * np.linalg.norm(modes.e_fields[0, 0]):
            for dimension in (0, 1):
                mismatch = (
                    np.linalg.norm(magnitude - np.flip(magnitude, dimension)) / norm
                )
                assert mismatch < 1e-8
