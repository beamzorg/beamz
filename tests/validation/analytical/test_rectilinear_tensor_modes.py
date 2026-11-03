"""Guided tensor modes on physical rectilinear Yee supports, all orientations."""

from functools import lru_cache

import numpy as np
import pytest

import beamz as bz
from beamz.design.discretization import MaterialGrid
from beamz.design.raster import (
    Box,
    Grid,
    Material,
    Object,
    RasterOptions,
    Scene,
    rasterize,
)
from beamz.devices.modes.plane import solve_mode_plane_3d
from beamz.devices.sources.planar_tfsf import (
    advance_incident_e_3d,
    advance_incident_h_3d,
    build_incident_3d_phasor_state,
)
from tests.validation.analytical.test_tensor_port_maxwell import field_profile


@lru_cache(maxsize=12)
def rectilinear_guide(axis="x", cells=20, axis_uniform=False):
    u = np.linspace(0, 1, cells + 1)
    edges = (
        np.linspace(0, 4e-6, 2 * cells + 1),
        2e-6 * (u + 0.10 * np.sin(2 * np.pi * u)),
        2e-6 * (u + 0.06 * np.sin(2 * np.pi * u)),
    )
    if axis_uniform:
        edges = (1.3 * edges[0], 2e-6 * u, 2e-6 * u)
    old_axes = {"x": "xyz", "y": "yxz", "z": "zxy"}[axis]
    order = [old_axes.index(a) for a in "xyz"]
    epsilon = np.array([[6.0, 0.2, 0.1], [0.2, 5.0, 0.3], [0.1, 0.3, 4.0]])[
        np.ix_(order, order)
    ]
    low = np.array([0.0, 0.7, 0.7])[order] * 1e-6
    high = np.array([5.2 if axis_uniform else 4.0, 1.3, 1.3])[order] * 1e-6
    material = MaterialGrid.from_raster_result(
        rasterize(
            Scene(
                (Material(), Material(epsilon_r=tuple(map(tuple, epsilon)))),
                (Object(Box(tuple(low), tuple(high)), 1),),
            ),
            Grid(*(edges[i] for i in order)),
            options=RasterOptions(smoothing="farjadpour_full"),
        )
    )
    dt = material.grid.cfl_time_step(0.6)
    return material, dt, bz.LIGHT_SPEED / 1.55e-6


def rectilinear_mode(axis="x", direction="+", cells=20, axis_uniform=False):
    material, dt, f = rectilinear_guide(axis, cells, axis_uniform)
    plane = cells // 2
    return solve_mode_plane_3d(
        material.permittivity,
        material.permeability,
        material_tensors=material.tensors,
        yee_materials=material.yee_materials,
        yee_tensors=material.yee_tensors,
        frequency=f,
        resolution=material.resolution,
        dt=dt,
        axis=axis,
        direction=direction + axis,
        grid_shape=material.shape,
        center=(1e-6,) * 3,
        width=1.6e-6,
        height=1.6e-6,
        plane_index=plane,
        offset_index=plane - 1 if direction == "+" else plane + 1,
        mode_index=0,
        polarization=None,
        target_neff=2.0,
        num_modes=2,
        snapped_region=None,
        # Keep the full reference supercell independent of the launch aperture.
        solver_padding_cells=cells,
        grid=material.grid,
    )


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("direction", ["+", "-"])
@pytest.mark.parametrize("axis_uniform", [False, True])
def test_rectilinear_tensor_eigenmode_matches_actual_leapfrog(
    axis, direction, axis_uniform
):
    material, dt, f = rectilinear_guide(axis, axis_uniform=axis_uniform)
    mode = rectilinear_mode(axis, direction, axis_uniform=axis_uniform)
    sim = bz.Simulation(
        material_grid=material, time=np.arange(3) * dt, setup_device="cpu"
    )
    fields = sim.compile(backend="jax").grid
    for backward in (False, True):
        state = build_incident_3d_phasor_state(
            field_profile(mode, f, backward=backward),
            fields,
            resolution=material.resolution,
            t_e=0,
            t_h=-dt / 2,
            masked=False,
            max_shift=8,
        )
        h = advance_incident_h_3d(fields, state, dt, resolution=material.resolution)
        e = advance_incident_e_3d(fields, state, h, dt, resolution=material.resolution)
        selection = [slice(2, -2)] * 3
        selection["zyx".index(axis)] = slice(8, 12)
        for name, value in {**e, **h}.items():
            target = state[name] * np.exp(1j * 2 * np.pi * f * dt)
            relative = np.linalg.norm(
                (value - target)[tuple(selection)]
            ) / np.linalg.norm(target[tuple(selection)])
            assert relative < 3e-6, (axis, direction, backward, name, relative)
    assert mode.diagnostics["tensor_ampere_residual"] < 1e-7
