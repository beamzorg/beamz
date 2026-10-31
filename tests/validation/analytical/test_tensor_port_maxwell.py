"""Tensor eigenfields checked against the actual source-free leapfrog update."""

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
from beamz.devices.sources.specs import FieldProfile3D


@lru_cache(maxsize=12)
def guide(axis="x", cells=20, cross=True):
    u = 1e-6
    axes = {"x": "xyz", "y": "yxz", "z": "zxy"}[axis]
    epsilon = np.array([[6.0, 0.2, 0.1], [0.2, 5.0, 0.3], [0.1, 0.3, 4.0]])
    if not cross:
        epsilon = np.diag(np.diag(epsilon))
    order = [axes.index(a) for a in "xyz"]
    epsilon = epsilon[np.ix_(order, order)]
    length = np.array([4.0, 2.0, 2.0])[order] * u
    low = np.array([0.0, 0.7, 0.7])[order] * u
    high = np.array([4.0, 1.3, 1.3])[order] * u
    shape = np.array([2 * cells, cells, cells])[order]
    scene = Scene(
        (Material(), Material(epsilon_r=tuple(map(tuple, epsilon)))),
        (Object(Box(tuple(low), tuple(high)), 1),),
    )
    materials = MaterialGrid.from_raster_result(
        rasterize(
            scene,
            Grid.uniform((0, 0, 0), tuple(length), tuple(shape)),
            options=RasterOptions(smoothing="farjadpour_full"),
        )
    )
    d = 2 * u / cells
    dt = 0.4 * d / bz.LIGHT_SPEED
    return materials, d, dt, bz.LIGHT_SPEED / (1.55 * u)


def mode_for(axis="x", direction="+", cells=20, cross=True):
    m, d, dt, f = guide(axis, cells, cross)
    plane = round(1e-6 / d)
    return solve_mode_plane_3d(
        np.asarray(m.permittivity),
        np.asarray(m.permeability),
        material_tensors=m.tensors,
        yee_materials=m.yee_materials,
        yee_tensors=m.yee_tensors,
        frequency=f,
        resolution=d,
        dt=dt,
        axis=axis,
        direction=direction + axis,
        grid_shape=m.shape,
        center=(1e-6,) * 3,
        width=1.6e-6,
        height=1.6e-6,
        plane_index=plane,
        offset_index=plane - 1 if direction == "+" else plane + 1,
        mode_index=0,
        polarization=None,
        target_neff=2.0,
        num_modes=3,
        snapped_region=None,
        # The Maxwell/MPB reference uses the full transverse supercell.
        solver_padding_cells=cells,
    )


def field_profile(mode, frequency, *, backward=False):
    sign = 1.0 if mode.direction[0] == "+" else -1.0
    if backward:
        sign = -sign
    return FieldProfile3D(
        components=mode.backward_profiles if backward else mode.profiles,
        indices=mode.component_indices,
        axis=mode.axis,
        direction_sign=sign,
        omega=2 * np.pi * frequency,
        k_axis=sign * mode.k_num_axis,
        phase_ref_coord=mode.phase_reference_coord,
        phase_plane_coord=mode.phase_plane_coord,
        power_weights=mode.integration_weights,
    )


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("direction", ["+", "-"])
def test_tensor_port_satisfies_actual_leapfrog_update(axis, direction):
    m, d, dt, f = guide(axis)
    mode = mode_for(axis, direction)
    assert mode.diagnostics["tensor_ampere_residual"] < 1e-7
    assert mode.diagnostics["tensor_faraday_residual"] < 1e-7
    assert abs(mode.neff - 1.7799819508924184) < 2e-6
    sim = bz.Simulation(material_grid=m, time=np.arange(3) * dt, setup_device="cpu")
    fields = sim.compile().grid
    # Check both the computed eigenmode and its reciprocal time-reversed partner.
    for backward in (False, True):
        state = build_incident_3d_phasor_state(
            field_profile(mode, f, backward=backward),
            fields,
            resolution=d,
            t_e=0,
            t_h=-0.5 * dt,
            masked=False,
            max_shift=8,
        )
        h = advance_incident_h_3d(fields, state, dt, resolution=d)
        e = advance_incident_e_3d(fields, state, h, dt, resolution=d)
        selection = [slice(2, -2)] * 3
        selection[{"z": 0, "y": 1, "x": 2}[axis]] = slice(8, 12)
        selection = tuple(selection)
        for name, values in {**e, **h}.items():
            target = state[name] * np.exp(1j * 2 * np.pi * f * dt)
            relative = np.linalg.norm((values - target)[selection]) / np.linalg.norm(
                target[selection]
            )
            assert relative < 3e-6, (axis, direction, backward, name, relative)
