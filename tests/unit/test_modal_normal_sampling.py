"""Recover known traveling waves after the actual 3D monitor sampler acts on them."""

from types import SimpleNamespace

import numpy as np
import pytest

from beamz.analysis.modal_projection.colocation import _normal_mode_sampling_factors_3d
from beamz.analysis.mode_projection import _project_modal_coefficients_3d_group
from beamz.design.grid import RectilinearGrid
from beamz.devices._placement import snap_plane_region_grid
from beamz.lattice import (
    compile_yee_plane_quadrature_3d,
    component_coordinates_rectilinear,
    component_shape_3d,
)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("fraction", [0.0, 0.23, 0.5])
def test_sampled_basis_recovers_bidirectional_waves(axis, fraction):
    edges = np.array([0, 0.9, 1.9, 3.0, 4.1, 5.2, 6.2, 7.1])
    grid = RectilinearGrid(edges, edges * 1.1, edges * 0.9)
    shape_zyx = tuple(reversed(grid.shape))
    axis_index = "xyz".index(axis)
    plane = grid.axis_edges(axis)[3] + fraction * grid.cell_widths(axis)[3]
    center = [3.5, 3.5, 3.5]
    center[axis_index] = plane
    size = [2.0, 2.0, 2.0]
    size[axis_index] = 0.0
    monitor = SimpleNamespace(center=tuple(center))
    sim = SimpleNamespace(coordinates=SimpleNamespace(grid=grid))
    components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    shapes = {c: component_shape_3d(c, shape_zyx) for c in components}
    region = snap_plane_region_grid(
        center=center, size=size, plane_normal=axis, grid=grid
    )
    quadrature = compile_yee_plane_quadrature_3d(
        center=center,
        size=size,
        normal_axis=axis,
        region=region,
        resolution=1.0,
        grid_shape=shape_zyx,
        component_shapes=shapes,
        grid=grid,
    )
    e_name, h_name = {"x": ("Ey", "Hz"), "y": ("Ez", "Hx"), "z": ("Ex", "Hy")}[axis]
    tangential = tuple(c for c in components if c[1].lower() != axis)
    amplitude = np.sqrt(2 / np.sum(quadrature.integration_weights))
    beta = 0.7
    forward, backward = 0.7 + 0.2j, 0.3 - 0.1j
    fields = {}
    for component in tangential:
        coords = component_coordinates_rectilinear(component, grid)[axis]
        along = np.ones(3, dtype=int)
        along["zyx".index(axis)] = len(coords)
        phase = beta * (coords - plane)
        if component == e_name:
            values = amplitude * (
                forward * np.exp(1j * phase) + backward * np.exp(-1j * phase)
            )
        elif component == h_name:
            values = amplitude * (
                forward * np.exp(1j * phase) - backward * np.exp(-1j * phase)
            )
        else:
            values = np.zeros(len(coords), dtype=complex)
        raw = np.broadcast_to(values.reshape(tuple(along)), shapes[component]).reshape(
            -1
        )
        indices, weights = quadrature.plan(component)
        fields[component] = np.sum(raw[indices] * weights, axis=1)
    bases = []
    for direction in (1, -1):
        factors = _normal_mode_sampling_factors_3d(
            sim,
            monitor,
            tangential,
            axis=axis,
            wave_number=beta,
            direction_sign=direction,
        )
        bases.append(
            {
                component: np.full(
                    quadrature.point_count,
                    amplitude
                    * (
                        1
                        if component == e_name
                        else direction
                        if component == h_name
                        else 0
                    )
                    * factors[component],
                    dtype=complex,
                )
                for component in tangential
            }
        )
    projection = {
        "components": tangential,
        "axis": axis,
        "direction_sign": 1.0,
        "integration_weights": quadrature.integration_weights,
        "mode_components": bases[0],
        "mode_components_bwd": bases[1],
    }
    recovered, residual, _, _ = _project_modal_coefficients_3d_group(
        fields, [projection]
    )
    np.testing.assert_allclose(
        recovered[0], [forward, backward], rtol=1e-12, atol=1e-12
    )
    assert residual < 1e-12
