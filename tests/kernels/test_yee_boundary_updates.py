"""Contracts for the canonical complete Yee representation and PEC masks."""

from __future__ import annotations

import numpy as np

import beamz as bz
from beamz.design.discretization import MaterialGrid
from beamz.design.grid import RectilinearGrid
from beamz.devices._boundary_compile import (
    compile_metallic_masks,
    resolve_metallic_edges,
)
from beamz.devices.boundaries import (
    PEC,
    PML,
    Periodic,
    normalize_boundaries,
    periodic_storage_axes,
    validate_boundary_compatibility,
)
from beamz.lattice import build_h_boundary_views_for_e_3d, component_shape_3d
from beamz.simulation.compile import _compile_derivative_metrics
from beamz.simulation.kernels import apply_post_source_boundaries
from tests.utils import compiled_grid


def _fields(shape=(4, 5, 6)):
    return compiled_grid(
        np.ones(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
        np.ones(shape, dtype=np.float32),
        resolution=1.0,
    )


def test_complete_yee_shapes_retain_every_domain_wall():
    fields = _fields()
    expected = {
        "Ex": (5, 6, 6),
        "Ey": (5, 5, 7),
        "Ez": (4, 6, 7),
        "Hx": (4, 5, 7),
        "Hy": (4, 6, 6),
        "Hz": (5, 5, 6),
    }

    for component, shape in expected.items():
        assert component_shape_3d(component, (4, 5, 6)) == shape
        assert getattr(fields, component).shape == shape


def test_default_boundary_is_explicit_six_sided_pec():
    boundaries = normalize_boundaries([])

    assert boundaries == (PEC(),)
    assert resolve_metallic_edges(boundaries, is_3d=True) == {
        "left",
        "right",
        "bottom",
        "top",
        "front",
        "back",
    }


def test_complete_yee_masks_cover_both_sides_of_every_pec_axis():
    fields = _fields()
    masks = compile_metallic_masks(
        fields.component_shapes, fields.material_grid.shape, [PEC()]
    )

    # Tangential E and normal H samples are constrained on both faces of each axis.
    for component in ("Ex", "Ey", "Hz"):
        assert np.asarray(masks[component])[0].all()
        assert np.asarray(masks[component])[-1].all()
    for component in ("Ex", "Ez", "Hy"):
        assert np.asarray(masks[component])[:, 0].all()
        assert np.asarray(masks[component])[:, -1].all()
    for component in ("Ey", "Ez", "Hx"):
        assert np.asarray(masks[component])[:, :, 0].all()
        assert np.asarray(masks[component])[:, :, -1].all()


def test_partial_pec_only_masks_selected_faces():
    fields = _fields()
    masks = compile_metallic_masks(
        fields.component_shapes,
        fields.material_grid.shape,
        [PEC(edges=("left", "top"))],
    )

    assert np.asarray(masks["Ez"])[:, :, 0].all()
    assert np.asarray(masks["Ez"])[:, -1, :].all()
    assert not np.asarray(masks["Ez"])[:, :-1, -1].any()
    assert not np.asarray(masks["Ez"])[:, 0, 1:].any()


def test_absorbing_boundary_opens_matching_pec_faces():
    boundaries = [PEC(), PML(edges=("left", "right"))]

    assert resolve_metallic_edges(boundaries, is_3d=True) == {
        "bottom",
        "top",
        "front",
        "back",
    }


def test_periodic_boundary_owns_paired_axes_and_rejects_face_conflicts():
    boundaries = [Periodic(axes=("x", "y")), PML(edges=("front", "back"))]

    validate_boundary_compatibility(boundaries, is_3d=True)
    assert periodic_storage_axes(boundaries, is_3d=True) == {1, 2}
    assert resolve_metallic_edges(boundaries, is_3d=True) == set()

    with np.testing.assert_raises_regex(ValueError, "conflicting edges"):
        validate_boundary_compatibility(
            [Periodic(axes="x"), PML(edges="left")], is_3d=True
        )


def test_periodic_h_ghosts_wrap_opposite_cell_in_3d():
    hz = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    hx = np.zeros((2, 3, 5), dtype=np.float32)
    hy = np.zeros((2, 4, 4), dtype=np.float32)

    views = build_h_boundary_views_for_e_3d(hx, hy, hz, periodic_axes=frozenset({2}))

    np.testing.assert_array_equal(np.asarray(views["hz_x"][:, :, 0]), hz[:, :, -1])
    np.testing.assert_array_equal(np.asarray(views["hz_x"][:, :, -1]), hz[:, :, 0])


def test_periodic_constraint_identifies_duplicate_yee_nodes():
    values = (
        np.arange(3 * 4, dtype=np.float32).reshape(3, 4),
        np.arange(2 * 5, dtype=np.float32).reshape(2, 5),
        np.arange(3 * 5, dtype=np.float32).reshape(3, 5),
    )
    masks = tuple(np.zeros_like(value, dtype=bool) for value in values)
    logical = {"Ex": (3, 4), "Ey": (2, 5), "Ez": (3, 5)}

    ex, ey, ez = apply_post_source_boundaries(
        values,
        masks,
        components=("Ex", "Ey", "Ez"),
        periodic_axes=frozenset({0, 1}),
        material_shape=(2, 4),
        logical_shapes=logical,
    )

    np.testing.assert_allclose(np.asarray(ex[0]), np.asarray(ex[-1]))
    np.testing.assert_allclose(np.asarray(ey[:, 0]), np.asarray(ey[:, -1]))
    np.testing.assert_allclose(np.asarray(ez[0]), np.asarray(ez[-1]))
    np.testing.assert_allclose(np.asarray(ez[:, 0]), np.asarray(ez[:, -1]))


def test_mixed_periodic_xy_and_cpml_z_compile_on_small_3d_grid():
    shape = (8, 6, 6)
    ones = np.ones(shape, dtype=np.float32)
    zeros = np.zeros(shape, dtype=np.float32)
    sim = bz.Simulation(
        material_grid=MaterialGrid(ones, zeros, ones, 0.1 * bz.um, shape),
        boundaries=[
            bz.Periodic(axes=("x", "y")),
            bz.PML(edges=("front", "back"), thickness=0.1 * bz.um, formulation="cpml"),
        ],
        run_time=1e-16,
    )

    program = sim.compile(num_steps=1, backend="jax")

    assert program.boundary.periodic_axes == {1, 2}
    assert program.boundary.cpml.enabled
    assert not program.boundary.cpml.metallic_edges

    state = sim.initial_state()
    state = state._replace(ez=state.ez.at[:, :, 0].set(1.0).at[:, :, -1].set(3.0))
    next_state = sim.step(state, backend="jax")
    np.testing.assert_allclose(
        np.asarray(next_state.ez[:, :, 0]),
        np.asarray(next_state.ez[:, :, -1]),
    )


def test_periodic_boundary_maps_physical_axes_in_2d_planes():
    assert periodic_storage_axes(
        [Periodic(axes=("x", "z"))], is_3d=False, plane_2d="xz"
    ) == {0, 1}

    with np.testing.assert_raises_regex(ValueError, "not active"):
        periodic_storage_axes([Periodic(axes="y")], is_3d=False, plane_2d="xz")


def test_periodic_boundary_executes_for_both_2d_polarizations():
    shape = (4, 5)
    ones = np.ones(shape, dtype=np.float32)
    zeros = np.zeros(shape, dtype=np.float32)

    for polarization in ("tm", "te"):
        sim = bz.Simulation(
            material_grid=MaterialGrid(
                ones,
                zeros,
                ones,
                0.1 * bz.um,
                shape,
                polarization=polarization,
            ),
            polarization=polarization,
            boundaries=[bz.Periodic(axes=("x", "y"))],
            run_time=1e-14,
        )
        state = sim.initial_state()
        if polarization == "tm":
            state = state._replace(ez=state.ez.at[0, :].set(1.0).at[-1, :].set(3.0))
        else:
            state = state._replace(
                ex=state.ex.at[0, :].set(1.0).at[-1, :].set(3.0),
                ey=state.ey.at[:, 0].set(2.0).at[:, -1].set(4.0),
            )

        next_state = sim.step(state, backend="jax")
        if polarization == "tm":
            np.testing.assert_allclose(next_state.ez[0, :], next_state.ez[-1, :])
            np.testing.assert_allclose(next_state.ez[:, 0], next_state.ez[:, -1])
        else:
            np.testing.assert_allclose(next_state.ex[0, :], next_state.ex[-1, :])
            np.testing.assert_allclose(next_state.ey[:, 0], next_state.ey[:, -1])


def test_periodic_2d_step_matches_center_of_tiled_domain_for_te_and_tm():
    ny, nx = 4, 5

    def make_sim(shape, polarization, *, periodic):
        ones = np.ones(shape, dtype=np.float32)
        zeros = np.zeros(shape, dtype=np.float32)
        return bz.Simulation(
            material_grid=MaterialGrid(
                ones,
                zeros,
                ones,
                0.1 * bz.um,
                shape,
                polarization=polarization,
            ),
            polarization=polarization,
            boundaries=[bz.Periodic(axes=("x", "y"))] if periodic else [],
            run_time=1e-14,
        )

    def seeded_state(sim, polarization, cells_y, cells_x):
        state = sim.initial_state()
        if polarization == "tm":
            rows = np.arange(cells_y + 1, dtype=float)[:, None]
            columns = np.arange(cells_x + 1, dtype=float)[None, :]
            ez = np.sin(2 * np.pi * columns / nx) + 0.4 * np.cos(2 * np.pi * rows / ny)
            return state._replace(ez=state.ez.at[:].set(ez))

        node_rows = np.arange(cells_y + 1, dtype=float)[:, None]
        cell_columns = (np.arange(cells_x, dtype=float) + 0.5)[None, :]
        cell_rows = (np.arange(cells_y, dtype=float) + 0.5)[:, None]
        node_columns = np.arange(cells_x + 1, dtype=float)[None, :]
        ex = np.sin(2 * np.pi * node_rows / ny) + 0.3 * np.cos(
            2 * np.pi * cell_columns / nx
        )
        ey = 0.2 * np.cos(2 * np.pi * cell_rows / ny) + np.sin(
            2 * np.pi * node_columns / nx
        )
        return state._replace(ex=state.ex.at[:].set(ex), ey=state.ey.at[:].set(ey))

    slices = {
        "tm": {
            "Ez": (slice(ny, 2 * ny + 1), slice(nx, 2 * nx + 1)),
            "Hx": (slice(ny, 2 * ny), slice(nx, 2 * nx + 1)),
            "Hy": (slice(ny, 2 * ny + 1), slice(nx, 2 * nx)),
        },
        "te": {
            "Ex": (slice(ny, 2 * ny + 1), slice(nx, 2 * nx)),
            "Ey": (slice(ny, 2 * ny), slice(nx, 2 * nx + 1)),
            "Hz": (slice(ny, 2 * ny), slice(nx, 2 * nx)),
        },
    }
    for polarization in ("tm", "te"):
        periodic_sim = make_sim((ny, nx), polarization, periodic=True)
        tiled_sim = make_sim((3 * ny, 3 * nx), polarization, periodic=False)
        periodic_state = seeded_state(periodic_sim, polarization, ny, nx)
        tiled_state = seeded_state(tiled_sim, polarization, 3 * ny, 3 * nx)

        periodic_next = periodic_sim.step(periodic_state, backend="jax")
        tiled_next = tiled_sim.step(tiled_state, backend="jax")
        for component, region in slices[polarization].items():
            np.testing.assert_allclose(
                np.asarray(getattr(periodic_next, component.lower())),
                np.asarray(getattr(tiled_next, component.lower())[region]),
                rtol=1e-6,
                atol=1e-6,
            )


def test_periodic_xy_cpml_z_step_matches_center_of_tiled_3d_domain():
    nz, ny, nx = 8, 4, 5

    def make_sim(shape, *, periodic):
        ones = np.ones(shape, dtype=np.float32)
        zeros = np.zeros(shape, dtype=np.float32)
        boundaries = [
            bz.PML(
                edges=("front", "back"),
                thickness=0.1 * bz.um,
                formulation="cpml",
            )
        ]
        if periodic:
            boundaries.insert(0, bz.Periodic(axes=("x", "y")))
        return bz.Simulation(
            material_grid=MaterialGrid(ones, zeros, ones, 0.1 * bz.um, shape),
            boundaries=boundaries,
            run_time=1e-14,
        )

    def seeded_state(sim, cells_y, cells_x):
        state = sim.initial_state()
        z_nodes = np.arange(nz + 1, dtype=float)[:, None, None]
        z_cells = (np.arange(nz, dtype=float) + 0.5)[:, None, None]
        y_nodes = np.arange(cells_y + 1, dtype=float)[None, :, None]
        y_cells = (np.arange(cells_y, dtype=float) + 0.5)[None, :, None]
        x_nodes = np.arange(cells_x + 1, dtype=float)[None, None, :]
        x_cells = (np.arange(cells_x, dtype=float) + 0.5)[None, None, :]
        ex = (
            0.1 * np.cos(np.pi * z_nodes / nz)
            + np.sin(2 * np.pi * y_nodes / ny)
            + 0.2 * np.cos(2 * np.pi * x_cells / nx)
        )
        ey = (
            0.2 * np.sin(np.pi * z_nodes / nz)
            + 0.3 * np.cos(2 * np.pi * y_cells / ny)
            + np.sin(2 * np.pi * x_nodes / nx)
        )
        ez = (
            0.3 * np.cos(np.pi * z_cells / nz)
            + 0.4 * np.sin(2 * np.pi * y_nodes / ny)
            + np.cos(2 * np.pi * x_nodes / nx)
        )
        return state._replace(
            ex=state.ex.at[:].set(ex),
            ey=state.ey.at[:].set(ey),
            ez=state.ez.at[:].set(ez),
        )

    periodic_sim = make_sim((nz, ny, nx), periodic=True)
    tiled_sim = make_sim((nz, 3 * ny, 3 * nx), periodic=False)
    periodic_next = periodic_sim.step(seeded_state(periodic_sim, ny, nx), backend="jax")
    tiled_next = tiled_sim.step(seeded_state(tiled_sim, 3 * ny, 3 * nx), backend="jax")

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        periodic_value = getattr(periodic_next, component.lower())
        y_count, x_count = periodic_value.shape[1:]
        tiled_center = getattr(tiled_next, component.lower())[
            :, ny : ny + y_count, nx : nx + x_count
        ]
        np.testing.assert_allclose(
            np.asarray(periodic_value),
            np.asarray(tiled_center),
            rtol=1e-6,
            atol=1e-6,
        )


def test_periodic_rectilinear_metric_uses_distance_across_seam():
    grid = RectilinearGrid(
        x_edges=np.array([0.0, 0.1, 0.3, 0.7]),
        y_edges=np.array([0.0, 0.3, 0.8]),
        z_edges=np.array([0.0, 1.0]),
    )
    shape = (2, 3)
    material_grid = MaterialGrid(1.0, 0.0, 1.0, 0.1, shape, grid=grid)

    metrics = _compile_derivative_metrics(material_grid, periodic_axes=frozenset({1}))

    expected_seam_inverse = 2.0 / (0.4 + 0.1)
    assert float(metrics.h_to_e_x[0]) == expected_seam_inverse
    assert float(metrics.h_to_e_x[-1]) == expected_seam_inverse
