import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

import beamz as bz
from beamz.analysis.plotting import (
    _plot_image,
    extract_axis_aligned_slice,
    plot_field_view,
)
from beamz.design import MaterialGrid
from beamz.design.raster import Grid, Material, Scene, rasterize


def test_generic_slice_and_field_primitives_preserve_axis_coordinates():
    values = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    section = extract_axis_aligned_slice(
        values,
        axis="y",
        step=0.5,
        position=1.0,
        lengths={"x": 3.0, "y": 2.5, "z": 2.0},
    )

    np.testing.assert_array_equal(section.values, values[:, 2, :])
    assert (section.vertical, section.horizontal) == ("z", "x")
    assert section.extent == (0.0, 3.0, 0.0, 2.0)

    fig, ax = plt.subplots()
    try:
        image, view = plot_field_view(ax, section.values * (1.0 + 1.0j), val="abs^2")
        np.testing.assert_allclose(image.get_array(), 2.0 * section.values**2)
        assert view.magnitude and view.power
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("extent", "location"),
    (
        ((0.0, 1.0, 0.0, 3.0), "right"),
        ((0.0, 3.0, 0.0, 1.0), "bottom"),
    ),
)
def test_field_plot_colorbar_follows_field_aspect_ratio(extent, location):
    fig, ax = _plot_image(np.ones((2, 2)), extent=extent, colorbar=True, show=False)
    try:
        colorbar_ax = fig.axes[-1]
        plot_bounds = ax.get_position()
        colorbar_bounds = colorbar_ax.get_position()
        if location == "right":
            assert colorbar_bounds.x0 > plot_bounds.x1
            assert colorbar_bounds.height > colorbar_bounds.width
        else:
            assert colorbar_bounds.y1 < plot_bounds.y0
            assert colorbar_bounds.width > colorbar_bounds.height
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            assert not ax.xaxis.label.get_window_extent(renderer).overlaps(
                colorbar_ax.get_tightbbox(renderer)
            )
    finally:
        plt.close(fig)


def test_3d_simulation_plot_uses_tidy_layout_cross_sections():
    design = bz.Design(background=bz.Material(2.25))
    design += bz.Box(
        center=(0.0, 0.0, 0.0),
        size=(1.0 * bz.um, 1.0 * bz.um, 0.4 * bz.um),
        material=bz.Material(12.0),
    )
    source = bz.ModeSource(
        center=(-1.0 * bz.um, 0.0, 0.0),
        size=(0.0, 1.0 * bz.um, 0.8 * bz.um),
        source_time=bz.GaussianPulse(
            freq0=bz.LIGHT_SPEED / (1.55 * bz.um), fwidth=2e13
        ),
        direction="+",
    )
    monitor = bz.FluxMonitor(
        center=(1.0 * bz.um, 0.0, 0.0),
        size=(0.0, 1.0 * bz.um, 0.8 * bz.um),
        freqs=[bz.LIGHT_SPEED / (1.55 * bz.um)],
        name="flux",
    )
    sim = bz.Simulation(
        domain=(4.0 * bz.um, 4.0 * bz.um, 2.0 * bz.um),
        design=design,
        sources=[source],
        monitors=[monitor],
        boundaries=[bz.PML(thickness=0.5 * bz.um)],
        resolution=0.5 * bz.um,
        time=np.array([0.0, 1e-15]),
    )

    fig, axes = sim.plot(z=0.0, y=0.0, show=False)

    try:
        assert len(axes) == 2
        assert len(fig.axes) == 2
        # Layout plots use vector geometry rather than a quantized material-grid
        # image, so curves and polygon edges remain smooth at any grid resolution.
        assert not axes[0].images
        assert not axes[1].images
        assert len(axes[0].patches) >= 6
        assert len(axes[1].patches) >= 6
        assert len(axes[0].lines) >= 2
        assert len(axes[1].lines) >= 2
        assert axes[0].get_xlim() == (-2.0, 2.0)
        assert axes[1].get_xlim() == (-2.0, 2.0)
        assert axes[0].get_ylim() == (-2.0, 2.0)
        assert axes[1].get_ylim() == (-1.0, 1.0)
        assert axes[0].get_xlabel() == "x (um)"
        assert axes[1].get_ylabel() == "z (um)"
    finally:
        plt.close(fig)

    fig, axes = sim.view3d(show=False)

    try:
        assert axes[0].get_title() == "cross section at z=0.00 (um)"
        assert axes[1].get_title() == "cross section at y=0.00 (um)"
    finally:
        plt.close(fig)


def test_3d_simulation_plot_renders_face_on_monitor_as_translucent_rectangle():
    design = bz.Design(background=bz.Material(2.25))
    monitor = bz.FluxMonitor(
        center=(0.0, 0.0, 0.0),
        size=(1.0 * bz.um, 0.8 * bz.um, 0.0),
        freqs=[bz.LIGHT_SPEED / (1.55 * bz.um)],
        name="top_view_flux",
    )
    sim = bz.Simulation(
        domain=(4.0 * bz.um, 4.0 * bz.um, 2.0 * bz.um),
        design=design,
        monitors=[monitor],
        boundaries=[bz.PML(thickness=0.5 * bz.um)],
        resolution=0.5 * bz.um,
        time=np.array([0.0, 1e-15]),
    )

    fig, axes = sim.plot(z=0.0, y=0.0, show=False)

    try:
        monitor_patches = [
            patch
            for patch in axes[0].patches
            if to_hex(patch.get_edgecolor()) == "#ff9800"
        ]
        assert len(monitor_patches) == 1
        patch = monitor_patches[0]
        assert patch.get_width() == pytest.approx(1.0)
        assert patch.get_height() == pytest.approx(0.8)
        assert to_hex(patch.get_facecolor()) == "#ff9800"
        assert patch.get_alpha() == pytest.approx(0.25)
    finally:
        plt.close(fig)


def test_3d_simulation_plot_renders_face_on_source_as_translucent_rectangle():
    design = bz.Design(background=bz.Material(2.25))
    source = bz.ModeSource(
        center=(0.0, 0.0, 0.0),
        size=(1.0 * bz.um, 0.8 * bz.um, 0.0),
        source_time=bz.GaussianPulse(
            freq0=bz.LIGHT_SPEED / (1.55 * bz.um), fwidth=2e13
        ),
        direction="+",
    )
    sim = bz.Simulation(
        domain=(4.0 * bz.um, 4.0 * bz.um, 2.0 * bz.um),
        design=design,
        sources=[source],
        boundaries=[bz.PML(thickness=0.5 * bz.um)],
        resolution=0.5 * bz.um,
        time=np.array([0.0, 1e-15]),
    )

    fig, axes = sim.plot(z=0.0, y=0.0, show=False)

    try:
        source_patches = [
            patch
            for patch in axes[0].patches
            if to_hex(patch.get_edgecolor()) == "#2ca02c"
        ]
        assert len(source_patches) == 1
        patch = source_patches[0]
        assert patch.get_width() == pytest.approx(1.0)
        assert patch.get_height() == pytest.approx(0.8)
        assert to_hex(patch.get_facecolor()) == "#2ca02c"
        assert patch.get_alpha() == pytest.approx(0.25)
        assert not [line for line in axes[0].lines if line.get_color() == "#2ca02c"]
    finally:
        plt.close(fig)


def test_3d_layout_plot_uses_antialiased_polygon_sections_without_compiling(
    monkeypatch,
):
    air = bz.Material(1.0)
    silicon = bz.Material(12.0)
    design = bz.Design(
        width=4 * bz.um, height=4 * bz.um, depth=2 * bz.um, background=air
    )
    design += bz.Ring(
        position=(2 * bz.um, 2 * bz.um, 0.9 * bz.um),
        inner_radius=0.35 * bz.um,
        outer_radius=0.8 * bz.um,
        depth=0.25 * bz.um,
        material=silicon,
    )
    sim = bz.Simulation(
        design=design,
        boundaries=[bz.PML(thickness=0.25 * bz.um)],
        # Deliberately coarse: layout geometry must not inherit these pixels.
        resolution=1.0 * bz.um,
        time=np.array([0.0, 1e-15]),
    )

    def fail_if_compiled(*_args, **_kwargs):
        raise AssertionError("layout plotting must not compile the FDTD grid")

    monkeypatch.setattr(bz.Simulation, "compile", fail_if_compiled)
    fig, axes = sim.plot(z=1.0 * bz.um, y=2.0 * bz.um, show=False)

    try:
        assert not axes[0].images
        assert not axes[1].images
        xy_core = [
            patch
            for patch in axes[0].patches
            if to_hex(patch.get_facecolor()) == "#d81b60"
        ]
        xz_core = [
            patch
            for patch in axes[1].patches
            if to_hex(patch.get_facecolor()) == "#d81b60"
        ]
        assert len(xy_core) == 1
        # The xz cut preserves the annulus hole as two distinct material spans.
        assert len(xz_core) == 2
        assert xy_core[0].get_antialiased()
        assert all(patch.get_antialiased() for patch in xz_core)
        assert xy_core[0].get_edgecolor()[3] == 0.0
        assert all(patch.get_edgecolor()[3] == 0.0 for patch in xz_core)
    finally:
        plt.close(fig)


def test_3d_simulation_plot_clips_x_pml_overlay_to_active_vertical_span():
    sim = bz.Simulation(
        domain=(4.0 * bz.um, 4.0 * bz.um, 2.0 * bz.um),
        design=bz.Design(background=bz.Material(2.25)),
        boundaries=[bz.PML(thickness=0.5 * bz.um)],
        resolution=0.5 * bz.um,
        time=np.array([0.0, 1e-15]),
    )

    fig, axes = sim.plot(z=0.0, y=0.0, show=False)

    try:
        xy_pml = [patch for patch in axes[0].patches if patch.get_hatch() == "///"]
        xz_pml = [patch for patch in axes[1].patches if patch.get_hatch() == "///"]

        assert len(xy_pml) == 4
        assert len(xz_pml) == 4

        xy_left = min(
            (patch for patch in xy_pml if patch.get_width() == pytest.approx(0.5)),
            key=lambda patch: patch.get_x(),
        )
        xz_left = min(
            (patch for patch in xz_pml if patch.get_width() == pytest.approx(0.5)),
            key=lambda patch: patch.get_x(),
        )

        assert xy_left.get_x() == pytest.approx(-2.0)
        assert xy_left.get_y() == pytest.approx(-1.5)
        assert xy_left.get_height() == pytest.approx(3.0)
        assert xz_left.get_x() == pytest.approx(-2.0)
        assert xz_left.get_y() == pytest.approx(-0.5)
        assert xz_left.get_height() == pytest.approx(1.0)
    finally:
        plt.close(fig)


def test_3d_layout_plot_only_draws_pml_faces_visible_in_each_section():
    sim = bz.Simulation(
        domain=(4.0 * bz.um, 4.0 * bz.um, 2.0 * bz.um),
        design=bz.Design(background=bz.Material(1.0)),
        boundaries=[bz.PML(edges=("left", "front"), thickness=0.5 * bz.um)],
        resolution=0.5 * bz.um,
        time=np.array([0.0, 1e-15]),
    )

    fig, axes = sim.plot(z=0.0, y=0.0, show=False)

    try:
        xy_pml = [patch for patch in axes[0].patches if patch.get_hatch() == "///"]
        xz_pml = [patch for patch in axes[1].patches if patch.get_hatch() == "///"]
        # ``front`` is normal to the xy view and therefore has no in-plane band.
        assert len(xy_pml) == 1
        assert len(xz_pml) == 2
        assert min(patch.get_x() for patch in xy_pml) == pytest.approx(-2.0)
        assert min(patch.get_y() for patch in xz_pml) == pytest.approx(-1.0)
    finally:
        plt.close(fig)


def test_3d_plot_keeps_square_limits_for_source_added_by_copy_update():
    sim0 = bz.Simulation(
        domain=(4.0 * bz.um, 4.0 * bz.um, 2.0 * bz.um),
        design=bz.Design(background=bz.Material(2.25)),
        sources=[],
        monitors=[],
        boundaries=[bz.PML(thickness=0.5 * bz.um)],
        resolution=0.5 * bz.um,
        time=np.array([0.0, 1e-15]),
    )
    source = bz.ModeSource(
        center=(-1.0 * bz.um, 0.0, 0.0),
        size=(0.0, 1.0 * bz.um, 0.8 * bz.um),
        source_time=bz.GaussianPulse(
            freq0=bz.LIGHT_SPEED / (1.55 * bz.um), fwidth=2e13
        ),
        direction="+",
    )

    sim = sim0.updated_copy(sources=[source])
    fig, axes = sim.plot(z=0.0, y=0.0, show=False)

    try:
        assert sim.sources[0].center == pytest.approx(
            (1.0 * bz.um, 2.0 * bz.um, 1.0 * bz.um)
        )
        assert source.center == pytest.approx((-1.0 * bz.um, 0.0, 0.0))
        assert axes[0].get_xlim() == pytest.approx((-2.0, 2.0))
        assert axes[0].get_ylim() == pytest.approx((-2.0, 2.0))
        source_line = next(
            line for line in axes[0].lines if line.get_color() == "#2ca02c"
        )
        np.testing.assert_allclose(source_line.get_xdata(), [-1.0, -1.0])
    finally:
        plt.close(fig)


def test_monitor_dft_field_plot_restores_tidy_plane_view():
    freq0 = 2.0e14
    design = bz.Design(background=bz.Material(2.25))
    design += bz.Box(
        center=(0.0, 0.0, 0.0),
        size=(1.0 * bz.um, 1.0 * bz.um, 0.4 * bz.um),
        material=bz.Material(12.0),
    )
    monitor = bz.FieldMonitor(
        center=(0.0, 0.0, 0.0),
        size=(4.0 * bz.um, 4.0 * bz.um, 0.0),
        freqs=[freq0],
        fields=("Ex", "Ey", "Ez"),
        name="field",
    )
    sim = bz.Simulation(
        domain=(4.0 * bz.um, 4.0 * bz.um, 2.0 * bz.um),
        design=design,
        sources=[],
        monitors=[monitor],
        # A non-integral cell count makes the staggered monitor plane one
        # sample larger per in-plane axis than the material grid. This mirrors
        # the modal_sources_monitors notebook regression.
        resolution=0.45 * bz.um,
        time=np.array([0.0, 1e-15]),
    )
    monitor = sim.monitors[0]
    fields = sim.compile().grid
    component_shapes = tuple(
        tuple(int(v) for v in getattr(fields, name).shape)
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    )
    monitor_base_shape = tuple(
        max(shape[axis] for shape in component_shapes) for axis in range(3)
    )
    coords0, coords1 = monitor.get_analysis_plane_coords_3d(
        dx=sim.resolution,
        dy=sim.resolution,
        dz=sim.resolution,
        field_shape=monitor_base_shape,
    )
    npoints = int(coords0.size * coords1.size)
    monitor_result = bz.MonitorResults(
        monitor=monitor,
        fields={},
        power_history=np.asarray([], dtype=float),
        power_timestamps=np.asarray([], dtype=float),
        power_spectrum=np.asarray([], dtype=np.complex64),
        dft_fields={
            "Ex": np.full((1, npoints), 10.0 / bz.um, dtype=np.complex128),
            "Ey": np.zeros((1, npoints), dtype=np.complex128),
            "Ez": np.zeros((1, npoints), dtype=np.complex128),
        },
        dft_frequencies=np.asarray([freq0]),
        dft_weight_sum=np.array([2.0]),
        dft_base_dt=0.0,
        resolution=float(sim.resolution),
    )
    results = bz.SimulationResults.from_run(
        sim,
        runtime_fields=fields,
        monitor_results={"field": monitor_result},
    )

    fig, ax = results.plot_field(
        field_monitor_name="field",
        field_name="E",
        val="abs^2",
        f=freq0,
        vmin=0,
        vmax=3000,
        cmap="magma",
        figsize=(6, 5),
        show=False,
    )

    try:
        image = ax.images[0].get_array()
        assert image.shape == (coords0.size, coords1.size)
        assert np.nanmax(image) == 100.0
        assert ax.get_xlabel() == "x (um)"
        assert ax.get_ylabel() == "y (um)"
        assert ax.get_xlim() == (-2.0, 2.0)
        assert ax.get_ylim() == (-2.0, 2.0)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("normal", "center", "size", "expected_x_edges", "expected_y_edges"),
    (
        ("x", (0.1, 0.5, 0.5), (0.0, 1.0, 1.0), (0.0, 0.3, 1.0), (0.0, 0.4, 1.0)),
        ("y", (0.5, 0.15, 0.5), (1.0, 0.0, 1.0), (0.0, 0.2, 1.0), (0.0, 0.4, 1.0)),
        ("z", (0.5, 0.5, 0.2), (1.0, 1.0, 0.0), (0.0, 0.2, 1.0), (0.0, 0.3, 1.0)),
    ),
)
def test_monitor_dft_field_plot_uses_exact_nonuniform_plane_coordinates(
    normal, center, size, expected_x_edges, expected_y_edges
):
    grid = Grid(
        np.asarray([0.0, 0.2, 1.0]) * bz.um,
        np.asarray([0.0, 0.3, 1.0]) * bz.um,
        np.asarray([0.0, 0.4, 1.0]) * bz.um,
    )
    material_grid = MaterialGrid.from_raster_result(
        rasterize(Scene((Material(),)), grid), dimensions=3
    )
    monitor = bz.FieldMonitor(
        center=tuple(value * bz.um for value in center),
        size=tuple(value * bz.um for value in size),
        freqs=[1.0],
        fields=("Ez",),
        name="field",
    )
    simulation = bz.Simulation(
        material_grid=material_grid,
        monitors=[monitor],
        time=np.asarray([0.0, 1e-16]),
    )
    program = simulation.compile()
    spec = program.monitors[0]
    from beamz.simulation.results import material_region_for_monitor

    result = bz.MonitorResults(
        monitor=simulation.monitors[0],
        fields={},
        power_history=np.empty(0),
        power_timestamps=np.empty(0),
        power_spectrum=np.empty(0, dtype=np.complex64),
        dft_fields={"Ez": np.ones((1, spec.dft_point_count), dtype=np.complex128)},
        dft_frequencies=np.asarray([1.0]),
        dft_weight_sum=np.ones(1),
        resolution=simulation.resolution,
        sample_region=spec.sample_region,
        material_region=material_region_for_monitor(
            simulation,
            simulation.monitors[0],
            runtime_fields=program.grid,
        ),
    )
    results = bz.SimulationResults.from_run(
        simulation,
        runtime_fields=program.grid,
        monitor_results={"field": result},
    )

    fig, ax = results.plot_field(
        field_monitor_name="field",
        field_name="Ez",
        show=False,
    )
    try:
        assert len(ax.collections) == 2
    finally:
        plt.close(fig)

    fig, ax = results.plot_field(
        field_monitor_name="field",
        field_name="Ez",
        show_grid=True,
        show=False,
    )

    try:
        assert not ax.images
        meshes = [
            collection
            for collection in ax.collections
            if hasattr(collection, "get_coordinates")
        ]
        assert len(meshes) == 2
        assert len(ax.collections) == 4
        for mesh in meshes:
            assert mesh.get_array().shape[:2] == (2, 2)
            coordinates = mesh.get_coordinates()
            np.testing.assert_allclose(coordinates[0, :, 0], expected_x_edges)
            np.testing.assert_allclose(coordinates[:, 0, 1], expected_y_edges)
        assert ax.get_xlim() == pytest.approx((0.0, 1.0))
        assert ax.get_ylim() == pytest.approx((0.0, 1.0))
    finally:
        plt.close(fig)


def test_flux_result_is_finite_for_notebook_style_line_plot():
    freqs = np.linspace(1.9e14, 2.1e14, 5)
    ldas = np.linspace(1.26, 1.36, freqs.size)
    monitor = bz.FluxMonitor(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 2.0 * bz.um, 2.0 * bz.um),
        freqs=freqs,
        name="flux",
    )
    npoints = 4
    result = bz.MonitorResults(
        monitor=monitor,
        fields={},
        power_history=np.asarray([], dtype=float),
        power_timestamps=np.asarray([], dtype=float),
        power_spectrum=np.asarray([], dtype=np.complex64),
        dft_fields={
            "Ex": np.zeros((freqs.size, npoints), dtype=np.complex128),
            "Ey": np.ones((freqs.size, npoints), dtype=np.complex128),
            "Ez": np.zeros((freqs.size, npoints), dtype=np.complex128),
            "Hx": np.zeros((freqs.size, npoints), dtype=np.complex128),
            "Hy": np.zeros((freqs.size, npoints), dtype=np.complex128),
            "Hz": np.ones((freqs.size, npoints), dtype=np.complex128),
        },
        dft_frequencies=freqs,
        dft_weight_sum=np.full(freqs.size, 2.0),
        dft_base_dt=0.0,
        resolution=1.0 * bz.um,
        power_scale=bz.um**2,
    )

    fig, ax = plt.subplots()
    try:
        flux_db = 10 * np.log10(result.flux)
        lines = ax.plot(ldas, flux_db, lw=3)
        ydata = np.asarray(lines[0].get_ydata(), dtype=float)
        assert len(lines) == 1
        assert ydata.shape == ldas.shape
        assert np.all(np.isfinite(ydata))
    finally:
        plt.close(fig)


@pytest.mark.parametrize("origin", [(0.0, 0.0, 0.0), (-2e-6, -1e-6, 0.0)])
def test_plot_eps_uses_material_grid_and_public_rectilinear_edges(origin):
    eps = np.array([[1.0, 2.0, 4.0], [3.0, 5.0, 6.0]])
    materials = MaterialGrid(
        permittivity=eps,
        conductivity=np.zeros_like(eps),
        permeability=np.ones_like(eps),
        resolution=1e-6,
        shape=eps.shape,
        origin=origin,
    )
    sim = bz.Simulation(material_grid=materials, time=np.array([0.0, 1e-16]))
    fig, ax = sim.plot_eps(source_markers=False, monitor_markers=False, vmin=1, vmax=6)
    try:
        # Material-grid simulations have metadata-only Design objects; the plot
        # must show their real optimized material data, not that empty geometry.
        image = ax.collections[0]
        np.testing.assert_array_equal(image.get_array(), eps)
        coords = image.get_coordinates()
        np.testing.assert_allclose(coords[0, :, 0], origin[0] / 1e-6 + np.arange(4))
        np.testing.assert_allclose(coords[:, 0, 1], origin[1] / 1e-6 + np.arange(3))
        assert image.get_clim() == (1, 6)
        assert len(fig.axes) == 2
        assert not ax.lines and not ax.patches
    finally:
        plt.close(fig)


@pytest.mark.parametrize("normal", ["y", "z"])
def test_plot_eps_3d_slices_actual_material_values(normal):
    eps = np.arange(8).reshape(2, 2, 2) + 1.0
    materials = MaterialGrid(
        permittivity=eps,
        conductivity=np.zeros_like(eps),
        permeability=np.ones_like(eps),
        resolution=1e-6,
        shape=eps.shape,
        origin=(-1e-6, -1e-6, -1e-6),
    )
    sim = bz.Simulation(material_grid=materials, time=np.array([0.0, 1e-16]))
    fig, ax = sim.plot_eps(**{normal: 0.0}, colorbar=False)
    try:
        expected = eps[1] if normal == "z" else eps[:, 1, :]
        np.testing.assert_array_equal(ax.collections[0].get_array(), expected)
        np.testing.assert_allclose(ax.get_xlim(), [-1, 1])
        assert ax.get_ylabel() == ("y (um)" if normal == "z" else "z (um)")
    finally:
        plt.close(fig)
    with pytest.raises(ValueError, match="outside"):
        sim.plot_eps(**{normal: 2e-6})


@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
def test_2d_layout_shares_3d_style_and_public_coordinates(monkeypatch, plane):
    from matplotlib.patches import Circle

    from beamz.lattice import grid_vector_to_physical_2d

    def physical(values):
        return grid_vector_to_physical_2d(values, plane)

    design = bz.Design(background=bz.Material(1))
    design += bz.Rectangle(
        position=(-0.8 * bz.um, -0.25 * bz.um),
        width=1.6 * bz.um,
        height=0.5 * bz.um,
        material=bz.Material(4),
    )
    source = bz.ModeSource(
        center=physical((-bz.um, 0, 0)),
        size=physical((0, bz.um, bz.um)),
        source_time=bz.GaussianPulse(freq0=2e14, fwidth=2e13),
        direction="+",
    )
    monitor = bz.ModeMonitor(
        center=physical((bz.um, 0, 0)),
        size=physical((0, bz.um, bz.um)),
        freqs=[2e14],
        name="out",
    )
    gaussian = bz.GaussianSource(
        position=physical((0, 0, 0)), width=0.2 * bz.um, signal=np.zeros(2)
    )
    sim = bz.Simulation(
        domain=(4 * bz.um, 3 * bz.um),
        design=design,
        plane_2d=plane,
        sources=[source, gaussian],
        monitors=[monitor],
        resolution=0.5 * bz.um,
        run_time=1e-15,
        boundaries=[bz.PML(thickness=0.5 * bz.um)],
    )

    def fail(*args, **kwargs):
        raise AssertionError("Analytic layout must not compile the numerical grid")

    monkeypatch.setattr(bz.Simulation, "compile", fail)
    fig, ax = sim.plot(show=False)
    try:
        assert not ax.images
        core = [p for p in ax.patches if to_hex(p.get_facecolor()) == "#d81b60"]
        assert len(core) == 1 and core[0].get_antialiased()
        assert len([p for p in ax.patches if p.get_hatch() == "///"]) == 4
        np.testing.assert_allclose(ax.get_xlim(), [-2, 2])
        np.testing.assert_allclose(ax.get_ylim(), [-1.5, 1.5])
        source_line = next(line for line in ax.lines if line.get_color() == "#2ca02c")
        np.testing.assert_allclose(source_line.get_xdata(), [-1, -1])
        circle = next(patch for patch in ax.patches if isinstance(patch, Circle))
        np.testing.assert_allclose(circle.center, [0, 0])
        assert any(line.get_color() == "#ff9800" for line in ax.lines)
        assert ax.get_xlabel() == f"{plane[0]} (um)"
        assert ax.get_ylabel() == f"{plane[1]} (um)"
    finally:
        plt.close(fig)
    fig, ax = plt.subplots()
    try:
        _, same_ax = sim.plot(
            ax=ax,
            source_markers=False,
            monitor_markers=False,
            xlim=(-1, 1),
            ylim=(-0.5, 0.5),
            show=False,
        )
        assert same_ax is ax and not ax.lines
        np.testing.assert_allclose(ax.get_xlim(), [-1, 1])
    finally:
        plt.close(fig)


def test_2d_layout_material_grid_preserves_geometry_and_origin():
    eps = np.array([[1, 4, 1], [1, 4, 1]], dtype=float)
    materials = MaterialGrid(
        permittivity=eps,
        conductivity=np.zeros_like(eps),
        permeability=np.ones_like(eps),
        resolution=1e-6,
        shape=eps.shape,
        origin=(-1e-6, -1e-6, 0.0),
    )
    sim = bz.Simulation(material_grid=materials, time=np.array([0.0, 1e-16]))
    fig, ax = sim.plot()
    try:
        mesh = ax.collections[0]
        assert np.unique(mesh.get_array()).size == 2
        np.testing.assert_allclose(mesh.get_coordinates()[0, :, 0], [-1, 0, 1, 2])
        np.testing.assert_allclose(ax.get_xlim(), [-1, 2])
        np.testing.assert_allclose(ax.get_ylim(), [-1, 1])
    finally:
        plt.close(fig)
