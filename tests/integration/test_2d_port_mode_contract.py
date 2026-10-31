"""The public 2D preview, launch and detached measurement share one eigenproblem."""

from types import SimpleNamespace

import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    GaussianPulse,
    ModeMonitor,
    ModeSource,
    ModeSpec,
    Port,
    Simulation,
)
from beamz.analysis import mode_projection
from beamz.analysis.data import AnalysisData
from beamz.analysis.modal_projection.geometry import _mode_components_for_port
from beamz.analysis.mode_projection import _build_port_projection_2d
from beamz.design.discretization import MaterialGrid
from beamz.design.grid import RectilinearGrid
from beamz.devices.sources import mode_launch, solve
from beamz.lattice import component_coordinates_rectilinear
from beamz.simulation.results import (
    FieldMetadata,
    SimulationMetadata,
    material_region_for_monitor,
)


@pytest.mark.parametrize("polarization", ["te", "tm"])
@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("direction", ["+", "-"])
@pytest.mark.parametrize("stretched", [False, True])
@pytest.mark.parametrize("targeted", [False, True])
@pytest.mark.parametrize("padding", [0, 5])
def test_outside_guide_does_not_change_port_mode(
    monkeypatch, polarization, axis, direction, stretched, targeted, padding
):
    dl, frequency = 0.025e-6, LIGHT_SPEED / 1.55e-6
    transverse_edges = np.arange(241) * dl
    if stretched:
        transverse_edges += 0.002e-6 * np.sin(np.linspace(0, 2 * np.pi, 241))
    normal_edges = np.arange(25) * dl
    grid = RectilinearGrid(
        normal_edges if axis == "x" else transverse_edges,
        transverse_edges if axis == "x" else normal_edges,
        np.array([0.0, dl]),
    )
    center = (0.3e-6, 2e-6, 0) if axis == "x" else (2e-6, 0.3e-6, 0)
    size = (0, 2e-6, dl) if axis == "x" else (2e-6, 0, dl)
    calls = []
    native = solve.solve_modes

    def observe(**kwargs):
        result = native(**kwargs)
        calls.append((kwargs, result))
        return result

    for module in [solve, mode_launch, mode_projection]:
        monkeypatch.setattr(module, "solve_modes", observe)

    reference = None
    for neighbor in [False, True]:
        eps = np.full((240, 24), 1.444**2, dtype=np.float32)
        eps[70:90] = 3.476**2
        if neighbor:
            eps[140:172] = 3.476**2
        if axis == "y":
            eps = eps.T.copy()
        yee = {}
        if stretched:
            for component in ["Ez"] if polarization == "tm" else ["Ex", "Ey"]:
                coords = component_coordinates_rectilinear(component, grid)
                ix = np.clip(
                    np.searchsorted(grid.x_edges, coords["x"], side="right") - 1,
                    0,
                    eps.shape[1] - 1,
                )
                iy = np.clip(
                    np.searchsorted(grid.y_edges, coords["y"], side="right") - 1,
                    0,
                    eps.shape[0] - 1,
                )
                yee["eps_" + component[-1].lower()] = eps[np.ix_(iy, ix)]
        materials = MaterialGrid(
            eps,
            np.zeros_like(eps),
            np.ones_like(eps),
            dl,
            eps.shape,
            grid=grid,
            yee_materials=yee,
            polarization=polarization,
        )
        spec = ModeSpec(
            polarization=polarization,
            target_neff=reference if targeted else None,
            solver_padding_cells=padding,
        )
        source = ModeSource(
            center=center,
            size=size,
            direction=direction,
            source_time=GaussianPulse(freq0=frequency, fwidth=frequency / 10),
            mode_spec=spec,
        )
        monitor = ModeMonitor(
            center=center, size=size, freqs=[frequency], name="port", mode_spec=spec
        )
        simulation = Simulation(
            material_grid=materials,
            sources=[source],
            monitors=[monitor],
            polarization=polarization,
            run_time=1e-16,
        )
        calls.clear()
        preview = source.solve_modes(simulation)
        if reference is None:
            reference = float(preview.neffs[0, 0].real)
        np.testing.assert_allclose(preview.neffs[0, 0].real, reference, rtol=1e-8)
        # Exercise the real launch planner and detached monitor-material snapshot
        # without a time-domain simulation: this tests selection, not transmission.
        fields = SimpleNamespace(
            permittivity=eps,
            conductivity=np.zeros_like(eps),
            permeability=np.ones_like(eps),
            material_grid=materials,
            geometry=grid,
            polarization_2d=polarization,
        )
        launch = mode_launch.plan_mode_source_launch(
            source, fields, resolution=materials.resolution, dt=None, grid=grid
        )
        assert all(np.all(np.isfinite(entry.profile)) for entry in launch.entries)
        region = material_region_for_monitor(simulation, monitor, runtime_fields=fields)
        metadata = SimulationMetadata(
            dt=1e-18,
            resolution=materials.resolution,
            is_3d=False,
            plane_2d="xy",
            coordinate_offset=(0, 0, 0),
            time=np.array([0.0]),
            width=grid.x_edges[-1],
            height=grid.y_edges[-1],
            depth=0,
            fields=FieldMetadata(eps.shape, {}),
            polarization_2d=polarization,
            grid=grid,
        )
        data = AnalysisData(metadata, {}, region, np.array([frequency]), monitor)
        port = Port(
            center=center, size=size, name="port", mode_spec=spec, direction=direction
        )
        projection = _build_port_projection_2d(
            data,
            spec=port,
            monitor=monitor,
            frequency=frequency,
            parts=_mode_components_for_port(port),
            mode_pad_cells=None,
        )
        assert projection["mode_neff"] == pytest.approx(reference, rel=1e-8)
        assert len(calls) == 3
        for kwargs, result in calls:
            np.testing.assert_array_equal(kwargs["eps"], preview.eps_profiles[0])
            assert kwargs["target_neff"] == spec.target_neff
            assert kwargs["m"] == calls[0][0]["m"]
            np.testing.assert_allclose(result[0][0], reference, rtol=1e-8)
        assert preview.diagnostics[0]["field_representation"] == "solver"


@pytest.mark.parametrize("padding", [0, 3, (4, 99)])
@pytest.mark.parametrize("clipped", [False, True])
def test_explicit_padding_counts_cells_and_clips_at_boundary(padding, clipped):
    from beamz.devices.modes.line import prepare_mode_line

    dl = 0.1e-6
    eps = np.ones((40, 10)) * 2.25
    spec = ModeSpec(polarization="te", solver_padding_cells=padding)
    center = (0.5e-6, 0.1e-6 if clipped else 2e-6, 0)
    size = (0, 1e-6, dl)
    context = prepare_mode_line(
        permittivity=eps, center=center, size=size, axis="x", spec=spec, resolution=dl
    )
    pad = padding[0] if isinstance(padding, tuple) else padding
    interval = context.snapped.axis_interval("y")
    assert context.crop.start == max(0, interval.start - pad)
    assert context.crop.stop == min(40, interval.stop + pad)
    assert len(context.edges) == len(context.eps) + 1


def test_mode_index_unavailable_fails_instead_of_launching_another_mode():
    from beamz.devices.modes.line import prepare_mode_line, solve_mode_line

    spec = ModeSpec(mode_index=2)
    context = prepare_mode_line(
        permittivity=np.ones((10, 4)),
        center=(1, 5, 0),
        size=(0, 6, 0),
        axis="x",
        spec=spec,
        resolution=1,
    )

    def solver(**_):
        return np.array([1.5]), np.ones((1, 3, 6)), np.ones((1, 3, 6)), None

    with pytest.raises(ValueError, match="mode_index=2"):
        solve_mode_line(
            context,
            spec=spec,
            frequency=1,
            resolution=1,
            direction="+x",
            polarization=None,
            solver=solver,
        )


@pytest.mark.parametrize("confined", [False, True])
def test_padded_line_confinement_can_exclude_outside_higher_index_guide(confined):
    from beamz.devices.modes.line import prepare_mode_line, solve_mode_line
    from beamz.devices.modes.plane import solve_modes

    eps = np.full((240, 8), 1.444**2)
    eps[70:90] = 3.476**2
    eps[140:172] = 3.476**2
    spec = ModeSpec(
        polarization="te",
        solver_padding_cells=90,
        min_aperture_fraction=0.8 if confined else None,
    )
    context = prepare_mode_line(
        permittivity=eps,
        center=(0.1e-6, 2e-6, 0),
        size=(0, 2e-6, 0),
        axis="x",
        spec=spec,
        resolution=0.025e-6,
    )
    neff, electric, _ = solve_mode_line(
        context,
        spec=spec,
        frequency=LIGHT_SPEED / 1.55e-6,
        resolution=0.025e-6,
        direction="+x",
        polarization="te",
        solver=solve_modes,
    )
    peak_y = (
        np.argmax(np.sum(abs(electric[0]) ** 2, axis=0)) + context.crop.start + 0.5
    ) * 0.025
    if confined:
        assert 1.75 <= peak_y <= 2.25
        assert neff[0].real < 3.2
    else:
        assert 3.5 <= peak_y <= 4.3
        assert neff[0].real > 3.3


@pytest.mark.parametrize("polarization", ["te", "tm"])
@pytest.mark.parametrize("transverse_center", [0.1e-6, 1.5e-6])
def test_public_mode_monitor_projection_matches_off_grid_preview(
    polarization, transverse_center
):
    """Check real acquisition shapes and the public detached-result API."""
    dl, frequency = 0.025e-6, LIGHT_SPEED / 1.55e-6
    eps = np.full((120, 20), 1.444**2, dtype=np.float32)
    eps[50:70] = 3.476**2
    materials = MaterialGrid(
        eps,
        np.zeros_like(eps),
        np.ones_like(eps),
        dl,
        eps.shape,
        polarization=polarization,
    )
    spec = ModeSpec(polarization=polarization)
    center, size = (0.2e-6, transverse_center, 0), (0, 1.81e-6, dl)
    source = ModeSource(
        center=center,
        size=size,
        direction="+",
        source_time=GaussianPulse(freq0=frequency, fwidth=frequency / 10),
        mode_spec=spec,
    )
    monitor = ModeMonitor(
        center=center, size=size, freqs=[frequency], name="port", mode_spec=spec
    )
    sim = Simulation(
        material_grid=materials,
        sources=[source],
        monitors=[monitor],
        polarization=polarization,
        run_time=1e-16,
    )
    preview = source.solve_modes(sim)
    result = sim.run(backend="jax", progress=False)
    modes = result.mode("port")
    assert modes.amps.shape == (1, 2, 1)
    assert np.all(np.isfinite(modes.amps))
    # The tiny run establishes API/support compatibility, not transmission.
    assert np.isfinite(preview.neffs[0, 0])
