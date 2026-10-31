"""Acquisition coordinates survive detachment and archives without re-snapping."""

import json

import numpy as np
import pytest

import beamz as bz
from beamz.analysis.data import analysis_data
from beamz.analysis.modal_projection.geometry import _monitor_analysis_plane_3d
from beamz.design import MaterialGrid
from beamz.devices.monitors import geometry as monitor_geometry
from beamz.simulation.execute import initial_program_state


@pytest.mark.parametrize("normal", ["x", "y", "z"])
def test_shifted_uniform_plane_retains_actual_sample_coordinates(
    tmp_path, monkeypatch, normal
):
    step = 2.0**-20
    shape = (4, 6, 8)
    materials = MaterialGrid(
        np.ones(shape),
        np.array(0.0),
        np.array(1.0),
        step,
        shape,
        origin=(2 * step,) * 3,
    )
    size = [4 * step, 3 * step, 2 * step]
    size["xyz".index(normal)] = 0.0
    monitor = bz.FieldMonitor(
        center=(6 * step, 5 * step, 4 * step),
        size=tuple(size),
        freqs=[2e14],
        fields=("Ez",),
        name="plane",
    )
    sim = bz.Simulation(
        material_grid=materials,
        monitors=[monitor],
        time=np.array([0.0, 1e-16]),
        normalize_source=None,
    )
    program = sim.compile(backend="jax")
    compiled = program.monitors[0].sample_region
    # Runtime coordinates are local to the material-grid origin. The legacy
    # uniform policy spaces its four y samples across the requested three-cell aperture.
    expected = {
        "x": ((1.5, 2.5), (1.875, 2.625, 3.375, 4.125)),
        "y": ((1.5, 2.5), (2.5, 3.5, 4.5, 5.5)),
        "z": ((1.875, 2.625, 3.375, 4.125), (2.5, 3.5, 4.5, 5.5)),
    }[normal]
    for actual, values in zip(compiled.sample_coordinates, expected, strict=True):
        np.testing.assert_array_equal(actual, np.array(values) * step)
    result = bz.MonitorResults.from_compiled_state(
        sim.monitors[0],
        program.monitors[0],
        initial_program_state(program, t=0.0, current_step=0),
        program.config,
    )
    results = bz.SimulationResults.from_run(
        sim, runtime_fields=program.grid, monitor_results={"plane": result}
    )
    path = tmp_path / "plane.npz"
    results.save(path)
    with monkeypatch.context() as patch:

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "New results must retain coordinates instead of reconstructing them"
            )

        patch.setattr(monitor_geometry, "_restore_sample_region", forbidden)
        loaded = bz.SimulationResults.load(path)
        data = analysis_data(loaded, "plane")
        coords = _monitor_analysis_plane_3d(data, data.monitor_geometry, normal)
        for actual, expected_coord in zip(
            coords, compiled.sample_coordinates, strict=True
        ):
            np.testing.assert_array_equal(actual, expected_coord)
            assert not actual.flags.writeable
        assert data.sample_region.sample_shape == compiled.sample_shape

    # Old archives have no geometry member. Normalize them at the result boundary.
    with np.load(path, allow_pickle=False) as saved:
        payload = {name: np.array(saved[name]) for name in saved.files}
    manifest = json.loads(str(payload["manifest"]))

    def remove_geometry(value):
        if isinstance(value, dict):
            if value.get("class") == "SnappedRegion":
                value["fields"].pop("sample_coordinates", None)
            for child in value.values():
                remove_geometry(child)
        elif isinstance(value, list):
            for child in value:
                remove_geometry(child)

    remove_geometry(manifest)
    manifest["format"] = "beamz.simulation.results/1"
    payload["manifest"] = np.asarray(json.dumps(manifest))
    np.savez(path, **payload)
    legacy = bz.SimulationResults.load(path)
    for actual, expected_coord in zip(
        legacy.monitors["plane"].sample_region.sample_coordinates,
        compiled.sample_coordinates,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected_coord)


def test_standalone_shifted_metadata_uses_realized_sampling():
    from types import SimpleNamespace

    from beamz.analysis.data import AnalysisData
    from beamz.design.grid import RectilinearGrid
    from beamz.devices.monitors.compiler import _monitor_plane_quadrature
    from beamz.lattice import component_shapes
    from beamz.simulation.results import FieldMetadata, SimulationMetadata

    step = 2.0**-20
    shape = (4, 6, 8)
    shapes = component_shapes(shape)
    grid = RectilinearGrid(*(step * (2 + np.arange(n + 1)) for n in reversed(shape)))
    monitor = bz.FieldMonitor(
        center=(6 * step, 5 * step, 4 * step),
        size=(0.0, 3 * step, 2 * step),
        freqs=[2e14],
        name="shifted",
    )
    fields = SimpleNamespace(
        permittivity=np.ones(shape), **{n: np.empty(s) for n, s in shapes.items()}
    )
    _, quadrature = _monitor_plane_quadrature(monitor, fields, step, grid)
    metadata = SimulationMetadata(
        dt=1e-16,
        resolution=step,
        is_3d=True,
        plane_2d="xy",
        coordinate_offset=(0, 0, 0),
        time=np.array([0.0, 1e-16]),
        width=8 * step,
        height=6 * step,
        depth=4 * step,
        fields=FieldMetadata(shape, shapes),
        grid=grid,
    )
    data = AnalysisData(metadata, {}, None, np.array([2e14]), monitor)
    actual = _monitor_analysis_plane_3d(data, monitor, "x")
    expected = (np.array([3.5, 4.5]) * step, np.array([3.5, 4.5, 5.5, 6.5]) * step)
    for value, truth, sampled in zip(
        actual, expected, quadrature.coordinates, strict=True
    ):
        np.testing.assert_array_equal(value, truth)
        np.testing.assert_array_equal(value, sampled)
