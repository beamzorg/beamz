"""Modal analysis must retain the constitutive coefficients used by propagation."""

import numpy as np
import pytest

from beamz import ModeMonitor, ModeSpec, Port, Simulation
from beamz.analysis import mode_projection
from beamz.analysis.data import AnalysisData
from beamz.analysis.modal_projection.geometry import _mode_components_for_port
from beamz.design.discretization import MaterialGrid
from beamz.lattice import component_shape_3d
from beamz.simulation.results import (
    SimulationMetadata,
    material_region_for_monitor,
)


@pytest.mark.contract
@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("full", [False, True])
def test_detached_projection_receives_actual_materials(axis, full, monkeypatch):
    shape = (10, 12, 14)
    dx = 0.1e-6
    names = dict(
        zip(
            ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
            ("eps_x", "eps_y", "eps_z", "mu_hx", "mu_hy", "mu_hz"),
            strict=True,
        )
    )
    yee = {
        name: (
            2
            + np.arange(np.prod(component_shape_3d(comp, shape))).reshape(
                component_shape_3d(comp, shape)
            )
            / 10000
        ).astype(np.float32)
        for comp, name in names.items()
    }
    tensors = {"epsilon": np.full((3, *shape), 2.5), "mu": np.ones((1, *shape))}
    grid = MaterialGrid(
        np.ones(shape),
        np.zeros(shape),
        np.ones(shape),
        dx,
        shape,
        yee_materials=yee,
        tensors=tensors,
        smoothing="farjadpour_diagonal",
    )
    size = tuple(0.0 if a == axis else 0.4e-6 for a in "xyz")
    monitor = ModeMonitor(
        center=(0.7e-6, 0.6e-6, 0.5e-6),
        size=size,
        freqs=[193e12],
        mode_spec=ModeSpec(polarization="te"),
        name="mode",
    )
    sim = Simulation(material_grid=grid, monitors=[monitor], time=[0, 1e-16])
    runtime = sim.compile().grid
    monitor = sim.monitors[0]
    metadata = SimulationMetadata.from_simulation(
        sim,
        runtime_fields=runtime,
        store_full_materials=full,
    )
    region = (
        metadata.fields.materials
        if full
        else material_region_for_monitor(
            sim,
            monitor,
            runtime_fields=runtime,
        )
    )
    normal = {"z": 0, "y": 1, "x": 2}[axis]
    assert region.permittivity.shape[normal] == (shape[normal] if full else 5)
    assert set(region.yee_materials) == set(yee)
    assert set(region.material_tensors) == set(tensors)
    for name, value in region.yee_materials.items():
        selection = tuple(
            slice(start, start + count)
            for start, count in zip(region.origin, value.shape, strict=True)
        )
        np.testing.assert_array_equal(value, yee[name][selection])
        assert not value.flags.writeable
        assert not np.shares_memory(value, grid.yee_materials[name])
    for name, value in region.material_tensors.items():
        selection = (
            slice(None),
            *(
                slice(start, start + count)
                for start, count in zip(
                    region.origin, region.permittivity.shape, strict=True
                )
            ),
        )
        np.testing.assert_array_equal(value, tensors[name][selection])
        assert not value.flags.writeable
        assert not np.shares_memory(value, grid.tensors[name])
    with pytest.raises(TypeError):
        region.yee_materials["eps_x"] = np.zeros(1)

    data = AnalysisData(metadata, {}, region, [193e12], monitor)
    port = Port(
        center=monitor.center,
        size=size,
        direction="+",
        name="mode",
        mode_spec=ModeSpec(polarization="te"),
    )
    captured = {}

    def capture(spec):
        captured.update(vars(spec))
        raise RuntimeError("captured mode materials")

    monkeypatch.setattr(mode_projection, "solve_beamz_mode", capture)
    with pytest.raises(RuntimeError, match="captured mode materials"):
        mode_projection._build_discrete_port_projection_3d(
            data,
            spec=port,
            monitor=monitor,
            frequency=193e12,
            parts=_mode_components_for_port(port),
            direction_sign=1.0,
            analysis_coords0=np.arange(4) * dx,
            analysis_coords1=np.arange(4) * dx,
        )
    # The scalar raster is one everywhere. Reconstructing Yee materials from it
    # (the pre-309 path) would silently discard these component coefficients.
    for value in captured["component_permittivity"].values():
        assert np.min(value) >= 2.0
    assert set(captured["diagonal_permittivity"]) == {"xx", "yy", "zz"}
    for value in captured["diagonal_permittivity"].values():
        np.testing.assert_array_equal(value, 2.5)
