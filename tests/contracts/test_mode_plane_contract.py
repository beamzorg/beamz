import numpy as np
import pytest

from beamz import ModeMonitor, ModeSpec, Port
from beamz.analysis import mode_projection
from beamz.analysis.data import AnalysisData
from beamz.analysis.modal_projection.geometry import _mode_components_for_port
from beamz.devices.modes.plane import (
    MODE_PLANE_APERTURE_PAD_CELLS,
    MODE_PLANE_APERTURE_WINDOW_ALPHA,
)
from beamz.lattice import component_shape_3d
from beamz.simulation.results import FieldMetadata, MaterialRegion, SimulationMetadata


@pytest.mark.unit
@pytest.mark.parametrize("force_symmetry", [False, True])
@pytest.mark.parametrize(
    "padding, expected_shape", [(None, (8, 10)), (0, (8, 10)), ((2, 4), (16, 14))]
)
def test_mode_monitor_uses_shared_finite_aperture_policy(
    monkeypatch, force_symmetry, padding, expected_shape
):
    resolution = 0.1
    shape = (40, 50, 60)
    monitor = ModeMonitor(
        center=(3.0, 2.5, 2.0),
        size=(0.0, 1.0, 0.8),
        freqs=np.asarray([1.0]),
        mode_spec=ModeSpec(
            num_modes=1, force_symmetry=force_symmetry, solver_padding_cells=padding
        ),
        name="mode",
    )
    materials = MaterialRegion(
        np.ones(shape, dtype=np.float32),
        np.ones(shape, dtype=np.float32),
        (0, 0, 0),
        shape,
    )
    metadata = SimulationMetadata(
        dt=0.01,
        resolution=resolution,
        is_3d=True,
        plane_2d="xy",
        coordinate_offset=(0.0, 0.0, 0.0),
        time=np.array([0.0, 0.01]),
        width=6.0,
        height=5.0,
        depth=4.0,
        fields=FieldMetadata(
            shape,
            {
                component: component_shape_3d(component, shape)
                for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            },
            materials,
        ),
    )
    sim = AnalysisData(metadata, {}, materials, (1.0,), monitor)
    port = Port(
        center=monitor.center,
        size=monitor.size,
        name="mode",
        direction="+",
        mode_spec=ModeSpec(polarization="te"),
    )
    spec = port
    seen = {}

    def fake_solve(spec):
        seen.update(vars(spec))
        raise RuntimeError("stop after contract capture")

    monkeypatch.setattr(mode_projection, "solve_beamz_mode", fake_solve)

    with pytest.raises(RuntimeError, match="stop after contract capture"):
        mode_projection._build_discrete_port_projection_3d(
            sim,
            spec=spec,
            monitor=monitor,
            frequency=1.0,
            parts=_mode_components_for_port(spec),
            direction_sign=1.0,
            analysis_coords0=np.arange(8, dtype=float),
            analysis_coords1=np.arange(10, dtype=float),
        )

    assert seen["force_symmetry"] is force_symmetry
    assert seen["aperture_pad_cells"] == MODE_PLANE_APERTURE_PAD_CELLS
    assert seen["aperture_window_alpha"] == MODE_PLANE_APERTURE_WINDOW_ALPHA
    assert seen["scalar_permittivity"].shape == expected_shape
