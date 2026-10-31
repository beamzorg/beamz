from dataclasses import fields, replace

import numpy as np
import pytest

from beamz import FieldMonitor, FieldRecorder
from beamz.analysis.adapters import monitor_to_xarray, to_xarray
from beamz.analysis.data import AnalysisData, analysis_data
from beamz.simulation.results import FieldMetadata, SimulationMetadata


def _metadata():
    return SimulationMetadata(
        dt=0.25,
        resolution=0.5,
        is_3d=False,
        plane_2d="xy",
        coordinate_offset=(0.0, 0.0, 0.0),
        time=np.empty(0),
        width=1.0,
        height=1.0,
        depth=0.0,
        fields=FieldMetadata((1, 1), {"Ez": (1, 1)}),
    )


def test_analysis_data_has_an_immutable_sampling_contract():
    monitor = FieldMonitor(
        center=(0.0, 0.0, 0.0),
        size=(1.0, 0.0, 0.0),
        freqs=[2.0],
        fields=("Ez",),
        name="field",
    )
    coordinates = _metadata()
    contract = AnalysisData(
        coordinates=coordinates,
        fields={"Ez": np.asarray([[3.0 + 4.0j]]), "flux": np.asarray([7.0])},
        materials=None,
        frequencies=np.asarray([2.0]),
        monitor_geometry=monitor,
    )

    data = analysis_data(contract)

    assert tuple(item.name for item in fields(AnalysisData)) == (
        "coordinates",
        "fields",
        "materials",
        "frequencies",
        "monitor_geometry",
        "sample_region",
    )
    assert data.name == "field"
    assert monitor_to_xarray(data).flux.dims == ("f",)
    assert monitor_to_xarray(data).flux.item() == 7
    with pytest.raises(ValueError):
        monitor_to_xarray(replace(data, frequencies=np.array([2.0, 3.0])))
    material = replace(
        data, frequencies=np.empty(0), fields={"permittivity": np.array([[4.0]])}
    )
    np.testing.assert_array_equal(to_xarray(material).x, [0.25])
    np.testing.assert_array_equal(to_xarray(material).y, [0.25])
    assert data.dt == 0.25
    assert data.resolution == 0.5
    for convert in (to_xarray, monitor_to_xarray):
        assert convert(data)["Ez"].dims == ("f", "sample_1")
        np.testing.assert_array_equal(convert(data)["Ez"].f, [2.0])
    np.testing.assert_allclose(data.field("Ez"), [[3.0 + 4.0j]])
    with pytest.raises(ValueError):
        data.frequencies[0] = 3.0
    with pytest.raises(TypeError):
        data.fields["Ez"] = np.zeros((1, 1))


def test_analysis_data_snapshots_field_recorder_geometry():
    monitor = FieldRecorder(
        ("Ey",),
        interval=3,
        name="slice",
        center=(0.0, 0.0, 0.2),
        size=(2.0, 1.0, 0.0),
    )

    data = AnalysisData(
        coordinates=_metadata(),
        fields={"Ey": np.ones((2, 3)), "power": np.array([1.0, 2.0])},
        materials=None,
        frequencies=np.empty(0),
        monitor_geometry=monitor,
    )

    assert data.monitor_geometry == monitor
    assert data.monitor_geometry.components == ("Ey",)
    assert data.monitor_geometry.region == "slice"
    assert not data.monitor_geometry.freqs.flags.writeable

    for convert in (to_xarray, monitor_to_xarray):
        samples = convert(data)["Ey"]
        assert samples.dims == ("frame", "s")
        assert samples.attrs["coordinate_frame"] == "sample"
        np.testing.assert_array_equal(samples.s, np.arange(3))
    np.testing.assert_array_equal(monitor_to_xarray(data).power[:, 0], [1.0, 2.0])
