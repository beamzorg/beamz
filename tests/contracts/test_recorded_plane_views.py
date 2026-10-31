"""Static and animated views use identical recorded-plane coordinates and values."""

import matplotlib

matplotlib.use("Agg")

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from beamz.analysis import adapters
from beamz.analysis._coordinates import recorded_plane
from beamz.analysis.plotting import _normalize_dft_components
from beamz.simulation.observe import SourceNormalization
from tests.integration.test_analysis_video import _results_3d


def test_static_and_video_planes_share_selection_without_xarray(monkeypatch):
    import beamz.analysis.adapters as adapters
    import beamz.analysis.plotting as plotting

    results = _results_3d()
    recording = results.monitors["volume"]
    y = 0.75e-6  # Exactly between y=0.5um and1um; nearest ties choose the higher.
    expected = np.asarray(recording.fields["Ez"])[0, :, 2, :]
    video = recorded_plane(results, recording, field="Ez", plane="y", index=2)
    np.testing.assert_array_equal(video.frames[0], expected)

    def forbidden(*args, **kwargs):
        raise AssertionError("Field rendering must not construct a labeled dataset")

    monkeypatch.setattr(adapters, "to_xarray", forbidden)
    monkeypatch.setattr(plotting, "analysis_inputs", forbidden)
    figure, axis = results.plot_field(
        field="Ez", frame=0, y=y, colorbar=False, show=False
    )
    try:
        np.testing.assert_array_equal(axis.images[0].get_array(), expected)
        np.testing.assert_allclose(
            axis.images[0].get_extent(), video.extent, rtol=0, atol=0
        )
        assert axis.get_xlabel() == "x (um)"
        assert axis.get_ylabel() == "z (um)"
    finally:
        plt.close(figure)


def test_static_recording_without_timestamps_still_renders():
    results = _results_3d()
    recording = replace(
        results.monitors["volume"], field_times=np.empty(0), field_steps=np.empty(0)
    )
    results = replace(results, monitors={"volume": recording})
    figure, axis = results.plot_field(field="Ez", frame=0, show=False)
    try:
        np.testing.assert_array_equal(
            axis.images[0].get_array(), recording.fields["Ez"][0, 1, :, :]
        )
    finally:
        plt.close(figure)
    with pytest.raises(RuntimeError, match="no complete"):
        recorded_plane(results, recording, field="Ez")


def test_unexpected_physical_coordinate_failures_propagate(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("invalid physical coordinates")

    monkeypatch.setattr(adapters, "field_coordinates", broken)
    with pytest.raises(RuntimeError, match="invalid physical coordinates"):
        adapters.to_xarray(_results_3d())


@pytest.mark.parametrize("normalization", [np.array([2j]), SourceNormalization([2j])])
def test_source_normalization_preserves_complex_field_phase(normalization):
    values = _normalize_dft_components([np.array([[2 + 2j, 4]])], normalization, 0, 1)
    np.testing.assert_array_equal(values, [[[1 - 1j, -2j]]])


@pytest.mark.parametrize("normalization", [[], [1, 2], [0], [np.nan], [np.inf]])
def test_requested_source_normalization_rejects_unusable_amplitudes(normalization):
    with pytest.raises(ValueError):
        _normalize_dft_components([np.ones((2, 2))], normalization, 0, 1)
