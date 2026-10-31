"""Static and animated views use identical recorded-plane coordinates and values."""

import matplotlib

matplotlib.use("Agg")

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from beamz.analysis._coordinates import recorded_plane
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
