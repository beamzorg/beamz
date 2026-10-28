"""Video export helpers for recorded time-domain fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from beamz.analysis._field_video import (
    _color_limits,
    _dynamic_limits,
    _format_time,
    _resolve_cmap,
    _select_recording,
    _set_even_pixel_canvas,
    _video_data,
)

__all__ = ["save_field_video"]


def _mpl_types():
    from matplotlib import pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.colors import LinearSegmentedColormap

    return plt, FFMpegWriter, LinearSegmentedColormap


def _video_axes(
    data,
    *,
    field,
    cmap,
    interpolation,
    low,
    high,
    clean_visualization,
    colorbar,
    plt,
    linear_segmented_colormap,
):
    width = max(data.extent[1] - data.extent[0], np.finfo(float).eps)
    height = max(data.extent[3] - data.extent[2], np.finfo(float).eps)
    figsize = (6.0 * width / height, 6.0) if clean_visualization else (7.0, 5.5)
    if clean_visualization:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
    else:
        fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(
        data.frames[0],
        origin="lower",
        extent=data.extent,
        cmap=_resolve_cmap(cmap, linear_segmented_colormap),
        interpolation=interpolation,
        vmin=low,
        vmax=high,
        aspect="equal",
    )
    ax.set_xlim(*data.extent[:2])
    ax.set_ylim(*data.extent[2:])
    ax.margins(0.0)
    if clean_visualization:
        ax.set_axis_off()
    else:
        ax.set_xlabel(f"{data.horizontal} (um)")
        ax.set_ylabel(f"{data.vertical} (um)")
        if colorbar:
            units = "V/m" if field.startswith("E") else "A/m"
            fig.colorbar(image, ax=ax, label=f"{field} ({units})")
        fig.tight_layout()
    return fig, ax, image


def save_field_video(
    results,
    filename,
    *,
    monitor_name=None,
    field="Ez",
    fps=30,
    dpi=150,
    cmap="twilight_zero",
    cmap_limits="dynamic",
    clean_visualization=False,
    interpolation="bicubic",
    vmin=None,
    vmax=None,
    plane="z",
    index=None,
    colorbar=True,
):
    """Save recorded time-domain fields from simulation results as an MP4 video.

    Parameters
    ----------
    results : SimulationResults
        Completed results containing at least one ``FieldRecorder`` acquisition.
    filename : path-like
        Destination video path. FFmpeg determines the container from its suffix.
    monitor_name : str, optional
        Recorder to export. Omit it when exactly one recorder contains ``field``.
    field : str, default="Ez"
        Recorded electric or magnetic field component.
    fps : int, default=30
        Encoded frames per second.
    dpi : int, default=150
        Output figure resolution.
    cmap : str or Colormap, default="twilight_zero"
        Matplotlib colormap. ``"twilight_zero"`` restores BeamZ's historical map.
    cmap_limits : {"dynamic", "global"} or pair, default="dynamic"
        Per-frame scaling, one symmetric global scale, or explicit limits.
    clean_visualization : bool, default=False
        Remove axes, title, and colorbar for a field-only video.
    interpolation : str, default="bicubic"
        Matplotlib image interpolation method.
    vmin, vmax : float, optional
        Explicit limits used with dynamic scaling. Do not combine with a limit pair
        or ``cmap_limits="global"``.
    plane : {"x", "y", "z"}, default="z"
        Slice normal for a full-domain 3D recorder. Ignored for 2D recordings and
        recorder planes.
    index : int, optional
        Slice index for a full-domain 3D recorder. The centered slice is the default.
    colorbar : bool, default=True
        Include a field-amplitude colorbar unless ``clean_visualization`` is true.

    Returns
    -------
    pathlib.Path
        The destination path after all frames have been written.
    """
    fps, dpi = int(fps), int(dpi)
    if fps <= 0 or dpi <= 0:
        raise ValueError("fps and dpi must be positive integers.")
    field = str(field)
    recording = _select_recording(
        results,
        monitor_name=monitor_name,
        field=field,
    )
    data = _video_data(
        results,
        recording,
        field=field,
        plane=plane,
        index=index,
    )
    low, high, dynamic = _color_limits(
        data.frames,
        cmap_limits,
        vmin=vmin,
        vmax=vmax,
    )
    if dynamic:
        low, high = _dynamic_limits(data.frames[0], vmin=low, vmax=high)

    plt, writer_type, linear_segmented_colormap = _mpl_types()
    if hasattr(writer_type, "isAvailable") and not writer_type.isAvailable():
        raise RuntimeError("Saving a field video requires FFmpeg to be installed.")
    output = Path(filename)
    fig, ax, image = _video_axes(
        data,
        field=field,
        cmap=cmap,
        interpolation=interpolation,
        low=low,
        high=high,
        clean_visualization=clean_visualization,
        colorbar=colorbar,
        plt=plt,
        linear_segmented_colormap=linear_segmented_colormap,
    )

    # Own the raw-frame dimensions before opening FFmpeg. Matplotlib's special
    # codec="h264" resizing differs across versions and can declare one height while
    # emitting another, which makes consecutive raw frames appear to slide vertically.
    _set_even_pixel_canvas(fig, dpi)
    writer = writer_type(
        fps=fps,
        # libx264 avoids Matplotlib's codec="h264" canvas adjustment while producing
        # the same H.264 bitstream. The pad remains a harmless encoder-side safeguard.
        codec="libx264",
        bitrate=5000,
        extra_args=[
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
        ],
    )
    try:
        with writer.saving(fig, str(output), dpi=dpi):
            for frame_index, frame in enumerate(data.frames):
                image.set_data(frame)
                if dynamic:
                    image.set_clim(*_dynamic_limits(frame, vmin=vmin, vmax=vmax))
                if not clean_visualization:
                    location = (
                        "" if data.slice_label is None else f", {data.slice_label}"
                    )
                    ax.set_title(
                        f"{field} at t = {_format_time(data.times[frame_index])}{location}"
                    )
                writer.grab_frame()
    finally:
        plt.close(fig)
    return output
