"""Video export helpers for recorded time-domain fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from beamz.analysis._coordinates import recorded_plane
from beamz.simulation.results import MonitorResults, SimulationResults

_UM = 1e-6


def _mpl_types():
    from matplotlib import pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.colors import LinearSegmentedColormap

    return plt, FFMpegWriter, LinearSegmentedColormap


def _resolve_cmap(cmap: Any, linear_segmented_colormap):
    if cmap != "twilight_zero":
        return cmap
    colors = [
        (1.0, 1.0, 1.0),
        (0.2, 0.3, 0.8),
        (0.1, 0.1, 0.5),
        (0.1, 0.1, 0.1),
        (0.5, 0.1, 0.1),
        (0.8, 0.3, 0.2),
        (1.0, 1.0, 1.0),
    ]
    return linear_segmented_colormap.from_list("twilight_zero", colors, N=256)


def _select_recording(results, *, monitor_name, field) -> MonitorResults:
    if not isinstance(results, SimulationResults):
        raise TypeError("save_field_video requires SimulationResults.")
    if monitor_name is not None:
        try:
            recording = results.monitor(str(monitor_name))
        except KeyError as exc:
            raise ValueError(f"Unknown monitor {monitor_name!r}.") from exc
        if field not in recording.fields:
            raise ValueError(
                f"Monitor {monitor_name!r} did not record {field!r}; "
                f"available fields: {sorted(recording.fields)}."
            )
        return recording

    matches = [
        recording
        for recording in results.monitors.values()
        if field in recording.fields
    ]
    if not matches:
        available = {
            name: sorted(recording.fields)
            for name, recording in results.monitors.items()
            if recording.fields
        }
        raise ValueError(
            f"No recorder contains field {field!r}; available fields: {available}."
        )
    if len(matches) > 1:
        names = [str(recording.monitor.name) for recording in matches]
        raise ValueError(
            f"Multiple recorders contain {field!r}: {names}; pass monitor_name."
        )
    return matches[0]


def _color_limits(frames, cmap_limits, *, vmin, vmax):
    explicit = vmin is not None or vmax is not None
    if cmap_limits is None:
        cmap_limits = "dynamic"
    if isinstance(cmap_limits, str):
        choice = cmap_limits.lower()
        if choice == "dynamic":
            return vmin, vmax, True
        if choice not in {"max", "global", "global_max", "field_max"}:
            raise ValueError(
                "cmap_limits must be 'dynamic', 'global', or a (vmin, vmax) pair."
            )
        if explicit:
            raise ValueError("Use either cmap_limits='global' or vmin/vmax, not both.")
        limit = float(np.nanmax(np.abs(frames)))
        if not np.isfinite(limit) or limit <= 0.0:
            limit = 1.0
        return -limit, limit, False
    if explicit:
        raise ValueError("Use either cmap_limits or vmin/vmax, not both.")
    try:
        low, high = cmap_limits
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "cmap_limits must be 'dynamic', 'global', or a (vmin, vmax) pair."
        ) from exc
    return (
        None if low is None else float(low),
        None if high is None else float(high),
        False,
    )


def _dynamic_limits(frame, *, vmin, vmax):
    finite = np.asarray(frame)[np.isfinite(frame)]
    if finite.size == 0:
        return (-1.0 if vmin is None else vmin, 1.0 if vmax is None else vmax)
    low = float(np.min(finite)) if vmin is None else float(vmin)
    high = float(np.max(finite)) if vmax is None else float(vmax)
    if low == high:
        margin = max(abs(low) * 1e-6, 1e-12)
        low, high = low - margin, high + margin
    return low, high


def _format_time(value):
    magnitude = abs(float(value))
    if magnitude < 1e-12:
        return f"{value * 1e15:.2f} fs"
    if magnitude < 1e-9:
        return f"{value * 1e12:.2f} ps"
    if magnitude < 1e-6:
        return f"{value * 1e9:.2f} ns"
    return f"{value:.3g} s"


def _set_even_pixel_canvas(fig, dpi):
    """Size a figure so raw RGBA frames have stable, even pixel dimensions."""
    size = np.asarray(fig.get_size_inches(), dtype=float)
    pixels = np.maximum(2, np.rint(size * float(dpi)).astype(int))
    pixels += pixels % 2
    # Move toward +inf so Matplotlib's integer conversion cannot turn an exact
    # quotient such as 826 / 150 back into 825 through floating-point roundoff.
    inches = np.nextafter(pixels.astype(float) / float(dpi), np.inf)
    fig.set_size_inches(inches, forward=True)
    return int(pixels[0]), int(pixels[1])


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
    data = recorded_plane(
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
    width = max(data.extent[1] - data.extent[0], np.finfo(float).eps)
    height = max(data.extent[3] - data.extent[2], np.finfo(float).eps)
    figsize = (6.0 * width / height, 6.0) if clean_visualization else (7.0, 5.5)
    if clean_visualization:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
    else:
        fig, ax = plt.subplots(figsize=figsize)
    from beamz.analysis.plotting import plot_field_view

    image, _ = plot_field_view(
        ax,
        data.frames[0],
        extent=data.extent,
        cmap=_resolve_cmap(cmap, linear_segmented_colormap),
        interpolation=interpolation,
        vmin=low,
        vmax=high,
        aspect="equal",
        symmetric_limits=False,
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


__all__ = ["save_field_video"]
