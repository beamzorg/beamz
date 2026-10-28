"""Recorded-field frames and color scales for MP4 export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from beamz.analysis._coordinates import monitor_plane_coordinates_3d
from beamz.lattice import (
    component_coordinates_2d_um,
    component_coordinates_3d_um,
    component_coordinates_rectilinear,
    coordinates_in_public_frame,
    plane_axes_3d,
)
from beamz.simulation.results import MonitorResults, SimulationResults

_UM = 1e-6
_AXIS_INDEX = {"z": 0, "y": 1, "x": 2}


@dataclass(frozen=True, slots=True)
class _VideoData:
    frames: np.ndarray
    times: np.ndarray
    extent: tuple[float, float, float, float]
    horizontal: str
    vertical: str
    slice_label: str | None = None


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


def _axis_extent(values, *, fallback_step):
    coordinates = np.asarray(values, dtype=float)
    if coordinates.size == 0:
        raise ValueError("Recorded field coordinates are empty.")
    step = (
        float(np.nanmedian(np.diff(coordinates)))
        if coordinates.size > 1
        else float(fallback_step)
    )
    return (
        float(coordinates[0] - 0.5 * step),
        float(coordinates[-1] + 0.5 * step),
    )


def _slice_recorder_video(
    metadata, recording, frames, times, field, offset_by_axis, resolution_um
):
    monitor = recording.monitor
    if not metadata.is_3d or frames.ndim != 3:
        raise ValueError(
            "save_field_video requires two-dimensional image frames; "
            "a 2D line recorder cannot be exported as a field video."
        )
    normal = monitor.plane_normal
    vertical, horizontal = plane_axes_3d(normal)
    region, coord0, coord1 = monitor_plane_coordinates_3d(
        metadata,
        monitor,
        region=recording.sample_region,
    )
    if frames.shape[1:] != (coord0.size, coord1.size):
        raise ValueError(
            f"Recorded {field!r} frame shape {frames.shape[1:]} does not match "
            f"the monitor plane {(coord0.size, coord1.size)}."
        )
    if region is not None and metadata.grid is not None:
        horizontal_interval = region.axis_interval(horizontal)
        vertical_interval = region.axis_interval(vertical)
        assert horizontal_interval is not None and vertical_interval is not None
        horizontal_edges = metadata.grid.axis_edges(horizontal)
        vertical_edges = metadata.grid.axis_edges(vertical)
        x_extent = (
            horizontal_edges[int(horizontal_interval.start)] / _UM
            - offset_by_axis[horizontal],
            horizontal_edges[int(horizontal_interval.stop)] / _UM
            - offset_by_axis[horizontal],
        )
        y_extent = (
            vertical_edges[int(vertical_interval.start)] / _UM
            - offset_by_axis[vertical],
            vertical_edges[int(vertical_interval.stop)] / _UM
            - offset_by_axis[vertical],
        )
    else:
        x_extent = _axis_extent(
            coord1 / _UM - offset_by_axis[horizontal],
            fallback_step=resolution_um,
        )
        y_extent = _axis_extent(
            coord0 / _UM - offset_by_axis[vertical],
            fallback_step=resolution_um,
        )
    plane_position = (
        float(region.plane_coord)
        if region is not None
        else float(monitor.plane_position)
    )
    position_um = plane_position / _UM - offset_by_axis[normal]
    return _VideoData(
        frames,
        times,
        (
            float(x_extent[0]),
            float(x_extent[1]),
            float(y_extent[0]),
            float(y_extent[1]),
        ),
        horizontal,
        vertical,
        f"{normal}={position_um:.3g} um",
    )


def _volume_recorder_video(
    metadata,
    frames,
    times,
    field,
    *,
    plane,
    index,
    offsets_um,
    offset_by_axis,
    resolution_um,
):
    if frames.ndim != 4:
        raise ValueError(f"Full-domain 3D frames must have rank 4, got {frames.shape}.")
    normal = str(plane).lower()
    if normal not in _AXIS_INDEX:
        raise ValueError("plane must be 'x', 'y', or 'z'.")
    frame_axis = _AXIS_INDEX[normal] + 1
    selected_index = frames.shape[frame_axis] // 2 if index is None else int(index)
    if not -frames.shape[frame_axis] <= selected_index < frames.shape[frame_axis]:
        raise IndexError(
            f"index {selected_index} is outside the {normal} axis with "
            f"length {frames.shape[frame_axis]}."
        )
    selected_index %= frames.shape[frame_axis]
    frames = np.take(frames, selected_index, axis=frame_axis)
    local_coordinates = (
        {
            axis: values / _UM
            for axis, values in component_coordinates_rectilinear(
                field, metadata.grid
            ).items()
        }
        if metadata.grid is not None
        else component_coordinates_3d_um(
            field,
            metadata.fields.grid_shape,
            resolution_um,
        )
    )
    coordinates = coordinates_in_public_frame(local_coordinates, offsets_um)
    vertical, horizontal = plane_axes_3d(normal)
    x_extent = _axis_extent(
        coordinates[horizontal],
        fallback_step=resolution_um,
    )
    y_extent = _axis_extent(
        coordinates[vertical],
        fallback_step=resolution_um,
    )
    position_um = coordinates[normal][selected_index]
    return _VideoData(
        frames,
        times,
        (*x_extent, *y_extent),
        horizontal,
        vertical,
        f"{normal}={position_um:.3g} um",
    )


def _domain_recorder_video(
    metadata, frames, times, field, offset_by_axis, resolution_um, offsets_um
):
    if frames.ndim != 3:
        raise ValueError(f"Full-domain 2D frames must have rank 3, got {frames.shape}.")
    vertical, horizontal = {
        "xy": ("y", "x"),
        "xz": ("z", "x"),
        "yz": ("z", "y"),
    }[metadata.plane_2d]
    local_coordinates = (
        {
            axis: values / _UM
            for axis, values in component_coordinates_rectilinear(
                field,
                metadata.grid,
                plane=metadata.plane_2d,
                polarization=metadata.polarization_2d,
            ).items()
        }
        if metadata.grid is not None
        else component_coordinates_2d_um(
            field,
            metadata.fields.grid_shape,
            resolution_um,
            metadata.plane_2d,
            metadata.polarization_2d,
        )
    )
    coordinates = coordinates_in_public_frame(local_coordinates, offsets_um)
    x_extent = _axis_extent(
        coordinates[horizontal],
        fallback_step=resolution_um,
    )
    y_extent = _axis_extent(
        coordinates[vertical],
        fallback_step=resolution_um,
    )
    return _VideoData(
        frames,
        times,
        (*x_extent, *y_extent),
        horizontal,
        vertical,
    )


def _video_data(results, recording, *, field, plane, index):
    metadata = results.metadata
    frames = np.asarray(recording.fields[field])
    times = np.asarray(recording.field_times, dtype=float)
    if frames.shape[0] != times.size or times.size == 0:
        raise RuntimeError(
            f"Recorder {recording.monitor.name!r} has no complete {field!r} frames."
        )

    offsets_um = tuple(value / _UM for value in metadata.coordinate_offset)
    offset_by_axis = dict(zip("xyz", offsets_um, strict=True))
    resolution_um = metadata.resolution / _UM
    monitor = recording.monitor

    if getattr(monitor, "region", "domain") == "slice":
        return _slice_recorder_video(
            metadata,
            recording,
            frames,
            times,
            field,
            offset_by_axis,
            resolution_um,
        )
    if metadata.is_3d:
        return _volume_recorder_video(
            metadata,
            frames,
            times,
            field,
            plane=plane,
            index=index,
            offsets_um=offsets_um,
            offset_by_axis=offset_by_axis,
            resolution_um=resolution_um,
        )
    return _domain_recorder_video(
        metadata,
        frames,
        times,
        field,
        offset_by_axis,
        resolution_um,
        offsets_um,
    )


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
