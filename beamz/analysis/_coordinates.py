"""Shared coordinate reconstruction for detached monitor analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beamz.design.grid import RectilinearGrid
from beamz.devices.monitors.geometry import retained_sample_region
from beamz.lattice import (
    component_coordinates_rectilinear,
    coordinates_in_public_frame,
    plane_axes_3d,
)


def field_coordinates(metadata, component, shape):
    """Return component coordinates in metres in the public frame."""
    grid = metadata.grid
    if grid is None:
        counts = tuple(reversed(metadata.fields.grid_shape))
        grid = RectilinearGrid.from_spacing(
            counts if len(counts) == 3 else (*counts, 1), metadata.resolution
        )
    coordinates = component_coordinates_rectilinear(
        component,
        grid,
        plane=None if len(shape) == 3 else metadata.plane_2d,
        polarization=metadata.polarization_2d,
    )
    return coordinates_in_public_frame(coordinates, metadata.coordinate_offset)


_UM = 1e-6
_AXIS_INDEX = {"z": 0, "y": 1, "x": 2}


@dataclass(frozen=True, slots=True)
class RecordedPlane:
    frames: np.ndarray
    times: np.ndarray
    extent: tuple[float, float, float, float]
    horizontal: str
    vertical: str
    slice_label: str | None = None


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


def recorded_plane(
    results, recording, *, field, plane="z", index=None, position=None, frame=None
):
    metadata = results.metadata
    frames = np.asarray(recording.fields[field])
    times = np.asarray(recording.field_times, dtype=float)
    if frame is None and (frames.shape[0] != times.size or times.size == 0):
        raise RuntimeError(
            f"Recorder {recording.monitor.name!r} has no complete {field!r} frames."
        )

    if frame is not None:
        selected = int(frame)
        times = times[selected][None] if times.size == frames.shape[0] else times[:0]
        frames = frames[selected][None, ...]

    offsets_um = tuple(value / _UM for value in metadata.coordinate_offset)
    offset_by_axis = dict(zip("xyz", offsets_um, strict=True))
    resolution_um = metadata.resolution / _UM
    monitor = recording.monitor

    if getattr(monitor, "region", "domain") == "slice":
        if not metadata.is_3d and frames.ndim == 2 and frame is not None:
            return RecordedPlane(
                frames[:, None, :],
                times,
                (0.0, float(max(frames.shape[-1] - 1, 1)), 0.0, 1.0),
                "sample",
                "sample",
            )
        if not metadata.is_3d or frames.ndim != 3:
            raise ValueError(
                "save_field_video requires two-dimensional image frames; "
                "a 2D line recorder cannot be exported as a field video."
            )
        normal = monitor.plane_normal
        vertical, horizontal = plane_axes_3d(normal)
        region = retained_sample_region(metadata, monitor, recording.sample_region)
        if region is None:
            raise ValueError("A recorded plane requires retained sampling geometry.")
        coord0, coord1 = region.sample_coordinates
        if frames.shape[1:] != (coord0.size, coord1.size):
            raise ValueError(
                f"Recorded {field!r} frame shape {frames.shape[1:]} does not match "
                f"the monitor plane {(coord0.size, coord1.size)}."
            )
        from beamz.analysis.plotting import _monitor_plane_display

        extent, _, _ = _monitor_plane_display(
            metadata,
            region,
            normal=normal,
            origin=metadata.coordinate_offset,
            coords=(coord0, coord1),
        )
        plane_position = float(region.plane_coord)
        position_um = plane_position / _UM - offset_by_axis[normal]
        return RecordedPlane(
            frames,
            times,
            extent,
            horizontal,
            vertical,
            f"{normal}={position_um:.3g} um",
        )

    spatial_shape = frames.shape[1:]
    if metadata.is_3d:
        if frames.ndim != 4:
            raise ValueError(
                f"Full-domain 3D frames must have rank 4, got {frames.shape}."
            )
        normal = str(plane).lower()
        if normal not in _AXIS_INDEX:
            raise ValueError("plane must be 'x', 'y', or 'z'.")
        frame_axis = _AXIS_INDEX[normal] + 1
        if position is not None:
            coordinates = field_coordinates(metadata, field, spatial_shape)[normal]
            # Match labeled nearest selection: an equidistant tie chooses the
            # larger coordinate. Video index selection remains exact indexing.
            distance = np.abs(coordinates - float(position))
            index = len(distance) - 1 - int(np.argmin(distance[::-1]))
        selected_index = frames.shape[frame_axis] // 2 if index is None else int(index)
        if not -frames.shape[frame_axis] <= selected_index < frames.shape[frame_axis]:
            raise IndexError(
                f"index {selected_index} is outside the {normal} axis with "
                f"length {frames.shape[frame_axis]}."
            )
        selected_index %= frames.shape[frame_axis]
        frames = np.take(frames, selected_index, axis=frame_axis)
        vertical, horizontal = plane_axes_3d(normal)
    else:
        if frames.ndim != 3:
            raise ValueError(
                f"Full-domain 2D frames must have rank 3, got {frames.shape}."
            )
        vertical, horizontal = {
            "xy": ("y", "x"),
            "xz": ("z", "x"),
            "yz": ("z", "y"),
        }[metadata.plane_2d]
    coordinates = {
        axis: values / _UM
        for axis, values in field_coordinates(metadata, field, spatial_shape).items()
    }
    x_extent = _axis_extent(coordinates[horizontal], fallback_step=resolution_um)
    y_extent = _axis_extent(coordinates[vertical], fallback_step=resolution_um)
    slice_label = (
        f"{normal}={coordinates[normal][selected_index]:.3g} um"
        if metadata.is_3d
        else None
    )
    return RecordedPlane(
        frames, times, (*x_extent, *y_extent), horizontal, vertical, slice_label
    )


__all__ = ["field_coordinates", "recorded_plane"]
