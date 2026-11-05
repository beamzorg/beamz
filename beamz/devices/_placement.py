from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class SnappedInterval:
    start: int
    stop: int
    step: float
    edges: tuple[float, ...] | None = None
    physical_bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "stop", int(self.stop))
        object.__setattr__(self, "step", float(self.step))
        if self.edges is not None:
            object.__setattr__(
                self, "edges", tuple(float(value) for value in self.edges)
            )
        if self.physical_bounds is not None:
            raw_bounds = tuple(float(value) for value in self.physical_bounds)
            if len(raw_bounds) != 2 or not np.all(np.isfinite(raw_bounds)):
                raise ValueError(
                    "SnappedInterval physical_bounds must contain two finite values."
                )
            bounds = (raw_bounds[0], raw_bounds[1])
            if bounds[1] < bounds[0]:
                raise ValueError(
                    "SnappedInterval physical_bounds must be nondecreasing."
                )
            object.__setattr__(self, "physical_bounds", bounds)

    @property
    def lower(self) -> float:
        if self.edges is not None:
            return float(self.edges[int(self.start)])
        if self.physical_bounds is not None:
            return self.physical_bounds[0]
        return float(self.start) * float(self.step)

    @property
    def upper(self) -> float:
        if self.edges is not None:
            return float(self.edges[int(self.stop)])
        if self.physical_bounds is not None:
            return self.physical_bounds[1]
        return float(self.stop) * float(self.step)

    @property
    def center(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def size(self) -> float:
        return max(0.0, self.upper - self.lower)

    def as_slice(self) -> slice:
        return slice(int(self.start), int(self.stop))


@dataclass(frozen=True)
class SnappedRegion:
    ndim: int
    normal_axis: str
    plane_index: int
    plane_coord: float
    intervals: Mapping[str, SnappedInterval] = field(default_factory=dict)
    companion_index: int | None = None
    companion_coord: float | None = None

    def __post_init__(self) -> None:
        normal_axis = str(self.normal_axis).lower()
        if normal_axis not in _AXES:
            raise ValueError(
                f"Snapped region normal axis must be one of {_AXES}; "
                f"got {self.normal_axis!r}."
            )
        intervals = {}
        for axis, interval in self.intervals.items():
            normalized_axis = str(axis).lower()
            if normalized_axis not in _AXES:
                raise ValueError(
                    f"Snapped interval axis must be one of {_AXES}; got {axis!r}."
                )
            if not isinstance(interval, SnappedInterval):
                raise TypeError(
                    "SnappedRegion intervals must be SnappedInterval values."
                )
            intervals[normalized_axis] = interval
        object.__setattr__(self, "ndim", int(self.ndim))
        object.__setattr__(self, "normal_axis", normal_axis)
        object.__setattr__(self, "plane_index", int(self.plane_index))
        object.__setattr__(self, "plane_coord", float(self.plane_coord))
        object.__setattr__(self, "intervals", MappingProxyType(intervals))
        if self.companion_index is not None:
            object.__setattr__(self, "companion_index", int(self.companion_index))
        if self.companion_coord is not None:
            object.__setattr__(self, "companion_coord", float(self.companion_coord))

    def axis_interval(self, axis: str) -> SnappedInterval | None:
        return self.intervals.get(str(axis).lower())

    def axis_coord(self, axis: str) -> float:
        axis = str(axis).lower()
        if axis == self.normal_axis:
            return float(self.plane_coord)
        interval = self.axis_interval(axis)
        if interval is None:
            return 0.0
        return float(interval.center)

    def axis_bounds(self, axis: str) -> tuple[float, float]:
        axis = str(axis).lower()
        if axis == self.normal_axis:
            coord = float(self.plane_coord)
            return coord, coord
        interval = self.axis_interval(axis)
        if interval is None:
            return 0.0, 0.0
        return float(interval.lower), float(interval.upper)

    @property
    def start(self) -> tuple[float, ...]:
        return tuple(self.axis_bounds(axis)[0] for axis in _AXES[: self.ndim])

    @property
    def end(self) -> tuple[float, ...]:
        return tuple(self.axis_bounds(axis)[1] for axis in _AXES[: self.ndim])

    @property
    def center(self) -> tuple[float, ...]:
        return tuple(self.axis_coord(axis) for axis in _AXES[: self.ndim])

    @property
    def size(self) -> tuple[float, ...]:
        return tuple(
            self.axis_bounds(axis)[1] - self.axis_bounds(axis)[0]
            for axis in _AXES[: self.ndim]
        )


def snap_cell_center(coord: float, step: float, count: int) -> tuple[int, float]:
    if count <= 0:
        return 0, 0.0
    idx = int(np_round_half_even((float(coord) / float(step)) - 0.5))
    idx = max(0, min(idx, count - 1))
    return idx, (float(idx) + 0.5) * float(step)


def snap_edge_interval(
    lower: float,
    upper: float,
    step: float,
    count: int,
    *,
    min_cells: int = 1,
) -> SnappedInterval:
    if count <= 0:
        return SnappedInterval(0, 0, float(step))

    lo = min(float(lower), float(upper))
    hi = max(float(lower), float(upper))
    eps = 1e-12 * max(abs(lo), abs(hi), abs(float(step)), 1.0)

    start = int(math.floor((lo + eps) / float(step)))
    stop = int(math.ceil((hi - eps) / float(step)))

    start = max(0, min(start, count))
    stop = max(0, min(stop, count))

    need = max(1, int(min_cells))
    if stop - start < need:
        center = 0.5 * (lo + hi)
        cell = int(math.floor(center / float(step)))
        cell = max(0, min(cell, count - 1))
        start = cell
        stop = min(count, cell + need)
        start = max(0, stop - need)

    return SnappedInterval(start=int(start), stop=int(stop), step=float(step))


def snap_rectilinear_cell_center(coord: float, edges) -> tuple[int, float]:
    """Snap a coordinate to the nearest physical cell center."""
    edge_array = np.asarray(edges, dtype=np.float64)
    centers = 0.5 * (edge_array[:-1] + edge_array[1:])
    if centers.size == 0:
        return 0, 0.0
    index = int(np.argmin(np.abs(centers - float(coord))))
    return index, float(centers[index])


def snap_rectilinear_edge_interval(
    lower: float,
    upper: float,
    edges,
    *,
    min_cells: int = 1,
) -> SnappedInterval:
    """Snap physical bounds to the overlapping rectilinear cells."""
    edge_array = np.asarray(edges, dtype=np.float64)
    count = max(0, edge_array.size - 1)
    if count == 0:
        return SnappedInterval(0, 0, 0.0, tuple(edge_array))
    lo, hi = sorted((float(lower), float(upper)))
    start = int(np.searchsorted(edge_array, lo, side="right") - 1)
    stop = int(np.searchsorted(edge_array, hi, side="left"))
    start = max(0, min(start, count))
    stop = max(0, min(stop, count))
    need = max(1, int(min_cells))
    if stop - start < need:
        cell, _ = snap_rectilinear_cell_center(0.5 * (lo + hi), edge_array)
        start = max(0, min(cell, count - need))
        stop = min(count, start + need)
    representative = float(np.min(np.diff(edge_array)))
    return SnappedInterval(
        int(start),
        int(stop),
        representative,
        tuple(float(value) for value in edge_array),
    )


def _aligned_line(start, end, tolerance):
    """Return ``(normal, center, varying bounds, tangential axis)`` for an axis-aligned line."""
    x0, y0, x1, y1 = (float(value) for value in (*start[:2], *end[:2]))
    if abs(x0 - x1) <= tolerance:
        return "x", 0.5 * (x0 + x1), (y0, y1), "y"
    if abs(y0 - y1) <= tolerance:
        return "y", 0.5 * (y0 + y1), (x0, x1), "x"
    return None


def snap_axis_aligned_line_region_grid(
    start: tuple[float, ...],
    end: tuple[float, ...],
    grid,
) -> SnappedRegion | None:
    """Snap a 2D line using realized x/y cell edges."""
    tolerance = 1e-12 * max(
        *map(abs, (*start[:2], *end[:2])), grid.minimum_spacing, 1.0
    )
    aligned = _aligned_line(start, end, tolerance)
    if aligned is None:
        return None
    normal, center, (lo, hi), tangential = aligned
    edges = {"x": grid.x_edges, "y": grid.y_edges}
    index, coordinate = snap_rectilinear_cell_center(center, edges[normal])
    return SnappedRegion(
        2,
        normal,
        index,
        coordinate,
        {tangential: snap_rectilinear_edge_interval(lo, hi, edges[tangential])},
    )


def snap_plane_region_grid(*, center, size, plane_normal: str, grid) -> SnappedRegion:
    """Snap a center/size plane using realized rectilinear edges."""
    center = tuple(float(value) for value in center)
    size = tuple(float(value) for value in size)
    axis = str(plane_normal).lower()
    edges = {"x": grid.x_edges, "y": grid.y_edges, "z": grid.z_edges}
    axis_index = {"x": 0, "y": 1, "z": 2}
    plane_index, plane_coord = snap_rectilinear_cell_center(
        center[axis_index[axis]], edges[axis]
    )
    intervals = {}
    for tangential in _AXES:
        if tangential == axis:
            continue
        index = axis_index[tangential]
        half = 0.5 * size[index]
        intervals[tangential] = snap_rectilinear_edge_interval(
            center[index] - half,
            center[index] + half,
            edges[tangential],
        )
    return SnappedRegion(3, axis, plane_index, plane_coord, intervals)


def _launch_companion(plane_index: int, direction_sign: float, count: int) -> int:
    limit = max(int(count) - 2, 0)
    if direction_sign > 0.0:
        return max(0, int(plane_index) - 1)
    return min(limit, int(plane_index) + 1)


def _mode_transverse_extents(axis: str, width: float, height: float | None):
    span = float(width if height is None else height)
    width = float(width)
    return {
        "x": (("y", width), ("z", span)),
        "y": (("x", width), ("z", span)),
        "z": (("x", width), ("y", span)),
    }[axis]


def snap_centered_extent(
    center: float,
    extent: float,
    step: float,
    count: int,
    *,
    min_cells: int = 1,
) -> SnappedInterval:
    half = 0.5 * float(extent)
    return snap_edge_interval(
        float(center) - half,
        float(center) + half,
        step,
        count,
        min_cells=min_cells,
    )


def snap_mode_source_region(
    *,
    center: tuple[float, ...],
    width: float,
    height: float | None,
    axis: str,
    direction_sign: float,
    grid_shape: tuple[int, ...],
    resolution: float,
    is_3d: bool,
) -> SnappedRegion:
    axis = str(axis).lower()
    counts = _counts_by_axis(grid_shape, is_3d)
    plane_index, plane_coord = snap_cell_center(
        center[_axis_pos(axis)], resolution, counts[axis]
    )
    companion_index = _launch_companion(plane_index, direction_sign, counts[axis])
    companion_coord = (float(companion_index) + 1.0) * float(resolution)

    intervals: dict[str, SnappedInterval] = {}
    for transverse, extent in _mode_transverse_extents(axis, width, height):
        if not is_3d and transverse == "z":
            continue
        if transverse == "z" and len(center) <= 2:
            coord = 0.5 * counts["z"] * float(resolution)
        else:
            coord = float(center[_axis_pos(transverse)])
        intervals[transverse] = snap_centered_extent(
            coord, extent, resolution, counts[transverse]
        )

    return SnappedRegion(
        ndim=3 if is_3d else 2,
        normal_axis=axis,
        plane_index=int(plane_index),
        plane_coord=float(plane_coord),
        intervals=intervals,
        companion_index=int(companion_index),
        companion_coord=float(companion_coord),
    )


def snap_mode_source_region_grid(
    *,
    center: tuple[float, ...],
    width: float,
    height: float | None,
    axis: str,
    direction_sign: float,
    grid,
    is_3d: bool,
) -> SnappedRegion:
    """Snap a mode launch to the realized rectilinear Yee grid."""
    axis = str(axis).lower()
    edges = {name: np.asarray(grid.axis_edges(name)) for name in _AXES}
    plane_index, plane_coord = snap_rectilinear_cell_center(
        center[_axis_pos(axis)], edges[axis]
    )
    counts = {name: int(values.size - 1) for name, values in edges.items()}
    companion_index = _launch_companion(plane_index, direction_sign, counts[axis])
    companion_coord = float(edges[axis][companion_index + 1])
    intervals = {}
    for transverse, extent in _mode_transverse_extents(axis, width, height):
        if not is_3d and transverse == "z":
            continue
        coord = float(center[_axis_pos(transverse)])
        intervals[transverse] = snap_rectilinear_edge_interval(
            coord - 0.5 * extent,
            coord + 0.5 * extent,
            edges[transverse],
        )
    return SnappedRegion(
        ndim=3 if is_3d else 2,
        normal_axis=axis,
        plane_index=plane_index,
        plane_coord=plane_coord,
        intervals=intervals,
        companion_index=companion_index,
        companion_coord=companion_coord,
    )


def snap_axis_aligned_line_region(
    start: tuple[float, ...],
    end: tuple[float, ...],
    dx: float,
    dy: float,
    shape: tuple[int, ...],
) -> SnappedRegion | None:
    if len(start) < 2 or len(end) < 2 or len(shape) != 2:
        raise ValueError(
            "A 2D line region requires two coordinates and a rank-2 shape."
        )
    ny, nx = shape
    aligned = _aligned_line(start, end, 1e-9 * max(abs(dx), abs(dy), 1.0))
    if aligned is None:
        return None
    normal, center, (lo, hi), tangential = aligned
    step = {"x": float(dx), "y": float(dy)}
    count = {"x": nx, "y": ny}
    plane_index, plane_coord = snap_cell_center(center, step[normal], count[normal])
    return SnappedRegion(
        ndim=2,
        normal_axis=normal,
        plane_index=int(plane_index),
        plane_coord=float(plane_coord),
        intervals={
            tangential: snap_edge_interval(lo, hi, step[tangential], count[tangential])
        },
    )


def snap_plane_region(
    *,
    start: tuple[float, ...],
    end: tuple[float, ...] | None,
    plane_normal: str,
    size: tuple[float, ...] | None,
    dx: float,
    dy: float,
    dz: float,
    shape: tuple[int, ...],
) -> SnappedRegion:
    if len(start) != 3 or (end is not None and len(end) != 3) or len(shape) != 3:
        raise ValueError("A 3D plane region requires rank-3 coordinates and shape.")
    if size is not None and len(size) != 2:
        raise ValueError("A 3D plane size requires two tangential extents.")
    nz, ny, nx = shape
    counts = {"x": nx, "y": ny, "z": nz}
    steps = {"x": float(dx), "y": float(dy), "z": float(dz)}
    axis = str(plane_normal).lower()

    origin = {name: float(value) for name, value in zip(_AXES, start, strict=True)}
    tangential = [name for name in _AXES if name != axis]
    if end is None:
        spans = (float(size[0]), float(size[1])) if size is not None else (0.0, 0.0)
        plane_pos = origin[axis]
        bounds = {
            name: (origin[name], origin[name] + span)
            for name, span in zip(tangential, spans, strict=True)
        }
    else:
        far = {name: float(value) for name, value in zip(_AXES, end, strict=True)}
        plane_pos = 0.5 * (origin[axis] + far[axis])
        bounds = {name: (origin[name], far[name]) for name in tangential}

    plane_index, plane_coord = snap_cell_center(plane_pos, steps[axis], counts[axis])
    intervals = {
        name: snap_edge_interval(lo, hi, steps[name], counts[name])
        for name, (lo, hi) in bounds.items()
    }
    return SnappedRegion(
        ndim=3,
        normal_axis=axis,
        plane_index=int(plane_index),
        plane_coord=float(plane_coord),
        intervals=intervals,
    )


def line_region_points(region: SnappedRegion) -> list[tuple[int, int]]:
    tangential = "y" if region.normal_axis == "x" else "x"
    interval = region.axis_interval(tangential)
    if interval is None:
        return []
    fixed = int(region.plane_index)
    varying = range(int(interval.start), int(interval.stop))
    if region.normal_axis == "x":
        return [(fixed, int(index)) for index in varying]
    return [(int(index), fixed) for index in varying]


def plane_region_slices(
    region: SnappedRegion,
) -> tuple[int | slice, int | slice, int | slice]:
    def slot(axis: str):
        if axis == region.normal_axis:
            return int(region.plane_index)
        interval = region.axis_interval(axis)
        return interval.as_slice() if interval is not None else slice(0, 0)

    z, y, x = (slot(axis) for axis in ("z", "y", "x"))
    return z, y, x


def np_round_half_even(value: float) -> int:
    return int(round(float(value)))


def _axis_pos(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[str(axis).lower()]


def _counts_by_axis(grid_shape: tuple[int, ...], is_3d: bool) -> dict[str, int]:
    if is_3d:
        nz, ny, nx = (int(v) for v in grid_shape)
        return {"x": nx, "y": ny, "z": nz}
    ny, nx = (int(v) for v in grid_shape)
    return {"x": nx, "y": ny, "z": 1}
