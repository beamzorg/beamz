from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from beamz.design.grid import GridAxis, RectilinearGrid

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


def _uniform_axis(step, count):
    return GridAxis(float(step) * np.arange(count + 1, dtype=float), "uniform")


def _axis_interval(axis, lower, upper, *, min_cells=1):
    start, stop = axis.interval(lower, upper, min_cells=min_cells)
    return SnappedInterval(
        start,
        stop,
        axis.step if axis.policy == "uniform" else float(np.min(np.diff(axis.edges))),
        physical_bounds=(axis.edge(start), axis.edge(stop)),
    )


def snap_edge_interval(lower, upper, step, count, *, min_cells=1):
    if count <= 0:
        return SnappedInterval(0, 0, float(step))
    return _axis_interval(_uniform_axis(step, count), lower, upper, min_cells=min_cells)


def snap_rectilinear_edge_interval(lower, upper, edges, *, min_cells=1):
    if len(edges) < 2:
        return SnappedInterval(0, 0, 0.0, tuple(edges))
    return _axis_interval(
        GridAxis(np.asarray(edges, dtype=float)), lower, upper, min_cells=min_cells
    )


def _grid_axes(grid, *, policy="edges"):
    return {name: grid.axis(name, policy=policy) for name in _AXES}


def _uniform_axes(shape, steps):
    grid = RectilinearGrid.from_spacing(tuple(reversed(shape)), tuple(steps))
    return _grid_axes(grid, policy="uniform")


def _snap_region(ndim, axis, coordinate, bounds, axes, *, companion_sign=None):
    index, coordinate = axes[axis].center(coordinate)
    companion = None
    if companion_sign is not None:
        companion = (
            max(0, index - 1)
            if companion_sign > 0.0
            else min(max(axes[axis].count - 2, 0), index + 1)
        )
    return SnappedRegion(
        ndim,
        axis,
        index,
        coordinate,
        {name: _axis_interval(axes[name], *limits) for name, limits in bounds.items()},
        companion,
        None if companion is None else axes[axis].edge(companion + 1),
    )


def _snap_line(start, end, axes, tolerance):
    for index, axis in enumerate(("x", "y")):
        if abs(float(start[index]) - float(end[index])) <= tolerance:
            tangent = "y" if axis == "x" else "x"
            other = 1 - index
            return _snap_region(
                2,
                axis,
                0.5 * (start[index] + end[index]),
                {tangent: (start[other], end[other])},
                axes,
            )
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
    return _snap_line(start, end, _grid_axes(grid), tolerance)


def snap_plane_region_grid(*, center, size, plane_normal: str, grid) -> SnappedRegion:
    """Snap a center/size plane using realized rectilinear edges."""
    axis = str(plane_normal).lower()
    bounds = {
        name: (float(center[i]) - 0.5 * size[i], float(center[i]) + 0.5 * size[i])
        for i, name in enumerate(_AXES)
        if name != axis
    }
    return _snap_region(3, axis, center[_axis_pos(axis)], bounds, _grid_axes(grid))


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


def mode_source_region(
    *,
    center,
    width,
    height,
    axis,
    direction_sign,
    grid_shape=None,
    resolution=None,
    is_3d,
    grid=None,
):
    if grid is None:
        if grid_shape is None or resolution is None:
            raise ValueError("Mode placement requires a grid or shape and resolution.")
        shape = grid_shape if is_3d else (1, *grid_shape)
        axes = _uniform_axes(shape, (resolution,) * 3)
        default_z = 0.5 * shape[0] * resolution
    else:
        axes = _grid_axes(grid)
        default_z = None
    return _snap_mode_region(
        center, width, height, axis, direction_sign, axes, is_3d, default_z
    )


def _snap_mode_region(
    center, width, height, axis, direction_sign, axes, is_3d, default_z=None
):
    axis = str(axis).lower()
    bounds = {}
    for name, extent in zip(
        (name for name in _AXES if name != axis),
        (width, height if height is not None else width),
        strict=True,
    ):
        if not is_3d and name == "z":
            continue
        coordinate = (
            default_z
            if name == "z" and len(center) < 3 and default_z is not None
            else float(center[_axis_pos(name)])
        )
        bounds[name] = (coordinate - 0.5 * extent, coordinate + 0.5 * extent)
    return _snap_region(
        3 if is_3d else 2,
        axis,
        center[_axis_pos(axis)],
        bounds,
        axes,
        companion_sign=direction_sign,
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
    axes = _uniform_axes((1, *shape), (dx, dy, 1.0))
    return _snap_line(start, end, axes, 1e-9 * max(abs(dx), abs(dy), 1.0))


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
    axes = _uniform_axes(shape, (dx, dy, dz))
    axis = str(plane_normal).lower()

    starts: dict[str, float] = dict(zip(_AXES, map(float, start), strict=True))
    transverse_axes = tuple(name for name in _AXES if name != axis)
    if end is None:
        plane_pos = starts[axis]
        bounds = {
            name: (starts[name], starts[name] + float(extent))
            for name, extent in zip(
                transverse_axes, size if size is not None else (0.0, 0.0), strict=True
            )
        }
    else:
        ends: dict[str, float] = dict(zip(_AXES, map(float, end), strict=True))
        plane_pos = 0.5 * (starts[axis] + ends[axis])
        bounds = {name: (starts[name], ends[name]) for name in transverse_axes}

    return _snap_region(3, axis, plane_pos, bounds, axes)


def line_region_coordinates(region, *, grid=None, steps=None, normal_coord=None):
    """Coordinates for a snapped line, with optional requested normal position."""
    normal = region.normal_axis
    tangent = "y" if normal == "x" else "x"
    interval = region.axis_interval(tangent)
    if interval is None:
        return None
    if grid is None:
        assert steps is not None
        values = (np.arange(interval.start, interval.stop, dtype=float) + 0.5) * steps[
            tangent
        ]
    else:
        values = np.asarray(grid.centers(tangent))[interval.start : interval.stop]
    constant = np.full(
        values.shape, region.plane_coord if normal_coord is None else normal_coord
    )
    return (constant, values) if normal == "x" else (values, constant)


def line_region_points(region: SnappedRegion) -> list[tuple[int, int]]:
    if region.normal_axis == "x":
        interval = region.axis_interval("y")
        if interval is None:
            return []
        return [
            (int(region.plane_index), int(y_idx))
            for y_idx in range(int(interval.start), int(interval.stop))
        ]
    interval = region.axis_interval("x")
    if interval is None:
        return []
    return [
        (int(x_idx), int(region.plane_index))
        for x_idx in range(int(interval.start), int(interval.stop))
    ]


def plane_region_slices(
    region: SnappedRegion,
) -> tuple[int | slice, int | slice, int | slice]:
    indices: list[int | slice] = []
    for axis in reversed(_AXES):
        if axis == region.normal_axis:
            indices.append(int(region.plane_index))
        else:
            interval = region.axis_interval(axis)
            indices.append(interval.as_slice() if interval is not None else slice(0, 0))
    return indices[0], indices[1], indices[2]


def _axis_pos(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[str(axis).lower()]
