"""Shared Yee-lattice geometry, coordinates, sampling, and material colocation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0, µm

_FIELD_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_STORED_AXES = ("z", "y", "x")
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_PHYSICAL_AXES_2D = {
    "xy": ("x", "y", "z"),
    "xz": ("x", "z", "y"),
    "yz": ("y", "z", "x"),
}
_TANGENTIAL_AXES = {"x": ("z", "y"), "y": ("z", "x"), "z": ("y", "x")}
# Row/column labels for a 2D plane. Samples still come from stored grid y, then x.
_PLANE_SAMPLE_AXES = {"xy": ("y", "x"), "yz": ("z", "y"), "xz": ("z", "x")}
_COMPONENT_AXIS_OFFSETS_3D = {
    "Ex": {"z": 0.0, "y": 0.0, "x": 0.5},
    "Ey": {"z": 0.0, "y": 0.5, "x": 0.0},
    "Ez": {"z": 0.5, "y": 0.0, "x": 0.0},
    "Hx": {"z": 0.5, "y": 0.5, "x": 0.0},
    "Hy": {"z": 0.5, "y": 0.0, "x": 0.5},
    "Hz": {"z": 0.0, "y": 0.5, "x": 0.5},
}
# Canonical xy offsets as (row, column). A zero offset keeps the extra wall sample.
_OFFSETS_2D = {
    "Ez": (0.0, 0.0),
    "Hx": (0.5, 0.0),
    "Hy": (0.0, 0.5),
    "Ex": (0.0, 0.5),
    "Ey": (0.5, 0.0),
    "Hz": (0.5, 0.5),
}
# public, canonical, orientation. One row per active component on a 2D plane.
_COMPONENT_2D = {
    "tm": {
        "xy": (("Ez", "Ez", 1.0), ("Hx", "Hx", 1.0), ("Hy", "Hy", 1.0)),
        "yz": (("Ex", "Ez", 1.0), ("Hy", "Hx", 1.0), ("Hz", "Hy", 1.0)),
        "xz": (("Ey", "Ez", -1.0), ("Hx", "Hx", 1.0), ("Hz", "Hy", 1.0)),
    },
    "te": {
        "xy": (("Ex", "Ex", 1.0), ("Ey", "Ey", 1.0), ("Hz", "Hz", 1.0)),
        "yz": (("Ey", "Ex", 1.0), ("Ez", "Ey", 1.0), ("Hx", "Hz", 1.0)),
        "xz": (("Ex", "Ex", 1.0), ("Ez", "Ey", 1.0), ("Hy", "Hz", -1.0)),
    },
}
# Magnetic curl components: (d field_i / stored_axis) - (d field_j / stored_axis).
# Field indices are (x, y, z) = (0, 1, 2); stored axes are (z, y, x) = (0, 1, 2).
_CURL_PAIRS = (
    ((2, 1), (1, 0)),
    ((0, 0), (2, 2)),
    ((1, 2), (0, 1)),
)


def _h_view_name(component_index: int, axis: int) -> str:
    return f"h{'xyz'[component_index]}_{'zyx'[axis]}"


def _component_rows_2d(plane: str, polarization: str):
    return _COMPONENT_2D[normalize_polarization_2d(polarization)][str(plane).lower()]


def component_axis_offsets_3d(component: str) -> dict[str, float]:
    """Return physical Yee offsets, in grid-cell units, for a 3D component."""
    try:
        return dict(_COMPONENT_AXIS_OFFSETS_3D[component])
    except KeyError as exc:
        raise ValueError(f"Unsupported component {component!r}") from exc


def component_shape_3d(
    component: str, grid_shape: tuple[int, ...]
) -> tuple[int, int, int]:
    """Return the complete stored 3D Yee support for a field component."""
    # Material cells are the common basis; complete supports retain both domain walls
    # instead of introducing a second expanded representation only for PEC cavities.
    if len(grid_shape) != 3:
        raise ValueError(f"A 3D grid shape must have three dimensions: {grid_shape!r}")
    offsets = component_axis_offsets_3d(component)
    shape = tuple(
        int(size) + int(offsets[axis] == 0.0)
        for axis, size in zip(_STORED_AXES, grid_shape, strict=True)
    )
    return shape[0], shape[1], shape[2]


def component_coordinates_3d_um(
    component: str,
    grid_shape: tuple[int, ...],
    dx_um: float,
) -> dict[str, np.ndarray]:
    """Return Beamz raw Yee sample coordinates for a stored 3D component."""

    # Derive coordinates from the same shape/offset contract used by storage allocation.
    shape = component_shape_3d(component, grid_shape)
    offsets = component_axis_offsets_3d(component)
    return {
        axis: (np.arange(length, dtype=np.float64) + offsets[axis]) * dx_um
        for axis, length in zip(_STORED_AXES, shape, strict=True)
    }


def _grid_offset_samples(
    grid, axis: str, offset: float, *, dtype: np.dtype | type | None = np.float64
):
    edges = grid.axis_edges(axis)
    edges = np.asarray(edges) if dtype is None else np.asarray(edges, dtype=dtype)
    return edges if offset == 0.0 else 0.5 * (edges[:-1] + edges[1:])


def component_coordinates_rectilinear(
    component: str,
    grid,
    *,
    plane: str | None = None,
    polarization: str = "tm",
) -> dict[str, np.ndarray]:
    """Return exact physical Yee coordinates from rectilinear grid edges."""
    if plane is None:
        offsets = component_axis_offsets_3d(component)
        return {
            axis: _grid_offset_samples(grid, axis, offsets[axis])
            for axis in _STORED_AXES
        }
    normalized_plane = str(plane).lower()
    canonical = canonical_component_2d(component, normalized_plane, polarization)
    if canonical is None:
        return {}
    row_offset, column_offset = _OFFSETS_2D[canonical]
    row_axis, column_axis = _PLANE_SAMPLE_AXES[normalized_plane]
    return {
        row_axis: _grid_offset_samples(grid, "y", row_offset),
        column_axis: _grid_offset_samples(grid, "x", column_offset),
    }


def grid_axes_in_physical_frame_2d(plane: str) -> tuple[str, str, str]:
    """Map stored rectilinear ``(x, y, z)`` axes to physical axes for a 2D plane."""
    try:
        return _PHYSICAL_AXES_2D[str(plane).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported 2D plane {plane!r}.") from exc


def grid_vector_to_physical_2d(
    values: tuple[float, float, float], plane: str
) -> tuple[float, float, float]:
    """Permute a stored-grid vector into public physical ``(x, y, z)`` order."""
    stored = tuple(float(value) for value in values)
    if len(stored) != 3:
        raise ValueError("A stored-grid vector must contain three values.")
    physical = dict.fromkeys("xyz", 0.0)
    for axis, value in zip(grid_axes_in_physical_frame_2d(plane), stored, strict=True):
        physical[axis] = value
    return physical["x"], physical["y"], physical["z"]


def physical_vector_to_grid_2d(
    values: tuple[float, float, float], plane: str
) -> tuple[float, float, float]:
    """Permute a public physical vector into stored-grid ``(x, y, z)`` order."""
    physical = dict(zip("xyz", map(float, values), strict=True))
    grid_x, grid_y, grid_z = grid_axes_in_physical_frame_2d(plane)
    return physical[grid_x], physical[grid_y], physical[grid_z]


def in_plane_vector_2d(
    values: tuple[float, float, float], plane: str
) -> tuple[float, float]:
    """Select the public offsets corresponding to stored grid ``(x, y)`` order."""
    grid_x, grid_y, _grid_z = physical_vector_to_grid_2d(values, plane)
    return grid_x, grid_y


def coordinates_in_public_frame(
    coordinates: Mapping[str, Any],
    coordinate_offset: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    """Translate solver-local axis coordinates into the public coordinate frame.

    Simulation devices are lowered according to ``local = public + offset``.
    Analysis therefore applies the inverse translation to exact Yee coordinates.
    """

    offsets = dict(zip("xyz", map(float, coordinate_offset), strict=True))
    return {
        axis: np.asarray(values, dtype=np.float64) - offsets.get(axis, 0.0)
        for axis, values in coordinates.items()
    }


def common_grid_shape_3d(fields) -> tuple[int, int, int]:
    """Return the largest logical Yee component support for a field container."""
    shapes = getattr(fields, "_logical_component_shapes", None)
    if not shapes:
        shapes = getattr(fields, "component_shapes", None)
    if not shapes:
        shapes = {
            name: tuple(getattr(fields, name).shape)
            for name in _FIELD_COMPONENTS
            if hasattr(fields, name)
        }
    rank3 = [
        tuple(int(v) for v in shape) for shape in shapes.values() if len(shape) == 3
    ]
    if not rank3:
        fallback = getattr(fields, "grid_shape", np.asarray(fields.permittivity).shape)
        if len(fallback) != 3:
            raise ValueError("A common 3D Yee shape requires rank-3 fields.")
        return int(fallback[0]), int(fallback[1]), int(fallback[2])
    maxima = tuple(max(shape[axis] for shape in rank3) for axis in range(3))
    return maxima[0], maxima[1], maxima[2]


def plane_axes_3d(normal_axis: str) -> tuple[str, str]:
    """Return tangential axes in stored plane-array order."""
    try:
        return _TANGENTIAL_AXES[str(normal_axis).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported plane normal {normal_axis!r}.") from exc


def _uniform_axis_centers(lower: float, upper: float, count: int) -> np.ndarray:
    count = max(0, int(count))
    if count == 0:
        return np.zeros((0,), dtype=np.float64)
    lower, upper = float(lower), float(upper)
    if count == 1:
        return np.asarray([0.5 * (lower + upper)], dtype=np.float64)
    step = (upper - lower) / float(count)
    return lower + (np.arange(count, dtype=np.float64) + 0.5) * step


def yee_plane_coordinates_3d(center, size, normal_axis: str, region, grid=None):
    """Return tangential cell-center coordinates for a snapped physical plane."""
    center = tuple(float(value) for value in center)
    size = tuple(float(value) for value in size)
    axis0, axis1 = plane_axes_3d(normal_axis)
    coordinates = []
    for axis in (axis0, axis1):
        index = _AXIS_INDEX[axis]
        interval = region.axis_interval(axis)
        if interval is None:
            raise ValueError(f"Plane is missing its {axis!r} tangential interval.")
        if grid is not None:
            coordinates.append(
                np.asarray(grid.centers(axis))[int(interval.start) : int(interval.stop)]
            )
        else:
            lower = center[index] - 0.5 * size[index]
            upper = center[index] + 0.5 * size[index]
            coordinates.append(
                _uniform_axis_centers(
                    lower, upper, int(interval.stop) - int(interval.start)
                )
            )
    return coordinates[0], coordinates[1]


def linear_interpolation_plan(source, target):
    source = np.asarray(source, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if source.size == 0:
        raise ValueError("Yee interpolation source coordinates cannot be empty.")
    if source.size == 1:
        indices = np.zeros(target.size, dtype=np.int32)
        return (
            indices,
            indices,
            np.ones(target.size, dtype=np.float32),
            np.zeros(target.size, dtype=np.float32),
        )
    high = np.clip(np.searchsorted(source, target, side="right"), 1, source.size - 1)
    low = high - 1
    low[target <= source[0]] = 0
    high[target <= source[0]] = 0
    low[target >= source[-1]] = source.size - 1
    high[target >= source[-1]] = source.size - 1
    denominator = source[high] - source[low]
    alpha = np.divide(
        target - source[low],
        denominator,
        out=np.zeros_like(target),
        where=(high != low) & (np.abs(denominator) > 0.0),
    )
    weight_high = alpha.astype(np.float32)
    weight_low = (1.0 - alpha).astype(np.float32)
    weight_low[high == low] = 1.0
    weight_high[high == low] = 0.0
    return low.astype(np.int32), high.astype(np.int32), weight_low, weight_high


def _uniform_component_coordinates_m(component, grid_shape, resolution):
    return {
        axis: values * float(µm)
        for axis, values in component_coordinates_3d_um(
            component, grid_shape, float(resolution / µm)
        ).items()
    }


def normal_plane_wave_transfer(
    component,
    *,
    axis,
    plane_position,
    wave_number,
    grid_shape,
    resolution,
    grid=None,
):
    """Normal gather response to exp(+i k (x - plane)) on a Yee component.

    ``wave_number`` is signed. Modal projection applies this measurement factor
    before normalizing the measured basis to unit discrete Poynting flux.
    Transverse quadrature and temporal leapfrog phases are separate operations.
    """
    coordinates = (
        component_coordinates_rectilinear(component, grid)
        if grid is not None
        else _uniform_component_coordinates_m(component, grid_shape, resolution)
    )[axis]
    low, high, wlow, whigh = linear_interpolation_plan(coordinates, [plane_position])
    phase = np.exp(1j * wave_number * (coordinates - plane_position))
    return complex((wlow * phase[low] + whigh * phase[high])[0])


def sampling_coordinates(indices, shape):
    """Decode static global indices before JAX can narrow them to 32 bits.

    Large sharded domains need wide host flat indices even when each dimension
    and each local native buffer fits 32-bit addressing. Coordinates stay small.
    """
    if isinstance(indices, np.ndarray) or math.prod(shape) > np.iinfo(np.int32).max:
        return tuple(
            jnp.asarray(value, dtype=jnp.int32)
            for value in np.unravel_index(np.asarray(indices), shape)
        )
    return jnp.unravel_index(indices, shape)


def _component_plane_plan_3d(
    component: str,
    *,
    normal_axis: str,
    plane_position: float,
    coordinates: tuple[np.ndarray, np.ndarray],
    resolution: float,
    grid_shape: tuple[int, int, int],
    field_shape: tuple[int, int, int],
    grid=None,
):
    axis0, axis1 = plane_axes_3d(normal_axis)
    if grid is None:
        component_coords = _uniform_component_coordinates_m(
            component, grid_shape, resolution
        )
    else:
        offsets = component_axis_offsets_3d(component)
        component_coords = {
            axis: _grid_offset_samples(grid, axis, offsets[axis], dtype=None)
            for axis in ("x", "y", "z")
        }
    lengths: dict[str, int] = dict(zip(_STORED_AXES, field_shape, strict=True))
    source0 = np.asarray(component_coords[axis0])[: lengths[axis0]]
    source1 = np.asarray(component_coords[axis1])[: lengths[axis1]]
    source_normal = np.asarray(component_coords[normal_axis])[: lengths[normal_axis]]
    grid0, grid1 = np.meshgrid(*coordinates, indexing="ij")
    target0, target1 = grid0.reshape(-1), grid1.reshape(-1)
    target_normal = np.full_like(target0, float(plane_position))
    plans = (
        linear_interpolation_plan(source0, target0),
        linear_interpolation_plan(source1, target1),
        linear_interpolation_plan(source_normal, target_normal),
    )
    corner_bits = np.asarray(
        [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)],
        dtype=np.int32,
    )
    indices = []
    weights = []
    for dimension, (low, high, weight_low, weight_high) in enumerate(plans):
        bit = corner_bits[:, dimension][None, :]
        indices.append(np.where(bit == 0, low[:, None], high[:, None]))
        weights.append(np.where(bit == 0, weight_low[:, None], weight_high[:, None]))
    index_map = {axis0: indices[0], axis1: indices[1], normal_axis: indices[2]}
    flat_indices = np.asarray(
        np.ravel_multi_index(
            (index_map["z"], index_map["y"], index_map["x"]),
            dims=field_shape,
        ),
        dtype=np.int64 if math.prod(field_shape) > np.iinfo(np.int32).max else np.int32,
    )
    combined_weights = np.prod(np.stack(weights), axis=0).astype(np.float32)
    nonzero = np.abs(combined_weights) > 4.0 * np.finfo(np.float32).eps
    combined_weights = np.where(nonzero, combined_weights, 0.0)
    row_sum = np.sum(combined_weights, axis=1, keepdims=True)
    combined_weights = np.divide(
        combined_weights,
        row_sum,
        out=np.zeros_like(combined_weights),
        where=np.abs(row_sum) > 0.0,
    )
    compact_width = int(np.max(np.sum(nonzero, axis=1), initial=0))
    if 0 < compact_width < combined_weights.shape[1]:
        order = np.argsort(~nonzero, axis=1, kind="stable")[:, :compact_width]
        flat_indices = np.take_along_axis(flat_indices, order, axis=1)
        combined_weights = np.take_along_axis(combined_weights, order, axis=1)
    return flat_indices, combined_weights


@dataclass(frozen=True, slots=True)
class YeePlaneQuadrature:
    """Compiled field colocation and area integration for one Yee-grid plane."""

    normal_axis: int
    sample_area: float
    integration_weights: np.ndarray
    coordinates: tuple[np.ndarray, np.ndarray]
    plans: Mapping[str, tuple[np.ndarray, np.ndarray]]

    @property
    def point_count(self) -> int:
        return int(self.coordinates[0].size * self.coordinates[1].size)

    def plan(self, component: str):
        return self.plans[component]

    def sample(self, fields: Mapping[str, np.ndarray]):
        return tuple(
            np.sum(
                np.asarray(fields[name]).reshape(-1)[self.plans[name][0]]
                * self.plans[name][1],
                axis=-1,
            )
            for name in _FIELD_COMPONENTS
        )


def yee_plane_area_weights(*, center, size, normal_axis, region, grid):
    """Physical cell areas clipped to the requested rectilinear aperture."""
    axis0, axis1 = plane_axes_3d(normal_axis)
    interval0 = region.axis_interval(axis0)
    interval1 = region.axis_interval(axis1)

    def clipped_widths(axis, interval):
        edges = np.asarray(grid.axis_edges(axis))
        index = _AXIS_INDEX[axis]
        lower = float(center[index]) - float(size[index]) / 2
        upper = float(center[index]) + float(size[index]) / 2
        widths = np.maximum(
            0.0, np.minimum(edges[1:], upper) - np.maximum(edges[:-1], lower)
        )
        return widths[int(interval.start) : int(interval.stop)]

    widths0 = clipped_widths(axis0, interval0)
    widths1 = clipped_widths(axis1, interval1)
    return (widths0[:, None] * widths1[None, :]).reshape(-1)


def compile_yee_plane_quadrature_3d(
    *,
    center,
    size,
    normal_axis: str,
    region,
    resolution: float,
    grid_shape: tuple[int, int, int],
    component_shapes: Mapping[str, tuple[int, int, int]],
    grid=None,
) -> YeePlaneQuadrature:
    """Compile the canonical plane colocation used by monitors and sources."""
    normal_axis = str(normal_axis).lower()
    coordinates = yee_plane_coordinates_3d(center, size, normal_axis, region, grid=grid)
    plane_position = float(tuple(center)[_AXIS_INDEX[normal_axis]])
    plans = {}
    for component in _FIELD_COMPONENTS:
        shape = component_shapes[component]
        plans[component] = _component_plane_plan_3d(
            component,
            normal_axis=normal_axis,
            plane_position=plane_position,
            coordinates=coordinates,
            resolution=float(resolution),
            grid_shape=grid_shape,
            field_shape=(int(shape[0]), int(shape[1]), int(shape[2])),
            grid=grid,
        )
    if grid is None:
        # These samples are cell centers inside the requested aperture, not
        # endpoint samples. An N/(N-1) endpoint correction overcounts its area.
        axis0, axis1 = plane_axes_3d(normal_axis)
        count = int(coordinates[0].size * coordinates[1].size)
        sample_area = float(size[_AXIS_INDEX[axis0]] * size[_AXIS_INDEX[axis1]]) / max(
            count, 1
        )
        integration_weights = np.empty((0,), dtype=np.float64)
    else:
        integration_weights = yee_plane_area_weights(
            center=center, size=size, normal_axis=normal_axis, region=region, grid=grid
        )
        sample_area = float(np.mean(integration_weights))
    return YeePlaneQuadrature(
        normal_axis=_AXIS_INDEX[normal_axis],
        sample_area=sample_area,
        integration_weights=integration_weights,
        coordinates=coordinates,
        plans=plans,
    )


def yee_flux(
    samples, normal_axis: int, *, normal_sign=1.0, measure: Any = 1.0, phasor=False
):
    """Integrate signed Poynting flux from six colocated Yee field samples."""
    ex, ey, ez, hx, hy, hz = (jnp.asarray(value) for value in samples)
    if phasor:
        hx, hy, hz = jnp.conjugate(hx), jnp.conjugate(hy), jnp.conjugate(hz)
    sx = ey * hz - ez * hy
    sy = ez * hx - ex * hz
    sz = ex * hy - ey * hx
    if int(normal_axis) < 0:
        flux = jnp.sqrt(jnp.abs(sx) ** 2 + jnp.abs(sy) ** 2 + jnp.abs(sz) ** 2)
    else:
        flux = (sx, sy, sz)[int(normal_axis)] * float(normal_sign)
    total = jnp.sum(flux * jnp.asarray(measure, dtype=flux.dtype))
    return 0.5 * jnp.real(total) if phasor else total


def _axis_difference(array, axis):
    high = [slice(None)] * array.ndim
    low = [slice(None)] * array.ndim
    high[axis] = slice(1, None)
    low[axis] = slice(None, -1)
    return array[tuple(high)] - array[tuple(low)]


def _paired_curl(fields, difference):
    return tuple(
        difference(fields[i], i_axis) - difference(fields[j], j_axis)
        for (i, i_axis), (j, j_axis) in _CURL_PAIRS
    )


def curl_e_to_h_3d(ex, ey, ez, resolution):
    """Differentiate the complete electric Yee supports onto H sites."""
    return tuple(
        curl / resolution for curl in _paired_curl((ex, ey, ez), _axis_difference)
    )


def _rectilinear_inverse_distances(grid, axis: str, *, backward: bool) -> np.ndarray:
    widths = np.asarray(grid.cell_widths(axis), dtype=np.float64)
    if not backward:
        return 1.0 / widths
    inverse = np.empty(widths.size + 1, dtype=np.float64)
    inverse[0], inverse[-1] = 1.0 / widths[0], 1.0 / widths[-1]
    if widths.size > 1:
        inverse[1:-1] = 2.0 / (widths[:-1] + widths[1:])
    return inverse


def _stored_inverse_distances(grid, *, backward: bool):
    return tuple(
        _rectilinear_inverse_distances(grid, axis, backward=backward)
        for axis in _STORED_AXES
    )


def curl_e_to_h_3d_metric(ex, ey, ez, grid):
    """Differentiate electric fields using exact rectilinear cell widths."""
    scales = _stored_inverse_distances(grid, backward=False)
    return _paired_curl(
        (ex, ey, ez),
        lambda array, axis: metric_adjacent_difference(array, axis, scales[axis]),
    )


def _scaled_adjacent_difference(array, axis, scale, *, divide: bool):
    values = jnp.asarray(array)
    scale = jnp.asarray(scale, dtype=values.dtype)
    moved = jnp.moveaxis(values, axis, 0)
    if scale.ndim == 1:
        scale = scale.reshape((scale.size,) + (1,) * (moved.ndim - 1))
    delta = moved[1:] - moved[:-1]
    scaled = delta / scale if divide else delta * scale
    return jnp.moveaxis(scaled, 0, axis)


def adjacent_difference(array, axis, resolution):
    return _scaled_adjacent_difference(array, axis, resolution, divide=True)


def metric_adjacent_difference(array, axis, inverse_distance):
    """Apply an adjacent difference scaled by a precomputed inverse distance."""
    return _scaled_adjacent_difference(array, axis, inverse_distance, divide=False)


def _pad_with_boundary_ghosts(
    array,
    axis,
    metallic_edges,
    *,
    logical_size: int | None = None,
    periodic: bool = False,
    periodic_phase=1.0,
):
    logical_size = int(array.shape[axis]) if logical_size is None else int(logical_size)
    if logical_size <= 0 or logical_size > int(array.shape[axis]):
        raise ValueError(
            f"Logical boundary size {logical_size} is invalid for shape {array.shape}."
        )
    physical_region = [slice(None)] * array.ndim
    physical_region[axis] = slice(0, logical_size)
    physical = array[tuple(physical_region)]
    storage_region = [slice(None)] * array.ndim
    storage_region[axis] = slice(logical_size, None)
    storage_padding = array[tuple(storage_region)]
    shape = list(physical.shape)
    shape[axis] = 1
    zero = jnp.zeros(tuple(shape), dtype=array.dtype)
    low_edge, high_edge = (("front", "back"), ("bottom", "top"), ("left", "right"))[
        axis
    ]
    if periodic:
        low = jnp.take(physical, jnp.array([logical_size - 1]), axis) / periodic_phase
        high = jnp.take(physical, jnp.array([0]), axis) * periodic_phase
    else:
        low = (
            zero
            if low_edge in metallic_edges
            else jnp.take(physical, jnp.array([0]), axis)
        )
        high = (
            zero
            if high_edge in metallic_edges
            else jnp.take(physical, jnp.array([logical_size - 1]), axis)
        )
    return jnp.concatenate((low, physical, high, storage_padding), axis=axis)


def build_h_boundary_views_for_e_3d(
    hx,
    hy,
    hz,
    metallic_edges=frozenset(),
    *,
    logical_shapes=None,
    periodic_axes=frozenset(),
    periodic_phases=(),
):
    """Create the six ghost-padded H views consumed by the 3D E curl."""
    fields = (hx, hy, hz)
    return {
        _h_view_name(index, axis): _pad_with_boundary_ghosts(
            fields[index],
            axis,
            metallic_edges,
            logical_size=(
                None
                if logical_shapes is None
                else logical_shapes[f"H{'xyz'[index]}"][axis]
            ),
            periodic=axis in periodic_axes,
            periodic_phase=periodic_phases[axis] if periodic_phases else 1.0,
        )
        for index, axis in (term for pair in _CURL_PAIRS for term in pair)
    }


def _e_target_shapes(hx, hy, hz, ex_shape, ey_shape, ez_shape):
    inferred = (
        (hz.shape[0], hz.shape[1] + 1, hz.shape[2]),
        (hx.shape[0] + 1, hx.shape[1], hx.shape[2]),
        (hy.shape[0], hy.shape[1], hy.shape[2] + 1),
    )
    given = (ex_shape, ey_shape, ez_shape)
    return tuple(
        item if item is not None else guess
        for item, guess in zip(given, inferred, strict=True)
    )


def _checked_curls(curls, expected):
    if any(curl.shape != shape for curl, shape in zip(curls, expected, strict=True)):
        raise ValueError(
            f"curl(H) shapes {tuple(curl.shape for curl in curls)} do not match {expected}"
        )
    return curls


def _curl_h_views(boundary_views, difference):
    return tuple(
        difference(boundary_views[_h_view_name(i, i_axis)], i_axis)
        - difference(boundary_views[_h_view_name(j, j_axis)], j_axis)
        for (i, i_axis), (j, j_axis) in _CURL_PAIRS
    )


def curl_h_to_e_3d(
    hx,
    hy,
    hz,
    resolution,
    ex_shape=None,
    ey_shape=None,
    ez_shape=None,
    *,
    boundary_views,
):
    """Differentiate H onto E sites using explicit boundary ghost views."""
    return _checked_curls(
        _curl_h_views(
            boundary_views,
            lambda array, axis: adjacent_difference(array, axis, resolution),
        ),
        _e_target_shapes(hx, hy, hz, ex_shape, ey_shape, ez_shape),
    )


def curl_h_to_e_3d_metric(
    hx,
    hy,
    hz,
    grid,
    ex_shape=None,
    ey_shape=None,
    ez_shape=None,
    *,
    boundary_views,
):
    """Differentiate magnetic fields using exact dual-grid distances."""
    scales = _stored_inverse_distances(grid, backward=True)
    return _checked_curls(
        _curl_h_views(
            boundary_views,
            lambda array, axis: metric_adjacent_difference(array, axis, scales[axis]),
        ),
        _e_target_shapes(hx, hy, hz, ex_shape, ey_shape, ez_shape),
    )


def _loss_step(field, curl, loss, scale, dt, *, subtract: bool):
    half = loss * dt / (2.0 * scale)
    denominator = 1.0 + half
    updated = field * (1.0 - half) / denominator
    coupled = (dt / scale) / denominator * curl
    return updated - coupled if subtract else updated + coupled


def advance_h_field(field, curl, sigma_m, dt):
    """Advance one magnetic Yee component through a source-free half-step."""
    return _loss_step(field, curl, sigma_m, MU_0, dt, subtract=True)


def advance_e_field(field, curl, conductivity, permittivity, dt, region):
    """Advance one electric Yee component through a source-free half-step."""
    updated = _loss_step(
        field[region],
        curl[region],
        conductivity,
        EPS_0 * permittivity,
        dt,
        subtract=False,
    )
    return field.at[region].set(updated)


def _stored_component_shape(component, grid_shape, stored_shape):
    if stored_shape is None:
        return component_shape_3d(component, grid_shape)
    return tuple(int(v) for v in stored_shape)


def _floor_index(offset, length, grid_dim):
    return np.clip(
        np.floor(np.arange(length, dtype=np.float64) + offset).astype(np.int32),
        0,
        int(grid_dim) - 1,
    )


def _take_indices(values, indices):
    sampled = jnp.asarray(values)
    for axis, index in enumerate(indices):
        sampled = jnp.take(sampled, jnp.asarray(index), axis=axis)
    return sampled


def component_index_arrays_3d(
    component: str,
    grid_shape: tuple[int, ...],
    *,
    stored_shape: tuple[int, ...] | None = None,
    region: tuple[slice, slice, slice] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return voxel indices sampled by a 3D Yee component on a cell-centered raster."""

    # Stored-shape overrides support padded/cropped views while grid_shape still bounds voxels.
    shape = _stored_component_shape(component, grid_shape, stored_shape)
    offsets = component_axis_offsets_3d(component)
    region = region or (slice(None), slice(None), slice(None))
    indices = tuple(
        _floor_index(offsets[axis], dim, grid_dim)[axis_region]
        for axis, dim, grid_dim, axis_region in zip(
            _STORED_AXES, shape, grid_shape, region, strict=False
        )
    )
    return indices[0], indices[1], indices[2]


def sample_voxel_grid_at_component_3d(
    grid,
    component: str,
    *,
    stored_shape: tuple[int, ...] | None = None,
    region: tuple[slice, slice, slice] | None = None,
):
    """Sample a cell-centered 3D raster on a Yee component lattice."""

    # Electric interfaces need symmetric averaging; magnetic ownership uses one voxel.
    if component in {"Ex", "Ey", "Ez"}:
        return sample_voxel_grid_at_e_component_3d_centered(
            grid,
            component,
            stored_shape=stored_shape,
            region=region,
        )
    return _take_indices(
        grid,
        component_index_arrays_3d(
            component,
            tuple(int(v) for v in jnp.asarray(grid).shape),
            stored_shape=stored_shape,
            region=region,
        ),
    )


def sample_voxel_grid_at_e_component_3d_centered(
    grid,
    component: str,
    *,
    stored_shape: tuple[int, ...] | None = None,
    region: tuple[slice, slice, slice] | None = None,
):
    """Sample a cell-centered 3D raster onto a staggered E site by symmetric averaging."""
    if component not in {"Ex", "Ey", "Ez"}:
        raise ValueError(
            "sample_voxel_grid_at_e_component_3d_centered only supports Ex/Ey/Ez"
        )
    grid_shape = tuple(int(v) for v in jnp.asarray(grid).shape)
    shape = _stored_component_shape(component, grid_shape, stored_shape)
    offsets = component_axis_offsets_3d(component)
    region = region or (slice(None), slice(None), slice(None))
    sampled = jnp.asarray(grid)
    for axis_index, (axis, dim, grid_dim, axis_region) in enumerate(
        zip(_STORED_AXES, shape, grid_shape, region, strict=False)
    ):
        if offsets[axis] == 0.5:
            index = _floor_index(offsets[axis], dim, grid_dim)[axis_region]
            sampled = jnp.take(sampled, jnp.asarray(index), axis=axis_index)
            continue
        if offsets[axis] != 0.0:
            raise ValueError(f"Unsupported Yee offset {offsets[axis]!r}")
        coord = np.arange(dim, dtype=np.float64) + offsets[axis]
        hi = np.floor(coord).astype(np.int32)
        lo = np.clip(hi - 1, 0, int(grid_dim) - 1)[axis_region]
        hi = np.clip(hi, 0, int(grid_dim) - 1)[axis_region]
        sampled = 0.5 * (
            jnp.take(sampled, jnp.asarray(lo), axis=axis_index)
            + jnp.take(sampled, jnp.asarray(hi), axis=axis_index)
        )
    return sampled


def normalize_polarization_2d(value: str) -> Literal["tm", "te"]:
    """Normalize a 2D electromagnetic polarization name."""
    polarization = str(value).strip().lower()
    if polarization not in {"tm", "te"}:
        raise ValueError("polarization must be 'tm' or 'te'.")
    return "tm" if polarization == "tm" else "te"


def canonical_component_2d(
    component: str, plane: str, polarization: str = "tm"
) -> str | None:
    """Map a public plane component to the selected canonical xy lattice."""
    try:
        rows = _component_rows_2d(plane, polarization)
    except KeyError as exc:
        raise ValueError(f"Unsupported plane {plane!r}") from exc
    for public, canonical, _sign in rows:
        if public == component:
            return canonical
    return None


def public_component_2d(
    component: str, plane: str, polarization: str = "tm"
) -> tuple[str, float]:
    """Map a canonical 2D component back to its public label and orientation."""
    try:
        rows = _component_rows_2d(plane, polarization)
        for public, canonical, sign in rows:
            if canonical == component:
                return public, sign
        raise KeyError(component)
    except KeyError as exc:
        raise ValueError(
            f"Component {component!r} is not active in canonical 2D plane {plane!r}"
        ) from exc


def component_shape_2d(
    component: str,
    grid_shape: tuple[int, ...],
    plane: str = "xy",
    polarization: str = "tm",
) -> tuple[int, int]:
    """Return a public component shape on the selected canonical 2D lattice."""
    if len(grid_shape) != 2:
        raise ValueError(f"A 2D grid shape must have two dimensions: {grid_shape!r}")
    canonical = canonical_component_2d(component, plane, polarization)
    if canonical is None:
        return (1, 1)
    ny, nx = (int(v) for v in grid_shape)
    row_offset, col_offset = _OFFSETS_2D[canonical]
    return ny + int(row_offset == 0.0), nx + int(col_offset == 0.0)


def component_shapes(
    grid_shape: tuple[int, ...], polarization: str = "tm"
) -> dict[str, tuple[int, ...]]:
    """Return every canonical field support for a material-grid shape."""
    if len(grid_shape) == 3:
        return {
            component: component_shape_3d(component, grid_shape)
            for component in _FIELD_COMPONENTS
        }
    return {
        component: component_shape_2d(component, grid_shape, "xy", polarization)
        for component in _FIELD_COMPONENTS
    }


def component_coordinates_2d_um(
    component: str,
    grid_shape: tuple[int, ...],
    dx_um: float,
    plane: str,
    polarization: str = "tm",
) -> dict[str, np.ndarray]:
    """Return public-axis coordinates for a canonical 2D Yee component."""
    shape = component_shape_2d(component, grid_shape, plane, polarization)
    canonical = canonical_component_2d(component, plane, polarization)
    row_axis, col_axis = _PLANE_SAMPLE_AXES[plane]
    row_offset, col_offset = (0.0, 0.0) if canonical is None else _OFFSETS_2D[canonical]
    return {
        row_axis: (np.arange(shape[0], dtype=np.float64) + row_offset) * dx_um,
        col_axis: (np.arange(shape[1], dtype=np.float64) + col_offset) * dx_um,
    }


def sample_voxel_grid_at_component_2d(
    grid,
    component: str,
    plane: str,
    polarization: str = "tm",
    *,
    stored_shape: tuple[int, ...] | None = None,
    region: tuple[slice, slice] | None = None,
):
    """Sample a material raster on the selected canonical 2D Yee support."""
    grid_values = jnp.asarray(grid)
    canonical = canonical_component_2d(component, plane, polarization)
    if canonical is None:
        return grid_values[:1, :1]
    shape = component_shape_2d(
        component,
        tuple(int(v) for v in grid_values.shape),
        plane,
        polarization,
    )
    if stored_shape is not None and tuple(int(v) for v in stored_shape) != shape:
        raise ValueError(f"stored_shape={stored_shape!r} does not match {shape!r}")
    region = region or (slice(None), slice(None))
    ny, nx = (int(v) for v in grid_values.shape)
    y_len, x_len = shape
    y_offset, x_offset = _OFFSETS_2D[canonical]
    return _take_indices(
        grid_values,
        (_floor_index(y_offset, y_len, ny), _floor_index(x_offset, x_len, nx)),
    )[region]


class MaterialCoefficients:
    """Read-only attribute view of material arrays on Yee component supports."""

    __slots__ = ("_values",)

    def __init__(self, **values):
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def items(self):
        return self._values.items()


def _total_conductivity(fields):
    # Sponge PML contributes physical conductivity; CPML damping lives in psi state.
    base_sigma = fields.conductivity
    if not (fields.has_pml and not fields.has_cpml and hasattr(fields, "pml_data")):
        return base_sigma
    keys = (
        ("sigma_x", "sigma_y", "sigma_z")
        if fields.permittivity.ndim == 3
        else ("sigma_x", "sigma_y")
    )
    sigma_pml = jnp.zeros_like(base_sigma)
    for key in keys:
        if key in fields.pml_data:
            sigma_pml = sigma_pml + fields.pml_data[key]
    material_grid = getattr(fields, "material_grid", None)
    if (
        getattr(material_grid, "metric_kind", "isotropic_uniform")
        != "isotropic_uniform"
    ):
        return base_sigma + sigma_pml
    return jnp.maximum(base_sigma, sigma_pml)


def _mu_3d(fields):
    if fields.permeability.ndim == 0:
        return (fields.permeability,) * 3
    return tuple(
        sample_voxel_grid_at_component_3d(
            fields.permeability,
            component,
            stored_shape=tuple(getattr(fields, component).shape),  # fmt: skip
        )
        for component in ("Hx", "Hy", "Hz")
    )


def _base_coefficients(total_sigma, e_terms, sigma_m, mu_terms):
    data = {"total_conductivity": total_sigma}
    for axis, (eps, sig, region) in zip("xyz", e_terms, strict=True):
        data[f"eps_{axis}"] = eps
        data[f"sig_{axis}"] = sig
        data[f"region_{axis}"] = region
    data.update({f"eps_e{axis}": data[f"eps_{axis}"] for axis in "xyz"})
    data.update(
        {f"sigma_m_h{axis}": sigma for axis, sigma in zip("xyz", sigma_m, strict=True)}
    )
    data.update({f"mu_h{axis}": mu for axis, mu in zip("xyz", mu_terms, strict=True)})
    return data


def _material_slice_for_e_3d(permittivity, conductivity, orientation):
    """Collocate material values with one staggered 3D electric component."""
    component = {"x": "Ex", "y": "Ey", "z": "Ez"}[orientation]
    eps = sample_voxel_grid_at_e_component_3d_centered(permittivity, component)
    sig = (
        jnp.asarray(conductivity)
        if jnp.asarray(conductivity).ndim == 0
        else sample_voxel_grid_at_e_component_3d_centered(conductivity, component)
    )
    return eps, sig, (slice(None), slice(None), slice(None))


def _magnetic_conductivity_terms_3d(
    conductivity, permeability, hx_shape, hy_shape, hz_shape
):
    """Collocate equivalent magnetic conductivity with the three H supports."""
    conductivity = jnp.asarray(conductivity)
    if conductivity.ndim == 0 and float(conductivity) == 0.0:
        zero = jnp.zeros((), dtype=conductivity.dtype)
        return zero, zero, zero
    sigma_base = conductivity * permeability * MU_0 / EPS_0
    if sigma_base.ndim == 0:
        return sigma_base, sigma_base, sigma_base
    return tuple(
        sample_voxel_grid_at_component_3d(sigma_base, component, stored_shape=shape)
        for component, shape in zip(
            ("Hx", "Hy", "Hz"), (hx_shape, hy_shape, hz_shape), strict=True
        )
    )


def _material_conductivities(fields):
    """Return total electric loss and the electric-equivalent sponge loss.

    Keep this split shared by static compilation and dynamic material binding.
    Material conductivity is electric-only; only the artificial sponge has a
    matched magnetic counterpart. CPML damping lives in its auxiliary state.
    """
    total_sigma = _total_conductivity(fields)
    magnetic_sigma = (
        total_sigma - fields.conductivity
        if fields.has_pml and not fields.has_cpml
        else jnp.zeros((), dtype=jnp.asarray(total_sigma).dtype)
    )
    return total_sigma, magnetic_sigma


def _active_supports_2d(polarization: str):
    if polarization == "tm":
        return ("Ez",), ("Hx", "Hy")
    return ("Ex", "Ey"), ("Hz",)


def _on_support_2d(value, component, polarization, active):
    array = jnp.asarray(value)
    if component not in active:
        return array[:1, :1]
    return sample_voxel_grid_at_component_2d(array, component, "xy", polarization)


def _supports_2d(value, components, active, polarization):
    if value.ndim == 0:
        return (value,) * len(components)
    return tuple(
        _on_support_2d(value, component, polarization, active)
        for component in components
    )


def build_material_coefficients(fields):
    """Collocate every material array once before runtime execution."""

    total_sigma, magnetic_sigma = _material_conductivities(fields)
    if fields.permittivity.ndim == 3:
        e_terms = tuple(
            _material_slice_for_e_3d(fields.permittivity, total_sigma, axis)
            for axis in "xyz"
        )
        sigma_m = _magnetic_conductivity_terms_3d(
            magnetic_sigma,
            fields.permeability,
            fields.Hx.shape,
            fields.Hy.shape,
            fields.Hz.shape,  # fmt: skip
        )
        return MaterialCoefficients(
            **_base_coefficients(total_sigma, e_terms, sigma_m, _mu_3d(fields))
        )

    # The runtime schema always contains six components; only one polarization's
    # three supports carry full arrays in a 2D program.
    polarization = normalize_polarization_2d(getattr(fields, "polarization_2d", "tm"))
    active_e, active_h = _active_supports_2d(polarization)
    region = (slice(None), slice(None))
    sigma_values = jnp.asarray(total_sigma)
    e_terms = tuple(
        (
            _on_support_2d(fields.permittivity, component, polarization, active_e),
            total_sigma
            if sigma_values.ndim == 0
            else _on_support_2d(sigma_values, component, polarization, active_e),
            region,
        )
        for component in ("Ex", "Ey", "Ez")
    )
    if jnp.asarray(magnetic_sigma).ndim == 0 and float(magnetic_sigma) == 0.0:
        sigma_m = (magnetic_sigma,) * 3
    else:
        base = (
            jnp.asarray(magnetic_sigma)
            * jnp.asarray(fields.permeability)
            * MU_0
            / EPS_0
        )
        sigma_m = _supports_2d(base, ("Hx", "Hy", "Hz"), active_h, polarization)
    mu_terms = _supports_2d(
        fields.permeability, ("Hx", "Hy", "Hz"), active_h, polarization
    )
    return MaterialCoefficients(
        **_base_coefficients(total_sigma, e_terms, sigma_m, mu_terms)
    )


def attach_material_coefficients(fields, materials) -> None:
    fields.materials = materials
    for name, value in materials.items():
        setattr(fields, name, value)


def material_for_component(materials, component: str):
    mapping = {
        "Ex": "eps_ex",
        "Ey": "eps_ey",
        "Ez": "eps_ez",
        "Hx": "mu_hx",
        "Hy": "mu_hy",
        "Hz": "mu_hz",
    }
    try:
        return getattr(materials, mapping[component])
    except KeyError as exc:
        raise ValueError(f"Unsupported field component {component!r}") from exc


def component_material_at(fields, component: str, index):
    materials = getattr(fields, "materials", fields)
    material = material_for_component(materials, component)
    return material if material.ndim == 0 else material[index]
