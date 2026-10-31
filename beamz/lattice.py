"""Shared Yee-lattice geometry, coordinates, sampling, and material colocation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0, µm

_COMPONENT_AXIS_OFFSETS_3D = {
    "Ex": {"z": 0.0, "y": 0.0, "x": 0.5},
    "Ey": {"z": 0.0, "y": 0.5, "x": 0.0},
    "Ez": {"z": 0.5, "y": 0.0, "x": 0.0},
    "Hx": {"z": 0.5, "y": 0.5, "x": 0.0},
    "Hy": {"z": 0.5, "y": 0.0, "x": 0.5},
    "Hz": {"z": 0.0, "y": 0.5, "x": 0.5},
}


_COMPONENT_AXIS_OFFSETS_2D = {
    component: (offsets["y"], offsets["x"])
    for component, offsets in _COMPONENT_AXIS_OFFSETS_3D.items()
}


_PUBLIC_COMPONENTS_2D = {
    "tm": {
        "xy": {"Ez": ("Ez", 1.0), "Hx": ("Hx", 1.0), "Hy": ("Hy", 1.0)},
        "yz": {"Ez": ("Ex", 1.0), "Hx": ("Hy", 1.0), "Hy": ("Hz", 1.0)},
        "xz": {"Ez": ("Ey", -1.0), "Hx": ("Hx", 1.0), "Hy": ("Hz", 1.0)},
    },
    "te": {
        "xy": {"Ex": ("Ex", 1.0), "Ey": ("Ey", 1.0), "Hz": ("Hz", 1.0)},
        "yz": {"Ex": ("Ey", 1.0), "Ey": ("Ez", 1.0), "Hz": ("Hx", 1.0)},
        "xz": {"Ex": ("Ex", 1.0), "Ey": ("Ez", 1.0), "Hz": ("Hy", -1.0)},
    },
}


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
        for axis, size in zip(("z", "y", "x"), grid_shape, strict=True)
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
        "z": (np.arange(shape[0], dtype=np.float64) + offsets["z"]) * dx_um,
        "y": (np.arange(shape[1], dtype=np.float64) + offsets["y"]) * dx_um,
        "x": (np.arange(shape[2], dtype=np.float64) + offsets["x"]) * dx_um,
    }


def component_coordinates_rectilinear(
    component: str,
    grid,
    *,
    plane: str | None = None,
    polarization: str = "tm",
) -> dict[str, np.ndarray]:
    """Return exact physical Yee coordinates from rectilinear grid edges."""

    def samples(axis: str, offset: float) -> np.ndarray:
        edges = np.asarray(grid.axis_edges(axis), dtype=np.float64)
        return edges if offset == 0.0 else 0.5 * (edges[:-1] + edges[1:])

    if plane is None:
        offsets = component_axis_offsets_3d(component)
        return {axis: samples(axis, offsets[axis]) for axis in ("z", "y", "x")}

    normalized_plane = str(plane).lower()
    canonical = canonical_component_2d(component, normalized_plane, polarization)
    if canonical is None:
        return {}
    row_offset, column_offset = _COMPONENT_AXIS_OFFSETS_2D[canonical]
    row_axis, column_axis = {
        "xy": ("y", "x"),
        "yz": ("z", "y"),
        "xz": ("z", "x"),
    }[normalized_plane]
    return {
        row_axis: samples("y", row_offset),
        column_axis: samples("x", column_offset),
    }


def grid_axes_in_physical_frame_2d(plane: str) -> tuple[str, str, str]:
    """Map stored rectilinear ``(x, y, z)`` axes to physical axes for a 2D plane."""
    try:
        return {
            "xy": ("x", "y", "z"),
            "xz": ("x", "z", "y"),
            "yz": ("y", "z", "x"),
        }[str(plane).lower()]
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
    physical = dict(zip("xyz", map(float, values), strict=True))
    grid_x, grid_y, _ = grid_axes_in_physical_frame_2d(plane)
    return physical[grid_x], physical[grid_y]


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


_FIELD_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


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
        return {
            "x": ("z", "y"),
            "y": ("z", "x"),
            "z": ("y", "x"),
        }[str(normal_axis).lower()]
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
    axis_position = {"x": 0, "y": 1, "z": 2}
    coordinates = []
    for axis in (axis0, axis1):
        index = axis_position[axis]
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
        else {
            key: value * float(µm)
            for key, value in component_coordinates_3d_um(
                component, grid_shape, float(resolution / µm)
            ).items()
        }
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
        component_coords = {
            axis: values * µm
            for axis, values in component_coordinates_3d_um(
                component,
                grid_shape,
                float(resolution / µm),
            ).items()
        }
    else:
        component_coords = component_coordinates_rectilinear(component, grid)
    lengths: dict[str, int] = dict(zip(("z", "y", "x"), field_shape, strict=True))
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
    index_map = {
        axis0: indices[0],
        axis1: indices[1],
        normal_axis: indices[2],
    }
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
        i = {"x": 0, "y": 1, "z": 2}[axis]
        lower = float(center[i]) - float(size[i]) / 2
        upper = float(center[i]) + float(size[i]) / 2
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
    plane_position = float(tuple(center)[{"x": 0, "y": 1, "z": 2}[normal_axis]])
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
        positions = {"x": 0, "y": 1, "z": 2}
        count = int(coordinates[0].size * coordinates[1].size)
        sample_area = float(size[positions[axis0]] * size[positions[axis1]]) / max(
            count, 1
        )
        integration_weights = np.empty((0,), dtype=np.float64)
    else:
        integration_weights = yee_plane_area_weights(
            center=center, size=size, normal_axis=normal_axis, region=region, grid=grid
        )
        sample_area = float(np.mean(integration_weights))
    return YeePlaneQuadrature(
        normal_axis={"x": 0, "y": 1, "z": 2}[normal_axis],
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


def curl_e_to_h_3d(ex, ey, ez, resolution):
    """Differentiate the complete electric Yee supports onto H sites."""
    return (
        ((ez[:, 1:, :] - ez[:, :-1, :]) - (ey[1:, :, :] - ey[:-1, :, :])) / resolution,
        ((ex[1:, :, :] - ex[:-1, :, :]) - (ez[:, :, 1:] - ez[:, :, :-1])) / resolution,
        ((ey[:, :, 1:] - ey[:, :, :-1]) - (ex[:, 1:, :] - ex[:, :-1, :])) / resolution,
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


def curl_e_to_h_3d_metric(ex, ey, ez, grid):
    """Differentiate electric fields using exact rectilinear cell widths."""
    ix = _rectilinear_inverse_distances(grid, "x", backward=False)
    iy = _rectilinear_inverse_distances(grid, "y", backward=False)
    iz = _rectilinear_inverse_distances(grid, "z", backward=False)
    return (
        metric_adjacent_difference(ez, 1, iy) - metric_adjacent_difference(ey, 0, iz),
        metric_adjacent_difference(ex, 0, iz) - metric_adjacent_difference(ez, 2, ix),
        metric_adjacent_difference(ey, 2, ix) - metric_adjacent_difference(ex, 1, iy),
    )


def adjacent_difference(array, axis, resolution):
    resolution = jnp.asarray(resolution, dtype=array.dtype)
    moved = jnp.moveaxis(array, axis, 0)
    if resolution.ndim == 1:
        resolution = resolution.reshape((resolution.size,) + (1,) * (moved.ndim - 1))
    return jnp.moveaxis((moved[1:] - moved[:-1]) / resolution, 0, axis)


def metric_adjacent_difference(array, axis, inverse_distance):
    """Apply an adjacent difference scaled by a precomputed inverse distance."""
    values = jnp.asarray(array)
    inverse_distance = jnp.asarray(inverse_distance, dtype=values.dtype)
    moved = jnp.moveaxis(values, axis, 0)
    if inverse_distance.ndim == 1:
        inverse_distance = inverse_distance.reshape(
            (inverse_distance.size,) + (1,) * (moved.ndim - 1)
        )
    return jnp.moveaxis(
        (moved[1:] - moved[:-1]) * inverse_distance,
        0,
        axis,
    )


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
    return {
        name: _pad_with_boundary_ghosts(
            field,
            axis,
            metallic_edges,
            logical_size=(
                None if logical_shapes is None else logical_shapes[component][axis]
            ),
            periodic=axis in periodic_axes,
            periodic_phase=periodic_phases[axis] if periodic_phases else 1.0,
        )
        for name, component, field, axis in (
            ("hz_y", "Hz", hz, 1),
            ("hy_z", "Hy", hy, 0),
            ("hx_z", "Hx", hx, 0),
            ("hz_x", "Hz", hz, 2),
            ("hy_x", "Hy", hy, 2),
            ("hx_y", "Hx", hx, 1),
        )
    }


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
    return _curl_h_to_e_3d(
        hx,
        hy,
        hz,
        ex_shape,
        ey_shape,
        ez_shape,
        boundary_views,
        adjacent_difference,
        (resolution,) * 3,
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
    distances = tuple(
        _rectilinear_inverse_distances(grid, axis, backward=True) for axis in "zyx"
    )
    return _curl_h_to_e_3d(
        hx,
        hy,
        hz,
        ex_shape,
        ey_shape,
        ez_shape,
        boundary_views,
        metric_adjacent_difference,
        distances,
    )


def _curl_h_to_e_3d(
    hx, hy, hz, ex_shape, ey_shape, ez_shape, boundary_views, difference, distances
):
    """Assemble the static Yee curl with uniform or dual-metric differences."""
    expected = (
        (hz.shape[0], hz.shape[1] + 1, hz.shape[2]) if ex_shape is None else ex_shape,
        (hx.shape[0] + 1, hx.shape[1], hx.shape[2]) if ey_shape is None else ey_shape,
        (hy.shape[0], hy.shape[1], hy.shape[2] + 1) if ez_shape is None else ez_shape,
    )
    dz, dy, dx = distances
    curls = (
        difference(boundary_views["hz_y"], 1, dy)
        - difference(boundary_views["hy_z"], 0, dz),
        difference(boundary_views["hx_z"], 0, dz)
        - difference(boundary_views["hz_x"], 2, dx),
        difference(boundary_views["hy_x"], 2, dx)
        - difference(boundary_views["hx_y"], 1, dy),
    )
    if any(curl.shape != shape for curl, shape in zip(curls, expected, strict=True)):
        raise ValueError(
            f"curl(H) shapes {tuple(curl.shape for curl in curls)} do not match {expected}"
        )
    return curls


def electric_update_factors(conductivity, permittivity, dt):
    """Backend-neutral FP32 coefficient algebra, in the eager evaluation order."""
    alpha = conductivity * np.float32(dt) / (np.float32(2 * EPS_0) * permittivity)
    denominator = np.float32(1) + alpha
    return (
        (np.float32(1) - alpha) / denominator,
        (np.float32(dt) / (np.float32(EPS_0) * permittivity)) / denominator,
    )


def magnetic_update_factors(sigma_m, dt):
    alpha = sigma_m * np.float32(dt) / np.float32(2 * MU_0)
    denominator = np.float32(1) + alpha
    return (np.float32(1) - alpha) / denominator, np.float32(dt / MU_0) / denominator


def advance_h_field(field, curl, sigma_m, dt):
    """Advance one magnetic Yee component through a source-free half-step."""
    denominator = 1.0 + sigma_m * dt / (2.0 * MU_0)
    return (
        field * (1.0 - sigma_m * dt / (2.0 * MU_0)) / denominator
        - (dt / MU_0) / denominator * curl
    )


def advance_e_field(field, curl, conductivity, permittivity, dt, region):
    """Advance one electric Yee component through a source-free half-step."""
    denominator = 1.0 + conductivity * dt / (2.0 * EPS_0 * permittivity)
    updated = (
        field[region]
        * (1.0 - conductivity * dt / (2.0 * EPS_0 * permittivity))
        / denominator
    )
    updated += (dt / (EPS_0 * permittivity)) / denominator * curl[region]
    return field.at[region].set(updated)


def component_index_arrays_3d(
    component: str,
    grid_shape: tuple[int, ...],
    *,
    stored_shape: tuple[int, ...] | None = None,
    region: tuple[slice, slice, slice] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return voxel indices sampled by a 3D Yee component on a cell-centered raster."""

    # Stored-shape overrides support padded/cropped views while grid_shape still bounds voxels.
    shape = (
        tuple(int(v) for v in stored_shape)
        if stored_shape is not None
        else component_shape_3d(component, grid_shape)
    )
    offsets = component_axis_offsets_3d(component)
    axes = ("z", "y", "x")
    region = region or (slice(None), slice(None), slice(None))

    indices: list[np.ndarray] = []
    for axis, dim, grid_dim, axis_region in zip(
        axes, shape, grid_shape, region, strict=False
    ):
        coord = np.arange(dim, dtype=np.float64) + offsets[axis]
        idx = np.floor(coord).astype(np.int32)
        idx = np.clip(idx, 0, int(grid_dim) - 1)
        indices.append(idx[axis_region])

    return (indices[0], indices[1], indices[2])


def sample_voxel_grid_at_component_3d(
    grid,
    component: str,
    *,
    stored_shape: tuple[int, ...] | None = None,
    region: tuple[slice, slice, slice] | None = None,
):
    """Sample a cell-centered 3D raster on a Yee component lattice."""

    return _sample_material_3d(jnp.asarray(grid), component, stored_shape, region)


def _sample_material_3d(grid, component, stored_shape=None, region=None, *, lazy=False):
    """One axis-ordered interpolation contract for eager and bounded host reads."""
    from beamz._region_array import RegionArray

    xp = np if lazy else jnp
    if grid.ndim == 0:
        return xp.asarray(grid, dtype=np.float32 if lazy else None)
    shape = stored_shape or component_shape_3d(component, grid.shape)
    offsets = component_axis_offsets_3d(component)

    def read(region):
        pairs, bounds = [], []
        for name, key, dim, n in zip("zyx", region, shape, grid.shape, strict=True):
            indices = np.arange(*key.indices(dim))
            hi = np.clip(indices, 0, n - 1)
            lo = (
                np.clip(indices - 1, 0, n - 1)
                if component[0] == "E" and offsets[name] == 0
                else hi
            )
            start = int(min(lo.min(), hi.min())) if hi.size else 0
            stop = int(max(lo.max(), hi.max())) + 1 if hi.size else 0
            bounds.append(slice(start, stop))
            pairs.append((lo - start, hi - start))
        values = xp.asarray(grid[tuple(bounds)], dtype=np.float32 if lazy else None)
        for axis, ((lo, hi), name) in enumerate(zip(pairs, "zyx", strict=True)):
            high = xp.take(values, hi, axis=axis)
            values = (
                0.5 * (xp.take(values, lo, axis=axis) + high)
                if component[0] == "E" and offsets[name] == 0
                else high
            )
        return values

    if lazy:
        return RegionArray(tuple(shape), np.dtype(np.float32), read)
    return read(region or (slice(None),) * 3)


def sample_voxel_grid_at_e_component_3d_centered(
    grid, component: str, *, stored_shape=None, region=None
):
    """Sample electric interfaces symmetrically, retaining source dtype and placement."""
    if component not in {"Ex", "Ey", "Ez"}:
        raise ValueError(
            "sample_voxel_grid_at_e_component_3d_centered only supports Ex/Ey/Ez"
        )
    return _sample_material_3d(jnp.asarray(grid), component, stored_shape, region)


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
    shape = tuple(
        int(size) + int(offset == 0.0)
        for size, offset in zip(
            grid_shape, _COMPONENT_AXIS_OFFSETS_2D[canonical], strict=True
        )
    )
    return shape[0], shape[1]


def component_shapes(
    grid_shape: tuple[int, ...], polarization: str = "tm"
) -> dict[str, tuple[int, ...]]:
    """Return every canonical field support for a material-grid shape."""
    if len(grid_shape) == 3:
        return {
            component: component_shape_3d(component, grid_shape)
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        }
    return {
        component: component_shape_2d(component, grid_shape, "xy", polarization)
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }


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
        mapping = _PUBLIC_COMPONENTS_2D[normalize_polarization_2d(polarization)][
            str(plane).lower()
        ]
    except KeyError as exc:
        raise ValueError(f"Unsupported plane {plane!r}") from exc
    return next(
        (
            canonical
            for canonical, (public, _) in mapping.items()
            if public == component
        ),
        None,
    )


def public_component_2d(
    component: str, plane: str, polarization: str = "tm"
) -> tuple[str, float]:
    """Map a canonical 2D component back to its public label and orientation."""
    try:
        return _PUBLIC_COMPONENTS_2D[normalize_polarization_2d(polarization)][
            str(plane).lower()
        ][component]
    except KeyError as exc:
        raise ValueError(
            f"Component {component!r} is not active in canonical 2D plane {plane!r}"
        ) from exc


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
    row_axis, col_axis = {"xy": ("y", "x"), "yz": ("z", "y"), "xz": ("z", "x")}[plane]
    offsets = (0.0, 0.0) if canonical is None else _COMPONENT_AXIS_OFFSETS_2D[canonical]
    return {
        row_axis: (np.arange(shape[0], dtype=np.float64) + offsets[0]) * dx_um,
        col_axis: (np.arange(shape[1], dtype=np.float64) + offsets[1]) * dx_um,
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
    y_offset, x_offset = _COMPONENT_AXIS_OFFSETS_2D[canonical]
    y_idx = np.clip(np.floor(np.arange(y_len) + y_offset).astype(np.int32), 0, ny - 1)
    x_idx = np.clip(np.floor(np.arange(x_len) + x_offset).astype(np.int32), 0, nx - 1)
    sampled = grid_values
    sampled = jnp.take(sampled, jnp.asarray(y_idx), axis=0)
    sampled = jnp.take(sampled, jnp.asarray(x_idx), axis=1)
    return sampled[region]


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


def _base_coefficients(total_sigma, e_terms, sigma_m, mu_terms):
    data = {"total_conductivity": total_sigma}
    for axis, (eps, sig, region) in zip("xyz", e_terms, strict=True):
        data[f"eps_{axis}"] = eps
        data[f"sig_{axis}"] = sig
        data[f"region_{axis}"] = region
    data.update(
        eps_ex=data["eps_x"],
        eps_ey=data["eps_y"],
        eps_ez=data["eps_z"],
        sigma_m_hx=sigma_m[0],
        sigma_m_hy=sigma_m[1],
        sigma_m_hz=sigma_m[2],
        mu_hx=mu_terms[0],
        mu_hy=mu_terms[1],
        mu_hz=mu_terms[2],
    )
    return data


def _material_conductivities(fields):
    """Return total electric loss and the electric-equivalent sponge loss.

    Keep this split shared by static compilation and dynamic material binding.
    Material conductivity is electric-only; only the artificial sponge has a
    matched magnetic counterpart. CPML damping lives in its auxiliary state.
    """
    total_sigma = _total_conductivity(fields)
    # Only the artificial sponge loss has a matched magnetic counterpart.
    # Material.conductivity describes electric current, not magnetic loss.
    # CPML applies its damping through the auxiliary fields instead.
    magnetic_sigma = (
        total_sigma - fields.conductivity
        if fields.has_pml and not fields.has_cpml
        else jnp.zeros((), dtype=jnp.asarray(total_sigma).dtype)
    )
    return total_sigma, magnetic_sigma


def build_material_coefficients(fields, *, lazy=False):
    """Collocate every material array once before runtime execution."""

    if lazy:
        if fields.has_pml and not fields.has_cpml:
            raise ValueError("Lazy material preparation requires CPML or no absorber.")
        total_sigma, magnetic_sigma = fields.conductivity, np.asarray(0, np.float32)
    else:
        total_sigma, magnetic_sigma = _material_conductivities(fields)
    if fields.permittivity.ndim == 3:

        def sample_3d(value, component):
            return _sample_material_3d(
                value, component, tuple(getattr(fields, component).shape), lazy=lazy
            )

        e_terms = tuple(
            (
                sample_3d(fields.permittivity, c),
                sample_3d(total_sigma, c),
                (slice(None),) * 3,
            )
            for c in ("Ex", "Ey", "Ez")
        )
        # Physical conductivity is electric-only. The artificial sponge alone
        # has an impedance-matched magnetic counterpart.
        sigma_base = magnetic_sigma
        if not (sigma_base.ndim == 0 and float(sigma_base) == 0):
            sigma_base = sigma_base * fields.permeability * MU_0 / EPS_0
        sigma_m = tuple(sample_3d(sigma_base, "H" + axis) for axis in "xyz")
        mu = tuple(sample_3d(fields.permeability, "H" + axis) for axis in "xyz")
        return MaterialCoefficients(
            **_base_coefficients(total_sigma, e_terms, sigma_m, mu)
        )

    # The runtime schema always contains six components; only one polarization's
    # three supports carry full arrays in a 2D program.
    polarization = normalize_polarization_2d(getattr(fields, "polarization_2d", "tm"))

    def sample(value, component):
        # The sampler owns active/inactive support; scalar coefficients stay scalar.
        if jnp.asarray(value).ndim == 0:
            return value
        return sample_voxel_grid_at_component_2d(value, component, "xy", polarization)

    e_terms = tuple(
        (
            sample(fields.permittivity, component),
            sample(total_sigma, component),
            (slice(None), slice(None)),
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
        sigma_m = tuple(sample(base, component) for component in ("Hx", "Hy", "Hz"))
    mu_terms = tuple(
        sample(fields.permeability, component) for component in ("Hx", "Hy", "Hz")
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
