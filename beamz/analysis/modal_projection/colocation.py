"""Field colocation helpers for modal projection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from beamz.analysis.data import AnalysisData, static_fields
from beamz.const import µm
from beamz.devices.modes.fields import _normalize_profiles
from beamz.lattice import (
    component_coordinates_3d_um,
    component_coordinates_rectilinear,
    normal_plane_wave_transfer,
    plane_axes_3d,
)

from .geometry import _monitor_analysis_plane_3d


def _interpolate_plane_matrix_2d(
    values: np.ndarray,
    src0: np.ndarray,
    src1: np.ndarray,
    dst0: np.ndarray,
    dst1: np.ndarray,
) -> np.ndarray:
    src0 = np.asarray(src0, dtype=np.float64).reshape(-1)
    src1 = np.asarray(src1, dtype=np.float64).reshape(-1)
    dst0 = np.asarray(dst0, dtype=np.float64).reshape(-1)
    dst1 = np.asarray(dst1, dtype=np.float64).reshape(-1)
    arr = np.asarray(values, dtype=np.complex128)
    if arr.shape != (src0.size, src1.size):
        raise ValueError(
            "Plane interpolation shape mismatch: "
            f"values={arr.shape}, src0={src0.size}, src1={src1.size}"
        )
    if np.array_equal(src0, dst0) and np.array_equal(src1, dst1):
        return arr.copy()
    mid = np.empty((src0.size, dst1.size), dtype=np.complex128)
    for row in range(src0.size):
        mid[row, :] = np.interp(dst1, src1, arr[row, :])
    out = np.empty((dst0.size, dst1.size), dtype=np.complex128)
    for col in range(dst1.size):
        out[:, col] = np.interp(dst0, src0, mid[:, col])
    return out


def _colocate_monitor_component_matrix_3d(
    sim,
    monitor,
    component: str,
    values: np.ndarray,
    *,
    axis: str,
    target0: np.ndarray,
    target1: np.ndarray,
) -> np.ndarray:
    src0, src1 = _monitor_analysis_plane_3d(
        sim,
        monitor,
        axis,
    )
    data = np.asarray(values, dtype=np.complex128)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        raise ValueError(
            f"Expected DFT matrix with shape (nfreq, npoints) for component '{component}', got {data.shape}."
        )
    n_src = int(src0.size * src1.size)
    if data.shape[1] != n_src:
        raise ValueError(
            f"Component '{component}' has {data.shape[1]} samples but expected {n_src} from monitor geometry."
        )
    out = np.empty(
        (data.shape[0], int(len(target0) * len(target1))), dtype=np.complex128
    )
    for idx in range(data.shape[0]):
        plane = data[idx].reshape(src0.size, src1.size)
        interp = _interpolate_plane_matrix_2d(
            plane,
            src0,
            src1,
            target0,
            target1,
        )
        out[idx, :] = interp.reshape(-1)
    return out


def _colocate_field_components_to_projection_3d(
    sim,
    monitor,
    field_components: Mapping[str, np.ndarray],
    projection: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    target0 = projection.get("analysis_coords0")
    target1 = projection.get("analysis_coords1")
    axis = projection.get("axis")
    if target0 is None or target1 is None or axis is None:
        return {
            name: np.asarray(value, dtype=np.complex128)
            for name, value in field_components.items()
        }
    colocated = {}
    for name, value in field_components.items():
        arr = np.asarray(value, dtype=np.complex128)
        was_vector = arr.ndim == 1
        if was_vector:
            arr = arr[None, :]
        interp = _colocate_monitor_component_matrix_3d(
            sim,
            monitor,
            name,
            arr,
            axis=str(axis),
            target0=np.asarray(target0, dtype=np.float64),
            target1=np.asarray(target1, dtype=np.float64),
        )
        colocated[name] = interp[0] if was_vector else interp
    return colocated


def _component_index_plane_coords_3d(sim, component, index, axis):
    if isinstance(sim, AnalysisData):
        grid_shape = sim.coordinates.fields.grid_shape
        grid = sim.coordinates.grid
    else:
        fields = static_fields(sim)
        grid = getattr(fields, "geometry", None)
        grid_shape = getattr(fields, "grid_shape", None)
        if grid_shape is None:
            grid_shape = np.asarray(fields.permittivity).shape
    if grid is None or grid.metric_kind == "isotropic_uniform":
        coords = {
            name: values * float(µm)
            for name, values in component_coordinates_3d_um(
                component,
                tuple(int(v) for v in grid_shape),
                float(sim.resolution / µm),
            ).items()
        }
    else:
        coords = component_coordinates_rectilinear(component, grid)
    axis0, axis1 = plane_axes_3d(axis)
    axis_indices = {"z": index[0], "y": index[1], "x": index[2]}
    coord0 = np.asarray(coords[axis0][axis_indices[axis0]], dtype=np.float64)
    coord1 = np.asarray(coords[axis1][axis_indices[axis1]], dtype=np.float64)
    return coord0.reshape(-1), coord1.reshape(-1)


def _discrete_mode_projection_grids_3d(
    sim,
    discrete_mode,
    profiles,
    *,
    monitor,
    axis,
    components,
    analysis_coords0,
    analysis_coords1,
):
    del monitor
    grids = {}
    samples = {}
    for name in components:
        if name not in profiles:
            continue
        arr = np.asarray(profiles[name], dtype=np.complex128)
        if arr.ndim == 1:
            arr = arr[:, None]
        index = discrete_mode.component_indices.get(name)
        if index is None:
            continue
        src0, src1 = _component_index_plane_coords_3d(sim, name, index, axis)
        rows = min(int(arr.shape[0]), int(src0.size))
        cols = min(int(arr.shape[1]), int(src1.size))
        if rows <= 0 or cols <= 0:
            continue
        grid = _interpolate_plane_matrix_2d(
            arr[:rows, :cols],
            src0[:rows],
            src1[:cols],
            np.asarray(analysis_coords0, dtype=np.float64),
            np.asarray(analysis_coords1, dtype=np.float64),
        )
        grids[name] = grid
        samples[name] = grid.reshape(-1)
    return grids, samples


def _sample_modal_basis_normal_3d(
    profiles,
    *,
    axis,
    plane_position,
    wave_number,
    direction,
    grid_shape,
    resolution,
    grid=None,
    measure,
    direction_sign=1.0,
):
    """Measure a directional mode, then normalize to one discrete flux watt.

    Opposite directions retain their distinct complex Yee interpolation factors.
    Scaling the entire measured E/H basis by a real scalar preserves that phase
    relationship and hence forward/backward discrimination. Its self-overlap
    then matches the Poynting quadrature used by flux monitors and sources.
    """
    sampled = {}
    for name, values in profiles.items():
        transfer = normal_plane_wave_transfer(
            name,
            axis=axis,
            plane_position=plane_position,
            wave_number=direction * wave_number,
            grid_shape=grid_shape,
            resolution=resolution,
            grid=grid,
        )
        sampled[name] = np.asarray(values, dtype=np.complex128) * transfer
    _normalize_profiles(
        sampled, axis=axis, measure=measure, direction_sign=direction_sign
    )
    return sampled
