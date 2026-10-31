from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from beamz.devices.modes.plane import SignedAxis, solve_modes
from beamz.devices.modes.specs import ModeData, ModeSpec

from .specs import plane_axis_and_spans


def _require_cell_mode_materials(material_grid, *, operation: str) -> None:
    """Reject material grids whose propagation coefficients cannot be reproduced."""

    if material_grid is not None and material_grid.uses_full_permittivity:
        raise ValueError(
            f"{operation} cannot yet reproduce full off-diagonal permittivity. "
            "Use farjadpour_diagonal or solve the mode with a full-tensor path."
        )


def _simulation_grid_view(simulation):
    grid = simulation._material_grid()
    _require_cell_mode_materials(grid, operation="ModeSource.solve_modes()")
    return grid


def _resolve_solver_direction(axis: str, direction=None) -> str:
    axis = str(axis).lower()
    if direction is None:
        return f"-{axis}"
    direction_str = str(direction).lower()
    if direction_str in {"+", "-"}:
        return f"{direction_str}{axis}"
    if direction_str not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise ValueError(
            "mode solve direction must be one of '+', '-', '+x', '-x', '+y', "
            f"'-y', '+z', or '-z', got {direction!r}."
        )
    if direction_str[-1] != axis:
        raise ValueError(
            f"mode solve direction {direction_str!r} does not match {axis!r}-normal plane."
        )
    return direction_str


@dataclass(frozen=True)
class ModePlaneContext:
    eps: np.ndarray
    axis: str
    center: tuple[float, float, float]
    axis_index: int
    grid_index: int
    eps_profile_full: np.ndarray
    crop_slices: tuple[slice, ...]
    diagonal_permittivity: dict[str, np.ndarray]
    diagonal_permeability: dict[str, np.ndarray]
    profile_edges: tuple[np.ndarray, ...] | None
    transverse_axes: tuple[str, ...]


def mode_plane_context(*, simulation, plane) -> ModePlaneContext:
    grid = _simulation_grid_view(simulation)
    conductivity = getattr(grid, "conductivity", None)
    if conductivity is not None and np.any(np.asarray(conductivity) != 0.0):
        raise ValueError(
            "Mode solving does not yet support conductive material profiles; "
            "the solved mode would not match lossy FDTD propagation."
        )
    eps = np.asarray(grid.permittivity)
    axis, center, _spans = plane_axis_and_spans(plane)
    offset = getattr(simulation, "coordinate_offset", (0.0, 0.0, 0.0))
    center = tuple(c + o for c, o in zip(center, offset, strict=True))
    axis_index = {"z": 0, "y": 1, "x": 2}[axis]
    geometry = grid.grid
    if geometry is None:
        raise ValueError("Mode-plane materials require realized grid geometry.")
    grid_index, _ = geometry.axis(axis).center(center["xyz".index(axis)])
    eps_profile_full = np.take(eps, grid_index, axis=axis_index)
    profile_axes = tuple(index for index in range(eps.ndim) if index != axis_index)
    storage_axis = {0: "z", 1: "y", 2: "x"}
    transverse_axes = tuple(storage_axis[int(index)] for index in profile_axes)
    crop_slices = geometry.crop(center, plane.size, axes=transverse_axes)
    profile_edges = tuple(
        geometry.axis_edges(axis)[selector.start : selector.stop + 1]
        for axis, selector in zip(transverse_axes, crop_slices, strict=True)
    )
    if len(center) != 3:
        raise ValueError(f"Mode-plane centers require three coordinates: {center!r}")
    tensors = dict(getattr(grid, "tensors", {}))

    def diagonal_profiles(name):
        if name not in tensors:
            return {}
        tensor = np.asarray(tensors[name])
        if (
            tensor.ndim != eps.ndim + 1
            or tensor.shape[0] not in (1, 3, 6)
            or tensor.shape[1:] != eps.shape
        ):
            raise ValueError(f"{name} tensor has invalid compact shape {tensor.shape}")
        return {
            component: np.take(
                tensor[0 if tensor.shape[0] == 1 else index],
                grid_index,
                axis=axis_index,
            )
            for index, component in enumerate(("xx", "yy", "zz"))
        }

    return ModePlaneContext(
        eps=eps,
        axis=axis,
        center=(center[0], center[1], center[2]),
        axis_index=axis_index,
        grid_index=grid_index,
        eps_profile_full=eps_profile_full,
        crop_slices=crop_slices,
        diagonal_permittivity=diagonal_profiles("epsilon"),
        diagonal_permeability=diagonal_profiles("mu"),
        profile_edges=profile_edges,
        transverse_axes=transverse_axes,
    )


def solve_mode_plane(
    *, simulation, plane, mode_spec: ModeSpec | None, freqs, direction=None
) -> ModeData:
    """Solve modal fields on a finite simulation plane."""
    spec = mode_spec or ModeSpec()
    frequencies = np.asarray(freqs, dtype=float).reshape(-1)
    if frequencies.size == 0:
        raise ValueError("Mode solving requires at least one frequency.")

    materials = simulation._material_grid()
    if len(materials.shape) == 3:
        from .tensor_preview import solve_tensor_mode_plane

        return solve_tensor_mode_plane(
            simulation=simulation,
            materials=materials,
            plane=plane,
            spec=spec,
            frequencies=frequencies,
            direction=direction,
        )
    context = mode_plane_context(simulation=simulation, plane=plane)
    coordinate_offset = tuple(
        float(value)
        for value in getattr(simulation, "coordinate_offset", (0.0, 0.0, 0.0))
    )
    eps_profile_full = context.diagonal_permittivity.get("xx", context.eps_profile_full)
    eps_profile = eps_profile_full[context.crop_slices]
    solver_direction = _resolve_solver_direction(context.axis, direction)

    def cropped(mapping, component):
        values = mapping.get(component)
        return None if values is None else values[context.crop_slices]

    solve_diagnostics = [{} for _ in frequencies]
    solved = [
        solve_modes(
            eps=eps_profile,
            omega=2.0 * np.pi * float(frequency),
            dL=simulation.resolution,
            m=int(spec.num_modes),
            direction=cast(SignedAxis, solver_direction),
            filter_pol=cast(Literal["te", "tm"] | None, spec.polarization),
            target_neff=spec.target_neff,
            return_fields=True,
            eps_yy=cropped(context.diagonal_permittivity, "yy"),
            eps_zz=cropped(context.diagonal_permittivity, "zz"),
            mu_xx=cropped(context.diagonal_permeability, "xx"),
            mu_yy=cropped(context.diagonal_permeability, "yy"),
            mu_zz=cropped(context.diagonal_permeability, "zz"),
            grid_edges=context.profile_edges,
            force_symmetry=spec.force_symmetry,
            mirror_center=tuple(
                context.center[{"x": 0, "y": 1, "z": 2}[a]]
                for a in context.transverse_axes
            ),
            diagnostics=solve_diagnostics[index],
        )
        for index, frequency in enumerate(frequencies)
    ]
    return ModeData(
        diagnostics=tuple(solve_diagnostics),
        frequencies=frequencies,
        neffs=np.asarray([item[0] for item in solved]),
        e_fields=np.asarray([item[1] for item in solved]),
        h_fields=np.asarray([item[2] for item in solved]),
        eps_profiles=np.repeat(eps_profile[None, ...], len(frequencies), axis=0),
        eps_profile_fulls=np.repeat(
            eps_profile_full[None, ...], len(frequencies), axis=0
        ),
        resolution=float(simulation.resolution),
        solver_direction=solver_direction,
        axis=context.axis,
        center=(
            context.center[0] - coordinate_offset[0],
            context.center[1] - coordinate_offset[1],
            context.center[2] - coordinate_offset[2],
        ),
        plane=plane,
        crop_slices=context.crop_slices,
        grid_edges=(
            None
            if context.profile_edges is None
            else tuple(
                np.asarray(edges, dtype=float)
                - coordinate_offset[{"x": 0, "y": 1, "z": 2}[axis]]
                for axis, edges in zip(
                    context.transverse_axes, context.profile_edges, strict=True
                )
            )
        ),
        transverse_axes=context.transverse_axes,
    )
