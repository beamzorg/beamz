"""Public 3D previews of the same discrete modes used by port launches."""

from __future__ import annotations

import numpy as np

from beamz.devices._placement import mode_source_region
from beamz.devices.modes.plane import bind_material_mode_plane
from beamz.devices.modes.specs import ModeData
from beamz.lattice import component_axis_offsets_3d


def solve_tensor_mode_plane(
    *, simulation, materials, plane, spec, frequencies, direction
):
    from .solve import _resolve_solver_direction
    from .specs import plane_axis_and_spans

    if np.any(np.asarray(materials.conductivity) != 0):
        raise ValueError("3D mode previews require lossless materials")
    axis, center, spans = plane_axis_and_spans(plane)
    offset = getattr(simulation, "coordinate_offset", (0.0, 0.0, 0.0))
    center = tuple(c + o for c, o in zip(center, offset, strict=True))
    # Match compilation's solver-local frame for imported, translated grids.
    grid = materials.grid.translated(tuple(-v for v in materials.grid.origin))
    signed = _resolve_solver_direction(axis, direction)
    transverse = tuple(a for a in ("z", "y", "x") if a != axis)
    sizes: dict[str, float] = dict(zip(("x", "y", "z"), plane.size, strict=True))
    width, height = (sizes[a] for a in transverse[::-1])
    snapped = mode_source_region(
        center=center,
        width=width,
        height=height,
        axis=axis,
        direction_sign=1.0 if signed[0] == "+" else -1.0,
        is_3d=True,
        grid=grid,
    )
    solved = []
    for frequency in frequencies:
        family = bind_material_mode_plane(
            materials,
            spec=spec,
            center=center,
            width=width,
            height=height,
            snapped=snapped,
            direction=signed,
            resolution=simulation.resolution,
            dt=simulation.dt,
            grid=grid,
            frequency=frequency,
        )
        solved.append([family.solve(index) for index in range(spec.num_modes)])

    def collocated(mode, prefix):
        fields = []
        for component in (prefix + a for a in "xyz"):
            values = np.asarray(mode.profiles[component])
            for dimension, a in enumerate(transverse):
                if component_axis_offsets_3d(component)[a] == 0:
                    lower = [slice(None)] * 2
                    upper = [slice(None)] * 2
                    lower[dimension], upper[dimension] = slice(None, -1), slice(1, None)
                    values = 0.5 * (values[tuple(lower)] + values[tuple(upper)])
            fields.append(values)
        return np.stack(fields)

    first = solved[0][0]
    normal_component = "E" + axis
    index = first.component_indices[normal_component]
    crop = tuple(
        slice(
            index[{"z": 0, "y": 1, "x": 2}[a]].start,
            index[{"z": 0, "y": 1, "x": 2}[a]].stop - 1,
        )
        for a in transverse
    )
    profile = np.take(
        np.asarray(materials.permittivity),
        snapped.plane_index,
        axis={"z": 0, "y": 1, "x": 2}[axis],
    )
    edges = tuple(
        np.asarray(grid.axis_edges(a))[s.start : s.stop + 1]
        - offset[{"x": 0, "y": 1, "z": 2}[a]]
        for a, s in zip(transverse, crop, strict=True)
    )

    return ModeData(
        frequencies=np.asarray(frequencies),
        neffs=np.array([[m.neff for m in modes] for modes in solved]),
        e_fields=np.array([[collocated(m, "E") for m in modes] for modes in solved]),
        h_fields=np.array([[collocated(m, "H") for m in modes] for modes in solved]),
        eps_profiles=np.repeat(profile[crop][None], len(frequencies), axis=0),
        eps_profile_fulls=np.repeat(profile[None], len(frequencies), axis=0),
        resolution=simulation.resolution,
        solver_direction=signed,
        axis=axis,
        center=(center[0] - offset[0], center[1] - offset[1], center[2] - offset[2]),
        plane=plane,
        crop_slices=crop,
        grid_edges=edges,
        transverse_axes=transverse,
        diagnostics=tuple(
            {
                **modes[0].diagnostics,
                "contract": "discrete_tensor_preview"
                if materials.uses_full_permittivity
                else "discrete_preview",
                "modes": tuple(m.diagnostics for m in modes),
            }
            for modes in solved
        ),
    )
