from typing import Any, cast

import numpy as np

from beamz.analysis.data import AnalysisData
from beamz.analysis.modal_projection.colocation import (
    _discrete_mode_projection_grids_3d,
    _sample_modal_basis_normal_3d,
)
from beamz.analysis.modal_projection.diagnostics import (
    _modal_projection_reconstruction_diagnostics_from_matrix,
)
from beamz.analysis.modal_projection.geometry import (
    _modal_projection_plane_delay_s,
    _mode_components_for_port,
    _monitor_analysis_plane_3d,
)
from beamz.devices._placement import (
    line_region_points,
    snap_axis_aligned_line_region_grid,
    snap_plane_region_grid,
)
from beamz.devices.modes.discrete import DISCRETE_MODE_CONTRACT
from beamz.devices.modes.discrete import (
    _solve_cached_beamz_mode as solve_beamz_mode,
)
from beamz.devices.modes.fields import _modal_overlap, _normalize_profiles
from beamz.devices.modes.plane import solve_mode_plane_3d, solve_modes
from beamz.devices.monitors.monitors import ModeMonitor
from beamz.lattice import yee_plane_area_weights


def _material_arrays(sim):
    if not isinstance(sim, AnalysisData):
        raise TypeError("Modal projection requires AnalysisData.")
    region = sim.materials
    if region is None:
        return None
    return (
        np.asarray(region.permittivity),
        np.asarray(region.permeability),
        tuple(int(v) for v in region.origin),
        tuple(int(v) for v in region.full_shape),
    )


def _line_crop(profile, sample_idx, pad_cells):
    lo = max(0, int(np.min(sample_idx)) - int(pad_cells))
    hi = min(len(profile), int(np.max(sample_idx)) + int(pad_cells) + 1)
    local_idx = np.clip(sample_idx - lo, 0, max(hi - lo - 1, 0))
    window = np.asarray(profile[lo:hi], dtype=np.complex128)
    return window, local_idx, lo, hi


def _monitor_profile_slice(sim, monitor, axis, pad_cells):
    materials = _material_arrays(sim)
    if materials is None:
        raise RuntimeError(
            "Detached modal analysis requires a monitor material region or "
            "store_full_materials=True."
        )
    perm, _mu, origin, full_shape = materials
    if perm.ndim != 2:
        raise NotImplementedError(
            "Modal line extraction supports 2D permittivity only."
        )
    grid = sim.coordinates.grid
    if grid is not None and grid.metric_kind_for(("x", "y")) != "isotropic_uniform":
        snapped = snap_axis_aligned_line_region_grid(monitor.start, monitor.end, grid)
        if snapped is None:
            raise ValueError(f"Monitor '{monitor.name}' is not axis aligned.")
        points = line_region_points(snapped)
        if not points:
            raise ValueError(f"Monitor '{monitor.name}' contains no sample points.")
        plane = int(snapped.plane_index)
        if axis == "x":
            profile = perm[:, plane - origin[1]]
            sample_idx = np.asarray(
                [point[1] - origin[0] for point in points], dtype=int
            )
            transverse_axis, sample_origin = "y", origin[0]
        else:
            profile = perm[plane - origin[0], :]
            sample_idx = np.asarray(
                [point[0] - origin[1] for point in points], dtype=int
            )
            transverse_axis, sample_origin = "x", origin[1]
        window, local_idx, lo, hi = _line_crop(profile, sample_idx, pad_cells)
        edges = np.asarray(grid.axis_edges(transverse_axis))[lo : hi + 1]
        weights = np.asarray(grid.cell_widths(transverse_axis))[
            sample_idx + sample_origin
        ]
        return window, local_idx, weights, edges
    points = monitor.get_grid_points_2d(sim.resolution, sim.resolution)
    if not points:
        raise ValueError(f"Monitor '{monitor.name}' contains no sample points.")
    coordinates = np.asarray(points, dtype=float)
    node_sampled = sim.coordinates.polarization_2d == "tm"

    def transverse_material(profile):
        if not node_sampled:
            return profile
        # Uniform TM monitors acquire Ez and tangential H on transverse nodes.
        # Solve their basis on that same support, not on cell centers shifted by
        # half a pixel. Do this before cropping so opposite ports retain exactly
        # reflected material neighborhoods.
        extended = np.pad(profile, (1, 1), mode="edge")
        return 0.5 * (extended[:-1] + extended[1:])

    if axis == "x":
        normal = int(
            np.clip(round(float(np.mean(coordinates[:, 0]))), 0, full_shape[1] - 1)
        )
        profile = transverse_material(perm[:, normal - origin[1]])
        component, limit, sample_origin = 1, full_shape[0], origin[0]
    else:
        normal = int(
            np.clip(round(float(np.mean(coordinates[:, 1]))), 0, full_shape[0] - 1)
        )
        profile = transverse_material(perm[normal - origin[0], :])
        component, limit, sample_origin = 0, full_shape[1], origin[1]
    sample_idx = np.asarray(
        [
            int(np.clip(point[component], 0, limit - int(not node_sampled)))
            - sample_origin
            for point in points
        ],
        dtype=int,
    )
    window, local_idx, _, _ = _line_crop(profile, sample_idx, pad_cells)
    if len(points) > 1:
        step_idx = np.diff(np.asarray(points, dtype=float), axis=0)
        dl = float(np.mean(np.linalg.norm(step_idx, axis=1))) * float(sim.resolution)
    else:
        dl = float(sim.resolution)
    return window, local_idx, max(dl, float(sim.resolution) * 1e-9), None


def _build_discrete_port_projection_3d(
    sim,
    *,
    spec,
    monitor,
    frequency,
    parts,
    direction_sign,
    analysis_coords0,
    analysis_coords1,
):
    if not isinstance(monitor, ModeMonitor):
        raise TypeError("3D modal projection requires a canonical ModeMonitor.")
    materials = _material_arrays(sim)
    if materials is None:
        raise RuntimeError(
            "3D modal projection requires monitor-local materials or "
            "store_full_materials=True."
        )
    perm, permeability, origin, full_shape = materials
    if perm.ndim != 3:
        raise ValueError("The discrete 3D modal contract requires 3D materials.")

    axis = parts["axis"]
    axis_index = {"z": 0, "y": 1, "x": 2}[axis]
    grid = sim.coordinates.grid
    if grid is not None and grid.metric_kind != "isotropic_uniform":
        snapped_region = snap_plane_region_grid(
            center=monitor.center,
            size=monitor.size,
            plane_normal=axis,
            grid=grid,
        )
        plane_index = int(snapped_region.plane_index)
    else:
        z_idx, y_idx, x_idx = monitor.get_grid_slice_3d(
            sim.resolution,
            sim.resolution,
            sim.resolution,
            full_shape,
        )
        normal_index = {"z": z_idx, "y": y_idx, "x": x_idx}[axis]
        if isinstance(normal_index, slice):
            start = 0 if normal_index.start is None else int(normal_index.start)
            stop = (
                full_shape[axis_index]
                if normal_index.stop is None
                else int(normal_index.stop)
            )
            plane_index = int(
                np.clip(
                    (start + max(start + 1, stop) - 1) // 2,
                    0,
                    full_shape[axis_index] - 1,
                )
            )
        else:
            plane_index = int(np.clip(int(normal_index), 0, full_shape[axis_index] - 1))
        snapped_region = monitor.get_snapped_region(
            dx=float(sim.resolution),
            dy=float(sim.resolution),
            dz=float(sim.resolution),
            field_shape=full_shape,
        )
    if direction_sign > 0.0:
        offset_index = max(0, plane_index - 1)
    else:
        offset_index = min(max(full_shape[axis_index] - 2, 0), plane_index + 1)

    mode_spec = monitor.mode_spec
    num_modes = int(
        max(
            int(mode_spec.num_modes or 0),
            int(spec.mode_index) + 1,
        )
    )
    center = tuple(float(value) for value in monitor.center)
    size = tuple(float(value) for value in monitor.size_spec)
    if axis == "x":
        width, height = size[1], size[2]
    elif axis == "y":
        width, height = size[0], size[2]
    else:
        width, height = size[0], size[1]

    # The scalar raster alone cannot reproduce smoothed or anisotropic Yee
    # coefficients. Use the same retained constitutive data as the source solve.
    discrete_mode = solve_mode_plane_3d(
        perm,
        permeability,
        material_tensors=sim.materials.material_tensors,
        yee_materials=sim.materials.yee_materials,
        yee_tensors=sim.materials.yee_tensors,
        frequency=frequency,
        resolution=sim.resolution,
        dt=sim.dt,
        axis=axis,
        grid_shape=full_shape,
        center=center,
        width=width,
        height=height,
        plane_index=plane_index,
        offset_index=offset_index,
        direction=spec.projection_direction,
        mode_index=spec.mode_index,
        polarization=None
        if spec.polarization is None
        else str(spec.polarization).lower(),
        target_neff=mode_spec.target_neff,
        force_symmetry=mode_spec.force_symmetry,
        num_modes=num_modes,
        snapped_region=snapped_region,
        material_origin_zyx=origin,
        solver=solve_beamz_mode,
        grid=grid
        if grid is not None and grid.metric_kind != "isotropic_uniform"
        else None,
    )
    proj_components = tuple(parts.get("projection_components_3d", ()))
    if not proj_components:
        raise ValueError(f"Port axis {axis!r} has no 3D projection components.")
    # Match the flux monitor's cell-center aperture quadrature.
    d_area = width * height / max(analysis_coords0.size * analysis_coords1.size, 1)
    if grid is not None and grid.metric_kind != "isotropic_uniform":
        if snapped_region is None:
            raise RuntimeError("Rectilinear mode projection requires a snapped region.")
        integration_weights = yee_plane_area_weights(
            center=center, size=size, normal_axis=axis, region=snapped_region, grid=grid
        )
    else:
        integration_weights = float(d_area)

    def sampled(profiles):
        raw = _discrete_mode_projection_grids_3d(
            sim,
            discrete_mode,
            profiles,
            axis=axis,
            components=proj_components,
            analysis_coords0=analysis_coords0,
            analysis_coords1=analysis_coords1,
        )
        return {
            name: np.asarray(raw[name], dtype=np.complex128)
            for name in proj_components
            if name in raw
        }

    plus_components = sampled(discrete_mode.backward_profiles)
    minus_components = sampled(discrete_mode.profiles)
    missing_components = [
        name
        for name in proj_components
        if name not in plus_components or name not in minus_components
    ]
    if missing_components:
        raise RuntimeError(
            "Discrete mode is missing projection components: "
            + ", ".join(missing_components)
        )
    if discrete_mode.diagnostics.get("full_tensor_operator", False):
        # Launch phasors use exp(+i omega t); recorded DFTs use exp(+i omega t)
        # in the integral, and therefore store their complex conjugates.
        plus_components = {
            name: np.conjugate(value) for name, value in plus_components.items()
        }
        minus_components = {
            name: np.conjugate(value) for name, value in minus_components.items()
        }
    signed_direction = float(direction_sign)
    for profiles in (plus_components, minus_components):
        _normalize_profiles(
            profiles,
            axis=axis,
            measure=integration_weights,
            direction_sign=signed_direction,
            max_scale=1e6,
        )
    # Preserve the directional measurement response, then normalize each
    # measured basis to the same discrete flux watt as the source and monitor.
    sampling = dict(
        axis=axis,
        plane_position=center[{"x": 0, "y": 1, "z": 2}[axis]],
        wave_number=discrete_mode.k_num_axis,
        measure=integration_weights,
        grid_shape=full_shape,
        resolution=sim.resolution,
        grid=grid,
    )
    plus_components, minus_components = (
        _sample_modal_basis_normal_3d(
            profiles,
            direction=direction,
            direction_sign=signed_direction,
            **sampling,
        )
        for profiles, direction in (
            (plus_components, -1),
            (minus_components, 1),
        )
    )
    if axis == "z" and discrete_mode.diagnostics.get("canonical_axis_rotation", False):
        # The canonical x-frame solver carries +z in its forward profiles.
        # Keep the established z-port convention: a_plus is the +z branch.
        plus_components, minus_components = minus_components, plus_components
    branches = (plus_components, minus_components)
    overlap_matrix = np.asarray(
        [
            [
                _modal_overlap(
                    left,
                    right,
                    axis,
                    integration_weights,
                    direction_sign=direction_sign,
                )
                for right in branches
            ]
            for left in branches
        ],
        dtype=np.complex128,
    )
    mode_neff = float(np.real(np.asarray(discrete_mode.neff)))
    if not np.isfinite(mode_neff) or mode_neff <= 0.0:
        raise RuntimeError(
            f"Discrete mode returned invalid effective index {mode_neff!r}."
        )
    projection = {
        "e_component": parts["e_component"],
        "h_component": parts["h_component"],
        "components": tuple(proj_components),
        "condition_number": float(np.linalg.cond(overlap_matrix)),
        "mode_neff": mode_neff,
        "mode_components": {
            name: np.asarray(plus_components[name], dtype=np.complex128)
            for name in proj_components
        },
        "mode_components_bwd": {
            name: np.asarray(minus_components[name], dtype=np.complex128)
            for name in proj_components
        },
        "axis": axis,
        "direction_sign": float(direction_sign),
        "d_area": float(d_area),
        "integration_weights": integration_weights,
        "discrete_contract": DISCRETE_MODE_CONTRACT,
        "analysis_coords0": np.asarray(analysis_coords0, dtype=np.float64),
        "analysis_coords1": np.asarray(analysis_coords1, dtype=np.float64),
    }
    projection["modal_plane_delay_s"] = _modal_projection_plane_delay_s(
        sim,
        spec,
        frequency,
        projection["mode_neff"],
    )
    return projection


def _build_port_projection_2d(
    sim,
    *,
    spec,
    monitor,
    frequency,
    parts,
    mode_pad_cells,
):
    from beamz.devices.modes.constitutive import require_diagonal_support

    if getattr(sim, "materials", None) is not None:
        require_diagonal_support(
            sim.materials.yee_tensors or sim.materials.material_tensors,
            operation="2D modal extraction",
        )
    mode_spec = getattr(monitor, "mode_spec", spec.mode_spec)
    # Production slices are (profile, index, spacing, edges). Callers and tests
    # may still omit edges, which means a uniform line.
    profile_slice = cast(
        Any, _monitor_profile_slice(sim, monitor, parts["axis"], mode_pad_cells)
    )
    if len(profile_slice) == 3:
        eps_profile, local_idx, dl = profile_slice
        grid_edges = None
    else:
        eps_profile, local_idx, dl, grid_edges = profile_slice
    eps_profile = np.asarray(eps_profile, dtype=np.complex128)
    target_neff = 0.98 * np.sqrt(max(float(np.max(np.real(eps_profile))), 1e-12))
    mode_count = int(spec.mode_index) + 1
    neffs, e_fields, h_fields, _ = solve_modes(
        eps=eps_profile,
        omega=2.0 * np.pi * float(frequency),
        dL=float(sim.resolution),
        m=mode_count,
        direction=spec.projection_direction,
        filter_pol=spec.polarization,
        target_neff=target_neff,
        force_symmetry=mode_spec.force_symmetry,
        return_fields=True,
        grid_edges=None
        if grid_edges is None
        else (np.asarray(grid_edges, dtype=np.float64),),
    )
    if len(neffs) <= int(spec.mode_index):
        raise ValueError(
            f"Mode solver returned {len(neffs)} modes for requested "
            f"mode_index={spec.mode_index} on port {spec.name!r}."
        )

    mode_index = int(spec.mode_index)
    e_profile = np.asarray(
        np.squeeze(e_fields[mode_index][parts["e_mode_index"]]),
        dtype=np.complex128,
    )
    h_profile = np.asarray(
        np.squeeze(h_fields[mode_index][parts["h_mode_index"]]),
        dtype=np.complex128,
    )
    if e_profile.ndim > 1:
        e_profile = e_profile[:, 0]
    if h_profile.ndim > 1:
        h_profile = h_profile[:, 0]
    e_profile = e_profile[local_idx]
    h_profile = h_profile[local_idx]

    measure = np.asarray(dl, dtype=float)
    weights = measure if measure.ndim == 0 else measure.reshape(-1)[: e_profile.size]
    power = 0.5 * np.real(
        np.sum(
            weights * parts["signed_flux_sign"] * e_profile * np.conjugate(h_profile)
        )
    )
    # The line-mode adapter retains legacy launch signs, including a magnetic
    # sign reversal for y-normal planes. A port's '+' basis must instead carry
    # positive physical flux along its normal, independent of that convention.
    if power < 0:
        h_profile = -h_profile
    normalization = np.sqrt(max(abs(power), 1e-30))
    e_forward = e_profile / normalization
    h_forward = h_profile / normalization
    mode_neff = float(np.real(np.asarray(neffs[mode_index])))
    plane_delay = _modal_projection_plane_delay_s(sim, spec, frequency, mode_neff)
    # Fit the two waves on the acquired Yee supports. Their E-to-H spatial
    # phases have opposite signs; a single phase applied to measured E assumes
    # a forward-only field and invents reflection for a backward wave.
    phase = np.exp(1j * 2.0 * np.pi * float(frequency) * plane_delay)
    mode_matrix = np.column_stack(
        [
            np.concatenate([e_forward / phase, h_forward]),
            np.concatenate([e_forward * phase, -h_forward]),
        ]
    )
    if np.ndim(weights) == 0:
        projection_weights = None
        measure_scale = np.ones(mode_matrix.shape[0])
    else:
        projection_weights = np.concatenate(
            [
                np.broadcast_to(np.sqrt(weights), e_forward.shape),
                np.broadcast_to(np.sqrt(weights), h_forward.shape),
            ]
        )
        measure_scale = projection_weights
    # Lorentz overlaps pair measured E with modal H, and measured H with modal
    # E. These reject power-orthogonal fields; a Euclidean fit dominated by E's
    # SI scale can mix radiation into the extracted TE mode and break reciprocity.
    overlap = np.zeros((2, mode_matrix.shape[0]), dtype=np.complex128)
    count = e_forward.size
    overlap[0, :count] = np.conjugate(h_forward)
    overlap[1, count:] = np.conjugate(e_forward)
    overlap *= measure_scale[None, :]
    pinv = np.linalg.pinv(overlap @ (measure_scale[:, None] * mode_matrix)) @ overlap
    projection = {
        "e_component": parts["e_component"],
        "h_component": parts["h_component"],
        "components": (parts["e_component"], parts["h_component"]),
        "mode_matrix": mode_matrix,
        "condition_number": float(np.linalg.cond(mode_matrix)),
        "pinv": pinv,
        "mode_neff": mode_neff,
    }
    if projection_weights is not None:
        projection["projection_weights"] = projection_weights
    # Spatial phase is already in each basis column, including its direction.
    projection["modal_plane_delay_s"] = 0.0
    return projection


def _build_port_projection(
    sim,
    spec,
    monitor,
    frequency,
    cache,
    mode_pad_cells=6,
):
    key = (
        spec.name,
        monitor.name,
        float(frequency),
        getattr(monitor, "mode_spec", spec.mode_spec),
    )
    cached = cache.get(key)
    if cached is not None:
        return cached

    parts = _mode_components_for_port(spec)
    if sim.is_3d:
        analysis_coords0, analysis_coords1 = _monitor_analysis_plane_3d(
            sim, monitor, parts["axis"]
        )
        projection = _build_discrete_port_projection_3d(
            sim,
            spec=spec,
            monitor=monitor,
            frequency=frequency,
            parts=parts,
            direction_sign=1.0,
            analysis_coords0=analysis_coords0,
            analysis_coords1=analysis_coords1,
        )
    else:
        projection = _build_port_projection_2d(
            sim,
            spec=spec,
            monitor=monitor,
            frequency=frequency,
            parts=parts,
            mode_pad_cells=mode_pad_cells,
        )

    cache[key] = projection
    return projection


_TANGENTIAL_PAIRS = {
    "x": (("Ey", "Hz"), ("Ez", "Hy")),
    "y": (("Ez", "Hx"), ("Ex", "Hz")),
    "z": (("Ex", "Hy"), ("Ey", "Hx")),
}


def _overlap_row(mode, components, pairs, measure, direction_sign):
    row = {}
    for (electric, magnetic), sign in zip(pairs, (1, -1), strict=True):
        weights = np.asarray(
            measure[electric] if isinstance(measure, dict) else measure
        ).reshape(-1)
        scale = 0.25 * direction_sign * sign * weights
        row[electric] = scale * np.conj(np.asarray(mode[magnetic]).ravel())
        row[magnetic] = scale * np.conj(np.asarray(mode[electric]).ravel())
    return np.concatenate([row[name] for name in components])


def _coupled_modal_system(projections):
    first = projections[0]
    components = first["components"]
    measure = first.get("integration_weights", first["d_area"])
    basis = [
        projection[key]
        for projection in projections
        for key in ("mode_components", "mode_components_bwd")
    ]
    rhs_rows = np.asarray(
        [
            _overlap_row(
                mode,
                components,
                _TANGENTIAL_PAIRS[first["axis"]],
                measure,
                first["direction_sign"],
            )
            for mode in basis
        ]
    )
    mode_matrix = np.column_stack(
        [
            np.concatenate([np.asarray(mode[name]).ravel() for name in components])
            for mode in basis
        ]
    )
    return rhs_rows, mode_matrix, rhs_rows @ mode_matrix


def _solve_coupled_system(system, rhs, condition=None):
    if condition is None:
        condition = np.linalg.cond(system)
    if condition < 1e8:
        return np.linalg.solve(system, rhs)
    return np.linalg.pinv(system) @ rhs


def _modal_coefficient_rows_3d(projections):
    """Linear rows for the ordinary coupled forward/backward overlap system.

    Rows multiply concatenated, colocated tangential E/H samples. Keeping this
    reduction explicit lets optimization differentiate the same modal measurement.
    """
    rhs_rows, _, system = _coupled_modal_system(projections)
    if not np.isfinite(system).all():
        raise ValueError("Invalid 3D modal overlap system.")
    return _solve_coupled_system(system, rhs_rows)


def _project_modal_coefficients_3d_group(field_components, projections):
    """Project one 3D monitor field onto a coupled forward/backward mode set."""
    projections = tuple(projections)
    if not projections:
        return [], np.nan, np.nan, {}

    first = projections[0]
    components = tuple(first.get("components", ()))
    axis = str(first.get("axis", "")).lower()
    if len(components) == 0 or axis not in {"x", "y", "z"}:
        raise ValueError("3D modal group projection is missing components or axis.")
    for proj in projections:
        if tuple(proj.get("components", ())) != components:
            raise ValueError("Grouped 3D modal projections must share components.")
        if str(proj.get("axis", "")).lower() != axis:
            raise ValueError("Grouped 3D modal projections must share an axis.")

    rhs_rows, mode_matrix, system = _coupled_modal_system(projections)
    offset = 0
    field_parts = []
    component_slices = []
    for name in components:
        part = np.asarray(field_components[name], dtype=np.complex128).reshape(-1)
        field_parts.append(part)
        next_offset = offset + int(part.size)
        component_slices.append((name, offset, next_offset))
        offset = next_offset
    field_vec = np.concatenate(field_parts)
    rhs = rhs_rows @ field_vec
    cond = float(np.linalg.cond(system))
    if (
        not np.all(np.isfinite(system))
        or not np.all(np.isfinite(rhs))
        or not np.isfinite(cond)
    ):
        raise ValueError("Invalid grouped 3D modal overlap system.")
    coeff = _solve_coupled_system(system, rhs, cond)
    diagnostics = _modal_projection_reconstruction_diagnostics_from_matrix(
        field_vec,
        mode_matrix,
        coeff,
        component_slices=component_slices,
    )
    diagnostics["projected_signed_power"] = float(
        np.real(np.dot(coeff, system.T @ np.conjugate(coeff)))
    )
    return (
        [
            (np.complex128(coeff[2 * idx]), np.complex128(coeff[2 * idx + 1]))
            for idx in range(len(projections))
        ],
        diagnostics["residual"],
        cond,
        diagnostics,
    )
