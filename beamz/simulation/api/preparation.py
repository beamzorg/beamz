"""Public immutable simulation specification and convenience API."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import replace
from importlib import import_module
from typing import Literal

import numpy as np

from beamz._helpers import env_bool, positive_float

# ``Simulation`` is an immutable specification; explicit ``SimulationState`` values
# carry all mutable runtime data.
from beamz.design.geometry.core import Design
from beamz.design.geometry.primitives import Box
from beamz.design.grid.discretization import MaterialGrid
from beamz.design.grid.realize import _realize_uniform_grid
from beamz.design.grid.spec import GridSpec, _validate_grid_budget
from beamz.design.materials import Material, MaterialProtocol
from beamz.devices.boundaries import (
    periodic_storage_axes,
)
from beamz.devices.monitors.monitors import (
    FieldMonitor,
    FieldRecorder,
    FluxMonitor,
    ModeMonitor,
)
from beamz.devices.sources import (
    CANONICAL_SOURCE_TYPES,
    GaussianBeamSource,
    PlaneWaveSource,
)
from beamz.lattice import (
    grid_vector_to_physical_2d,
    in_plane_vector_2d,
)

_MONITOR_TYPES = (FieldMonitor, FieldRecorder, FluxMonitor, ModeMonitor)


def _analysis_function(module: str, name: str):
    """Resolve analysis lazily so solver imports remain dependency-safe."""
    return getattr(import_module(f"beamz.analysis.{module}"), name)


def _design_depth(design) -> float:
    return float(design.depth)


def _design_domain(design):
    # Assemble extents in public x/y/z order rather than the z/y/x array-storage order.
    return (float(design.width), float(design.height), _design_depth(design))


def _design_is_3d(design) -> bool:
    return bool(design.is_3d)


def _normalize_domain(domain, size):
    # Collapse flexible public inputs here so internal code handles one validated representation.
    if domain is not None and size is not None and tuple(domain) != tuple(size):
        raise ValueError("Pass only one of domain=... or size=....")
    sim_size = size if domain is None else domain
    if sim_size is None:
        return None
    sim_size = tuple(float(v) for v in sim_size)
    if len(sim_size) == 2:
        sim_size = (sim_size[0], sim_size[1], 0.0)
    if len(sim_size) != 3:
        raise ValueError("Simulation size must be a 2D or 3D tuple.")
    return sim_size


def _structure_to_domain(structure, offset, domain_size):
    # Clip infinite boxes to the finite simulation domain before shifting centered geometry.
    if structure.shape_kind == "box":
        if any(not np.isfinite(v) for v in structure.size):
            structure = Box(
                center=structure.center,
                size=tuple(
                    float(d) if not np.isfinite(s) else min(float(s), float(d))
                    for s, d in zip(structure.size, domain_size, strict=True)
                ),
                material=structure.material,
            )
        return structure.to_rectangle(offset=offset, material=structure.material)
    return structure.shift(*offset)


def _shift_device_to_domain(device, offset, *, plane_2d):
    # Simulation domains always use public xyz offsets. A 2D Gaussian source
    # intentionally stores only its in-plane position, so lower the offset at
    # this domain-normalization boundary rather than widening its standalone API.
    position = getattr(device, "position", None)
    if position is not None and len(position) == 2 and len(offset) == 3:
        offset = in_plane_vector_2d(offset, plane_2d)
    return device.shifted(offset)


def _material_index(material) -> float:
    # Use the largest real permittivity to obtain a conservative index for grid resolution.
    if not isinstance(material, MaterialProtocol):
        raise TypeError("Simulation materials must satisfy MaterialProtocol.")
    eps = material.max_permittivity
    if eps is None:
        raise ValueError(
            "GridSpec.auto requires materials with finite max_permittivity."
        )
    return float(np.sqrt(max(float(np.max(np.real(np.asarray(eps)))), 1.0)))


def _max_index_for_specs(background, structures) -> float:
    # Include background and every structure so automatic meshing resolves the worst case.
    materials = [s.material for s in structures or ()]
    return max(_material_index(m) for m in (background, *materials) if m is not None)


def _resolve_grid_resolution(grid_spec, background, structures) -> float:
    if grid_spec.resolution is not None:
        resolution = grid_spec._target_spacing()
    else:
        resolution = grid_spec._target_spacing(
            max_index=_max_index_for_specs(background, structures)
        )
    return positive_float(resolution, name="Simulation resolution")


def _devices_require_uniform_grid(sources, monitors) -> bool:
    """Return whether current device operators require isotropic spacing."""
    del monitors
    return any(
        isinstance(source, GaussianBeamSource)
        and not isinstance(source, PlaneWaveSource)
        for source in sources or ()
    )


def _normalize_plane_2d(plane) -> str:
    if not isinstance(plane, str):
        raise TypeError("plane_2d must be 'xy', 'yz', or 'xz'.")
    plane = plane.lower()
    if plane not in {"xy", "yz", "xz"}:
        raise ValueError("plane_2d must be 'xy', 'yz', or 'xz'.")
    return plane


def _time_from_run_time(time, run_time, grid_spec, resolution, dims):
    # Preserve an explicit time grid; otherwise derive a stable step and include the endpoint.
    if time is not None or run_time is None:
        return time
    spec = grid_spec
    if spec is None:
        representative = (
            resolution.minimum_spacing
            if hasattr(resolution, "minimum_spacing")
            else resolution
        )
        spec = GridSpec.uniform(representative)
    dt = spec.resolve_time_step(resolution, dims=dims)
    return np.arange(0.0, float(run_time) + 0.5 * dt, dt)


def _resolve_design_time_and_grid(
    design,
    *,
    grid_spec,
    resolution,
    time,
    run_time,
    require_uniform_grid=False,
):
    # Compute and cache the authoritative setup value so later stages do not repeat expensive work.
    use_realized_grid = bool(
        grid_spec is not None and grid_spec.is_automatic and not require_uniform_grid
    )
    if use_realized_grid:
        assert grid_spec is not None
        grid = grid_spec.realize(design)
        resolution = grid.minimum_spacing
    else:
        if grid_spec is not None:
            resolution = _resolve_grid_resolution(
                grid_spec, design.background, list(design.structures)
            )
        grid = (
            grid_spec.realize(design)
            if grid_spec is not None and not grid_spec.is_automatic
            else _realize_uniform_grid(design, resolution, spec=grid_spec)
        )
    dims = 3 if _design_is_3d(design) else 2
    if grid_spec is not None:
        grid = _validate_grid_budget(grid, grid_spec, dimensions=dims)
    time = _time_from_run_time(time, run_time, grid_spec, grid, dims)
    return resolution, grid, time, use_realized_grid


def _shift_grid_spec_to_domain(grid_spec, offset):
    """Translate public mesh constraints with their geometry and devices."""
    return replace(
        grid_spec,
        overrides=tuple(
            replace(
                item,
                center=tuple(
                    value + delta
                    for value, delta in zip(item.center, offset, strict=True)
                ),
            )
            for item in grid_spec.overrides
        ),
        snapping_points=tuple(
            tuple(
                None if value is None else value + delta
                for value, delta in zip(point, offset, strict=True)
            )
            for point in grid_spec.snapping_points
        ),
    )


def _prepare_design(
    design,
    *,
    domain,
    size,
    background,
    grid_spec,
    resolution,
    time,
    run_time,
    require_uniform_grid,
    plane_2d,
):
    # 1. Resolve the optional domain/size alias. An already domain-based design can keep its coordinates and only needs grid/time normalization.
    sim_size = _normalize_domain(domain, size)
    if design is not None and not isinstance(design, Design):
        raise TypeError("Simulation design must be a canonical Design.")
    offset = (0.0, 0.0, 0.0)
    if design is None or (sim_size is not None and design._centered_coordinates):
        # 2. A centered-coordinate design or explicit size must be converted into Beamz's
        # positive-domain coordinate system; reject the request when neither form is present.
        if sim_size is None:
            raise ValueError("Simulation requires either design=... or domain=....")
        design_structures = list(design.structures) if design is not None else []
        source_structures = design_structures
        background = background or (
            design.background if design is not None else Material(1.0)
        )
        offset = (0.5 * sim_size[0], 0.5 * sim_size[1], 0.5 * sim_size[2])
        # 3. Rebuild the design at the requested domain size and translate every structure by the half-domain offset so its physical placement is preserved.
        new_design = Design(
            width=sim_size[0],
            height=sim_size[1],
            depth=sim_size[2],
            background=background,
        )
        for structure in source_structures:
            new_design += _structure_to_domain(structure, offset, sim_size)
        if grid_spec is not None:
            # Mesh requests use the same public coordinates as geometry and devices.
            grid_spec = replace(
                grid_spec,
                overrides=tuple(
                    replace(
                        item,
                        center=tuple(
                            value + delta
                            for value, delta in zip(item.center, offset, strict=True)
                        ),
                    )
                    for item in grid_spec.overrides
                ),
                snapping_points=tuple(
                    tuple(
                        None if value is None else value + delta
                        for value, delta in zip(point, offset, strict=True)
                    )
                    for point in grid_spec.snapping_points
                ),
            )
        design = new_design
    # Resolve grid/time once after any centered-domain translation.
    resolution, grid, time, use_realized_grid = _resolve_design_time_and_grid(
        design,
        grid_spec=grid_spec,
        resolution=resolution,
        time=time,
        run_time=run_time,
        require_uniform_grid=require_uniform_grid,
    )
    public_offset = (
        offset if design.is_3d else grid_vector_to_physical_2d(offset, plane_2d)
    )
    return design, resolution, grid, time, public_offset, use_realized_grid


def _prepare_material_grid(
    material_grid,
    *,
    design,
    domain,
    size,
    background,
    grid_spec,
    raster_options,
    time,
    run_time,
    plane_2d,
):
    """Create domain metadata for an already rasterized solver grid."""

    if not isinstance(material_grid, MaterialGrid):
        raise TypeError("Simulation material_grid must be a MaterialGrid.")
    if design is not None:
        raise ValueError("Pass either design=... or material_grid=..., not both.")
    if background is not None:
        raise ValueError("background cannot be combined with material_grid.")
    if grid_spec is not None:
        raise ValueError(
            "grid_spec cannot be combined with material_grid; the grid has "
            "already been rasterized."
        )
    if raster_options is not None:
        raise ValueError(
            "raster_options cannot be combined with material_grid; smoothing has "
            "already been applied."
        )

    assert material_grid.grid is not None
    inferred_size = (
        material_grid.grid.extent
        if len(material_grid.shape) == 3
        else (*material_grid.grid.extent[:2], 0.0)
    )
    requested_size = _normalize_domain(domain, size)
    if requested_size is not None and not np.allclose(
        requested_size,
        inferred_size,
        rtol=32.0 * np.finfo(float).eps,
        atol=0.0,
    ):
        raise ValueError(
            "Simulation domain/size must match the material_grid grid extents."
        )

    metadata_design = Design(
        width=inferred_size[0],
        height=inferred_size[1],
        depth=inferred_size[2],
        background=Material(1.0),
    )
    dimensions = 3 if len(material_grid.shape) == 3 else 2
    resolved_time = _time_from_run_time(
        time,
        run_time,
        None,
        material_grid.grid,
        dimensions,
    )
    origin = material_grid.grid.origin
    grid_offset = (-float(origin[0]), -float(origin[1]), -float(origin[2]))
    offset = (
        grid_offset
        if dimensions == 3
        else grid_vector_to_physical_2d(grid_offset, plane_2d)
    )
    return (
        metadata_design,
        material_grid.resolution,
        material_grid.grid,
        resolved_time,
        offset,
        False,
    )


def _resolved_raster_options(
    options, sources, *, boundaries=(), is_3d=True, plane_2d="xy"
):
    """Resolve source frequency and periodic raster topology from the simulation."""
    from beamz.design.raster import RasterOptions

    if options is None:
        options = RasterOptions()
    if not isinstance(options, RasterOptions):
        raise TypeError("raster_options must be RasterOptions or None.")
    if options.reference_frequency is None:
        frequencies = [
            getattr(getattr(s, "source_time", None), "freq0", None)
            for s in (() if sources is None else sources)
        ]
        positive = [
            float(f) for f in frequencies if f is not None and np.isfinite(f) and f > 0
        ]
        if positive:
            options = replace(options, reference_frequency=float(np.mean(positive)))
    storage_axes = periodic_storage_axes(boundaries, is_3d=is_3d, plane_2d=plane_2d)
    axes = "zyx" if is_3d else "yx"
    return replace(options, periodic_axes=tuple(axes[i] for i in sorted(storage_axes)))


def _rasterize_scene_for_simulation(
    scene,
    raster_grid,
    raster_options,
    polarization: Literal["tm", "te"],
    *,
    sources=None,
    boundaries=(),
    plane_2d="xy",
):
    """Convert an imported scene into the same solver grid used by Design."""

    from beamz.design.raster import Grid, Scene

    if not isinstance(scene, Scene):
        raise TypeError("Simulation scene must be a raster Scene.")
    if not isinstance(raster_grid, Grid):
        raise TypeError("Simulation raster_grid must be a raster Grid.")
    dimensions: Literal[2, 3] = 2 if raster_grid.shape[2] == 1 else 3
    options = _resolved_raster_options(
        raster_options,
        sources,
        boundaries=boundaries,
        is_3d=dimensions == 3,
        plane_2d=plane_2d,
    )
    if dimensions == 3 and polarization != "tm":
        raise ValueError("polarization applies only to 2D simulations.")
    from beamz.design.material.dispersion import PoleResidue

    has_dispersion = any(isinstance(m, PoleResidue) for m in scene.materials)
    options = replace(
        options,
        components=(f"two_dimensional_{polarization}" if dimensions == 2 else "all"),
    )
    material_grid = MaterialGrid.from_raster_result(
        scene.rasterize(raster_grid, options=options),
        dimensions=dimensions,
        polarization=polarization,
    )
    if has_dispersion:
        from beamz.design.raster.dispersion import rasterize_dispersion

        regions, interfaces = rasterize_dispersion(
            scene,
            raster_grid,
            kind="3d" if dimensions == 3 else "2d",
            polarization=polarization,
            quality=options.quality,
            resolution=material_grid.resolution,
            cache_directory=None,
            smoothing=options.smoothing,
            metal_smoothing=options.metal_smoothing,
            reference_frequency=options.reference_frequency,
            periodic_axes=options.periodic_axes,
        )
        material_grid = replace(
            material_grid, dispersion=regions, dispersion_interfaces=interfaces
        )
    return material_grid


def _setup_device_policy_label(policy) -> str:
    # An explicit argument overrides the environment; both become one normalized label later.
    return os.getenv("BEAMZ_SETUP_DEVICE", "auto") if policy is None else str(policy)


def _resolve_setup_device(policy):
    # Resolve policy once; allocation failures must surface in the setup context.
    normalized = policy.strip().lower()
    if normalized not in {"auto", "cpu", "default"}:
        raise ValueError(f"setup_device must be auto, cpu, or default, got {policy!r}.")
    return (
        "cpu"
        if normalized == "cpu"
        or (normalized == "auto" and env_bool("BEAMZ_SETUP_CPU", False))
        else "default"
    )


def _normalize_devices(*, sources, monitors):
    def dedupe(devices, *, expected_types, kind):
        # Deduplicate by identity, not equality, because equal devices may be intentional repeats.
        seen, out = set(), []
        for device in devices:
            if type(device) not in expected_types:
                expected = ", ".join(cls.__name__ for cls in expected_types)
                raise TypeError(
                    f"Simulation {kind} must use canonical immutable device types "
                    f"({expected}); got {type(device).__name__}."
                )
            if id(device) not in seen:
                seen.add(id(device))
                out.append(device)
        return out

    # Normalize both collections through the same identity/type policy before name checks.
    sources = dedupe(sources, expected_types=CANONICAL_SOURCE_TYPES, kind="sources")
    monitors = dedupe(monitors, expected_types=_MONITOR_TYPES, kind="monitors")
    duplicates = sorted(
        name for name, count in Counter(m.name for m in monitors).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate monitor names: {', '.join(duplicates)}.")
    return tuple(sources), tuple(monitors)


def _normalize_time(time):
    # Collapse flexible public inputs here so internal code handles one validated representation.
    if time is None:
        raise ValueError("Simulation requires time=... or run_time=....")
    values = np.array(time, dtype=float, copy=True)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("FDTD requires a 1D time array with at least two entries")
    steps = np.diff(values)
    if not np.all(np.isfinite(values)) or np.any(steps <= 0.0):
        raise ValueError(
            "Simulation time values must be finite and strictly increasing"
        )
    if not np.allclose(steps, steps[0], rtol=1e-7, atol=0.0):
        raise ValueError("Simulation time values must be uniformly spaced")
    values.setflags(write=False)
    return values
