"""Opt-in region assembly for sparse, isotropic, nondispersive scenes.

Local rasters use the original grid edges and guarded strips. Callers prepare
their own scene; this module does not simplify imported geometry or change the
normal scene raster path.
"""

from __future__ import annotations

from time import perf_counter
from types import SimpleNamespace

import numpy as np

from beamz.design.dispersion import PoleResidue
from beamz.design.materials import Material

from .engine import RasterOptions, compile_scene
from .result import RasterResult
from .schema import Box, ExtrudedPolygon, Grid, Mesh, Object, Scene, _integer


def _validate_region_inputs(scene: Scene, grid: Grid, options: RasterOptions) -> None:
    if not isinstance(scene, Scene) or not isinstance(grid, Grid):
        raise TypeError("Region rasterization requires a Scene and Grid.")
    if not isinstance(options, RasterOptions):
        raise TypeError("options must be RasterOptions.")
    if options.periodic_axes:
        raise ValueError("Region rasterization does not support periodic axes.")
    if not scene.objects:
        raise ValueError("Region rasterization requires at least one object.")
    for material in scene.materials:
        if not isinstance(material, Material) or isinstance(material, PoleResidue):
            raise ValueError("Region rasterization requires nondispersive materials.")
        for value in (
            material.permittivity,
            material.permeability,
            material.conductivity,
        ):
            if np.asarray(value).ndim != 0:
                raise ValueError("Region rasterization requires isotropic materials.")
    for obj in scene.objects:
        geometry_bounds(obj.geometry)


def rasterize_regions(
    scene: Scene,
    grid: Grid,
    *,
    options: RasterOptions | None = None,
    tile_x_cells: int = 100,
    guard_cells: int = 2,
    substrate_object: Object | None = None,
) -> RasterResult:
    """Raster guarded x strips and return a validated full-grid result.

    ``substrate_object`` may identify one disjoint full-width box whose remote
    interface is rasterized separately. This path does not use the normal
    full-raster cache and must be chosen explicitly for a prepared scene.
    """
    options = RasterOptions() if options is None else options
    _validate_region_inputs(scene, grid, options)
    tile_x_cells = _integer(tile_x_cells, "tile_x_cells", minimum=1)
    guard_cells = _integer(guard_cells, "guard_cells", minimum=2)
    if substrate_object is None:
        assembled, details = rasterize_tiled_x(
            scene, grid, options, tile_cells=tile_x_cells, guard_cells=guard_cells
        )
    else:
        assembled, details = rasterize_split_substrate(
            scene,
            grid,
            options,
            core_tile_cells=tile_x_cells,
            substrate_object=substrate_object,
            guard_cells=guard_cells,
        )
    return RasterResult._from_arrays(
        grid_edges=grid.edges,
        smoothing=assembled.smoothing,
        tensors=assembled.tensors,
        yee_tensors=assembled.yee_tensors,
        diagnostics={"strategy": "regions", "details": details},
        scene_hash=compile_scene(scene).hash,
    )


def geometry_bounds(geometry) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(geometry, Box):
        return np.asarray(geometry.minimum), np.asarray(geometry.maximum)
    if isinstance(geometry, ExtrudedPolygon):
        xy = np.asarray(geometry.polygon.exterior)
        return np.r_[xy.min(axis=0), geometry.z_min], np.r_[
            xy.max(axis=0), geometry.z_max
        ]
    if isinstance(geometry, Mesh):
        vertices = np.asarray(geometry.vertices)
        return vertices.min(axis=0), vertices.max(axis=0)
    raise TypeError(f"Unsupported raster geometry: {type(geometry).__name__}")


def occupied_region(scene, grid, guard_cells: int = 2) -> tuple[tuple[int, int], ...]:
    if not scene.objects:
        raise ValueError("The occupied-region method requires at least one object")
    lower, upper = zip(
        *(geometry_bounds(obj.geometry) for obj in scene.objects), strict=True
    )
    bounds_min = np.min(lower, axis=0)
    bounds_max = np.max(upper, axis=0)
    region = []
    for edges, minimum, maximum in zip(grid.edges, bounds_min, bounds_max, strict=True):
        first = max(
            0, int(np.searchsorted(edges, minimum, side="right")) - 1 - guard_cells
        )
        last = min(
            len(edges) - 1,
            int(np.searchsorted(edges, maximum, side="left")) + guard_cells,
        )
        if first >= last:
            raise ValueError("Scene object is outside the raster grid")
        region.append((first, last))
    return tuple(region)


def _isotropic_array(shape, fill: float, dtype) -> np.ndarray:
    array = np.zeros(shape, dtype=dtype)
    array[: min(3, shape[0])] = fill
    return array


def _promote_components(array: np.ndarray, count: int) -> np.ndarray:
    """Expand compact tensors without copying diagonal values off diagonal."""
    if array.shape[0] == count:
        return array
    expanded = np.zeros((count, *array.shape[1:]), dtype=array.dtype)
    expanded[:3] = array[:3]
    return expanded


def padded_array(
    local: np.ndarray, region: tuple[tuple[int, int], ...], grid, fill: float
) -> np.ndarray:
    local_counts = tuple(end - start for start, end in region)
    global_counts = tuple(len(edges) - 1 for edges in grid.edges)
    full_shape = (local.shape[0],) + tuple(
        global_count + local_count - region_count
        for global_count, local_count, region_count in zip(
            global_counts[::-1], local.shape[1:], local_counts[::-1], strict=True
        )
    )
    full = _isotropic_array(full_shape, fill, local.dtype)
    x, y, z = region
    full[
        :,
        z[0] : z[0] + local.shape[1],
        y[0] : y[0] + local.shape[2],
        x[0] : x[0] + local.shape[3],
    ] = local
    return full


def _tile_y_bounds(scene, x_min: float, x_max: float) -> tuple[float, float] | None:
    """Conservative y bounds of objects crossing an x strip."""
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry import box as shapely_box

    from beamz.design.raster import Box, ExtrudedPolygon, Mesh

    intervals = []
    for obj in scene.objects:
        geometry = obj.geometry
        lower, upper = geometry_bounds(geometry)
        if upper[0] < x_min or lower[0] > x_max:
            continue
        if isinstance(geometry, ExtrudedPolygon):
            polygon = ShapelyPolygon(geometry.polygon.exterior)
            clipped = polygon.intersection(
                shapely_box(x_min, lower[1], x_max, upper[1])
            )
            if clipped.is_empty:
                continue
            intervals.append((clipped.bounds[1], clipped.bounds[3]))
        elif isinstance(geometry, Box):
            intervals.append((lower[1], upper[1]))
        elif isinstance(geometry, Mesh):
            vertices = np.asarray(geometry.vertices)
            triangles = np.asarray(geometry.triangles)
            triangle_x = vertices[triangles, 0]
            crossing = (triangle_x.min(axis=1) <= x_max) & (
                triangle_x.max(axis=1) >= x_min
            )
            if np.any(crossing):
                crossing_y = vertices[triangles[crossing], 1]
                intervals.append((float(crossing_y.min()), float(crossing_y.max())))
            else:
                intervals.append((lower[1], upper[1]))
        else:
            intervals.append((lower[1], upper[1]))
    if not intervals:
        return None
    return min(start for start, _ in intervals), max(stop for _, stop in intervals)


def rasterize_tiled_x(scene, grid, options, tile_cells: int, guard_cells: int = 2):
    """Raster x strips with a tight y range, then fill the known background."""
    from beamz.design.raster import Grid, compile_scene

    if tile_cells < 1:
        raise ValueError("tile_cells must be positive")
    occupied = occupied_region(scene, grid, guard_cells=guard_cells)
    z_region = occupied[2]
    background = scene.materials[scene.background_material]
    fill_values = {
        "epsilon": float(background.permittivity),
        "mu": float(background.permeability),
        "conductivity": float(background.conductivity),
    }
    compiled = compile_scene(scene)
    tensors = {}
    yee_tensors = {}
    tiles = []
    tile_diagnostics = []
    native_seconds = 0.0
    expand_seconds = 0.0
    smoothing = None
    for x_start in range(occupied[0][0], occupied[0][1], tile_cells):
        x_stop = min(x_start + tile_cells, occupied[0][1])
        local_x = (
            max(0, x_start - guard_cells),
            min(len(grid.x_edges) - 1, x_stop + guard_cells),
        )
        y_bounds = _tile_y_bounds(
            scene, grid.x_edges[local_x[0]], grid.x_edges[local_x[1]]
        )
        if y_bounds is None:
            continue
        y_start = max(
            0,
            int(np.searchsorted(grid.y_edges, y_bounds[0], side="right"))
            - 1
            - guard_cells,
        )
        y_stop = min(
            len(grid.y_edges) - 1,
            int(np.searchsorted(grid.y_edges, y_bounds[1], side="left")) + guard_cells,
        )
        region = (local_x, (y_start, y_stop), z_region)
        local_grid = Grid(
            *(
                edges[start : end + 1]
                for edges, (start, end) in zip(grid.edges, region, strict=True)
            )
        )
        started = perf_counter()
        local = compiled.rasterize(local_grid, options=options)
        native_seconds += perf_counter() - started
        tile_diagnostics.append(dict(local.diagnostics))
        started = perf_counter()
        smoothing = local.smoothing
        for group_name, destination in (
            ("tensors", tensors),
            ("yee_tensors", yee_tensors),
        ):
            for name, local_array in getattr(local, group_name).items():
                if name not in destination:
                    local_counts = tuple(end - start for start, end in region)
                    global_counts = tuple(len(edges) - 1 for edges in grid.edges)
                    global_shape = (local_array.shape[0],) + tuple(
                        global_count + local_count - region_count
                        for global_count, local_count, region_count in zip(
                            global_counts[::-1],
                            local_array.shape[1:],
                            local_counts[::-1],
                            strict=True,
                        )
                    )
                    destination[name] = _isotropic_array(
                        global_shape,
                        fill_values[name.split("_")[0]],
                        local_array.dtype,
                    )
                component_count = max(destination[name].shape[0], local_array.shape[0])
                destination[name] = _promote_components(
                    destination[name], component_count
                )
                local_array = _promote_components(local_array, component_count)
                global_x_stop = x_stop + int(
                    x_stop == len(grid.x_edges) - 1
                    and local_array.shape[3] > local_x[1] - local_x[0]
                )
                local_start = x_start - local_x[0]
                local_stop = global_x_stop - local_x[0]
                destination[name][
                    :,
                    z_region[0] : z_region[0] + local_array.shape[1],
                    y_start : y_start + local_array.shape[2],
                    x_start:global_x_stop,
                ] = local_array[:, :, :, local_start:local_stop]
        expand_seconds += perf_counter() - started
        tiles.append(
            {
                "x_start": x_start,
                "x_stop": x_stop,
                "region_cells_xyz": [end - start for start, end in region],
            }
        )
    result = SimpleNamespace(
        grid_edges=grid.edges,
        smoothing=smoothing,
        tensors=tensors,
        yee_tensors=yee_tensors,
        diagnostics={"tiles": tile_diagnostics},
    )
    if not tensors:
        raise ValueError("No occupied strips were rasterized.")
    return result, {
        "tiles": tiles,
        "native_raster_s": native_seconds,
        "expand_s": expand_seconds,
    }


def rasterize_split_substrate(
    scene,
    grid,
    options,
    core_tile_cells: int = 0,
    substrate_object: Object | None = None,
    guard_cells: int = 2,
):
    """Raster a remote, full-width substrate interface apart from the device."""
    from beamz.design.raster import Box, Grid, Scene, compile_scene

    if substrate_object is None or not any(
        obj is substrate_object for obj in scene.objects
    ):
        raise ValueError("substrate_object must be an object in the scene.")
    substrate = substrate_object
    if not isinstance(substrate.geometry, Box):
        raise ValueError("Split substrate must be a Box.")
    if not np.allclose(
        substrate.geometry.minimum[:2],
        (grid.x_edges[0], grid.y_edges[0]),
        rtol=0,
        atol=1e-15,
    ) or not np.allclose(
        substrate.geometry.maximum[:2],
        (grid.x_edges[-1], grid.y_edges[-1]),
        rtol=0,
        atol=1e-15,
    ):
        raise ValueError("Split substrate must span the entire x-y grid")
    if not np.isclose(
        substrate.geometry.minimum[2], grid.z_edges[0], rtol=0, atol=1e-15
    ):
        raise ValueError("Split substrate must start at the lower z grid edge")
    core_scene = Scene(
        scene.materials,
        tuple(obj for obj in scene.objects if obj is not substrate),
        scene.background_material,
    )
    substrate_scene = Scene(scene.materials, (substrate,), scene.background_material)
    core_region = occupied_region(core_scene, grid, guard_cells=guard_cells)
    interface_edge = int(
        np.searchsorted(grid.z_edges, substrate.geometry.maximum[2], side="left")
    )
    substrate_region = (
        (0, len(grid.x_edges) - 1),
        (0, len(grid.y_edges) - 1),
        (max(0, interface_edge - 3), min(len(grid.z_edges) - 1, interface_edge + 3)),
    )
    if substrate_region[2][1] >= core_region[2][0]:
        raise ValueError("Substrate and core regions overlap")

    def raster_local(part_scene, region):
        local_grid = Grid(
            *(
                edges[start : end + 1]
                for edges, (start, end) in zip(grid.edges, region, strict=True)
            )
        )
        started = perf_counter()
        result = compile_scene(part_scene).rasterize(local_grid, options=options)
        return result, perf_counter() - started

    if core_tile_cells:
        core_grid = Grid(
            *(
                edges[start : end + 1]
                for edges, (start, end) in zip(grid.edges, core_region, strict=True)
            )
        )
        started = perf_counter()
        core, core_tiles = rasterize_tiled_x(
            core_scene, core_grid, options, core_tile_cells, guard_cells
        )
        core_seconds = perf_counter() - started
    else:
        core, core_seconds = raster_local(core_scene, core_region)
        core_tiles = None
    lower, substrate_seconds = raster_local(substrate_scene, substrate_region)
    started = perf_counter()
    background = scene.materials[scene.background_material]
    silicon = scene.materials[substrate.material_id]
    filled_tensors = {}
    filled_yee = {}
    for group_name, destination in (
        ("tensors", filled_tensors),
        ("yee_tensors", filled_yee),
    ):
        core_group = getattr(core, group_name)
        substrate_group = getattr(lower, group_name)
        if set(core_group) != set(substrate_group):
            raise ValueError("Split material fields differ")
        for name, core_array in core_group.items():
            lower_array = substrate_group[name]
            material_name = name.split("_")[0]
            component_count = max(core_array.shape[0], lower_array.shape[0])
            template = _promote_components(core_array, component_count)
            fill = float(
                getattr(
                    background,
                    {
                        "epsilon": "permittivity",
                        "mu": "permeability",
                        "conductivity": "conductivity",
                    }[material_name],
                )
            )
            silicon_value = float(
                getattr(
                    silicon,
                    {
                        "epsilon": "permittivity",
                        "mu": "permeability",
                        "conductivity": "conductivity",
                    }[material_name],
                )
            )
            global_array = padded_array(template, core_region, grid, fill)
            substrate_start = substrate_region[2][0]
            global_array[:, :substrate_start] = _isotropic_array(
                (component_count, 1, 1, 1), silicon_value, global_array.dtype
            )
            z_start = substrate_region[2][0]
            z_stop = z_start + lower_array.shape[1]
            global_array[:, z_start:z_stop] = _promote_components(
                lower_array, component_count
            )
            destination[name] = global_array
    expand_seconds = perf_counter() - started
    result = SimpleNamespace(
        grid_edges=grid.edges,
        smoothing=core.smoothing,
        tensors=filled_tensors,
        yee_tensors=filled_yee,
        diagnostics={
            "core": dict(core.diagnostics),
            "substrate": dict(lower.diagnostics),
        },
    )
    details = {
        "core_region_cells_xyz": [end - start for start, end in core_region],
        "substrate_region_cells_xyz": [end - start for start, end in substrate_region],
        "core_native_s": core_seconds,
        "core_tiles": core_tiles,
        "substrate_native_s": substrate_seconds,
        "expand_s": expand_seconds,
    }
    return result, details
