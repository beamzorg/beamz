"""Import and export GDS layouts as BeamZ designs.

The module has no import-time dependency on gdsfactory. Install ``beamz[gds]``
before calling its public functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, cast

import numpy as np

from beamz.const import µm
from beamz.design.core import Design
from beamz.design.materials import Material
from beamz.design.structures import Box, Polygon, Rectangle
from beamz.devices.ports import Port

Layer = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ImportedComponent:
    """A layout component converted into simulation-ready BeamZ objects.

    Attributes
    ----------
    design
        Imported and extruded simulation geometry in local design coordinates.
    ports
        Canonical modal ports at the original component port planes.
    component_name
        Name reported by the source layout component.
    world_origin
        Translation from local design coordinates to centered world coordinates.
    source_origin
        Add this metre-valued translation to local coordinates to recover the
        original layout/LayerStack frame.
    report
        Immutable per-level statuses, source z bounds, effective mesh orders,
        profile policy and associated optical ports. Empty in simple mode.
    """

    design: Design
    ports: tuple[Port, ...]
    component_name: str
    world_origin: tuple[float, float, float]
    source_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    report: tuple[Mapping[str, Any], ...] = ()
    _prepare: Callable[..., ImportedComponent] | None = field(
        default=None, repr=False, compare=False
    )

    def port(self, name: str) -> Port:
        """Return a port by name."""
        for port in self.ports:
            if port.name == name:
                return port
        available = ", ".join(port.name for port in self.ports) or "none"
        raise KeyError(f"Unknown port {name!r}; available ports: {available}.")


@dataclass(frozen=True, slots=True)
class _VerticalProfile:
    depth: float
    core_zmin: float
    core_zmax: float
    active_zmin: float
    active_zmax: float
    world_z_origin: float
    background: Material
    sidewall_angle: float = 0.0
    width_to_z: float = 0.0


def _gdsfactory(*, activate_pdk=True):
    try:
        import gdsfactory as gf  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "GDS support requires BeamZ's optional layout dependency. "
            "Install it with `pip install beamz[gds]`."
        ) from exc
    if not activate_pdk:
        return gf
    try:
        gf.get_active_pdk()
    except ValueError:
        generic_pdk = getattr(getattr(gf, "gpdk", None), "get_generic_pdk", None)
        if generic_pdk is None:
            raise
        generic_pdk().activate()
    return gf


def _resolve_component(component: Any, settings: Mapping[str, Any] | None):
    gf = _gdsfactory()
    try:
        return gf.get_component(component, settings=dict(settings or {}))
    except Exception as exc:
        raise ValueError(f"Could not resolve layout component {component!r}.") from exc


def _polygons_on_layer(component: Any, layer: Layer) -> tuple[np.ndarray, ...]:
    polygons_by_layer = component.get_polygons_points(by="tuple")
    polygons = tuple(
        np.asarray(points, dtype=float)[:, :2]
        for points in polygons_by_layer.get(tuple(layer), ())
    )
    if not polygons:
        available = sorted(polygons_by_layer)
        raise ValueError(
            f"Layer {tuple(layer)} is absent from component {component.name!r}; "
            f"available layers: {available}."
        )
    return polygons


def _vertical_profile(*, core_thickness, clad_below, clad_above, z_padding, n_clad):
    core_zmin = float(z_padding) + float(clad_below)
    core_zmax = core_zmin + float(core_thickness)
    depth = core_zmax + float(clad_above) + float(z_padding)
    return _VerticalProfile(
        depth=depth,
        core_zmin=core_zmin,
        core_zmax=core_zmax,
        active_zmin=float(z_padding),
        active_zmax=depth - float(z_padding),
        world_z_origin=-core_zmin,
        background=Material(float(n_clad) ** 2),
    )


def _component_ports(
    component: Any,
    *,
    xmin_um: float,
    ymin_um: float,
    xy_padding: float,
) -> tuple[dict[str, Any], ...]:
    cached = getattr(component, "_beamz_ports", None)
    if cached is not None:
        return tuple(
            {
                **p,
                "center": (
                    p["center"][0] - xmin_um * µm + xy_padding,
                    p["center"][1] - ymin_um * µm + xy_padding,
                ),
            }
            for p in cached
        )
    ports = []
    for port in component.ports:
        if getattr(port, "port_type", "optical") != "optical":
            continue
        orientation = float(port.orientation) % 360.0
        if not np.isfinite(orientation):
            raise ValueError(f"Port {port.name!r} orientation must be finite.")
        rounded = int(round(orientation / 90.0) * 90) % 360
        if not np.isclose(
            (orientation - rounded + 180) % 360 - 180, 0, atol=1e-6, rtol=0
        ):
            raise ValueError(
                f"Port {port.name!r} has unsupported orientation {orientation}; "
                "BeamZ requires axis-aligned ports."
            )
        direction = {180: "+x", 0: "-x", 90: "-y", 270: "+y"}[rounded]
        info = component.kcl.layout.get_info(port.layer)
        center_um = getattr(port, "dcenter", port.center)
        width_um = getattr(port, "dwidth", port.width)
        ports.append(
            {
                "name": str(port.name),
                "layer": (info.layer, info.datatype),
                "center": (
                    (float(center_um[0]) - xmin_um) * µm + float(xy_padding),
                    (float(center_um[1]) - ymin_um) * µm + float(xy_padding),
                ),
                "width": float(width_um) * µm,
                "direction": direction,
            }
        )
    return tuple(ports)


def _extend_ports(
    design: Design,
    ports: tuple[dict[str, Any], ...],
    *,
    profile: _VerticalProfile,
    material: Material,
    enabled: bool,
    overlap: float,
) -> Design:
    if not enabled:
        return design
    edges = {"+x": design.width, "-x": 0.0, "+y": design.height, "-y": 0.0}
    for port in ports:
        direction = str(port["direction"])
        outward = ("-" if direction.startswith("+") else "+") + direction[1]
        cx, cy = port["center"]
        shift = -float(overlap)
        sx = cx + shift * ({"+x": 1, "-x": -1}.get(outward, 0))
        sy = cy + shift * ({"+y": 1, "-y": -1}.get(outward, 0))
        width = float(port["width"])
        if outward.endswith("x"):
            edge = edges[outward]
            position = (min(sx, edge), cy - 0.5 * width, profile.core_zmin)
            size = (abs(edge - sx), width)
        else:
            edge = edges[outward]
            position = (cx - 0.5 * width, min(sy, edge), profile.core_zmin)
            size = (width, abs(edge - sy))
        design += Rectangle(
            position=position,
            width=size[0],
            height=size[1],
            depth=profile.core_zmax - profile.core_zmin,
            material=material,
            sidewall_angle=profile.sidewall_angle,
            width_to_z=profile.width_to_z,
        )
    return design


def _beamz_ports(
    ports: tuple[dict[str, Any], ...], profile: _VerticalProfile
) -> tuple[Port, ...]:
    core_center = 0.5 * (profile.core_zmin + profile.core_zmax)
    z_span = 2.0 * max(
        core_center - profile.active_zmin,
        profile.active_zmax - core_center,
    )
    result = []
    for item in ports:
        axis = str(item["direction"])[1]
        direction: Literal["+", "-"] = (
            "+" if str(item["direction"]).startswith("+") else "-"
        )
        width = float(item["width"])
        size = (0.0, width, z_span) if axis == "x" else (width, 0.0, z_span)
        result.append(
            Port(
                center=(*item["center"], core_center),
                size=size,
                name=str(item["name"]),
                direction=direction,
            )
        )
    return tuple(result)


def _source_layers(expression: Any):
    if expression is None:
        return set()
    if isinstance(expression, (tuple, list)) and len(expression) == 2:
        return {tuple(expression)}
    if isinstance(expression, (int, str)):
        return {tuple(_gdsfactory().get_layer_tuple(cast(Any, expression)))}
    layer = getattr(expression, "layer", None)
    if layer is not None:
        return _source_layers(layer)
    # Subtracted layers do not contribute material to the resulting footprint.
    layers = _source_layers(getattr(expression, "layer1", None))
    if getattr(expression, "operation", None) != "not":
        layers |= _source_layers(getattr(expression, "layer2", None))
    return layers


def _extend_stack_polygon(polygon, ports, bounds):
    """Continue each intersecting process footprint through a port datum.

    Union before tapering keeps longitudinal seams out of the physical geometry
    and continues slab/cladding footprints along with the port's core.
    """
    from shapely.geometry import LineString, box
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    from .raster.importers.gdsfactory import _shapely_polygons

    area = ShapelyPolygon(polygon.exterior, polygon.holes)
    additions = [area]
    for port in ports:
        x, y = port["source_center"]
        axis = 0 if port["direction"].endswith("x") else 1
        plane = (x, y)[axis]
        if plane < area.bounds[axis] or plane > area.bounds[axis + 2]:
            continue
        # Only take a cross section at the actual datum; never bridge a gap.
        line = LineString(
            [(plane, bounds[0][1]), (plane, bounds[1][1])]
            if axis == 0
            else [(bounds[0][0], plane), (bounds[1][0], plane)]
        )
        section = area.intersection(line)
        edge = bounds[0][axis] if port["direction"].startswith("+") else bounds[1][axis]
        for part in getattr(section, "geoms", (section,)):
            if part.geom_type != "LineString" or part.is_empty:
                continue
            start, stop = sorted((plane, edge))
            if start == stop:
                continue
            low, high = part.bounds[1 - axis], part.bounds[3 - axis]
            if high > low:
                additions.append(
                    box(start, low, stop, high)
                    if axis == 0
                    else box(low, start, high, stop)
                )
    return _shapely_polygons(unary_union(additions))


def _snapshot_component(component, *, stack=False):
    from types import SimpleNamespace

    raw = {} if stack else component.get_polygons_points(by="tuple")
    polygons = {
        layer: tuple(np.array(p, dtype=float, copy=True) for p in points)
        for layer, points in raw.items()
    }
    return SimpleNamespace(
        name=str(component.name),
        _beamz_ports=_component_ports(component, xmin_um=0, ymin_um=0, xy_padding=0),
        get_polygons_points=lambda **_: polygons,
    )


def _snapshot_stack(component, layer_stack, fill_levels):
    from types import SimpleNamespace

    from .raster.importers.gdsfactory import (
        _exclusion_polygons,
        _mesh_order,
        _polygons_by_layer,
        _region_polygons,
        _z_to_bias_profile,
    )

    polygons = _polygons_by_layer(component)
    levels = {}
    for name, level in layer_stack.layers.items():
        fill = name in fill_levels or bool(getattr(level, "background", False))
        masks = () if fill else tuple(_region_polygons(component, level, polygons, µm))
        exclusions = _exclusion_polygons(component, level, polygons, µm) if fill else ()
        profile = getattr(level, "z_to_bias", None)
        sampled = callable(profile)
        if profile is not None and (masks or fill):
            try:
                z, bias = _z_to_bias_profile(profile)
            except ValueError as exc:
                raise ValueError(f"Layer {name!r}: {exc}") from exc
            profile = (tuple(z), tuple(bias))
        else:
            profile = None
        levels[name] = SimpleNamespace(
            layer=None,
            _polygons=masks,
            _exclusions=tuple(exclusions),
            _port_layers=tuple(
                _source_layers(getattr(level, "layer", None))
                | _source_layers(getattr(level, "derived_layer", None))
            ),
            zmin=float(level.zmin),
            thickness=float(level.thickness),
            material=str(getattr(level, "material", "")),
            sidewall_angle=float(getattr(level, "sidewall_angle", 0) or 0),
            width_to_z=float(getattr(level, "width_to_z", 0) or 0),
            bias=getattr(level, "bias", None),
            z_to_bias=profile,
            _sampled=sampled,
            mesh_order=_mesh_order(level),
            background=fill,
        )
    return SimpleNamespace(layers=levels)


def _import_stack(
    component,
    layer_stack,
    *,
    material_map,
    bounds,
    port_levels,
    xy_padding,
    z_padding,
    extend_ports,
    port_overlap,
    unify,
):
    from .raster.importers.gdsfactory import _resolve_gdsfactory
    from .raster.schema import ExtrudedPolygon, TaperedExtrudedPolygon

    scene, report, bounds = _resolve_gdsfactory(
        component,
        layer_stack,
        material_map=material_map,
        bounds=bounds,
        _padding=(xy_padding, z_padding),
    )
    origin = np.asarray(bounds[0])
    extent = np.subtract(bounds[1], origin)
    raw_ports = _component_ports(
        component, xmin_um=origin[0] / µm, ymin_um=origin[1] / µm, xy_padding=0
    )
    port_levels = dict(port_levels or {})
    if set(port_levels) - {p["name"] for p in raw_ports}:
        raise ValueError("port_levels contains unknown optical port names.")
    ports, selected = [], []
    associations = {}
    for port in raw_ports:
        candidates = [
            entry
            for entry in report
            if entry["status"] == "included"
            and (port["layer"] in layer_stack.layers[entry["name"]]._port_layers)
        ]
        if port["name"] in port_levels:
            candidates = [
                entry
                for entry in report
                if entry["status"] == "included"
                and entry["name"] == port_levels[port["name"]]
            ]
        if not candidates:
            raise ValueError(
                f"Port {port['name']!r} has no physical level; set port_levels."
            )
        # Overlapping core/slab levels share a port: the winning level supplies
        # its datum. Vertically separated levels require an explicit selection.
        if max(entry["z_bounds"][0] for entry in candidates) >= min(
            entry["z_bounds"][1] for entry in candidates
        ):
            raise ValueError(
                f"Port {port['name']!r} maps to multiple heights; set port_levels."
            )
        entry = min(reversed(candidates), key=lambda entry: entry["mesh_order"])
        associations.setdefault(entry["name"], []).append(port["name"])
        z = np.mean(entry["z_bounds"]) - origin[2]
        x, y = port["center"]
        if not (0 <= x <= extent[0] and 0 <= y <= extent[1] and 0 <= z <= extent[2]):
            raise ValueError(f"Port {port['name']!r} lies outside bounds.")
        profile = _VerticalProfile(
            depth=extent[2],
            core_zmin=entry["z_bounds"][0] - origin[2],
            core_zmax=entry["z_bounds"][1] - origin[2],
            active_zmin=0,
            active_zmax=extent[2],
            world_z_origin=origin[2],
            background=scene.materials[scene.background_material],
        )
        ports.extend(_beamz_ports((port,), profile))
        selected.append({**port, "source_center": (x + origin[0], y + origin[1])})
    frame = Design(
        width=extent[0],
        height=extent[1],
        depth=extent[2],
        background=scene.materials[scene.background_material],
    )
    structures = []
    objects_by_id = {obj.id: obj for obj in scene.objects}
    for entry in sorted(
        (r for r in report if r["status"] == "included"),
        key=lambda r: objects_by_id[r["object_ids"][0]].priority,
    ):
        group = []
        for object_id in entry["object_ids"]:
            obj = objects_by_id[object_id]
            geometry = obj.geometry
            assert isinstance(geometry, (ExtrudedPolygon, TaperedExtrudedPolygon))
            footprints = (
                _extend_stack_polygon(geometry.polygon, selected, bounds)
                if extend_ports and not layer_stack.layers[entry["name"]].background
                else [geometry.polygon]
            )
            for polygon in footprints:
                z = geometry.z_min - origin[2]

                def ring(points, z=z):
                    return tuple((x - origin[0], y - origin[1], z) for x, y in points)

                group.append(
                    Polygon(
                        vertices=ring(polygon.exterior),
                        interiors=tuple(ring(h) for h in polygon.holes),
                        z=z,
                        depth=geometry.z_max - geometry.z_min,
                        material=scene.materials[obj.material_id],
                        sidewall_angle=getattr(geometry, "sidewall_angle_degrees", 0),
                        width_to_z=getattr(geometry, "width_to_z", 0),
                    )
                )
        # Unify only within a process level: merging across priorities changes ownership.
        design = frame.updated_copy(structures=group)
        structures.extend(design.unified_polygons().structures if unify else group)
    design = frame.updated_copy(structures=structures)
    from types import MappingProxyType

    report = tuple(
        MappingProxyType(
            {
                **entry,
                "ports": tuple(associations.get(entry["name"], ())),
            }
        )
        for entry in report
    )
    return ImportedComponent(
        design,
        tuple(ports),
        str(component.name),
        (-0.5 * design.width, -0.5 * design.height, float(origin[2])),
        (float(origin[0]), float(origin[1]), float(origin[2])),
        report,
    )


def import_component(
    component: Any = "mmi1x2",
    *,
    layer: Layer = (1, 0),
    n_core: float = 3.47,
    n_clad: float = 1.44,
    core_thickness: float = 0.22e-6,
    clad_below: float = 1.0e-6,
    clad_above: float = 1.0e-6,
    xy_padding: float = 0.0,
    z_padding: float = 0.0,
    extend_ports: bool = False,
    port_overlap: float = 0.0,
    settings: Mapping[str, Any] | None = None,
    layer_stack: Any | None = None,
    material_map: Mapping[str, Material | float] | None = None,
    unify: bool = True,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    fill_levels: Sequence[str] = (),
    port_levels: Mapping[str, str] | None = None,
) -> ImportedComponent:
    """Convert a gdsfactory component or active-PDK cell into a BeamZ design.

    Optional PDK stack interpretation is explicit: pass a gdsfactory
    ``LayerStack`` through ``layer_stack``. Cell names are resolved by
    ``gdsfactory.get_component``, so the active PDK remains the authority for
    component discovery.

    Parameters
    ----------
    component
        Component object, callable, or cell specification understood by
        gdsfactory.
    layer
        GDS layer and datatype containing the imported core geometry.
    n_core, n_clad
        Refractive indices used for core and simple cladding materials.
    core_thickness, clad_below, clad_above
        Simple vertical geometry in metres, used when ``layer_stack`` is absent.
    xy_padding, z_padding
        Legacy geometry-view padding in metres. For simulation preparation,
        specify padding once through gdsfactory.Settings instead.
    extend_ports
        Extend imported waveguides from their port planes to the domain edges.
    port_overlap
        Inward overlap between imported polygons and port extensions.
    settings
        Settings forwarded to gdsfactory component resolution.
    layer_stack
        Optional explicit gdsfactory PDK layer stack.
    material_map
        Mapping from PDK material names to BeamZ materials or permittivities.
        Required with layer_stack. Supplied objects are preserved, including
        tensors and dispersion; process material names are never guessed.
    bounds
        Optional ((xmin, ymin, zmin), (xmax, ymax, zmax)) crop in metres in the
        source stack frame. Mutually exclusive with padding. Source taper
        references are retained outside the crop.
    fill_levels
        Names of levels to fill across the domain, including older PDKs without
        background metadata. Background exclusions remain holes.
    port_levels
        Explicit port-name to physical-level mapping when several heights share
        a mask. Overlapping core/slab levels otherwise use mesh-order priority.
    unify
        Merge compatible touching polygons within each physical level. Levels
        retain their separate priority. Port extension continues full footprints;
        port_overlap is only needed by the simple no-stack extrusion.
    """
    for name, value in (
        ("xy_padding", xy_padding),
        ("z_padding", z_padding),
        ("port_overlap", port_overlap),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if bounds is not None:
        from .raster.importers.gdsfactory import _validate_bounds

        bounds = _validate_bounds(bounds)
        if xy_padding or z_padding:
            raise ValueError("Choose bounds or padding, not both.")
    component = _resolve_component(component, settings)
    if layer_stack is not None:
        if material_map is None:
            raise ValueError("layer_stack requires an explicit material_map.")
        unknown = set(fill_levels) - set(layer_stack.layers)
        if unknown:
            raise ValueError(f"Unknown fill_levels: {sorted(unknown)}.")
        stack = _snapshot_stack(component, layer_stack, fill_levels)
        component = _snapshot_component(component, stack=True)
        materials = dict(material_map)
        port_levels = dict(port_levels or {})

        render = partial(
            _import_stack,
            component,
            stack,
            material_map=materials,
            port_levels=port_levels,
            unify=unify,
        )
    else:
        if material_map is not None:
            raise ValueError(
                "material_map requires layer_stack; use n_core/n_clad for simple extrusion."
            )
        if bounds is not None or fill_levels or port_levels:
            raise ValueError("bounds, fill_levels and port_levels require layer_stack.")
        component = _snapshot_component(component)

        render = partial(
            _import_simple,
            component,
            layer=layer,
            n_core=n_core,
            n_clad=n_clad,
            core_thickness=core_thickness,
            clad_below=clad_below,
            clad_above=clad_above,
            unify=unify,
        )

    def build(xy_padding, z_padding, bounds=bounds, overlap=0.0, extend=True):
        if bounds is not None and (xy_padding or z_padding):
            raise ValueError(
                "Choose bounds or padding, not both; use zero padding with explicit bounds."
            )
        result = render(
            xy_padding=xy_padding,
            z_padding=z_padding,
            bounds=bounds,
            port_overlap=overlap,
            extend_ports=extend,
        )

        def prepare_geometry(
            *, design, xy_padding, z_padding, bounds=bounds, overlap=0.0
        ):
            if design is not result.design:
                raise ValueError(
                    "Imported geometry was edited; use a native Simulation for the edited Design."
                )
            return build(xy_padding, z_padding, bounds, overlap)

        return replace(result, _prepare=prepare_geometry)

    return build(xy_padding, z_padding, bounds, port_overlap, extend_ports)


def _import_simple(
    component,
    *,
    layer,
    n_core,
    n_clad,
    core_thickness,
    clad_below,
    clad_above,
    xy_padding,
    z_padding,
    extend_ports,
    port_overlap,
    unify,
    bounds,
):
    if bounds is not None:
        raise ValueError("bounds requires layer_stack.")

    polygons = _polygons_on_layer(component, layer)
    points = np.vstack(polygons)
    xmin_um, ymin_um = np.min(points, axis=0)
    xmax_um, ymax_um = np.max(points, axis=0)
    profile = _vertical_profile(
        n_clad=n_clad,
        core_thickness=core_thickness,
        clad_below=clad_below,
        clad_above=clad_above,
        z_padding=z_padding,
    )
    design = Design(
        width=float(xmax_um - xmin_um) * µm + 2.0 * float(xy_padding),
        height=float(ymax_um - ymin_um) * µm + 2.0 * float(xy_padding),
        depth=profile.depth,
        background=profile.background,
    )
    core_material = Material(float(n_core) ** 2)
    for points_um in polygons:
        vertices = tuple(
            (
                (float(x) - xmin_um) * µm + float(xy_padding),
                (float(y) - ymin_um) * µm + float(xy_padding),
                profile.core_zmin,
            )
            for x, y in points_um
        )
        design += Polygon(
            vertices=vertices,
            z=profile.core_zmin,
            depth=profile.core_zmax - profile.core_zmin,
            material=core_material,
            sidewall_angle=profile.sidewall_angle,
            width_to_z=profile.width_to_z,
        )

    raw_ports = _component_ports(
        component,
        xmin_um=float(xmin_um),
        ymin_um=float(ymin_um),
        xy_padding=xy_padding,
    )
    design = _extend_ports(
        design,
        raw_ports,
        profile=profile,
        material=core_material,
        enabled=bool(extend_ports),
        overlap=float(port_overlap),
    )
    if unify:
        design = design.unified_polygons()
    return ImportedComponent(
        design=design,
        ports=_beamz_ports(raw_ports, profile),
        component_name=str(component.name),
        world_origin=(
            -0.5 * design.width,
            -0.5 * design.height,
            profile.world_z_origin,
        ),
        source_origin=(
            float(xmin_um) * µm - xy_padding,
            float(ymin_um) * µm - xy_padding,
            profile.world_z_origin,
        ),
    )


def import_gds(path: str | Path, **kwargs: Any) -> ImportedComponent:
    """Read a GDS file and convert its top component into a BeamZ design.

    Additional keyword arguments are forwarded to :func:`import_component`.
    """
    gf = _gdsfactory()
    source = Path(path)
    try:
        component = gf.read.import_gds(source)
    except Exception as exc:
        raise ValueError(f"Could not read GDS file {source}.") from exc
    return import_component(component, **kwargs)


def _material_key(structure: Any) -> tuple[float, float, float] | None:
    material = getattr(structure, "material", None)
    if material is None:
        return None
    return (
        float(getattr(material, "permittivity", 1.0)),
        float(getattr(material, "permeability", 1.0)),
        float(getattr(material, "conductivity", 0.0)),
    )


def _add_polygon(cell, structure, layer_index, *, kdb, precision):
    vertices = getattr(structure, "vertices", ())
    if not vertices:
        raise TypeError(f"GDS export does not support {type(structure).__name__}.")
    rings = [vertices, *getattr(structure, "interiors", ())]
    points = []
    for ring in rings:
        xy = np.asarray(ring, dtype=float)[:, :2]
        if not np.isfinite(xy).all():
            raise ValueError("GDS coordinates must be finite.")
        if np.any(np.abs(xy) > precision * (np.iinfo(np.int32).max - 1)):
            raise ValueError("GDS precision is too fine for the exported coordinates.")
        points.append([kdb.DPoint(float(x / µm), float(y / µm)) for x, y in xy])
    polygon = kdb.DPolygon(points[0])
    for hole in points[1:]:
        polygon.insert_hole(hole)
    cell.shapes(layer_index).insert(polygon)


def export_gds(
    design: Design,
    path: str | Path,
    *,
    cell_name: str = "beamz_design",
    precision: float = 1e-9,
    layer_map: Mapping[Material, Layer] | None = None,
) -> Path:
    """Export planar structures to GDS with an explicit coordinate precision.

    ``precision`` is the database grid in metres (default 1 nm). A private
    KLayout database avoids quantization through gdsfactory's global grid.
    Coordinates remain in GDS micrometres. Interior rings remain holes.

    By default, equal materials share automatically assigned layers. An explicit
    ``layer_map={material: (layer, datatype)}`` selects only those materials,
    e.g. the core mask without low-index region-clearing polygons. Volumetric
    structures without a planar projection are rejected explicitly.
    """
    if not np.isfinite(precision) or precision <= 0:
        raise ValueError("precision must be finite and positive (metres).")
    if layer_map is not None:
        for layer in layer_map.values():
            if len(layer) != 2 or any(
                isinstance(v, bool)
                or not isinstance(v, (int, np.integer))
                or not 0 <= v <= 65535
                for v in layer
            ):
                raise ValueError("GDS layer/datatype must be integers in [0, 65535].")
    gf = _gdsfactory(activate_pdk=False)
    layout = gf.kdb.Layout()
    layout.dbu = precision / µm
    cell = layout.create_cell(cell_name)
    material_layers: dict[tuple[float, float, float], Layer] = {}
    count = 0
    for structure in design.unified_polygons().structures:
        if getattr(structure, "is_pml", False):
            continue
        if (
            layer_map is not None
            and getattr(structure, "material", None) not in layer_map
        ):
            continue
        if isinstance(structure, Box):
            structure = structure.to_rectangle()
        material_key = _material_key(structure)
        if material_key is None:
            continue
        layer = (
            layer_map[structure.material]
            if layer_map is not None
            else material_layers.setdefault(material_key, (len(material_layers), 0))
        )
        _add_polygon(
            cell, structure, layout.layer(*layer), kdb=gf.kdb, precision=precision
        )
        count += 1
    if not count:
        raise ValueError("The design has no planar material structures to export.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    layout.write(str(destination))
    return destination


__all__ = ["ImportedComponent", "export_gds", "import_component", "import_gds"]
