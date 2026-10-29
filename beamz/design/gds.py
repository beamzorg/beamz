"""Import and export GDS layouts as BeamZ designs.

The module has no import-time dependency on gdsfactory. Install ``beamz[gds]``
before calling its public functions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from beamz.const import µm
from beamz.design.core import Design
from beamz.design.materials import Material
from beamz.design.structures import Box, Polygon, Rectangle
from beamz.devices.ports import Port

Layer = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ImportedComponent:
    """Layout geometry, ports, and the shift into centered world coordinates."""

    design: Design
    ports: tuple[Port, ...]
    component_name: str
    world_origin: tuple[float, float, float]

    def port(self, name: str) -> Port:
        """Return a port by name."""
        for port in self.ports:
            if port.name == name:
                return port
        available = ", ".join(port.name for port in self.ports) or "none"
        raise KeyError(f"Unknown port {name!r}; available ports: {available}.")


@dataclass(frozen=True, slots=True)
class _StackLayer:
    name: str
    zmin: float
    zmax: float
    material: Material
    sidewall_angle: float = 0.0
    width_to_z: float = 0.0


@dataclass(frozen=True, slots=True)
class _VerticalProfile:
    depth: float
    core_zmin: float
    core_zmax: float
    active_zmin: float
    active_zmax: float
    world_z_origin: float
    background: Material
    layers: tuple[_StackLayer, ...] = ()
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


def _material(value: Material | float) -> Material:
    return value if isinstance(value, Material) else Material(float(value))


def _stack_layers(
    layer_stack: Any,
    *,
    n_core: float,
    n_clad: float,
    material_map: Mapping[str, Material | float] | None,
) -> tuple[_StackLayer, ...]:
    materials: dict[str, Material] = {
        "si": Material(float(n_core) ** 2),
        "silicon": Material(float(n_core) ** 2),
        "sio2": Material(float(n_clad) ** 2),
        "oxide": Material(float(n_clad) ** 2),
        "air": Material(1.0),
    }
    materials.update(
        {
            str(name).strip().lower(): _material(value)
            for name, value in (material_map or {}).items()
        }
    )

    layers = []
    for name, level in getattr(layer_stack, "layers", {}).items():
        material = materials.get(str(getattr(level, "material", "")).lower())
        if material is None or str(name).lower() == "substrate":
            continue
        zmin = float(level.zmin) * µm
        zmax = zmin + float(level.thickness) * µm
        if zmax <= zmin:
            continue
        layers.append(
            _StackLayer(
                name=str(name),
                zmin=zmin,
                zmax=zmax,
                material=material,
                sidewall_angle=float(getattr(level, "sidewall_angle", 0.0) or 0.0),
                width_to_z=float(getattr(level, "width_to_z", 0.0) or 0.0),
            )
        )
    return tuple(layers)


def _vertical_profile(
    *,
    n_core: float,
    n_clad: float,
    core_thickness: float,
    clad_below: float,
    clad_above: float,
    z_padding: float,
    layer_stack: Any | None,
    material_map: Mapping[str, Material | float] | None,
) -> _VerticalProfile:
    if layer_stack is None:
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

    layers = _stack_layers(
        layer_stack,
        n_core=n_core,
        n_clad=n_clad,
        material_map=material_map,
    )
    if not layers:
        raise ValueError("The supplied layer stack has no recognized optical layers.")

    core_epsilon = float(n_core) ** 2
    core_candidates = [
        layer
        for layer in layers
        if layer.name.lower() == "core"
        or np.isclose(layer.material.permittivity, core_epsilon)
    ]
    if not core_candidates:
        raise ValueError(
            "The supplied layer stack has no core layer. Add its material to "
            "material_map or use the simple core_thickness/cladding inputs."
        )
    core = min(
        core_candidates,
        key=lambda item: (
            item.name.lower() != "core",
            abs((item.zmax - item.zmin) - float(core_thickness)),
        ),
    )
    stack_zmin = min(layer.zmin for layer in layers)
    stack_zmax = max(layer.zmax for layer in layers)
    z_offset = float(z_padding) - stack_zmin
    shifted_layers = tuple(
        replace(layer, zmin=layer.zmin + z_offset, zmax=layer.zmax + z_offset)
        for layer in layers
        if not np.isclose(layer.material.permittivity, core_epsilon)
    )
    depth = stack_zmax - stack_zmin + 2.0 * float(z_padding)
    return _VerticalProfile(
        depth=depth,
        core_zmin=core.zmin + z_offset,
        core_zmax=core.zmax + z_offset,
        active_zmin=float(z_padding),
        active_zmax=depth - float(z_padding),
        world_z_origin=-z_offset,
        background=Material(1.0),
        layers=shifted_layers,
        sidewall_angle=core.sidewall_angle,
        width_to_z=core.width_to_z,
    )


def _component_ports(
    component: Any,
    *,
    xmin_um: float,
    ymin_um: float,
    xy_padding: float,
) -> tuple[dict[str, Any], ...]:
    ports = []
    for port in component.ports:
        orientation = float(port.orientation) % 360.0
        rounded = int(round(orientation / 90.0) * 90) % 360
        try:
            direction = {180: "+x", 0: "-x", 90: "-y", 270: "+y"}[rounded]
        except KeyError as exc:
            raise ValueError(
                f"Port {port.name!r} has unsupported orientation {orientation}; "
                "BeamZ requires axis-aligned ports."
            ) from exc
        center_um = getattr(port, "dcenter", port.center)
        width_um = getattr(port, "dwidth", port.width)
        ports.append(
            {
                "name": str(port.name),
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
    """Grow each waveguide from its port plane out to the domain edge."""
    if not enabled:
        return design
    edges = {"+x": design.width, "-x": 0.0, "+y": design.height, "-y": 0.0}
    depth = profile.core_zmax - profile.core_zmin
    for port in ports:
        inward = str(port["direction"])
        outward = ("-" if inward.startswith("+") else "+") + inward[1]
        cx, cy = port["center"]
        width = float(port["width"])
        along_x = outward.endswith("x")
        center = cx if along_x else cy
        start = center - float(overlap) * (1 if outward.startswith("+") else -1)
        edge = edges[outward]
        span = abs(edge - start)
        position = (
            (min(start, edge), cy - 0.5 * width, profile.core_zmin)
            if along_x
            else (cx - 0.5 * width, min(start, edge), profile.core_zmin)
        )
        design += Rectangle(
            position=position,
            width=span if along_x else width,
            height=width if along_x else span,
            depth=depth,
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


def import_component(
    component: Any = "mmi1x2",
    *,
    layer: Layer = (1, 0),
    n_core: float = 2.0,
    n_clad: float = 1.44,
    core_thickness: float = 0.22e-6,
    clad_below: float = 0.5e-6,
    clad_above: float = 0.5e-6,
    xy_padding: float = 0.0,
    z_padding: float = 0.0,
    extend_ports: bool = False,
    port_overlap: float = 0.0,
    settings: Mapping[str, Any] | None = None,
    layer_stack: Any | None = None,
    material_map: Mapping[str, Material | float] | None = None,
    unify: bool = True,
) -> ImportedComponent:
    """Convert a gdsfactory component or active-PDK cell into a BeamZ design.

    Pass ``layer_stack`` to interpret a PDK stack; otherwise the simple
    ``core_thickness`` / cladding arguments apply. Cell names go through
    ``gdsfactory.get_component``. Lengths are metres. ``extend_ports`` draws
    each waveguide out to the domain edge, and ``unify`` merges touching
    polygons of the same material.
    """
    component = _resolve_component(component, settings)
    polygons = _polygons_on_layer(component, layer)
    points = np.vstack(polygons)
    xmin_um, ymin_um = np.min(points, axis=0)
    xmax_um, ymax_um = np.max(points, axis=0)
    profile = _vertical_profile(
        n_core=n_core,
        n_clad=n_clad,
        core_thickness=core_thickness,
        clad_below=clad_below,
        clad_above=clad_above,
        z_padding=z_padding,
        layer_stack=layer_stack,
        material_map=material_map,
    )
    design = Design(
        width=float(xmax_um - xmin_um) * µm + 2.0 * float(xy_padding),
        height=float(ymax_um - ymin_um) * µm + 2.0 * float(xy_padding),
        depth=profile.depth,
        background=profile.background,
    )
    for stack_layer in profile.layers:
        design += Rectangle(
            position=(0.0, 0.0, stack_layer.zmin),
            width=design.width,
            height=design.height,
            depth=stack_layer.zmax - stack_layer.zmin,
            material=stack_layer.material,
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
    """Export planar structures to GDS.

    ``precision`` is the database grid in metres (default 1 nm) on a private
    KLayout layout, so it does not follow gdsfactory's global grid. Coordinates
    are GDS micrometres and interior rings stay holes. Equal materials share a
    layer unless ``layer_map`` selects specific materials. Shapes with no planar
    polygon are rejected.
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
