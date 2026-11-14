"""Translate imported layout port metadata into BeamZ ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from beamz.const import µm
from beamz.design.geometry.core import Design
from beamz.design.geometry.primitives import Rectangle
from beamz.design.io.decode import material as _material  # noqa: F401
from beamz.design.materials import Material
from beamz.devices.ports import Port


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
