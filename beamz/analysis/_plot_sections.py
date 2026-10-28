"""Vector cross-sections of structures, devices, and PML bands."""

from __future__ import annotations

import numpy as np

from beamz.analysis._plot_canvas import _PLANE_AXES, _UM
from beamz.analysis._plot_design import _vertices_2d
from beamz.analysis._plot_fields import AxisSlice
from beamz.analysis._plot_materials import _layout_color
from beamz.devices.visualization import visual_spec_from_device


def _pml_section_layers(sim, normal):
    """Yield PML thickness and visible edges for a plotted section."""
    from beamz.devices.boundaries import PML

    edge_map = (
        {"left": "left", "right": "right", "bottom": "bottom", "top": "top"}
        if normal == "z"
        else {"left": "left", "right": "right", "front": "bottom", "back": "top"}
    )
    for boundary in getattr(sim, "boundaries", ()) or ():
        if not isinstance(boundary, PML):
            continue
        thickness = boundary.thickness
        if thickness is None or thickness <= 0.0:
            continue
        boundary_edges = boundary._get_edges_for_dimensionality(sim.is_3d)
        visible_edges = tuple(
            edge_map[edge] for edge in boundary_edges if edge in edge_map
        )
        if visible_edges:
            yield float(thickness), visible_edges


def _device_geometry(device):
    """Return normalized axis, center, size, and direction for an overlay device."""
    spec = visual_spec_from_device(device)
    if spec is None or len(spec.center) < 3:
        return None
    center, size = np.asarray(spec.center), np.asarray(spec.size)
    direction = str(spec.direction or "").lower()
    axis = next((name for name in _PLANE_AXES if direction.endswith(name)), None)
    if axis is None and size.size == 3 and np.any(np.isclose(size, 0.0)):
        axis = ("x", "y", "z")[int(np.argmin(np.abs(size)))]
    if axis is None:
        return None
    if size.size == 2:
        size = {"x": (0.0, *size), "y": (size[0], 0.0, size[1]), "z": (*size, 0.0)}[
            axis
        ]
    return axis, center.astype(float), np.asarray(size, dtype=float), direction


def _draw_circular_device_overlay(ax, spec, *, normal, origin, color):
    from matplotlib.patches import Circle

    vertical, horizontal = _PLANE_AXES[normal]
    indices = tuple("xyz".index(axis) for axis in (horizontal, vertical))
    center = (
        spec.center if len(spec.center) == 2 else tuple(spec.center[i] for i in indices)
    )
    patch = Circle(
        tuple(
            (value - origin[i]) / _UM for value, i in zip(center, indices, strict=True)
        ),
        radius=0.5 * spec.size[0] / _UM,
        fill=False,
        edgecolor=color,
        linewidth=2.0,
    )
    ax.add_patch(patch)
    return patch


def _draw_parallel_device_footprint(
    ax, *, horizontal, vertical, center, size, origin, color
):
    """Draw the full footprint of a device whose plane faces the section."""
    from matplotlib.patches import Rectangle

    origin_by_axis = dict(zip(("x", "y", "z"), map(float, origin), strict=True))
    indices = {"x": 0, "y": 1, "z": 2}
    horizontal_index = indices[horizontal]
    vertical_index = indices[vertical]
    lower = (
        (center[horizontal_index] - 0.5 * size[horizontal_index])
        - origin_by_axis[horizontal],
        (center[vertical_index] - 0.5 * size[vertical_index])
        - origin_by_axis[vertical],
    )
    rectangle = Rectangle(
        tuple(value / _UM for value in lower),
        size[horizontal_index] / _UM,
        size[vertical_index] / _UM,
        facecolor=color,
        edgecolor=color,
        alpha=0.25,
        linewidth=2.0,
    )
    ax.add_patch(rectangle)
    return rectangle


def _draw_edge_on_device(
    ax,
    *,
    normal,
    axis,
    horizontal,
    vertical,
    center,
    size,
    direction,
    origin,
    color,
    source,
):
    if axis == horizontal:
        segments = [(False, vertical, horizontal, True)]
    elif axis == vertical:
        segments = [(True, horizontal, vertical, True)]
    elif normal == "z":
        segments = [
            (True, horizontal, vertical, True),
            (False, vertical, horizontal, False),
        ]
    else:
        return
    origin_by_axis: dict[str, float] = dict(
        zip(("x", "y", "z"), map(float, origin), strict=True)
    )
    indices = {"x": 0, "y": 1, "z": 2}
    for horizontal_line, span_axis, fixed_axis, arrow in segments:
        span_index, fixed_index = indices[span_axis], indices[fixed_axis]
        half = 0.5 * size[span_index]
        span = (
            center[span_index] - half - origin_by_axis[span_axis],
            center[span_index] + half - origin_by_axis[span_axis],
        )
        fixed = (center[fixed_index] - origin_by_axis[fixed_axis]) / _UM
        span = tuple(value / _UM for value in span)
        x, y = (span, (fixed, fixed)) if horizontal_line else ((fixed, fixed), span)
        ax.plot(x, y, color=color, lw=2.0)
        if not (source and arrow):
            continue
        anchor = (center[span_index] - origin_by_axis[span_axis]) / _UM
        delta = -0.55 if direction.startswith("-") else 0.55
        start_xy = (anchor, fixed) if horizontal_line else (fixed, anchor)
        end_xy = (anchor, fixed + delta) if horizontal_line else (fixed + delta, anchor)
        ax.annotate(
            "",
            xy=end_xy,
            xytext=start_xy,
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.0},
        )


def _draw_device_slice_overlay(ax, device, *, normal, origin, color, source):
    spec = visual_spec_from_device(device)
    if spec is not None and (spec.style or {}).get("shape") == "circle":
        return _draw_circular_device_overlay(
            ax, spec, normal=normal, origin=origin, color=color
        )
    geometry = _device_geometry(device)
    if geometry is None:
        return
    axis, center, size, direction = geometry
    vertical, horizontal = _PLANE_AXES[normal]
    if axis == normal:
        # A source or monitor whose plane is parallel to the displayed cross
        # section is best represented by its full footprint, rather than the
        # two center lines used for edge-on devices.
        return _draw_parallel_device_footprint(
            ax,
            horizontal=horizontal,
            vertical=vertical,
            center=center,
            size=size,
            origin=origin,
            color=color,
        )
    _draw_edge_on_device(
        ax,
        normal=normal,
        axis=axis,
        horizontal=horizontal,
        vertical=vertical,
        center=center,
        size=size,
        direction=direction,
        origin=origin,
        color=color,
        source=source,
    )


def overlay_boundaries(ax, extent, thickness, *, edges=None, **style):
    """Draw cross-section PML bands without letting side bands cover corners."""
    from matplotlib.patches import Rectangle

    x0, x1, y0, y1 = (float(v) for v in extent)
    width = max(x1 - x0, 0.0)
    height = max(y1 - y0, 0.0)
    if width <= 0.0 or height <= 0.0:
        return ()

    px = min(max(float(thickness), 0.0), 0.5 * width)
    py = min(max(float(thickness), 0.0), 0.5 * height)
    if px <= 0.0 and py <= 0.0:
        return ()

    selected = frozenset(("left", "right", "bottom", "top") if edges is None else edges)
    specs = []
    side_height = max(height - 2.0 * py, 0.0)
    if px > 0.0 and side_height:
        if "left" in selected:
            specs.append((x0, y0 + py, px, side_height))
        if "right" in selected:
            specs.append((x1 - px, y0 + py, px, side_height))
    if py > 0.0:
        if "bottom" in selected:
            specs.append((x0, y0, width, py))
        if "top" in selected:
            specs.append((x0, y1 - py, width, py))
    rectangles = tuple(Rectangle(spec[:2], *spec[2:], **style) for spec in specs)
    for rect in rectangles:
        ax.add_patch(rect)
    return rectangles


def _layout_slice(normal, position, *, design, origin) -> AxisSlice:
    """Describe a displayed design plane without quantizing it to the grid."""
    vertical, horizontal = _PLANE_AXES[normal]
    extents: dict[str, float] = {
        "x": float(design.width),
        "y": float(design.height),
        "z": float(design.depth or 0.0),
    }
    origin_by_axis: dict[str, float] = dict(zip(("x", "y", "z"), origin, strict=True))
    extent = (
        -float(origin_by_axis[horizontal]) / _UM,
        (extents[horizontal] - float(origin_by_axis[horizontal])) / _UM,
        -float(origin_by_axis[vertical]) / _UM,
        (extents[vertical] - float(origin_by_axis[vertical])) / _UM,
    )
    return AxisSlice(
        values=np.empty((0, 0)),
        normal=normal,
        vertical=vertical,
        horizontal=horizontal,
        index=0,
        position=float(position),
        extent=extent,
    )


def _structure_z_bounds(structure, design) -> tuple[float, float]:
    """Return the physical z extent used by the design rasterization bridge."""
    lower = float(getattr(structure, "z", 0.0) or 0.0)
    depth = float(getattr(structure, "depth", 0.0) or 0.0)
    if depth == 0.0:
        depth = float(design.depth)
    return lower, lower + depth


def _planar_structure_geometry(structure):
    """Build a Shapely geometry for a planar BeamZ structure when available."""
    vertices = _vertices_2d(getattr(structure, "vertices", ()))
    if len(vertices) < 3:
        return None
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient

    holes = [
        _vertices_2d(interior)
        for interior in getattr(structure, "interiors", ())
        if len(interior) >= 3
    ]
    geometry = Polygon(vertices, holes=holes)
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    return orient(geometry, sign=1.0) if not geometry.is_empty else None


def _polygon_geometries(geometry):
    """Yield the polygon leaves of a Shapely geometry collection."""
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
        return
    for child in getattr(geometry, "geoms", ()):
        yield from _polygon_geometries(child)


def _line_segments(geometry):
    """Yield non-zero x spans from a Shapely line intersection."""
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        coordinates = np.asarray(geometry.coords, dtype=float)
        if coordinates.shape[0] >= 2:
            low = float(np.min(coordinates[:, 0]))
            high = float(np.max(coordinates[:, 0]))
            if high > low:
                yield low, high
        return
    for child in getattr(geometry, "geoms", ()):
        yield from _line_segments(child)


def _path_patch(ax, geometry, *, normal, origin, facecolor):
    """Draw a Shapely polygon as an antialiased vector patch in a layout plane."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    vertical, horizontal = _PLANE_AXES[normal]
    origin_by_axis: dict[str, float] = dict(zip(("x", "y", "z"), origin, strict=True))
    axis_index: dict[str, int] = {"x": 0, "y": 1, "z": 2}
    h_index = axis_index[horizontal]
    v_index = axis_index[vertical]
    patches = []
    for polygon in _polygon_geometries(geometry):
        coords = []
        codes = []
        for ring in (polygon.exterior, *polygon.interiors):
            ring_coords = np.asarray(ring.coords[:-1], dtype=float)
            if ring_coords.shape[0] < 3:
                continue
            if normal == "z":
                transformed = np.column_stack(
                    (
                        (ring_coords[:, h_index] - origin_by_axis[horizontal]) / _UM,
                        (ring_coords[:, v_index] - origin_by_axis[vertical]) / _UM,
                    )
                )
            else:
                # xz sections arrive as x/z polygons, whose two coordinates are
                # already in the horizontal/vertical order for a y-normal view.
                transformed = np.column_stack(
                    (
                        (ring_coords[:, 0] - origin_by_axis[horizontal]) / _UM,
                        (ring_coords[:, 1] - origin_by_axis[vertical]) / _UM,
                    )
                )
            coords.extend(transformed)
            codes.extend(
                (Path.MOVETO, *(Path.LINETO,) * (len(transformed) - 1), Path.CLOSEPOLY)
            )
            coords.append(transformed[0])
        if not coords:
            continue
        patch = PathPatch(
            Path(np.asarray(coords), np.asarray(codes)),
            facecolor=facecolor,
            # Filled structures use painter's order. An edge stroke would leave
            # a light seam when a later structure overlaps an earlier one.
            edgecolor="none",
            linewidth=0.0,
            antialiased=True,
            joinstyle="round",
        )
        ax.add_patch(patch)
        patches.append(patch)
    return tuple(patches)


def _rectangle_geometry(x0, x1, y0, y1):
    from shapely.geometry import box

    return box(float(x0), float(y0), float(x1), float(y1))


def _draw_sphere_slice(ax, structure, *, normal, position, origin, facecolor):
    """Draw the exact circular section of a sphere, if it intersects the plane."""
    from matplotlib.patches import Circle

    center = np.asarray(structure.center, dtype=float)
    radius = float(structure.radius)
    fixed_axis = {"x": 0, "y": 1, "z": 2}[normal]
    squared_radius = radius**2 - (float(position) - center[fixed_axis]) ** 2
    if squared_radius < 0.0:
        return None
    radius = float(np.sqrt(squared_radius))
    if normal == "z":
        horizontal, vertical = "x", "y"
        coordinates = center[0], center[1]
    elif normal == "y":
        horizontal, vertical = "x", "z"
        coordinates = center[0], center[2]
    else:
        horizontal, vertical = "y", "z"
        coordinates = center[1], center[2]
    origin_by_axis = dict(zip(("x", "y", "z"), origin, strict=True))
    patch = Circle(
        (
            (coordinates[0] - origin_by_axis[horizontal]) / _UM,
            (coordinates[1] - origin_by_axis[vertical]) / _UM,
        ),
        radius / _UM,
        facecolor=facecolor,
        edgecolor="none",
        linewidth=0.0,
        antialiased=True,
    )
    ax.add_patch(patch)
    return patch


def _draw_structure_slice(
    ax,
    structure,
    *,
    normal,
    position,
    design,
    origin,
    facecolor,
):
    """Draw one faithful geometric slice of a public structure specification."""
    from beamz.design.structures import Box, Sphere

    if isinstance(structure, Sphere):
        return _draw_sphere_slice(
            ax,
            structure,
            normal=normal,
            position=position,
            origin=origin,
            facecolor=facecolor,
        )
    if isinstance(structure, Box):
        lower = structure.lower
        upper = structure.upper
        if normal == "z" and lower[2] <= position <= upper[2]:
            return _path_patch(
                ax,
                _rectangle_geometry(lower[0], upper[0], lower[1], upper[1]),
                normal=normal,
                origin=origin,
                facecolor=facecolor,
            )
        if normal == "y" and lower[1] <= position <= upper[1]:
            return _path_patch(
                ax,
                _rectangle_geometry(lower[0], upper[0], lower[2], upper[2]),
                normal=normal,
                origin=origin,
                facecolor=facecolor,
            )
        return ()

    geometry = _planar_structure_geometry(structure)
    if geometry is None:
        return ()
    z_min, z_max = _structure_z_bounds(structure, design)
    if normal == "z":
        if z_min <= position <= z_max:
            return _path_patch(
                ax, geometry, normal=normal, origin=origin, facecolor=facecolor
            )
        return ()
    if normal != "y" or not (z_min <= z_max):
        return ()

    from shapely.geometry import LineString

    x_min, _, x_max, _ = geometry.bounds
    line = LineString(((x_min, float(position)), (x_max, float(position))))
    patches = []
    for left, right in _line_segments(geometry.intersection(line)):
        patches.extend(
            _path_patch(
                ax,
                _rectangle_geometry(left, right, z_min, z_max),
                normal=normal,
                origin=origin,
                facecolor=facecolor,
            )
        )
    return tuple(patches)


def _draw_geometry_cross_section(ax, sim, *, normal, position, origin, colors):
    """Paint background and structures in painter's order without using the mesh."""
    design = sim.design
    if normal == "z":
        background = _rectangle_geometry(0.0, design.width, 0.0, design.height)
    else:
        background = _rectangle_geometry(0.0, design.width, 0.0, design.depth)
    _path_patch(
        ax,
        background,
        normal=normal,
        origin=origin,
        facecolor=_layout_color(design.background, colors),
    )
    for structure in design.structures:
        material = getattr(structure, "material", None)
        if material is None:
            continue
        if not design.is_3d:
            _path_patch(
                ax,
                _planar_structure_geometry(structure),
                normal=normal,
                origin=origin,
                facecolor=_layout_color(material, colors),
            )
            continue
        _draw_structure_slice(
            ax,
            structure,
            normal=normal,
            position=position,
            design=design,
            origin=origin,
            facecolor=_layout_color(material, colors),
        )


def overlay_simulation_devices(
    axes, sim, origin, *, source_markers=True, monitor_markers=True
):
    """Overlay projected sources and monitors on xy/xz slice axes."""
    for ax, normal in zip(axes, ("z", "y"), strict=True):
        _overlay_layout_devices(
            ax,
            sim,
            normal=normal,
            origin=origin,
            source_markers=source_markers,
            monitor_markers=monitor_markers,
        )


def _overlay_layout_devices(
    ax, sim, *, normal, origin, source_markers, monitor_markers
):
    """Use the same device colors and glyphs in 2D and 3D layouts."""
    groups = []
    if source_markers:
        groups.append((getattr(sim, "sources", ()), "#2ca02c", True))
    if monitor_markers:
        groups.append((getattr(sim, "monitors", ()), "#ff9800", False))
    for devices, color, source in groups:
        for device in devices:
            _draw_device_slice_overlay(
                ax, device, normal=normal, origin=origin, color=color, source=source
            )


def _overlay_layout_pml(ax, sim, *, normal, extent):
    for thickness, edges in _pml_section_layers(sim, normal):
        overlay_boundaries(
            ax,
            extent,
            thickness / _UM,
            edges=edges,
            facecolor="#666666",
            alpha=0.22,
            hatch="///",
            edgecolor="#666666",
            linewidth=0.0,
        )
