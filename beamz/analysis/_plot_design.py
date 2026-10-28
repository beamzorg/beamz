"""Two-dimensional design layouts with sources and monitors."""

from __future__ import annotations

import colorsys
import hashlib

import numpy as np

from beamz.analysis._plot_canvas import _axis_labels, _figure_axes, _maybe_show
from beamz.devices.visualization import visual_spec_from_device


def _material_color(structure, index):
    explicit = getattr(structure, "color", None)
    if explicit:
        return explicit
    material = getattr(structure, "material", None)
    key = repr(
        (
            getattr(material, "permittivity", None),
            getattr(material, "permeability", None),
            getattr(material, "conductivity", None),
            index,
        )
    )
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    hue = (seed % 1000) / 1000.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.45, 0.78)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _vertices_2d(vertices):
    return [(float(v[0]), float(v[1])) for v in vertices or ()]


def _draw_polygon(ax, structure, *, index=0, fill=True, alpha=None, edgecolor="black"):
    vertices = _vertices_2d(getattr(structure, "vertices", ()))
    if len(vertices) < 3:
        return None

    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    coords = []
    codes = []
    for path in [
        vertices,
        *[_vertices_2d(p) for p in getattr(structure, "interiors", ())],
    ]:
        if len(path) < 3:
            continue
        coords.append(path[0])
        codes.append(Path.MOVETO)
        coords.extend(path[1:])
        codes.extend([Path.LINETO] * (len(path) - 1))
        coords.append(path[0])
        codes.append(Path.CLOSEPOLY)

    facecolor = _material_color(structure, index) if fill else "none"
    patch = PathPatch(
        Path(np.asarray(coords), np.asarray(codes)),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
        alpha=0.22
        if alpha is None and index == 0
        else (0.65 if alpha is None else alpha),
    )
    ax.add_patch(patch)
    return patch


def draw_geometry(ax, design, *, sources=(), monitors=()):
    for index, structure in enumerate(getattr(design, "structures", ())):
        _draw_polygon(ax, structure, index=index)
    for source in sources or ():
        _draw_source(ax, source)
    for monitor in monitors or ():
        _draw_monitor(ax, monitor)


def _draw_source(ax, source):
    spec = visual_spec_from_device(source)
    if spec is None:
        return None
    center = tuple(float(v) for v in spec.center[:2])
    direction = str(spec.direction or "").lower()
    size = np.asarray(spec.size, dtype=float).reshape(-1)
    width = float(np.max(np.abs(size[:2]))) if size.size else 0.25e-6
    color = "crimson"
    if direction in {"+x", "-x"}:
        x = [center[0], center[0]]
        y = [center[1] - 0.5 * width, center[1] + 0.5 * width]
    elif direction in {"+y", "-y"}:
        x = [center[0] - 0.5 * width, center[0] + 0.5 * width]
        y = [center[1], center[1]]
    else:
        from matplotlib.patches import Circle

        circle = Circle(
            (center[0], center[1]),
            radius=0.5 * width,
            fill=False,
            ec=color,
            lw=1.5,
        )
        ax.add_patch(circle)
        return circle
    (line,) = ax.plot(x, y, color=color, linewidth=2.0, solid_capstyle="round")
    return line


def _monitor_rect(spec):
    center = np.asarray(spec.center, dtype=float).reshape(-1)
    size = np.asarray(spec.size, dtype=float).reshape(-1)
    if center.size < 2 or size.size < 2:
        return None
    return (
        float(center[0] - 0.5 * abs(size[0])),
        float(center[1] - 0.5 * abs(size[1])),
        float(abs(size[0])),
        float(abs(size[1])),
    )


def _draw_monitor(ax, monitor):
    spec = visual_spec_from_device(monitor)
    if spec is None:
        return None
    if spec.kind == "monitor-line":
        style = spec.style or {}
        start = style.get("start")
        end = style.get("end")
        if start is None or end is None:
            return None
        (line,) = ax.plot(
            [float(start[0]), float(end[0])],
            [float(start[1]), float(end[1])],
            color="navy",
            linewidth=2.0,
        )
        return line

    rect = _monitor_rect(spec)
    if rect is None:
        return None
    from matplotlib.patches import Rectangle

    patch = Rectangle(
        rect[:2],
        rect[2],
        rect[3],
        facecolor="none",
        edgecolor="navy",
        linestyle="--",
        linewidth=1.5,
    )
    ax.add_patch(patch)
    return patch


def _draw_boundaries(ax, sim):
    from matplotlib.patches import Rectangle

    design = sim.design
    width, height = float(design.width), float(design.height)
    for boundary in getattr(sim, "boundaries", ()) or ():
        thickness = float(getattr(boundary, "thickness", 0.0) or 0.0)
        if thickness <= 0:
            continue
        edges = getattr(boundary, "_get_edges_for_dimensionality", lambda _is_3d: ())(
            bool(getattr(design, "is_3d", False))
        )
        rectangles = {
            "left": (0.0, 0.0, thickness, height),
            "right": (width - thickness, 0.0, thickness, height),
            "bottom": (0.0, 0.0, width, thickness),
            "top": (0.0, height - thickness, width, thickness),
        }
        for edge in edges:
            if edge not in rectangles:
                continue
            ax.add_patch(
                Rectangle(
                    rectangles[edge][:2],
                    *rectangles[edge][2:],
                    facecolor="none",
                    edgecolor="red",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.6,
                )
            )


def plot_design(
    design,
    *,
    sources=None,
    monitors=None,
    ax=None,
    figsize=None,
    show=True,
    title="Design Layout",
):
    """Plot a design layout and optional source/monitor overlays."""
    if figsize is None:
        width = max(float(design.width), 1e-30)
        figsize = (6.0, max(2.0, 6.0 * float(design.height) / width))
    fig, ax = _figure_axes(ax, figsize=figsize)
    draw_geometry(ax, design, sources=sources or (), monitors=monitors or ())
    ax.set_xlim(0.0, float(design.width))
    ax.set_ylim(0.0, float(design.height))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    _axis_labels(ax, width=design.width, height=design.height)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax
