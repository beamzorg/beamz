"""
Mesh-building logic for the 3D preview, factored out from the Qt widget
(`preview_3d.py`) so it can be unit-tested without an interactive VTK
render context.

Note on why this split exists: `pv.Plotter(off_screen=True)` renders fine
in a headless/no-GPU environment, but the *interactive* `QtInteractor`
widget the dock uses needs a working OpenGL context and does not degrade
gracefully without one — it fails silently (blank/garbage framebuffer)
rather than raising. So mesh CONSTRUCTION is verified here, headlessly;
actual on-screen rendering can only be confirmed on a machine with a real
GPU/display.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pyvista as pv

from .model import SceneObject

# Structures render solid; sources/monitors render as translucent planes
# so they read as annotations rather than physical material.
DEFAULT_STRUCTURE_COLOR = "#6699cc"
SOURCE_COLOR = "#e67e22"
MONITOR_COLOR = "#2980b9"
REGION_COLOR = "#f1c40f"  # bright yellow — deliberately the most eye-
# catching color in the scene, so the simulation domain is impossible to
# miss (a direct ask: "add a box highlight for simulation region such
# that its easy to notice for users").


def structure_mesh(so: SceneObject) -> Optional["pv.PolyData"]:
    """Extrude a structure's 2D outline (`obj.vertices`, at `obj.z` for
    `obj.depth`) into a solid. Uses the same generic vertices-based
    approach as the 2D canvas — works uniformly across every structure
    kind, including a rotated one (beamz represents that as a `Polygon`
    with different vertices, not a transformed Rectangle; see model.py).

    Known v1 limitation: holes (`obj.interiors`, e.g. a Ring's bore) are
    NOT cut in 3D — `delaunay_2d` triangulates the outer boundary only, so
    a Ring currently renders as a solid disc extrusion rather than a true
    annulus. The 2D canvas does cut holes correctly (via QPainterPath's
    odd-even fill); matching that in 3D needs constrained
    triangulation-with-holes, which is a reasonable follow-up rather than
    something to rush here.
    """
    obj = so.obj
    vertices = getattr(obj, "vertices", None)
    if not vertices or len(vertices) < 3:
        return None

    depth = getattr(obj, "depth", 0.0) or 0.0
    z0 = getattr(obj, "z", 0.0) or 0.0

    pts = np.array([[v[0], v[1], z0] for v in vertices], dtype=np.float32)
    poly = pv.PolyData(pts)
    try:
        surface = poly.delaunay_2d()
    except Exception:
        return None
    if surface.n_points == 0:
        return None

    if depth <= 0:
        return surface
    return surface.extrude((0, 0, depth), capping=True)


def plane_mesh(so: SceneObject) -> "pv.PolyData":
    """A flat quad for a source/monitor, spanning `center` +/- `size`/2.
    Whichever axis has ~zero size collapses to a point, so the quad
    naturally lies in the right plane (XY/XZ/YZ) without needing to know
    which axis that is ahead of time.
    """
    center = so.recipe.get("center", (0.0, 0.0, 0.0))
    size = so.recipe.get("size", (0.0, 0.0, 0.0))
    cx, cy, cz = center
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2

    corners = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float32,
    )
    poly = pv.PolyData(corners)
    poly.faces = np.hstack([[4], [0, 1, 2, 3]])
    return poly


def region_box_mesh(so: SceneObject) -> "pv.PolyData":
    """A wireframe box outlining the FDTD simulation region — rendered as
    a plain pv.Box (its own center+size, no corner/rotation concerns; the
    region is a GUI-only concept, always center-based — see
    model.REGION_KIND). Drawn wireframe-only (no fill) so it never
    obscures the actual structures/sources/monitors inside it.
    """
    r = so.recipe
    cx, cy, cz = r.get("x", 0.0), r.get("y", 0.0), r.get("z", 0.0)
    sx, sy, sz = r.get("x_span", 0.0), r.get("y_span", 0.0), r.get("z_span", 0.0)
    bounds = (cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, cz - sz / 2, cz + sz / 2)
    return pv.Box(bounds=bounds)


def mesh_and_style_for(so: SceneObject) -> Optional[tuple[Any, dict]]:
    """Single entry point the widget calls: returns (mesh, add_mesh
    kwargs) for a SceneObject, or None if it can't be rendered in v1
    (e.g. a Box/Sphere structure with no planar `vertices`).
    """
    if so.category == "structure":
        mesh = structure_mesh(so)
        if mesh is None:
            return None
        color = getattr(so.obj, "color", None) or DEFAULT_STRUCTURE_COLOR
        return mesh, dict(color=color, opacity=0.9, pickable=True)

    if so.category == "source":
        return plane_mesh(so), dict(color=SOURCE_COLOR, opacity=0.45, pickable=True)

    if so.category == "monitor":
        return plane_mesh(so), dict(
            color=MONITOR_COLOR, opacity=0.35, style="wireframe", line_width=2, pickable=True
        )

    if so.category == "region":
        return region_box_mesh(so), dict(
            color=REGION_COLOR, opacity=1.0, style="wireframe", line_width=3, pickable=False
        )

    return None