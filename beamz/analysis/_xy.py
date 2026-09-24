"""Small geometry adapters for XY's pyplot compatibility boundary."""

import numpy as np


def add_patch(ax, patch):
    """Render a Matplotlib patch in XY, retaining polygon holes and hatching."""
    from matplotlib.patches import PathPatch
    from shapely.geometry import Polygon
    from shapely.ops import triangulate

    paths = patch.get_path().transformed(patch.get_patch_transform()).to_polygons()
    face = patch.get_facecolor()
    if isinstance(patch, PathPatch) and len(paths) > 1 and face[3] > 0:
        # XY's add_patch deliberately skips fills for nested rings. Splitting
        # the filled region into simple polygons keeps holes transparent.
        geometry = Polygon(paths[0], paths[1:])
        for triangle in triangulate(geometry):
            clipped = triangle.intersection(geometry)
            pieces = getattr(clipped, "geoms", (clipped,))
            for piece in pieces:
                if isinstance(piece, Polygon) and piece.area > 0:
                    points = np.asarray(piece.exterior.coords)
                    ax.fill(points[:, 0], points[:, 1], color=face, linewidth=0)
        from copy import copy

        patch = copy(patch)
        patch.set_facecolor("none")
    artist = ax.add_patch(patch)
    if patch.get_hatch() and paths:
        from matplotlib.collections import LineCollection
        from shapely.geometry import LineString

        geometry = Polygon(paths[0], paths[1:])
        x0, y0, x1, y1 = geometry.bounds
        # Match the diagonal direction and approximate screen density of ///.
        span = max(x1 - x0, y1 - y0)
        if span > 0:
            segments = []
            for intercept in np.arange(-(x1 - x0), y1 - y0, span / 45):
                line = LineString(
                    [(x0, y0 + intercept), (x1, y0 + intercept + x1 - x0)]
                )
                clipped = line.intersection(geometry)
                for piece in getattr(clipped, "geoms", (clipped,)):
                    if piece.geom_type == "LineString" and not piece.is_empty:
                        segments.append(np.asarray(piece.coords))
            ax.add_collection(
                LineCollection(segments, colors=[patch.get_edgecolor()], linewidths=0.5)
            )
    return artist


def field_colormap(values, cmap, norm, vmin, vmax):
    """Keep custom/cyclic Matplotlib color tables via XY's discrete-map path."""
    from matplotlib import colormaps
    from matplotlib.colors import BoundaryNorm, Normalize

    if isinstance(cmap, str) and cmap not in ("twilight", "twilight_shifted"):
        return cmap, norm, vmin, vmax
    color_table = colormaps[cmap] if isinstance(cmap, str) else cmap.copy()
    if norm is None:
        scale = Normalize(vmin=vmin, vmax=vmax)
        scale.autoscale_None(np.ma.masked_invalid(values))
        assert scale.vmin is not None and scale.vmax is not None
        lo, hi = float(scale.vmin), float(scale.vmax)
        if lo == hi:
            hi = lo + max(abs(lo), 1.0) * 1e-12
        norm = BoundaryNorm(np.linspace(lo, hi, color_table.N + 1), color_table.N)
    elif not isinstance(norm, BoundaryNorm):
        raise ValueError("XY custom colormaps require a BoundaryNorm or linear limits.")
    # XY validates the map name before consulting BoundaryNorm's exact RGBA
    # table. A recognized carrier name unlocks that path; its colors are unused.
    color_table.name = "viridis"
    return color_table, norm, None, None
