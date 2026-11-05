"""Figure helpers and shared coordinates for BeamZ plots."""

from __future__ import annotations

import numpy as np

from beamz._helpers import get_si_scale_and_label
from beamz.analysis._coordinates import (
    _PLANE_AXES as _PLANE_AXES,
)
from beamz.analysis._coordinates import (
    _STORAGE_AXIS,
    _XYZ_INDEX,
)

_UM = 1e-6
_AXIS_INDEX = _STORAGE_AXIS
_DESIGN_EXTENT: dict[str, str] = {
    "x": "width",
    "y": "height",
    "z": "depth",
}


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _figure_axes(ax, *, figsize):
    if ax is not None:
        return ax.figure, ax
    return _pyplot().subplots(figsize=figsize)


def _maybe_show(fig, *, show):
    if show:
        _pyplot().show()
    return fig


def _axis_labels(ax, *, width, height, x_label="x", y_label="y"):
    scale, unit = get_si_scale_and_label(max(float(width), float(height), 1e-30))
    ax.set_xlabel(f"{x_label} ({unit})")
    ax.set_ylabel(f"{y_label} ({unit})")
    ax.xaxis.set_major_formatter(lambda value, _pos: f"{value * scale:g}")
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value * scale:g}")


def _index_from_position(position, *, step, count, default):
    if position is None:
        return int(default)
    idx = int(round(float(position) / max(float(step), 1e-30)))
    return int(np.clip(idx, 0, int(count) - 1))


def _origin_map(origin) -> dict[str, float]:
    return dict(zip("xyz", map(float, origin), strict=True))


def _display_extent(origins, horizontal, vertical, lengths, *, unit=1.0):
    def span(axis):
        origin = float(origins[axis])
        return -origin / unit, (float(lengths[axis]) - origin) / unit

    return (*span(horizontal), *span(vertical))


def _region_edges_um(grid, region, axis, origin):
    interval = region.axis_interval(axis)
    if interval is None:
        return None
    edges = np.asarray(grid.axis_edges(axis), dtype=float)
    return (edges[interval.start : interval.stop + 1] - float(origin)) / _UM


def _plane_edges_um(grid, region, vertical, horizontal, origins):
    """Return extent, grid edges, and plot edges when spacing is nonuniform."""
    x_edges = _region_edges_um(grid, region, horizontal, origins[horizontal])
    y_edges = _region_edges_um(grid, region, vertical, origins[vertical])
    if x_edges is None or y_edges is None:
        return None
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    uniform = _edges_are_uniform(x_edges) and _edges_are_uniform(y_edges)
    plot_x, plot_y = (None, None) if uniform else (x_edges, y_edges)
    return extent, x_edges, y_edges, plot_x, plot_y


def _shifted_grid_edges(grid, origin, axes):
    return {
        axis: np.asarray(grid.axis_edges(axis))
        - grid.origin[_XYZ_INDEX[axis]]
        - origin[_XYZ_INDEX[axis]]
        for axis in axes
    }


def _simulation_origin(sim):
    offset = getattr(sim, "coordinate_offset", None)
    if offset is not None:
        values = tuple(float(v) for v in offset)
        if len(values) == 3:
            return values
    return (0.0, 0.0, 0.0)


def _field_display_scale_and_units(field):
    family = str(field)[:1].upper()
    if family == "E":
        return _UM, "V/um"
    if family == "H":
        return _UM, "A/um"
    return 1.0, ""


def _coordinate_extent(coords0, coords1, *, origin0, origin1, fallback_step):
    def bounds(coords, origin):
        coords = np.asarray(coords, dtype=float)
        step = (
            float(np.mean(np.diff(coords))) if coords.size > 1 else float(fallback_step)
        )
        return (
            (float(coords[0]) - 0.5 * step - float(origin)) / _UM,
            (float(coords[-1]) + 0.5 * step - float(origin)) / _UM,
        )

    x0, x1 = bounds(coords1, origin1)
    y0, y1 = bounds(coords0, origin0)
    return x0, x1, y0, y1


def _edges_are_uniform(edges) -> bool:
    edges = np.asarray(edges, dtype=float)
    widths = np.diff(edges)
    if widths.size < 2:
        return True
    scale = max(1.0, float(np.max(np.abs(edges), initial=0.0)))
    return bool(
        np.allclose(
            widths,
            widths[0],
            rtol=1e-12,
            atol=8.0 * np.finfo(float).eps * scale,
        )
    )


def _field_colorbar_label(
    field, view, *, display_field=None, units="", show_units=True
):
    label = display_field or field
    if view.power:
        text = f"|{label}|" + r"$^2$"
    elif view.magnitude:
        text = f"|{label}|"
    elif view.kind in {"imag", "imaginary", "im"}:
        text = f"Im({label})"
    elif view.kind == "phase":
        text = f"phase({label})"
    else:
        text = f"Re({label})"
    if show_units and units:
        text += f" ({f'({units})^2' if view.power else units})"
    return text


def _add_field_colorbar(fig, ax, image, **kwargs):
    """Add a compact colorbar whose orientation follows the visible field extent."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_span = abs(x1 - x0)
    y_span = abs(y1 - y0)
    if y_span >= x_span:
        return fig.colorbar(
            image,
            ax=ax,
            location="right",
            fraction=0.05,
            pad=0.04,
            aspect=30,
            **kwargs,
        )
    return fig.colorbar(
        image,
        ax=ax,
        location="bottom",
        fraction=0.075,
        # Keep the colorbar clear of the x-axis label after tight_layout().
        pad=0.2,
        aspect=35,
        **kwargs,
    )
