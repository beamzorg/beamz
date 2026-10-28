"""Figure helpers and shared coordinates for BeamZ plots."""

from __future__ import annotations

import numpy as np

from beamz._helpers import get_si_scale_and_label

_UM = 1e-6
_AXIS_INDEX: dict[str, int] = {"z": 0, "y": 1, "x": 2}
_PLANE_AXES: dict[str, tuple[str, str]] = {
    "x": ("z", "y"),
    "y": ("z", "x"),
    "z": ("y", "x"),
}
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
