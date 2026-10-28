"""Axis-aligned slices and the shared field-image primitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from beamz.analysis._plot_canvas import (
    _AXIS_INDEX,
    _PLANE_AXES,
    _add_field_colorbar,
    _figure_axes,
    _index_from_position,
    _maybe_show,
)


@dataclass(frozen=True, slots=True)
class AxisSlice:
    """An axis-aligned 2D array plus its plotting coordinates."""

    values: np.ndarray
    normal: str
    vertical: str
    horizontal: str
    index: int
    position: float
    extent: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class FieldView:
    """Scalar display data derived from a scalar or complex field."""

    values: np.ndarray
    kind: str
    magnitude: bool
    power: bool


def extract_axis_aligned_slice(
    values,
    *,
    axis,
    step,
    position=None,
    index=None,
    origin=(0.0, 0.0, 0.0),
    display_position=False,
    lengths=None,
    unit=1.0,
) -> AxisSlice:
    """Extract a 3D array plane and attach consistent display coordinates."""
    array = np.asarray(values)
    normal = str(axis).lower()
    if array.ndim != 3 or normal not in _PLANE_AXES:
        raise ValueError("Axis-aligned slices require a 3D array and x, y, or z axis.")
    axis_index = _AXIS_INDEX[normal]
    origin_by_axis: dict[str, float] = dict(
        zip(("x", "y", "z"), map(float, origin), strict=True)
    )
    absolute_position = position
    if position is not None and display_position:
        absolute_position = float(position) + origin_by_axis[normal]
    if index is None:
        index = _index_from_position(
            absolute_position,
            step=step,
            count=array.shape[axis_index],
            default=array.shape[axis_index] // 2,
        )
    index = int(np.clip(index, 0, array.shape[axis_index] - 1))
    vertical, horizontal = _PLANE_AXES[normal]
    if lengths is None:
        lengths = {
            name: array.shape[_AXIS_INDEX[name]] * float(step)
            for name in ("x", "y", "z")
        }
    extent = (
        (0.0 - origin_by_axis[horizontal]) / unit,
        (float(lengths[horizontal]) - origin_by_axis[horizontal]) / unit,
        (0.0 - origin_by_axis[vertical]) / unit,
        (float(lengths[vertical]) - origin_by_axis[vertical]) / unit,
    )
    shown_position = (
        float(position)
        if position is not None and display_position
        else index * float(step) - origin_by_axis[normal]
    )
    return AxisSlice(
        values=np.take(array, index, axis=axis_index),
        normal=normal,
        vertical=vertical,
        horizontal=horizontal,
        index=index,
        position=shown_position,
        extent=extent,
    )


def _field_view(values, *, val="real", vector=False) -> FieldView:
    """Reduce scalar or vector complex data to one plottable scalar array."""
    key = str(val or "real").lower().replace(" ", "")
    power = key in {"abs^2", "abs2", "abs_sq", "abssq", "power", "intensity"}
    magnitude = key in {"abs", "magnitude"} or power
    arrays = [np.asarray(value) for value in values] if vector else [np.asarray(values)]
    if vector:
        if not magnitude:
            raise ValueError("Derived vector fields support val='abs' or val='abs^2'.")
        reduced = np.sum([np.abs(array) ** 2 for array in arrays], axis=0)
        return FieldView(reduced if power else np.sqrt(reduced), key, True, power)
    array = arrays[0]
    transforms = {
        "real": np.real,
        "re": np.real,
        "imag": np.imag,
        "imaginary": np.imag,
        "im": np.imag,
        "abs": np.abs,
        "magnitude": np.abs,
        "phase": np.angle,
    }
    if power:
        reduced = np.abs(array) ** 2
    elif key in transforms:
        reduced = transforms[key](array)
    else:
        raise ValueError(
            "val must be one of 'real', 'imag', 'abs', 'abs^2', or 'phase'."
        )
    return FieldView(np.asarray(reduced), key, magnitude, power)


def plot_field_view(
    ax,
    values,
    *,
    val=None,
    vector=False,
    extent=None,
    cmap: Any = "RdBu",
    vmin=None,
    vmax=None,
    percentile=None,
    aspect=None,
    interpolation="nearest",
    norm=None,
    x_edges=None,
    y_edges=None,
):
    """Plot scalar or complex field data through the shared image primitive."""
    view = (
        values
        if isinstance(values, FieldView)
        else (
            _field_view(values, val="real" if val is None else val, vector=vector)
            if val is not None or vector or np.iscomplexobj(values)
            else FieldView(np.asarray(values), "scalar", False, False)
        )
    )
    array = view.values
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if percentile is not None and vmax is None:
        scale = float(np.nanpercentile(np.abs(array), float(percentile)))
        vmax = scale if np.isfinite(scale) and scale > 0.0 else 1.0
    if vmax is not None and vmin is None:
        vmin = 0.0 if view.magnitude else -float(vmax)
    if (x_edges is None) != (y_edges is None):
        raise ValueError("x_edges and y_edges must be provided together.")
    if x_edges is not None and y_edges is not None:
        x_edges = np.asarray(x_edges, dtype=float)
        y_edges = np.asarray(y_edges, dtype=float)
        if x_edges.shape != (array.shape[1] + 1,) or y_edges.shape != (
            array.shape[0] + 1,
        ):
            raise ValueError(
                "Rectilinear plot edges must have one more entry than their "
                f"field axis; got field {array.shape}, x {x_edges.shape}, "
                f"and y {y_edges.shape}."
            )
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            array,
            shading="flat",
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_aspect(aspect or ("auto" if array.shape[0] == 1 else "equal"))
    else:
        image = ax.imshow(
            array,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            aspect=aspect or ("auto" if array.shape[0] == 1 else "equal"),
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
        )
    return image, view


def _coord_extent_um(da, *, sim=None):
    offset = (
        (0.0, 0.0, 0.0)
        if da.attrs.get("coordinate_frame") == "public"
        else getattr(sim, "coordinate_offset", (0.0, 0.0, 0.0))
    )
    offset_by_dim = {"x": offset[0], "y": offset[1], "z": offset[2]}
    dims = [dim for dim in da.dims if dim in {"x", "y", "z", "s"}]
    if len(dims) < 2:
        width = max(int(da.shape[-1]) - 1, 1)
        height = max(int(da.shape[-2]) - 1, 1) if da.ndim > 1 else 1
        return (0.0, float(width), 0.0, float(height)), ("sample", "sample")
    y_dim, x_dim = dims[-2], dims[-1]

    def axis_extent(dim):
        coord = np.asarray(da.coords[dim], dtype=float)
        if coord.size == 0:
            return (0.0, 1.0)
        values = (coord - float(offset_by_dim.get(dim, 0.0))) / 1e-6
        if values.size == 1:
            return (float(values[0] - 0.5), float(values[0] + 0.5))
        step = float(np.nanmedian(np.diff(values)))
        return (float(values[0] - 0.5 * step), float(values[-1] + 0.5 * step))

    x0, x1 = axis_extent(x_dim)
    y0, y1 = axis_extent(y_dim)
    return (x0, x1, y0, y1), (x_dim, y_dim)


def _plot_image(
    values,
    *,
    ax=None,
    figsize=None,
    extent=None,
    xlabel="x (um)",
    ylabel="y (um)",
    title=None,
    cmap="RdBu",
    colorbar=True,
    show=True,
    vmin=None,
    vmax=None,
    xlim=None,
    ylim=None,
    interpolation="nearest",
):
    fig, ax = _figure_axes(ax, figsize=(6.0, 4.0) if figsize is None else figsize)
    im, _view = plot_field_view(
        ax,
        values,
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if colorbar:
        _add_field_colorbar(fig, ax, im)
    fig.tight_layout(pad=0.5)
    _maybe_show(fig, show=show)
    return fig, ax


def _select_saved_field(da, *, frame=None, z=None, y=None):
    selected = da
    if "t" in selected.dims:
        selected = selected.isel(t=-1 if frame is None else int(frame))
    elif "frame" in selected.dims:
        selected = selected.isel(frame=-1 if frame is None else int(frame))
    if "z" in selected.dims:
        if z is None:
            selected = selected.isel(z=selected.sizes["z"] // 2)
        else:
            selected = selected.sel(z=float(z), method="nearest")
    if selected.ndim > 2 and "y" in selected.dims and y is not None:
        selected = selected.sel(y=float(y), method="nearest")
    while selected.ndim > 2:
        selected = selected.isel(
            {selected.dims[0]: selected.sizes[selected.dims[0]] // 2}
        )
    return selected
