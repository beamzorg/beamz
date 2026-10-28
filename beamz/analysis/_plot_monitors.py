"""Frequency-domain, recorded, and mode-field figures."""

from __future__ import annotations

import numpy as np

from beamz.analysis._coordinates import monitor_plane_coordinates_3d
from beamz.analysis._plot_canvas import (
    _UM,
    _add_field_colorbar,
    _coordinate_extent,
    _edges_are_uniform,
    _field_colorbar_label,
    _field_display_scale_and_units,
    _figure_axes,
    _maybe_show,
    _simulation_origin,
)
from beamz.analysis._plot_fields import (
    _coord_extent_um,
    _field_view,
    _plot_image,
    _select_saved_field,
    plot_field_view,
)
from beamz.analysis._plot_materials import _overlay_material_slice
from beamz.analysis.data import AnalysisData, analysis_inputs
from beamz.simulation.observe import source_field_amplitude_normalization


def _monitor_frequency_index(data, frequency):
    freqs = np.asarray(data.frequencies, dtype=float)
    if freqs.size == 0:
        raise ValueError(f"Monitor {data.name!r} has no DFT data.")
    if frequency is None:
        return 0, freqs
    return int(np.argmin(np.abs(freqs - float(frequency)))), freqs


def _available_dft_component(data, component):
    try:
        return np.asarray(data.field(component), dtype=np.complex128)
    except ValueError:
        return None


def _monitor_dft_values(data, field_name, *, frequency=None, val="real"):
    field_name = str(field_name or "Ez")
    idx, freqs = _monitor_frequency_index(data, frequency)
    if field_name in {"E", "H"}:
        arrays = [
            comp_values
            for comp in (f"{field_name}{axis}" for axis in "xyz")
            if (comp_values := _available_dft_component(data, comp)) is not None
        ]
        if not arrays:
            raise ValueError(f"No DFT components found for field {field_name!r}.")
        vector_val = (
            val if str(val).lower() in {"abs^2", "power", "intensity"} else "abs"
        )
        return _field_view(
            [array[idx] for array in arrays], val=vector_val, vector=True
        ).values, freqs[idx]

    values = _available_dft_component(data, field_name)
    if values is None:
        raise ValueError(f"No DFT component found for field {field_name!r}.")
    return _field_view(values[idx], val=val).values, freqs[idx]


def _monitor_plane_shape_and_coords(simulation, monitor, *, region=None):
    plane_shape = getattr(monitor, "_compiled_dft_shape_3d", None)
    region, coords0, coords1 = monitor_plane_coordinates_3d(
        simulation, monitor, region=region
    )
    if plane_shape is None:
        plane_shape = (int(coords0.size), int(coords1.size))
    else:
        plane_shape = tuple(int(v) for v in plane_shape)
        coords0 = coords0[: plane_shape[0]]
        coords1 = coords1[: plane_shape[1]]
    return plane_shape, coords0, coords1, region


def _reshape_dft_plane(values, f_idx, plane_shape):
    shape = tuple(int(v) for v in plane_shape)
    samples = np.asarray(values[f_idx], dtype=np.complex128)
    expected = int(np.prod(shape))
    if samples.size != expected:
        raise ValueError(
            f"Recorded DFT plane has {samples.size} samples, but its exact "
            f"rectilinear coordinates require shape {shape} ({expected} samples)."
        )
    return samples.reshape(shape)


def _dft_plane_components(data, field, frequency_index, plane_shape):
    """Return one or more reshaped component planes for a DFT field request."""
    field = str(field)
    if field not in {"E", "H"}:
        values = np.asarray(data.field(field), dtype=np.complex128)
        return [_reshape_dft_plane(values, frequency_index, plane_shape)], False
    names = tuple(f"{field}{axis}" for axis in "xyz")
    components = [
        _reshape_dft_plane(values, frequency_index, plane_shape)
        for name in names
        if (values := _available_dft_component(data, name)) is not None
    ]
    if not components:
        raise ValueError(
            f"No DFT data recorded for derived field {field!r}. "
            f"Expected at least one of: {', '.join(names)}."
        )
    return components, True


def _normalize_dft_components(components, normalization, frequency_index, count):
    if normalization is None:
        return components
    try:
        normalization = source_field_amplitude_normalization(normalization)
    except Exception:
        normalization = np.asarray(normalization, dtype=np.complex128).reshape(-1)
    if (
        normalization is None
        or normalization.size != count
        or abs(normalization[frequency_index]) <= 1e-12
    ):
        return components
    return [value / normalization[frequency_index] for value in components]


def _dft_frequency(data, frequency, frequency_index):
    freqs = np.asarray(data.frequencies, dtype=float)
    selected_frequency = (
        freqs[int(np.clip(frequency_index, 0, max(freqs.size - 1, 0)))]
        if frequency is None and freqs.size
        else frequency
    )
    return _monitor_frequency_index(data, selected_frequency)


def _dft_plane_extent(
    coordinate_context,
    sample_region,
    *,
    axis0,
    axis1,
    origin_by_axis,
    target0,
    target1,
    resolution,
):
    """Return the image extent and, when needed, rectilinear plot and grid edges."""
    plot_x_edges = None
    plot_y_edges = None
    grid_x_edges = None
    grid_y_edges = None
    grid = getattr(coordinate_context, "grid", None)
    if grid is None:
        extent = _coordinate_extent(
            target0,
            target1,
            origin0=origin_by_axis[axis0],
            origin1=origin_by_axis[axis1],
            fallback_step=resolution,
        )
        return extent, plot_x_edges, plot_y_edges, grid_x_edges, grid_y_edges
    interval0 = sample_region.axis_interval(axis0)
    interval1 = sample_region.axis_interval(axis1)
    if interval0 is None or interval1 is None:
        raise ValueError("A 3D DFT monitor is missing a tangential grid interval.")
    edges0 = grid.axis_edges(axis0)
    edges1 = grid.axis_edges(axis1)
    x_edges = (
        edges1[interval1.start : interval1.stop + 1] - origin_by_axis[axis1]
    ) / _UM
    y_edges = (
        edges0[interval0.start : interval0.stop + 1] - origin_by_axis[axis0]
    ) / _UM
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    if not _edges_are_uniform(x_edges) or not _edges_are_uniform(y_edges):
        plot_x_edges, plot_y_edges = x_edges, y_edges
    return extent, plot_x_edges, plot_y_edges, x_edges, y_edges


def _draw_cell_edges(ax, x_edges, y_edges, *, color, linewidth, alpha):
    ax.vlines(
        x_edges,
        y_edges[0],
        y_edges[-1],
        colors=color,
        linewidths=float(linewidth),
        alpha=float(alpha),
    )
    ax.hlines(
        y_edges,
        x_edges[0],
        x_edges[-1],
        colors=color,
        linewidths=float(linewidth),
        alpha=float(alpha),
    )


def plot_dft_field(
    simulation,
    monitor=None,
    *,
    field="Ey",
    display_field=None,
    frequency=None,
    frequency_index=0,
    val="real",
    source_normalization=None,
    source_normalize=True,
    sample_region=None,
    origin=None,
    percentile=99.5,
    vmin=None,
    vmax=None,
    ax=None,
    figsize=(6, 4),
    cmap="RdBu",
    colorbar=True,
    overlay_core=True,
    eps_alpha=0.2,
    core_permittivity=None,
    xlim=None,
    ylim=None,
    show_grid=False,
    grid_color="black",
    grid_linewidth=0.2,
    grid_alpha=0.18,
    show_units=True,
    show=True,
):
    """Plot a frequency-domain scalar or vector field on a monitor plane.

    Pass ``show_grid=True`` to draw the physical cell edges; grid lines are
    hidden by default.
    """
    if not isinstance(simulation, AnalysisData):
        raise TypeError("plot_dft_field requires AnalysisData.")
    data = simulation
    monitor = data.monitor_geometry
    if monitor is None:
        raise ValueError("A monitor geometry is required to plot a DFT field.")
    if not getattr(monitor, "is_3d", False):
        raise ValueError("plot_dft_field expects a 3D plane monitor.")
    axis = str(getattr(monitor, "plane_normal", "z")).lower()
    plane_axes = {"x": ("z", "y"), "y": ("z", "x"), "z": ("y", "x")}
    if axis not in plane_axes:
        raise ValueError("DFT field monitor plane_normal must be one of x, y, or z.")
    axis0, axis1 = plane_axes[axis]

    f_idx, freqs = _dft_frequency(data, frequency, frequency_index)

    field_key = str(field)
    coordinate_context = data.coordinates if data is not None else simulation
    plane_shape, target0, target1, sample_region = _monitor_plane_shape_and_coords(
        coordinate_context, monitor, region=sample_region
    )
    components, vector = _dft_plane_components(data, field_key, f_idx, plane_shape)

    if source_normalize:
        components = _normalize_dft_components(
            components, source_normalization, f_idx, freqs.size
        )

    field_scale, field_units = _field_display_scale_and_units(field_key)
    components = [component * field_scale for component in components]
    view = _field_view(components if vector else components[0], val=val, vector=vector)

    if origin is None:
        origin = _simulation_origin(coordinate_context)
    ox, oy, oz = (float(v) for v in origin)
    origin_by_axis = {"x": ox, "y": oy, "z": oz}
    extent, plot_x_edges, plot_y_edges, grid_x_edges, grid_y_edges = _dft_plane_extent(
        coordinate_context,
        sample_region,
        axis0=axis0,
        axis1=axis1,
        origin_by_axis=origin_by_axis,
        target0=target0,
        target1=target1,
        resolution=simulation.resolution,
    )
    fig, ax = _figure_axes(ax, figsize=figsize)
    im, _ = plot_field_view(
        ax,
        view,
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        percentile=percentile,
        aspect="equal",
        x_edges=plot_x_edges,
        y_edges=plot_y_edges,
    )

    if overlay_core:
        _overlay_material_slice(
            ax,
            simulation,
            normal=axis,
            position=float(getattr(monitor, "plane_position", 0.0)),
            origin=origin,
            reverse=view.magnitude,
            alpha=eps_alpha,
            core_permittivity=core_permittivity,
            sample_region=sample_region if plot_x_edges is not None else None,
        )

    if bool(show_grid) and grid_x_edges is not None and grid_y_edges is not None:
        _draw_cell_edges(
            ax,
            grid_x_edges,
            grid_y_edges,
            color=grid_color,
            linewidth=grid_linewidth,
            alpha=grid_alpha,
        )

    ax.set_xlabel(f"{axis1} (um)")
    ax.set_ylabel(f"{axis0} (um)")
    plane_value = float(getattr(monitor, "plane_position", 0.0)) - origin_by_axis[axis]
    ax.set_title(f"cross section at {axis}={plane_value / _UM:.2f} (um)")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if colorbar:
        _add_field_colorbar(
            fig,
            ax,
            im,
            label=_field_colorbar_label(
                field_key,
                view,
                display_field=display_field,
                units=field_units,
                show_units=show_units,
            ),
            extend="max" if view.magnitude else "both",
        )
    fig.tight_layout(pad=0.5)
    _maybe_show(fig, show=show)
    return fig, ax


def _monitor_extent_um(monitor, values, *, sim=None):
    arr = np.asarray(values)
    if arr.ndim == 1:
        length = max(arr.size - 1, 1)
        return (0.0, float(length), 0.0, 1.0), "sample", "sample"
    start = getattr(monitor, "start", None)
    size = getattr(monitor, "size", None)
    if start is not None and size is not None:
        start = np.asarray(start, dtype=float).reshape(-1)
        size = np.asarray(size, dtype=float).reshape(-1)
        offset = np.asarray(
            getattr(sim, "coordinate_offset", (0.0, 0.0, 0.0)), dtype=float
        ).reshape(-1)
        if start.size >= 2 and size.size >= 2:
            return (
                (
                    float((start[0] - offset[0]) / 1e-6),
                    float((start[0] + size[0] - offset[0]) / 1e-6),
                    float((start[1] - offset[1]) / 1e-6),
                    float((start[1] + size[1] - offset[1]) / 1e-6),
                ),
                "x",
                "y",
            )
    return (0.0, float(arr.shape[-1]), 0.0, float(arr.shape[-2])), "sample", "sample"


def _result_field_request(
    results,
    args,
    *,
    field_monitor_name,
    monitor_name,
    field_name,
    field,
    frequency,
    f,
    val,
    cmap,
):
    inputs = analysis_inputs(results)
    if args:
        first_arg = str(args[0])
        if first_arg in results.monitors or len(args) > 1:
            monitor_name = first_arg
        else:
            field_name = first_arg
    if len(args) > 1:
        field_name = args[1]
    monitor_name = field_monitor_name or monitor_name
    field_name = field_name or field
    frequency = f if frequency is None else frequency
    image_cmap = cmap or ("magma" if str(val).lower() == "abs^2" else "RdBu")
    return inputs, monitor_name, field_name, frequency, image_cmap


def _monitor_field_image(data, monitor, *, field_name, frequency, val):
    values, freq = _monitor_dft_values(
        data,
        field_name or "Ez",
        frequency=frequency,
        val=val,
    )
    extent, x_dim, y_dim = _monitor_extent_um(
        monitor,
        values,
        sim=data.coordinates,
    )
    title = f"{field_name or 'field'} at {freq * 1e-12:g} THz"
    return values, extent, x_dim, y_dim, title


def _saved_field_image(results, inputs, *, field_name, val, frame, z, y):
    from beamz.analysis.adapters import to_xarray

    fields = to_xarray(results)
    if fields is None or not getattr(fields, "data_vars", None):
        raise RuntimeError("No saved fields are available to plot.")
    name = field_name or next(iter(fields.data_vars))
    selected = _select_saved_field(fields[str(name)], frame=frame, z=z, y=y)
    values = _field_view(selected, val=val).values
    data = next(iter(inputs.values()))
    extent, (x_dim, y_dim) = _coord_extent_um(selected, sim=data.coordinates)
    return values, extent, x_dim, y_dim, str(name)


def plot_result_field(
    results,
    *args,
    field_monitor_name=None,
    monitor_name=None,
    field_name=None,
    field=None,
    frequency=None,
    f=None,
    val="real",
    frame=None,
    z=None,
    y=None,
    ax=None,
    figsize=None,
    cmap=None,
    show=True,
    colorbar=True,
    vmin=None,
    vmax=None,
    xlim=None,
    ylim=None,
    show_grid=False,
    grid_color="black",
    grid_linewidth=0.2,
    grid_alpha=0.18,
    **_kwargs,
):
    """Plot saved simulation fields or a monitor DFT field."""
    inputs, monitor_name, field_name, frequency, image_cmap = _result_field_request(
        results,
        args,
        field_monitor_name=field_monitor_name,
        monitor_name=monitor_name,
        field_name=field_name,
        field=field,
        frequency=frequency,
        f=f,
        val=val,
        cmap=cmap,
    )

    if monitor_name is not None:
        data = inputs[str(monitor_name)]
        monitor = data.monitor_geometry
        if getattr(monitor, "is_3d", False) and data.frequencies.size:
            dft_freqs = data.frequencies
            if dft_freqs.size:
                return plot_dft_field(
                    data,
                    field=field_name or "Ez",
                    frequency=frequency,
                    val=val,
                    source_normalization=None,
                    sample_region=results.monitors[str(monitor_name)].sample_region,
                    ax=ax,
                    figsize=figsize,
                    cmap=image_cmap,
                    show=show,
                    vmin=vmin,
                    vmax=vmax,
                    xlim=xlim,
                    ylim=ylim,
                    show_grid=show_grid,
                    grid_color=grid_color,
                    grid_linewidth=grid_linewidth,
                    grid_alpha=grid_alpha,
                    colorbar=colorbar,
                )
        values, extent, x_dim, y_dim, title = _monitor_field_image(
            data,
            monitor,
            field_name=field_name,
            frequency=frequency,
            val=val,
        )
    else:
        values, extent, x_dim, y_dim, title = _saved_field_image(
            results,
            inputs,
            field_name=field_name,
            val=val,
            frame=frame,
            z=z,
            y=y,
        )
    return _plot_image(
        values,
        ax=ax,
        figsize=figsize,
        extent=extent,
        xlabel=f"{x_dim} (um)",
        ylabel=f"{y_dim} (um)",
        title=title,
        cmap=image_cmap,
        colorbar=colorbar,
        show=show,
        vmin=vmin,
        vmax=vmax,
        xlim=xlim,
        ylim=ylim,
    )


def _mode_cmap(value_kind):
    if value_kind in {"real", "imag"}:
        return "RdBu"
    if value_kind == "phase":
        return "twilight"
    return "magma"


def _mode_component_view(raw_field, *, grid_edges, transverse_axes):
    """Return the array, plot edges, and axis labels for one mode component."""
    field = (
        raw_field
        if grid_edges is not None and raw_field.ndim == len(grid_edges)
        else np.squeeze(raw_field)
    )
    plot_edges = None
    if grid_edges is not None and len(grid_edges) == field.ndim:
        plot_edges = tuple(np.asarray(edge, dtype=float) / _UM for edge in grid_edges)
    if field.ndim == 2 and plot_edges is not None:
        y_edges, x_edges = plot_edges
        y_axis, x_axis = (
            tuple(transverse_axes)
            if transverse_axes is not None and len(transverse_axes) == 2
            else ("axis 0", "axis 1")
        )
        return field, x_edges, y_edges, f"{x_axis} (um)", f"{y_axis} (um)"
    if field.ndim == 1 and plot_edges is not None:
        x_axis = (
            str(transverse_axes[0])
            if transverse_axes is not None and len(transverse_axes) == 1
            else "axis 0"
        )
        return field, plot_edges[0], np.asarray([-0.5, 0.5]), f"{x_axis} (um)", ""
    return field, None, None, "grid index", "grid index"


def plot_mode_field_components(
    modes,
    *,
    field_names=("Ey", "Ez"),
    mode_indices=(0,),
    val="abs",
    f=None,
    ax=None,
    figsize=None,
    cmap=None,
    show=True,
):
    """Plot selected electric or magnetic components on their physical grid."""
    import matplotlib.pyplot as plt

    value_kind = str(val).lower()
    fields = tuple(map(str, field_names))
    selected_modes = tuple(map(int, mode_indices))
    f_idx = modes._frequency_index(f)
    shape = (len(selected_modes), len(fields))
    if ax is None:
        figsize = figsize or (4.0 * shape[1], 3.2 * shape[0])
        fig, axes = plt.subplots(*shape, squeeze=False, figsize=figsize)
    else:
        axes = np.asarray(ax, dtype=object).reshape(shape)
        fig = axes.flat[0].figure
    component_index = {name: idx for idx, name in enumerate(("Ex", "Ey", "Ez"))}
    component_index.update({name: idx for idx, name in enumerate(("Hx", "Hy", "Hz"))})
    grid_edges = getattr(modes, "grid_edges", None)
    transverse_axes = getattr(modes, "transverse_axes", None)
    neffs = []
    for row, mode_index in enumerate(selected_modes):
        neff = np.asarray(modes.neffs)[f_idx, mode_index]
        neffs.append(neff)
        for col, field_name in enumerate(fields):
            if field_name not in component_index:
                raise ValueError(f"Unsupported modal field component {field_name!r}.")
            source = modes.e_fields if field_name.startswith("E") else modes.h_fields
            values = np.asarray(source)[f_idx, mode_index, component_index[field_name]]
            field, x_edges, y_edges, xlabel, ylabel = _mode_component_view(
                np.asarray(values),
                grid_edges=grid_edges,
                transverse_axes=transverse_axes,
            )
            plot_field_view(
                axes[row, col],
                field,
                val=value_kind,
                cmap=cmap or _mode_cmap(value_kind),
                x_edges=x_edges,
                y_edges=y_edges,
            )
            axes[row, col].set(
                title=f"{field_name} mode {mode_index}, n_eff={np.real(neff):.4g}",
                xlabel=xlabel,
                ylabel=ylabel,
            )
    fig.suptitle(f"Mode fields at f={float(modes.frequencies[f_idx]):.6g} Hz")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes, np.asarray(neffs)
