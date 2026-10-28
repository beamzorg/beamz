"""Simulation layout figures and permittivity rasters."""

from __future__ import annotations

import numpy as np

from beamz.analysis._plot_canvas import (
    _AXIS_INDEX,
    _DESIGN_EXTENT,
    _PLANE_AXES,
    _UM,
    _add_field_colorbar,
    _axis_labels,
    _figure_axes,
    _maybe_show,
    _pyplot,
    _simulation_origin,
)
from beamz.analysis._plot_design import _draw_boundaries, plot_design
from beamz.analysis._plot_fields import extract_axis_aligned_slice, plot_field_view
from beamz.analysis._plot_materials import (
    _layout_material_colors,
    _material_category_array,
    _material_cmap,
    _material_levels,
)
from beamz.analysis._plot_sections import (
    _draw_geometry_cross_section,
    _layout_slice,
    _overlay_layout_devices,
    _overlay_layout_pml,
    _pml_section_layers,
    overlay_boundaries,
    overlay_simulation_devices,
)
from beamz.analysis.data import static_fields


def _cross_section_arrays(
    sim, eps, slices, *, z, resolution, origin, categorical, cmap
):
    """Return display arrays plus the colormap and norm for two sections."""
    if not categorical:
        return (item.values for item in slices), cmap, None
    core_eps, substrate_eps = _material_levels(sim)
    xy, xz = (
        _material_category_array(
            item.values,
            core_permittivity=core_eps,
            substrate_permittivity=substrate_eps,
        )
        for item in slices
    )
    if abs(float(z)) <= 0.5 * resolution:
        xy = np.where(xy >= 2, xy, 1)
    z_coords = np.arange(eps.shape[0], dtype=float) * resolution
    substrate_height = float(origin[2])
    xz = np.where((z_coords > substrate_height)[:, None] & (xz == 1), 0, xz)
    xz = np.where((z_coords <= substrate_height)[:, None] & (xz == 0), 1, xz)
    material_cmap, norm = _material_cmap()
    return (xy, xz), material_cmap, norm


def _label_cross_sections(fig, axes, images, slices, *, categorical, lengths):
    if categorical:
        for ax, item in zip(axes, slices, strict=True):
            ax.set(
                title=f"cross section at {item.normal}={item.position / _UM:.2f} (um)",
                xlabel=f"{item.horizontal} (um)",
                ylabel=f"{item.vertical} (um)",
            )
        axes[0].set_xlim(slices[0].extent[:2])
        axes[1].set_xlim(slices[1].extent[:2])
        return
    for ax, image, item in zip(axes, images, slices, strict=True):
        ax.set_title(f"{item.normal} slice {item.index}")
        _axis_labels(
            ax,
            width=lengths[item.horizontal],
            height=lengths[item.vertical],
            y_label=item.vertical,
        )
        fig.colorbar(image, ax=ax, label="permittivity")


def _plot_simulation_slices(
    sim,
    *,
    z,
    y,
    origin,
    display_position,
    width_ratios,
    figsize,
    show,
    cmap,
    categorical,
    overlays,
    source_markers=True,
    monitor_markers=True,
    xlim=None,
    ylim=None,
    zlim=None,
):
    """Compose two simulation cross sections from the generic slice primitives."""
    fields = static_fields(sim)
    eps = np.asarray(fields.permittivity)
    if eps.ndim != 3:
        return plot_design(
            sim.design,
            sources=sim.sources,
            monitors=sim.monitors,
            figsize=figsize,
            show=show,
            title="Simulation Layout",
        )
    lengths = {
        axis: float(getattr(sim.design, attr)) for axis, attr in _DESIGN_EXTENT.items()
    }
    resolution = float(getattr(sim, "resolution", 1.0))
    steps = {
        "z": float(getattr(fields, "dz", resolution)),
        "y": float(getattr(fields, "dy", resolution)),
    }
    unit = _UM if categorical else 1.0
    slices = [
        extract_axis_aligned_slice(
            eps,
            axis=axis,
            step=steps[axis],
            position=position,
            origin=origin,
            display_position=display_position,
            lengths=lengths,
            unit=unit,
        )
        for axis, position in (("z", z), ("y", y))
    ]
    values, cmap, norm = _cross_section_arrays(
        sim,
        eps,
        slices,
        z=z,
        resolution=resolution,
        origin=origin,
        categorical=categorical,
        cmap=cmap,
    )

    fig, axes = _pyplot().subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": width_ratios} if width_ratios else None,
    )
    axes = list(np.ravel(axes))
    images = [
        plot_field_view(
            ax,
            value,
            extent=item.extent,
            cmap=cmap,
            norm=norm,
            aspect="equal" if categorical or item.normal == "z" else "auto",
        )[0]
        for ax, value, item in zip(axes, values, slices, strict=True)
    ]
    if overlays:
        overlay_simulation_devices(
            axes,
            sim,
            origin,
            source_markers=source_markers,
            monitor_markers=monitor_markers,
        )
        style = dict(
            facecolor="#9a9a9a",
            alpha=0.35,
            hatch="xx",
            edgecolor="#777777",
            linewidth=0.0,
        )
        for ax, item in zip(axes, slices, strict=True):
            for thickness, edges in _pml_section_layers(sim, item.normal):
                overlay_boundaries(
                    ax,
                    item.extent,
                    thickness / _UM,
                    edges=edges,
                    **style,
                )

    _label_cross_sections(
        fig,
        axes,
        images,
        slices,
        categorical=categorical,
        lengths=lengths,
    )
    if xlim is not None:
        axes[0].set_xlim(*xlim)
        axes[1].set_xlim(*xlim)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    if zlim is not None:
        axes[1].set_ylim(*zlim)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, axes


def _plot_3d_material_cross_sections(
    sim,
    *,
    z=None,
    y=None,
    width_ratios=None,
    figsize=None,
    show=True,
    cmap="Grays",
):
    fig, axes = _plot_simulation_slices(
        sim,
        z=z,
        y=y,
        origin=(0.0, 0.0, 0.0),
        display_position=False,
        width_ratios=width_ratios,
        figsize=figsize or (10.0, 4.0),
        show=show,
        cmap=cmap,
        categorical=False,
        overlays=False,
    )
    return fig, np.asarray(axes)


def _plot_3d_geometry_cross_sections(
    sim,
    *,
    z,
    y,
    origin,
    source_markers,
    monitor_markers,
    width_ratios,
    figsize,
    show,
    xlim,
    ylim,
    zlim,
):
    """Draw setup cross sections directly from the design geometry.

    This intentionally does not call :meth:`Simulation.compile`: material-grid
    slices are valuable for inspecting discretization, but a setup plot should
    preserve the exact polygon edges supplied by the user.
    """
    design = sim.design
    slices = (
        _layout_slice("z", z, design=design, origin=origin),
        _layout_slice("y", y, design=design, origin=origin),
    )
    fig, axes = _pyplot().subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": width_ratios} if width_ratios else None,
        constrained_layout=True,
    )
    axes = list(np.ravel(axes))
    colors = _layout_material_colors(design)
    for ax, item in zip(axes, slices, strict=True):
        physical_position = item.position + float(
            origin[{"x": 0, "y": 1, "z": 2}[item.normal]]
        )
        _draw_geometry_cross_section(
            ax,
            sim,
            normal=item.normal,
            position=physical_position,
            origin=origin,
            colors=colors,
        )

    for ax, item in zip(axes, slices, strict=True):
        _overlay_layout_pml(ax, sim, normal=item.normal, extent=item.extent)

    overlay_simulation_devices(
        axes,
        sim,
        origin,
        source_markers=source_markers,
        monitor_markers=monitor_markers,
    )
    for ax, item in zip(axes, slices, strict=True):
        ax.set(
            title=f"cross section at {item.normal}={item.position / _UM:.2f} (um)",
            xlabel=f"{item.horizontal} (um)",
            ylabel=f"{item.vertical} (um)",
            xlim=item.extent[:2],
            ylim=item.extent[2:],
        )
        ax.grid(False)
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].set_aspect("auto")
    if xlim is not None:
        axes[0].set_xlim(*xlim)
        axes[1].set_xlim(*xlim)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    if zlim is not None:
        axes[1].set_ylim(*zlim)
    _maybe_show(fig, show=show)
    return fig, axes


def _plot_3d_cross_sections(
    sim,
    *,
    z=0.0,
    y=0.0,
    origin=None,
    source_markers=True,
    monitor_markers=True,
    width_ratios=None,
    figsize=None,
    show=True,
    xlim=None,
    ylim=None,
    zlim=None,
):
    if origin is None:
        origin = _simulation_origin(sim)
    # Imported scenes and user-provided MaterialGrid instances only retain a
    # rasterized material field, so they keep the established grid-slice view.
    # Native Design simulations instead use their analytic geometry.
    if getattr(sim, "material_grid", None) is None:
        return _plot_3d_geometry_cross_sections(
            sim,
            z=z,
            y=y,
            origin=origin,
            source_markers=source_markers,
            monitor_markers=monitor_markers,
            width_ratios=width_ratios,
            figsize=figsize or (11.0, 4.0),
            show=show,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
        )
    return _plot_simulation_slices(
        sim,
        z=z,
        y=y,
        origin=origin,
        display_position=True,
        width_ratios=width_ratios,
        figsize=figsize or (11.0, 4.0),
        show=show,
        cmap=None,
        categorical=True,
        overlays=True,
        source_markers=source_markers,
        monitor_markers=monitor_markers,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
    )


def _permittivity_plane(sim, *, x, y, z):
    """Select one public-frame permittivity plane and its display edges."""
    materials = sim._material_grid()
    eps = np.real(np.asarray(materials.permittivity))
    grid = materials.grid
    origin = _simulation_origin(sim)
    edges = {
        axis: np.asarray(grid.axis_edges(axis)) - grid.origin[index] - origin[index]
        for index, axis in enumerate(("x", "y", "z"))
    }
    selected = [
        (axis, value)
        for axis, value in (("x", x), ("y", y), ("z", z))
        if value is not None
    ]
    if len(selected) > 1:
        raise ValueError("Specify one permittivity slice, x, y, or z.")
    normal, position = selected[0] if selected else ("z", 0.0)
    if eps.ndim == 3:
        position = float(position)
        if not edges[normal][0] <= position <= edges[normal][-1]:
            raise ValueError("Permittivity slice lies outside the simulation.")
        index = np.clip(
            np.searchsorted(edges[normal], position, side="right") - 1,
            0,
            len(edges[normal]) - 2,
        )
        eps = np.take(eps, int(index), axis=_AXIS_INDEX[normal])
    elif sim.plane_2d != "xy" or normal != "z":
        raise ValueError("2D permittivity plotting currently supports the xy plane.")
    vertical, horizontal = _PLANE_AXES[normal]
    return (
        eps,
        origin,
        normal,
        edges[horizontal] / _UM,
        edges[vertical] / _UM,
        horizontal,
        vertical,
    )


def plot_simulation_permittivity(
    sim,
    *,
    x=None,
    z=None,
    y=None,
    ax=None,
    figsize=None,
    show=True,
    source_markers=True,
    monitor_markers=True,
    colorbar=True,
    cmap="Greys",
    vmin=None,
    vmax=None,
    xlim=None,
    ylim=None,
    title="Relative permittivity",
):
    """Plot the actual material raster, including inverse-design material grids.

    Positions are in metres in the public simulation frame; display axes are
    micrometres. A 3D simulation accepts one x, y, or z slice (z=0 by default).
    """
    eps, origin, normal, x_edges, y_edges, horizontal, vertical = _permittivity_plane(
        sim, x=x, y=y, z=z
    )
    fig, ax = _figure_axes(ax, figsize=figsize or (7.0, 4.5))
    image, _ = plot_field_view(
        ax, eps, x_edges=x_edges, y_edges=y_edges, cmap=cmap, vmin=vmin, vmax=vmax
    )
    if colorbar:
        _add_field_colorbar(fig, ax, image, label="Relative permittivity")
    _overlay_layout_devices(
        ax,
        sim,
        normal=normal,
        origin=origin,
        source_markers=source_markers,
        monitor_markers=monitor_markers,
    )
    ax.set(
        xlabel=f"{horizontal} (um)",
        ylabel=f"{vertical} (um)",
        title=title,
        xlim=(x_edges[0], x_edges[-1]) if xlim is None else xlim,
        ylim=(y_edges[0], y_edges[-1]) if ylim is None else ylim,
    )
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def _draw_2d_material_grid(ax, sim, local_origin):
    materials = sim._material_grid()
    grid = materials.grid
    edges = [
        (np.asarray(grid.axis_edges(axis)) - grid.origin[i] - local_origin[i]) / _UM
        for i, axis in enumerate(("x", "y"))
    ]
    cmap, norm = _material_cmap()
    plot_field_view(
        ax,
        _material_category_array(materials.permittivity),
        x_edges=edges[0],
        y_edges=edges[1],
        cmap=cmap,
        norm=norm,
    )


def _plot_2d_layout(
    sim,
    *,
    origin,
    ax,
    figsize,
    show,
    title,
    source_markers,
    monitor_markers,
    xlim,
    ylim,
):
    """Draw the stored 2D plane with the same visual language as 3D sections."""
    from beamz.lattice import (
        grid_axes_in_physical_frame_2d,
        physical_vector_to_grid_2d,
    )

    origin = _simulation_origin(sim) if origin is None else origin
    local_origin = physical_vector_to_grid_2d(origin, sim.plane_2d)
    horizontal, vertical, normal = grid_axes_in_physical_frame_2d(sim.plane_2d)
    item = _layout_slice("z", 0.0, design=sim.design, origin=local_origin)
    fig, ax = _figure_axes(ax, figsize=figsize or (7.0, 4.5))
    if sim.material_grid is None:
        _draw_geometry_cross_section(
            ax,
            sim,
            normal="z",
            position=0.0,
            origin=local_origin,
            colors=_layout_material_colors(sim.design),
        )
    else:
        # Imported/material-only simulations must show their actual design,
        # rather than the placeholder analytic background.
        _draw_2d_material_grid(ax, sim, local_origin)
    # Boundary names refer to the stored plane; device positions are physical.
    _overlay_layout_pml(ax, sim, normal="z", extent=item.extent)
    _overlay_layout_devices(
        ax,
        sim,
        normal=normal,
        origin=origin,
        source_markers=source_markers,
        monitor_markers=monitor_markers,
    )
    ax.set(
        title=title,
        xlabel=f"{horizontal} (um)",
        ylabel=f"{vertical} (um)",
        xlim=item.extent[:2] if xlim is None else xlim,
        ylim=item.extent[2:] if ylim is None else ylim,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def plot_simulation(
    sim,
    *,
    z=None,
    y=None,
    categorical_cross_sections=True,
    source_markers=True,
    monitor_markers=True,
    width_ratios=None,
    xlim=None,
    ylim=None,
    zlim=None,
    origin=None,
    ax=None,
    figsize=None,
    show=True,
    title="Simulation Layout",
    **_kwargs,
):
    """Plot styled 2D layouts or 3D cross sections, with axes in micrometres."""
    if not sim.is_3d:
        return _plot_2d_layout(
            sim,
            origin=origin,
            ax=ax,
            figsize=figsize,
            show=show,
            title=title,
            source_markers=source_markers,
            monitor_markers=monitor_markers,
            xlim=xlim,
            ylim=ylim,
        )
    if z is not None or y is not None:
        if ax is not None:
            raise ValueError("3D cross-section plots create their own axes.")
        if not categorical_cross_sections:
            return _plot_3d_material_cross_sections(
                sim,
                z=z,
                y=y,
                width_ratios=width_ratios,
                figsize=figsize,
                show=show,
            )
        return _plot_3d_cross_sections(
            sim,
            z=0.0 if z is None else z,
            y=0.0 if y is None else y,
            origin=origin,
            source_markers=source_markers,
            monitor_markers=monitor_markers,
            width_ratios=width_ratios,
            figsize=figsize,
            show=show,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
        )

    fig, ax = plot_design(
        sim.design,
        sources=getattr(sim, "sources", ()),
        monitors=getattr(sim, "monitors", ()),
        ax=ax,
        figsize=figsize,
        show=False,
        title=title,
    )
    _draw_boundaries(ax, sim)
    fig.tight_layout()
    _maybe_show(fig, show=show)
    return fig, ax


def view_simulation_3d(sim, *, mode="auto", open_browser=True, show=False, **kwargs):
    """Return a static view for notebooks without restoring the old viewer stack."""
    del mode, open_browser
    if bool(getattr(sim, "is_3d", False)):
        origin = _simulation_origin(sim)
        if any(abs(value) > 0.0 for value in origin):
            kwargs.setdefault("z", 0.0)
            kwargs.setdefault("y", 0.0)
        else:
            kwargs.setdefault("z", 0.5 * float(getattr(sim.design, "depth", 0.0)))
            kwargs.setdefault("y", 0.5 * float(getattr(sim.design, "height", 0.0)))
    return plot_simulation(sim, show=show, **kwargs)
