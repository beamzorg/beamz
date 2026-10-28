"""Layout colours and permittivity overlays."""

from __future__ import annotations

import numpy as np

from beamz.analysis._plot_canvas import _PLANE_AXES, _UM, _edges_are_uniform
from beamz.analysis._plot_fields import extract_axis_aligned_slice
from beamz.analysis.data import AnalysisData


def _material_category_array(
    eps, *, core_permittivity=None, substrate_permittivity=None
):
    eps = np.real(np.asarray(eps))
    finite = eps[np.isfinite(eps)]
    if finite.size == 0:
        return np.zeros(eps.shape, dtype=int)
    eps_min = float(np.nanmin(finite))
    eps_max = float(np.nanmax(finite))
    if np.isclose(eps_max, eps_min):
        return np.zeros(eps.shape, dtype=int)
    if substrate_permittivity is None:
        unique = np.unique(np.round(finite, decimals=8))
        substrate_permittivity = float(unique[1]) if unique.size >= 3 else eps_min
    if core_permittivity is None:
        core_permittivity = eps_max

    sub_mid = 0.5 * (eps_min + float(substrate_permittivity))
    core_mid = 0.5 * (float(substrate_permittivity) + float(core_permittivity))
    return np.where(eps >= core_mid, 2, np.where(eps >= sub_mid, 1, 0))


def _material_cmap():
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(["#f5f5f5", "#83abc0", "#d86c96"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    return cmap, norm


# These are deliberately semantic layout colours, rather than colours sampled from
# the FDTD material grid.  A setup plot is a drawing of the design and should stay
# clean when the numerical mesh changes.
_LAYOUT_MATERIAL_COLORS = (
    "#f7fbff",  # air / lowest-index background
    "#9ecae1",  # oxide / intermediate-index material
    "#d81b60",  # silicon / highest-index material in the common SOI stack
    "#7b2cbf",
    "#f6bd60",
    "#2a9d8f",
)


def _material_display_key(material):
    """Return a stable, physically meaningful colour key for a material."""
    values = np.asarray(getattr(material, "permittivity", 1.0), dtype=float)
    values = np.real(values).reshape(-1)
    return tuple(float(value) for value in np.round(values, decimals=12))


def _material_display_value(material) -> float:
    values = np.asarray(getattr(material, "permittivity", 1.0), dtype=float)
    finite = np.real(values[np.isfinite(values)])
    return float(np.max(finite)) if finite.size else 1.0


def _layout_material_colors(design) -> dict[tuple[float, ...], str]:
    """Assign one consistent fake colour to every distinct design material."""
    materials = [getattr(design, "background", None)]
    materials.extend(
        getattr(structure, "material", None) for structure in design.structures
    )
    unique = {
        _material_display_key(material): material
        for material in materials
        if material is not None
    }
    ordered = sorted(
        unique.items(), key=lambda item: (_material_display_value(item[1]), item[0])
    )
    if len(ordered) == 1:
        palette = (_LAYOUT_MATERIAL_COLORS[0],)
    elif len(ordered) == 2:
        # A two-material SOI design normally means SiO2 + Si, whereas air + Si
        # should keep its white background.  Both should make the core prominent.
        palette = (
            (_LAYOUT_MATERIAL_COLORS[0], _LAYOUT_MATERIAL_COLORS[2])
            if _material_display_value(ordered[0][1]) <= 1.1
            else (_LAYOUT_MATERIAL_COLORS[1], _LAYOUT_MATERIAL_COLORS[2])
        )
    else:
        palette = _LAYOUT_MATERIAL_COLORS
    return {
        key: palette[index % len(palette)]
        for index, (key, _material) in enumerate(ordered)
    }


def _layout_color(material, colors) -> str:
    return colors.get(_material_display_key(material), _LAYOUT_MATERIAL_COLORS[0])


def _field_permittivity(sim):
    if isinstance(sim, AnalysisData):
        return (
            np.asarray(())
            if sim.materials is None
            else np.asarray(sim.materials.permittivity)
        )
    fields = getattr(sim, "fields", None)
    materials = getattr(fields, "materials", None)
    value = (
        getattr(materials, "permittivity", None)
        if materials is not None
        else getattr(fields, "permittivity", None)
    )
    return np.asarray(()) if value is None else np.asarray(value)


def _material_levels(sim):
    eps = _field_permittivity(sim)
    finite = np.real(eps[np.isfinite(eps)]) if eps.size else np.asarray(())
    if finite.size == 0:
        return None, None
    unique = np.unique(np.round(finite, decimals=8))
    substrate = float(unique[1]) if unique.size >= 3 else None
    core = float(unique[-1])
    return core, substrate


def _field_eps_slice(simulation, *, plane="z", index=None, plane_position=None):
    eps = _field_permittivity(simulation)
    if eps.size == 0:
        return None
    try:
        normal = {"xy": "z", "xz": "y", "yz": "x"}.get(
            str(plane).lower(), str(plane).lower()
        )
        return extract_axis_aligned_slice(
            eps,
            axis=normal,
            step=float(simulation.resolution),
            position=plane_position,
            index=index,
        ).values
    except Exception:
        return None


def _draw_field_eps_overlay(
    ax,
    eps_slice,
    *,
    extent,
    reverse=False,
    alpha=0.2,
    core_permittivity=None,
    x_edges=None,
    y_edges=None,
):
    if eps_slice is None:
        return None
    eps_real = np.real(np.asarray(eps_slice))
    finite = eps_real[np.isfinite(eps_real)]
    if finite.size == 0:
        return None
    eps_min = float(np.nanmin(finite))
    eps_max = (
        float(core_permittivity)
        if core_permittivity is not None
        else float(np.nanmax(finite))
    )
    eps_span = max(eps_max - eps_min, 1e-5)
    eps_fraction = np.clip((eps_real - eps_min) / eps_span, 0.0, 1.0)
    eps_color = eps_fraction if reverse else 1.0 - eps_fraction
    core = eps_real >= 0.5 * eps_max
    if not np.any(core):
        return None
    rgba = np.zeros((*core.shape, 4), dtype=float)
    rgba[core, :3] = eps_color[core, None]
    rgba[core, 3] = np.clip(float(alpha), 0.0, 1.0)
    if x_edges is not None and y_edges is not None:
        overlay = ax.pcolormesh(
            x_edges,
            y_edges,
            rgba,
            shading="flat",
        )
        ax.set_aspect("equal")
        return overlay
    return ax.imshow(
        rgba,
        origin="lower",
        extent=(extent[0], extent[1], extent[2], extent[3]),
        aspect="equal",
        interpolation="nearest",
    )


def _overlay_material_slice(
    ax,
    simulation,
    *,
    normal,
    position,
    origin,
    reverse,
    alpha,
    core_permittivity,
    sample_region=None,
):
    vertical, horizontal = _PLANE_AXES[normal]
    origin_by_axis: dict[str, float] = dict(
        zip(("x", "y", "z"), map(float, origin), strict=True)
    )
    grid = getattr(simulation.coordinates, "grid", None)
    plot_x_edges = None
    plot_y_edges = None
    if grid is not None and sample_region is not None:
        materials = simulation.materials
        eps = (
            np.asarray(materials.permittivity)
            if materials is not None
            else np.asarray(())
        )
        if materials is not None and eps.ndim == 3:
            storage_axes = ("z", "y", "x")
            selectors = []
            for dimension, axis in enumerate(storage_axes):
                data_origin = int(materials.origin[dimension])
                if axis == normal:
                    selector = int(sample_region.plane_index) - data_origin
                    if not 0 <= selector < eps.shape[dimension]:
                        return None
                else:
                    interval = sample_region.axis_interval(axis)
                    if interval is None:
                        return None
                    start = int(interval.start) - data_origin
                    stop = int(interval.stop) - data_origin
                    if start < 0 or stop > eps.shape[dimension] or stop <= start:
                        return None
                    selector = slice(
                        start,
                        stop,
                    )
                selectors.append(selector)
            eps_slice = eps[tuple(selectors)]
        else:
            eps_slice = None
        vertical_interval = sample_region.axis_interval(vertical)
        horizontal_interval = sample_region.axis_interval(horizontal)
        if vertical_interval is None or horizontal_interval is None:
            return None
        x_edges = (
            grid.axis_edges(horizontal)[
                horizontal_interval.start : horizontal_interval.stop + 1
            ]
            - origin_by_axis[horizontal]
        ) / _UM
        y_edges = (
            grid.axis_edges(vertical)[
                vertical_interval.start : vertical_interval.stop + 1
            ]
            - origin_by_axis[vertical]
        ) / _UM
        extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
        if not _edges_are_uniform(x_edges) or not _edges_are_uniform(y_edges):
            plot_x_edges, plot_y_edges = x_edges, y_edges
    else:
        eps_slice = _field_eps_slice(
            simulation, plane=normal, plane_position=float(position)
        )
        if eps_slice is None:
            return None
        shape = np.asarray(eps_slice).shape
        metadata = simulation.coordinates
        lengths: dict[str, float] = {
            "x": metadata.width,
            "y": metadata.height,
            "z": metadata.depth,
        }

        def length(axis, fallback):
            return float(lengths[axis] or fallback)

        extent = (
            (0.0 - origin_by_axis[horizontal]) / _UM,
            (
                length(horizontal, shape[1] * simulation.resolution)
                - origin_by_axis[horizontal]
            )
            / _UM,
            (0.0 - origin_by_axis[vertical]) / _UM,
            (
                length(vertical, shape[0] * simulation.resolution)
                - origin_by_axis[vertical]
            )
            / _UM,
        )
    if eps_slice is None:
        return None
    return _draw_field_eps_overlay(
        ax,
        eps_slice,
        extent=extent,
        reverse=reverse,
        alpha=alpha,
        core_permittivity=core_permittivity,
        x_edges=plot_x_edges,
        y_edges=plot_y_edges,
    )
