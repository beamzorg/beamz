"""Small plotting and acquisition helpers for the two metasurface notebooks.

Geometry, materials, sources, boundaries, and experiment parameters live in the
notebooks. These helpers do not contain precomputed or surrogate spectra.
"""

from __future__ import annotations

import gc
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from beamz.analysis import monitor_dft_component


def layered_axis(lower, upper, interfaces, fine, bulk):
    """Snap layer interfaces and smoothly grow cells into homogeneous buffers."""
    interfaces = np.asarray(interfaces, dtype=float)
    inside = []
    for start, end in zip(interfaces[:-1], interfaces[1:], strict=True):
        count = max(1, int(np.ceil((end - start) / fine)))
        inside.extend(np.linspace(start, end, count + 1)[:-1])
    inside.append(interfaces[-1])

    def buffer(start, end, sign):
        distance = abs(end - start)
        widths, step = [], fine
        while sum(widths) + step < distance:
            widths.append(step)
            step = min(step * 1.15, bulk)
        if widths:
            remainder = distance - sum(widths)
            if remainder < widths[-1] / 1.15:
                widths[-1] += remainder
            else:
                widths.append(remainder)
        else:
            widths = [distance]
        return start + sign * np.cumsum(widths)

    left = buffer(interfaces[0], lower, -1)[::-1]
    right = buffer(interfaces[-1], upper, 1)
    return np.r_[left, inside, right]


def run_case(simulation, label):
    """Acquire raw monitor fields so empty/device normalization is consistent."""
    start = time.monotonic()
    result = simulation.run(backend="jax", progress=False, performance=False)
    raw = result.renormalize(None)
    for monitor in raw.monitors.values():
        if any(not np.isfinite(value).all() for value in monitor.dft_fields.values()):
            raise RuntimeError(f"Nonfinite fields in {label} / {monitor.monitor.name}")
    print(f"{label}: {time.monotonic() - start:.1f} s; {len(simulation.time):,} steps")
    return raw


def plane_amplitude(monitor, component="Ex"):
    """Area-average the complex zeroth-order amplitude at normal incidence."""
    fields = monitor_dft_component(monitor, component)
    weights = np.asarray(monitor.integration_weights)
    if not weights.size:
        return fields.mean(axis=1)
    return np.sum(fields * weights[None, :], axis=1) / weights.sum()


def release_device_memory():
    """Release unused executables between independent mesh/geometry runs."""
    import jax

    gc.collect()
    jax.clear_caches()


def plot_setup(
    grid,
    *,
    radius,
    height,
    layer_bottom,
    centers=((0, 0),),
    source_z=None,
    transmission_z=None,
    reflection_z=None,
    unit=1e-6,
    unit_label="µm",
    title="Geometry and mesh",
):
    """Draw exact cylinder/layer sections with realized Yee cell edges."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    x, y, z = (grid.axis_edges(axis) / unit for axis in "xyz")
    r, h, bottom = radius / unit, height / unit, layer_bottom / unit
    for ax, horizontal, direction in zip(axes[:2], (x, y), (0, 1), strict=True):
        ax.add_patch(
            Rectangle(
                (horizontal[0], z[0]),
                np.ptp(horizontal),
                bottom - z[0],
                facecolor="#eeeeee",
                edgecolor="none",
            )
        )
        ax.add_patch(
            Rectangle(
                (horizontal[0], bottom),
                np.ptp(horizontal),
                -bottom,
                facecolor="#c8d7e8",
                edgecolor="none",
            )
        )
        for center in centers:
            distance = center[1 - direction]
            if abs(distance) < radius:
                chord = np.sqrt(radius**2 - distance**2) / unit
                ax.add_patch(
                    Rectangle(
                        (center[direction] / unit - chord, 0),
                        2 * chord,
                        h,
                        facecolor="#e5aa52",
                        edgecolor="k",
                    )
                )
        ax.vlines(horizontal, z[0], z[-1], color="k", alpha=0.09, linewidth=0.3)
        ax.hlines(
            z, horizontal[0], horizontal[-1], color="k", alpha=0.09, linewidth=0.3
        )
        for position, color, label in [
            (source_z, "#a2448e", "Source"),
            (transmission_z, "#246bb2", "Transmission"),
            (reflection_z, "#555555", "Reflection"),
        ]:
            if position is not None:
                ax.axhline(position / unit, color=color, ls="--", lw=1, label=label)
        ax.set(
            xlim=(horizontal[0], horizontal[-1]),
            ylim=(z[0], z[-1]),
            xlabel=f"{'xy'[direction]} ({unit_label})",
            ylabel=f"z ({unit_label})",
        )
        ax.set_title(f"{'xy'[direction]}–z section")
    for center in centers:
        axes[2].add_patch(
            Circle(
                np.asarray(center) / unit,
                r,
                facecolor="#e5aa52",
                edgecolor="k",
                linewidth=0.6,
            )
        )
    axes[2].vlines(x, y[0], y[-1], color="k", alpha=0.09, linewidth=0.3)
    axes[2].hlines(y, x[0], x[-1], color="k", alpha=0.09, linewidth=0.3)
    axes[2].set(
        xlim=(x[0], x[-1]),
        ylim=(y[0], y[-1]),
        xlabel=f"x ({unit_label})",
        ylabel=f"y ({unit_label})",
        aspect="equal",
    )
    axes[2].set_title("x–y section through disks")
    if source_z is not None:
        axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(title)
    return fig, axes


def field_intensity(result, name, *, frequency=None, title=None):
    """Plot the vector electric intensity from an actual DFT acquisition."""
    spans = [extent for extent in result[name].monitor.size if extent > 0]
    figsize = (10, 3) if spans[0] / spans[1] > 4 else (6, 5)
    fig, ax = plt.subplots(figsize=figsize)
    result.plot_field(
        monitor_name=name,
        field="E",
        frequency=frequency,
        val="abs^2",
        ax=ax,
        show=False,
    )
    if title:
        ax.set_title(title.replace(": ", ":\n"), fontsize=11)
        fig.tight_layout()
    return fig, ax


def plot_disk_3d(*, radius, height, period, sheet_thickness, unit=1e-6):
    """Render the unit cell without an interactive viewer dependency."""
    fig = plt.figure(figsize=(6, 4.5), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    angle, axial = np.meshgrid(np.linspace(0, 2 * np.pi, 65), [0, height / unit])
    ax.plot_surface(
        radius / unit * np.cos(angle),
        radius / unit * np.sin(angle),
        axial,
        color="#e5aa52",
        alpha=0.85,
    )
    radial, angle = np.meshgrid([0, radius / unit], np.linspace(0, 2 * np.pi, 65))
    ax.plot_surface(
        radial * np.cos(angle),
        radial * np.sin(angle),
        np.full_like(radial, height / unit),
        color="#e5aa52",
    )
    x, y = np.meshgrid(
        np.array([-period / 2, period / 2]) / unit,
        np.array([-period / 2, period / 2]) / unit,
    )
    ax.plot_surface(
        x, y, np.full_like(x, -sheet_thickness / unit), color="#c8d7e8", alpha=0.6
    )
    ax.set(
        xlabel="x (µm)",
        ylabel="y (µm)",
        zlabel="z (µm)",
        zlim=(-sheet_thickness / unit, 1.1 * height / unit),
        title="Silicon resonator on the PDMS sheet",
    )
    ax.set_box_aspect((period, period, height * 2))
    return fig, ax
