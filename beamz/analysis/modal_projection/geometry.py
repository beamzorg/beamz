"""Geometry and phase helpers for modal projection."""

from __future__ import annotations

from typing import cast

import numpy as np

from beamz.analysis._coordinates import monitor_plane_coordinates_3d
from beamz.analysis.data import AnalysisData, static_fields
from beamz.const import LIGHT_SPEED
from beamz.design.grid import Axis


def _analysis_grid(sim):
    if isinstance(sim, AnalysisData):
        return sim.coordinates.grid
    try:
        fields = static_fields(sim)
    except TypeError:
        return None
    return getattr(fields, "geometry", None)


def _monitor_projection_phase(component, frequencies, dt):
    """Phase-align raw monitor phasors to the E-field sample time.

    BeamZ's DFT uses the phasor convention

        f(t) = Re{F exp(-i omega t)}
        F ~= 2 sum_t f(t) exp(+i omega t) / sum_t 1

    Monitors are sampled after the E update at timestamp T = t + dt. At
    that instant E is stored at T, while the leapfrog H fields are stored at
    T - dt/2. If a component is actually sampled at T + tau but accumulated
    with exp(+i omega T), the accumulator returns F exp(-i omega tau). To
    recover the common-time modal phasor F, multiply by exp(+i omega tau).
    Therefore E has tau = 0 and H has tau = -dt/2.
    """
    freq_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
    comp = str(component)
    if comp.startswith("H"):
        return np.exp(-1j * np.pi * freq_arr * float(dt))
    return np.ones_like(freq_arr, dtype=np.complex128)


def _modal_projection_spatial_phase(component, frequencies, plane_delay_s):
    """Phase-align E components from their Yee plane to the H-referenced mode.

    Mode profiles are gauged to the dominant H component, matching the
    ModeSource launch convention. After the temporal Yee correction, E
    samples still need the spatial propagation phase from the E Yee plane
    to that H reference plane.
    """
    freq_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
    comp = str(component)
    delay = float(plane_delay_s)
    if comp.startswith("E") and delay != 0.0:
        return np.exp(1j * 2.0 * np.pi * freq_arr * delay)
    return np.ones_like(freq_arr, dtype=np.complex128)


def _modal_projection_plane_delay_s(sim, spec, frequency, mode_neff):
    """Return the E-to-H modal-plane delay used by S-parameter projection."""
    if getattr(sim, "is_3d", False):
        # 3D monitors interpolate every recorded component onto the same
        # physical analysis plane. There is no remaining normal-direction
        # Yee half-cell offset to compensate during modal extraction.
        return 0.0
    freq = float(frequency)
    neff = float(np.real(np.asarray(mode_neff)))
    if (not np.isfinite(freq)) or freq <= 0.0:
        return 0.0
    if (not np.isfinite(neff)) or neff <= 0.0:
        return 0.0
    grid = _analysis_grid(sim)
    if grid is not None and grid.metric_kind_for(("x", "y")) != "isotropic_uniform":
        axis = cast(Axis, str(getattr(spec, "axis", "x")))
        center = tuple(
            float(value) for value in getattr(spec, "center", (0.0, 0.0, 0.0))
        )
        coordinate = center[{"x": 0, "y": 1, "z": 2}[axis]]
        index = int(np.argmin(np.abs(np.asarray(grid.centers(axis)) - coordinate)))
        d_axis = float(grid.cell_widths(axis)[index])
    else:
        d_axis = float(getattr(sim, "resolution", 0.0) or 0.0)
    if (not np.isfinite(d_axis)) or d_axis <= 0.0:
        return 0.0

    direction_sign = +1.0
    delta_s = direction_sign * 0.5 * d_axis
    return float(delta_s * neff / LIGHT_SPEED)


def _apply_modal_projection_spatial_phase(component, values, frequency, projection):
    phase = _modal_projection_spatial_phase(
        component,
        np.asarray([float(frequency)], dtype=float),
        float(projection.get("modal_plane_delay_s", 0.0)),
    )[0]
    return np.asarray(values, dtype=np.complex128) * phase


def _mode_components_for_port(spec):
    axis = spec.axis
    tm_map = {
        "x": ("Ez", "Hy", 2, 1, -1.0),
        "y": ("Ez", "Hx", 2, 0, 1.0),
        "z": ("Ey", "Hx", 1, 0, -1.0),
    }
    te_map = {
        "x": ("Ey", "Hz", 1, 2, 1.0),
        "y": ("Ex", "Hz", 0, 2, -1.0),
        "z": ("Ex", "Hy", 0, 1, 1.0),
    }
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported port axis '{axis}'.")
    e_comp, h_comp, e_idx, h_idx, sign = (
        tm_map[axis] if spec.polarization == "tm" else te_map[axis]
    )
    proj_components_3d = {
        "x": ("Ey", "Ez", "Hy", "Hz"),
        "y": ("Ex", "Ez", "Hx", "Hz"),
        "z": ("Ex", "Ey", "Hx", "Hy"),
    }[axis]
    return {
        "axis": axis,
        "e_component": e_comp,
        "h_component": h_comp,
        "e_mode_index": e_idx,
        "h_mode_index": h_idx,
        "signed_flux_sign": sign,
        "projection_components": (e_comp, h_comp),
        "projection_components_3d": proj_components_3d,
    }


def _monitor_analysis_plane_3d(
    sim: AnalysisData,
    monitor,
    axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    if axis != monitor.plane_normal:
        raise ValueError("Port axis must match the monitor plane normal.")
    grid = sim.coordinates.grid
    if grid is None or grid.metric_kind == "isotropic_uniform":
        return monitor.get_analysis_plane_coords_3d(
            dx=sim.resolution,
            dy=sim.resolution,
            dz=sim.resolution,
            field_shape=sim.coordinates.fields.grid_shape,
        )
    _, coord0, coord1 = monitor_plane_coordinates_3d(sim.coordinates, monitor)
    return coord0, coord1
