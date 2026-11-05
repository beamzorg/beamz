"""Narrowband oblique TF/SF sheets on physical Yee supports."""

import numpy as np

from beamz.const import LIGHT_SPEED
from beamz.lattice import component_axis_offsets_3d


def lower_bloch_plane_wave(source, ctx):
    # Import plan records lazily to keep the singledispatch compiler acyclic.
    from .compiler import (
        CompiledInjectionPlan,
        TemporalWaveform,
        _append_tangential_sheet,
        _interior_plane_index,
        _require_transverse_span,
        _sheet_impedance_and_area,
        _validate_plane_wave_background,
    )
    from .time import sample_source_waveforms

    if np.asarray(ctx.fields.permittivity).ndim != 3:
        raise ValueError("Oblique PlaneWaveSource currently requires a 3D grid.")
    pulse = source.source_time
    frequency = getattr(pulse, "freq0", None)
    width = getattr(pulse, "fwidth", None)
    if (
        not hasattr(pulse, "sample")
        or frequency is None
        or width is None
        or width > 0.1 * frequency
    ):
        raise ValueError(
            "Oblique PlaneWaveSource requires a narrowband quadrature pulse (fwidth <= 0.1 * freq0)."
        )
    grid = ctx.grid
    normal_id = "xyz".index(source.direction[-1])
    array_axis = 2 - normal_id
    sign = 1 if source.direction[0] == "+" else -1
    transverse = [i for i in range(3) if i != normal_id]
    _require_transverse_span(
        source,
        grid,
        normal_id,
        "Oblique PlaneWaveSource must span the full transverse domain.",
    )
    omega = 2 * np.pi * frequency
    magnitude = omega * source.background_index / LIGHT_SPEED
    wavevector = np.asarray(source.transverse_wavevector)
    tangential = np.linalg.norm(wavevector)
    if tangential >= magnitude:
        raise ValueError(
            "Oblique PlaneWaveSource requires a propagating wavevector below n * omega / c."
        )
    wavevector = wavevector.copy()
    wavevector[normal_id] = sign * np.sqrt(magnitude**2 - tangential**2)
    khat = wavevector / magnitude
    normal = np.eye(3)[normal_id]
    s = np.cross(normal, khat)
    s /= np.linalg.norm(s)
    p = np.cross(s, khat)
    ehat = np.cos(source.pol_angle) * s + np.sin(source.pol_angle) * p
    hhat = np.cross(khat, ehat)
    # Surface equivalence currents use only tangential incident E/H components.
    currents_e = -sign * np.cross(normal, hhat)
    currents_h = sign * np.cross(normal, ehat)
    k, edges = _interior_plane_index(
        grid.axis_edges("xyz"[normal_id]),
        source.center[normal_id],
        "Plane-wave injection sheet must be inside the grid, away from its walls.",
    )
    hi = k - 1 if sign > 0 else k
    h_coordinate = 0.5 * (edges[hi] + edges[hi + 1])
    impedance, area = _sheet_impedance_and_area(source, grid, transverse)
    cosine = abs(khat[normal_id])
    amplitude = np.sqrt(2 * source.power * impedance / (area * cosine))
    entries = []
    for kind, current, plane, spacing, incident_coordinate, offset in (
        ("H", currents_h, hi, edges[hi + 1] - edges[hi], edges[k], 0.0),
        (
            "E",
            currents_e,
            k,
            0.5 * (edges[k + 1] - edges[k - 1]),
            h_coordinate,
            0.5 * ctx.dt,
        ),
    ):
        delay = wavevector[normal_id] * (incident_coordinate - edges[k]) / omega
        signal, quadrature = sample_source_waveforms(
            pulse,
            t0=ctx.t0,
            dt=ctx.dt,
            num_steps=ctx.num_steps,
            total_steps=ctx.total_steps,
            offset_fn=lambda t, dt, shift=offset - delay: t + shift,
        )
        index = [slice(None)] * 3
        index[array_axis] = slice(plane, plane + 1)
        index = tuple(index)

        def values_for(component, target, scaled, index=index):
            _validate_plane_wave_background(source, ctx, component, index)
            # E currents sample incident H, and H currents sample incident E.
            # Their transverse offsets coincide for each tangential component.
            phase = np.zeros(target[index].shape)
            offsets = component_axis_offsets_3d(component)
            for ti in transverse:
                axis = "xyz"[ti]
                coordinates = (
                    grid.centers(axis)
                    if offsets[axis] == 0.5
                    else grid.axis_edges(axis)
                )
                shape = [1] * 3
                shape[2 - ti] = len(coordinates)
                phase += source.transverse_wavevector[ti] * (
                    coordinates.reshape(shape) - source.center[ti]
                )
            return scaled * np.exp(1j * phase)

        _append_tangential_sheet(
            entries,
            ctx,
            kind=kind,
            vector=current,
            index=index,
            spacing=spacing,
            waveform=TemporalWaveform(signal, quadrature),
            amplitude=amplitude,
            impedance=impedance,
            launched_power=source.power,
            values_for=values_for,
        )
    return CompiledInjectionPlan(tuple(entries))
