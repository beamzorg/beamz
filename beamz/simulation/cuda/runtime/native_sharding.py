"""Partition resident CUDA packets and assemble their sparse observations."""

from __future__ import annotations

import threading
import weakref
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from math import prod
from os import getenv
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from beamz.simulation._cuda_abi import CUDA_GRAPH_CACHE
from beamz.simulation.cuda.engine._capabilities import _Unsupported, _validate
from beamz.simulation.cuda.engine._contracts import _CPML, _DFT, _FIELDS, _YEE
from beamz.simulation.cuda.engine._plan import (
    _admit_rolling,
    _resident_packet_bytes,
    _rolling_band,
)
from beamz.simulation.model import CompiledProgram, SimulationState
from beamz.simulation.sharding import _select_sharding_axis, normalize_sharding_config

_FIELD_ORDER = (*_FIELDS[3:], *_FIELDS[:3])


@contextmanager
def _on_device(module, ordinal):
    previous = module._set_device(ordinal)
    try:
        yield
    finally:
        module._set_device(previous)


def _slice(axis, lo, hi):
    return tuple(
        slice(lo, hi) if dimension == axis else slice(None) for dimension in range(3)
    )


def _material_slice(value, axis, lo, hi):
    from beamz._region_array import RegionArray

    if not isinstance(value, RegionArray):
        return value[_slice(axis, lo, hi)]
    shape = list(value.shape)
    shape[axis] = min(hi, shape[axis]) - lo

    def read(box):
        key = list(box)
        key[axis] = slice(box[axis].start + lo, box[axis].stop + lo)
        return value[tuple(key)]

    return RegionArray(tuple(shape), value.dtype, read)


def _compact_psi(term, axis, lo, hi, shape):
    """Keep packed CPML entries whose physical coordinates this rank owns."""
    packed_shape = list(shape)
    packed_shape[term.axis] = term.slab.low + term.slab.high
    if term.axis != axis:
        packed_shape[axis] = min(hi, shape[axis]) - lo
        return packed_shape, 0
    low, high = term.slab.low, term.slab.high
    stop = term.slab.logical_stop
    high_origin = stop - high
    low_start, low_stop = min(lo, low), min(hi, low)
    high_start = low + min(high, max(0, lo - high_origin))
    high_stop = low + min(high, max(0, hi - high_origin))
    start = low_start if low_stop > low_start else high_start
    end = high_stop if high_stop > high_start else max(start, low_stop)
    packed_shape[axis] = end - start
    return packed_shape, start


def _local_sources(groups, axis, lo, stops, runtime):
    """Clip owned source slabs before upload, leaving time samples intact."""
    arguments = []
    coincident, disjoint = 0, 0
    for index, group in enumerate(groups):
        if group is None:
            arguments.extend(runtime._EMPTY_SOURCE_GROUP)
            continue
        coefficients, samples, starts = (
            np.asarray(value) for value in (group.coeffs, group.waveforms, group.starts)
        )
        field = _FIELDS[index % 3 + (0 if 3 <= index < 6 else 3)]
        stop = stops[field]
        span = coefficients.shape[axis + 1]
        intersections = [
            (i, max(lo, int(start)), min(stop, int(start) + span))
            for i, start in enumerate(starts[:, axis])
            if max(lo, int(start)) < min(stop, int(start) + span)
        ]
        if not intersections:
            arguments.extend(runtime._EMPTY_SOURCE_GROUP)
            continue
        shape = list(coefficients.shape)
        shape[0] = len(intersections)
        shape[axis + 1] = max(end - begin for _, begin, end in intersections)
        local = np.zeros(shape, dtype=np.float32)
        local_starts = np.copy(starts[[i for i, _, _ in intersections]])
        for dst, (src, begin, end) in enumerate(intersections):
            start = int(starts[src, axis])
            aligned = min(begin, stop - shape[axis + 1])
            local[(dst, *_slice(axis, begin - aligned, end - aligned))] = coefficients[
                (src, *_slice(axis, begin - start, end - start))
            ]
            local_starts[dst, axis] = aligned - lo
        if np.all(local_starts == local_starts[0]):
            coincident |= 1 << index
        # Padded slab boxes can overlap after clipping even if the original
        # contribution regions did not. Certify the actual launch boxes.
        elif all(
            np.any(np.abs(start - previous) >= shape[1:])
            for n, start in enumerate(local_starts)
            for previous in local_starts[:n]
        ):
            disjoint |= 1 << index
        arguments.extend(
            (local, samples[[i for i, _, _ in intersections]], local_starts)
        )
    return tuple(arguments), coincident, disjoint


def _local_monitors(monitors, logical, owned, axis, lo, hi, origin=None):
    """Remap only owned interpolation neighbors into ragged local DFT rows."""
    origin = lo if origin is None else origin
    local_specs, scatter = [], []
    value_offset = weight_offset = 0
    for spec in monitors:
        indices, weights = [], []
        active = np.zeros(spec.dft_point_count, dtype=bool)
        component_mask = np.asarray(spec.dft_component_mask) > 0
        for index, (component, raw, weight) in enumerate(
            zip(_FIELD_ORDER, spec.dft_flat_idx, spec.dft_weights, strict=True)
        ):
            shape = logical[component]
            raw = np.asarray(raw)
            coordinate = list(
                np.unravel_index(np.clip(raw, 0, np.prod(shape) - 1), shape)
            )
            local = (
                (coordinate[axis] >= lo)
                & (coordinate[axis] < hi)
                & (raw >= 0)
                & (raw < np.prod(shape))
                & (np.asarray(weight) != 0)
                & component_mask[index]
            )
            active |= np.any(local, axis=1)
            coordinate[axis] = np.clip(
                coordinate[axis] - origin, 0, owned[component][axis] - 1
            )
            flat = np.where(
                local, np.ravel_multi_index(coordinate, owned[component]), -1
            )
            if np.any(flat > np.iinfo(np.int32).max):
                raise _Unsupported(
                    "A local CUDA monitor index exceeds int32; increase the shard count"
                )
            indices.append(flat.astype(np.int32))
            weights.append(np.where(local, weight, np.float32(0)))
        points = np.flatnonzero(active)
        if not len(points):
            if lo:
                continue
            # The first rank carries the reference window clock even when it
            # owns no nonzero component samples from this monitor.
            points = np.zeros(1, dtype=np.int32)
            # Preserve a valid clock-only gather, owned by this first rank.
            for flat in indices:
                flat[0, 0] = 0
        local_specs.append(
            replace(
                spec,
                dft_flat_idx=tuple(x[points] for x in indices),
                dft_weights=tuple(x[points] for x in weights),
                dft_point_count=len(points),
                dft_value_offset=value_offset,
                dft_weight_offset=weight_offset,
            )
        )
        scatter.append((spec, points))
        value_offset += 6 * spec.freq_count * len(points)
        weight_offset += spec.freq_count
    return tuple(local_specs), tuple(scatter)


def _local_scatters(sources, logical, shapes, axis, lo):
    packets = []
    for timing, component, indices, ends, coefficients, waveform in sources:
        name = _FIELDS[(0 if timing == 1 else 3) + component]
        coordinates = list(np.unravel_index(indices, logical[name]))
        coordinates[axis] -= lo
        keep = (coordinates[axis] >= 0) & (coordinates[axis] < shapes[name][axis])
        if not np.any(keep):
            continue
        local = np.ravel_multi_index(
            tuple(values[keep] for values in coordinates), shapes[name]
        )
        if np.max(local) > np.iinfo(np.int32).max:
            raise _Unsupported("A local CUDA sparse source index exceeds int32")
        if ends.size:
            counts = np.diff(ends, prepend=np.int32(0))
            coefficients = coefficients[np.repeat(keep, counts)]
            ends = np.cumsum(counts[keep], dtype=np.int32)
        else:
            coefficients = coefficients[keep]
        packets.append(
            (timing, component, local.astype(np.int32), ends, coefficients, waveform)
        )
    return packets


def _local_recorders(monitors, state, logical, shapes, axis, lo, begin, end, window):
    plans, outputs = [], []
    payload = 0
    for spec in monitors:
        if spec.recorder_index < 0:
            continue
        interval = spec.record_interval
        for component, sign, index, raw, weights in zip(
            spec.canonical_components,
            spec.component_signs,
            spec.field_buffer_indices,
            spec.field_interp_flat_idx,
            spec.field_interp_weights,
            strict=True,
        ):
            name = component.lower()
            shape = logical[name]
            if not np.size(raw):
                extent = min(end, shape[axis]) - begin
                if extent <= 0:
                    continue
                region = tuple(extent if k == axis else n for k, n in enumerate(shape))
                positions = np.unravel_index(np.arange(prod(region)), region)
                global_positions = tuple(
                    pos + begin if k == axis else pos for k, pos in enumerate(positions)
                )
                rows = np.ravel_multi_index(global_positions, shape)
                positions = tuple(
                    pos + begin - lo if k == axis else pos
                    for k, pos in enumerate(positions)
                )
                indices = np.ravel_multi_index(positions, shapes[name]).reshape(-1, 1)
                weights = np.ones(indices.shape, dtype=np.float32)
            else:
                raw, weights = np.asarray(raw), np.asarray(weights)
                coordinates = list(np.unravel_index(np.maximum(raw, 0), shape))
                owned = (raw >= 0) & (weights != 0)
                owned &= (coordinates[axis] >= begin) & (coordinates[axis] < end)
                rows = np.flatnonzero(np.any(owned, axis=1))
                if not rows.size:
                    continue
                positions = [coordinate[rows] for coordinate in coordinates]
                positions[axis] = np.clip(
                    positions[axis] - lo, 0, shapes[name][axis] - 1
                )
                indices = np.ravel_multi_index(positions, shapes[name])
                keep = owned[rows]
                # Zero-weight neighbors must still have a valid local index.
                anchor = np.take_along_axis(
                    indices, np.argmax(keep, axis=1)[:, None], axis=1
                )
                indices = np.where(keep, indices, anchor)
                weights = np.ascontiguousarray(
                    np.where(keep, weights[rows], 0), dtype=np.float32
                )
            if np.max(indices) > np.iinfo(np.int32).max:
                raise _Unsupported("A local CUDA recorder index exceeds int32")
            indices = indices.astype(np.int32)
            count = state.recorded_fields[index].shape[0]
            history = np.zeros((count, rows.size), dtype=np.float32)
            plans.append(
                (_FIELDS.index(name), interval, sign, indices, weights, history)
            )
            outputs.append((index, rows, history, interval))
            payload += (
                indices.nbytes
                + weights.nbytes
                + 4 * rows.size * min(count, (window + interval - 1) // interval)
            )
    return plans, outputs, payload


def _local_coefficients(
    program, coefficient_table, axis, lo, stops, logical, *, pack_lossless=True
):
    from beamz._region_array import uniform_value
    from beamz.const import EPS_0
    from beamz.simulation.cuda.storage.material_blocks import pack_material_blocks

    changes = {}
    for name in _FIELDS:
        hi = stops[name]
        h_or_e, component = name
        decay_name = f"{h_or_e}_decay_{component}"
        source_name = f"{h_or_e}_source_{component}"
        if h_or_e == "e" and pack_lossless:
            conductivity = getattr(program.grid, "sig_" + component)
            if uniform_value(conductivity) == 0:
                value = getattr(program.grid, "eps_" + name)
                if value.shape == logical[name]:
                    epsilon = _material_slice(value, axis, lo, hi)
                else:
                    epsilon = np.broadcast_to(np.asarray(value), logical[name])[
                        _slice(axis, lo, hi)
                    ]
                packed = pack_material_blocks(
                    (epsilon,), max_bytes=prod(epsilon.shape) * 4, host=True
                )
                if packed is not None:
                    tables, codes = packed
                    changes[decay_name] = coefficient_table(
                        np.ascontiguousarray(tables[0], dtype=np.float32),
                        np.float32(program.config.dt),
                        np.float32(EPS_0),
                    )
                    changes[source_name] = codes[0]
                    continue
        for kind in (decay_name, source_name):
            value = getattr(program.coefficients, kind)
            if value.ndim == 3 and value.shape[axis] != 1:
                value = _material_slice(value, axis, lo, hi)
            scalar = uniform_value(value)
            changes[kind] = scalar if scalar is not None else np.asarray(value)
    return program.coefficients._replace(**changes)


def _local_cpml_profiles(terms, profiles, axis, runtime):
    metadata = np.asarray(runtime._term_metadata(terms)).copy()
    values = []
    for index, (term, (shape, offset)) in enumerate(zip(terms, profiles, strict=True)):
        if term.axis == axis:
            low = min(shape[axis], max(0, term.slab.low - offset))
            metadata[index, 2:4] = (low, shape[axis] - low)
        for value in (term.a, term.b, term.inv_kappa):
            if not value.size or not shape[term.axis]:
                empty = tuple(
                    0 if dimension == term.axis else 1 for dimension in range(3)
                )
                values.append(np.empty(empty, dtype=np.float32))
            elif term.axis == axis:
                values.append(
                    np.asarray(value)[_slice(axis, offset, offset + shape[axis])]
                )
            else:
                values.append(value)
    return metadata, *values


def _local_tensor(payload, module, axis, start, stop):
    cross, origin, *metrics = payload
    origin = np.asarray(origin, dtype=np.int32).copy()
    cross = np.asarray(cross)
    begin = max(start, int(origin[axis]))
    end = min(stop + 1, int(origin[axis]) + cross.shape[axis])
    if end - begin < 2:
        # A one-node intersection still affects a local Yee edge. Pad it on
        # the available side, while an empty intersection has no correction.
        if end <= begin:
            origin[:] = 0
            cross = np.zeros((2, 2, 2, 3), dtype=np.float32)
        else:
            original = int(origin[axis])
            pads = [(0, 0)] * 4
            if end <= stop:
                pads[axis] = (0, 1)
            else:
                pads[axis] = (1, 0)
            cross = np.pad(cross[_slice(axis, begin - original, end - original)], pads)
            origin[axis] = begin - start - pads[axis][0]
    else:
        original = int(origin[axis])
        cross = cross[_slice(axis, begin - original, end - original)]
        origin[axis] = begin - start
    widths = [np.asarray(x) for x in metrics[:3]]
    if widths[axis].ndim:
        widths[axis] = widths[axis][start:stop]
    scratch = [
        module.ZeroBuffer(
            [n - int(a == 2 - c) for a, n in enumerate(cross.shape[:3])], "float32"
        )
        for c in range(3)
    ]
    return np.ascontiguousarray(cross), origin, *widths, *scratch


def _local_effects(
    module,
    program,
    state,
    logical,
    args,
    axis,
    lo,
    hi,
    ownership,
    temporal,
    tensor_payload,
    scatter_sources,
):
    if not temporal or not (tensor_payload or scatter_sources or state.recorded_fields):
        return (), (), (), (), 0
    shapes = {name: arg.shape for name, arg in zip(_FIELDS, args[:6], strict=True)}
    tensor = (
        _local_tensor(tensor_payload, module, axis, lo, hi) if tensor_payload else ()
    )
    scatters = _local_scatters(scatter_sources, logical, shapes, axis, lo)
    assert ownership is not None
    recorders, outputs, bytes_ = _local_recorders(
        program.monitors,
        state,
        logical,
        shapes,
        axis,
        lo,
        ownership[0],
        ownership[1],
        temporal[0],
    )
    bytes_ += sum(array.nbytes for array in tensor)
    bytes_ += sum(array.nbytes for packet in scatters for array in packet[2:])
    return tensor, scatters, recorders, outputs, bytes_


def _local_packet(
    program,
    context,
    groups,
    initial,
    module,
    axis,
    lo,
    hi,
    logical,
    ownership=None,
    staged=(True, True),
    coefficient_table=None,
    tensor_coefficients=None,
    fixed_cpml=False,
):
    from beamz.simulation import _cuda_abi as abi
    from beamz.simulation import execute
    from beamz.simulation.cuda import runtime as rt

    cpml = program.boundary.cpml
    psi_zero = np.zeros((), dtype=np.float32)
    # Each halo uses the staggered supports of a standalone Yee grid.
    minimum = min(shape[axis] for shape in logical.values())
    stops = {
        name: hi + shape[axis] - minimum if ownership else min(hi, shape[axis])
        for name, shape in logical.items()
    }
    owned = {
        name: tuple(stops[name] - lo if k == axis else n for k, n in enumerate(shape))
        for name, shape in logical.items()
    }
    if any(min(shape) < 1 for shape in owned.values()):
        raise _Unsupported("native shard count leaves an empty Yee component")
    relative = {}
    local_profiles = {}
    geometry = []
    for phase in ("h", "e"):
        terms = getattr(cpml, phase + "_terms")
        if fixed_cpml:
            profiles = [
                (
                    tuple(
                        term.slab.low + term.slab.high if k == term.axis else n
                        for k, n in enumerate(owned[term.component.lower()])
                    ),
                    0,
                )
                for term in terms
            ]
        else:
            profiles = [
                _compact_psi(
                    term,
                    axis,
                    lo,
                    stops[term.component.lower()],
                    logical[term.component.lower()],
                )
                for term in terms
            ]
        if ownership and not fixed_cpml and terms:
            local_profiles[phase] = _local_cpml_profiles(terms, profiles, axis, rt)
        relative[f"cpml_psi_{phase}_terms"] = tuple(
            np.broadcast_to(psi_zero, shape) for shape, _ in profiles
        )
        target_names = tuple(phase + c for c in "xyz")
        other = "e" if phase == "h" else "h"
        source_names = tuple(other + c for c in "xyz")
        geometry.append(
            np.asarray(
                [
                    (axis, lo, 0),
                    *(logical[n] for n in (*target_names, *source_names)),
                    tuple(offset for _, offset in profiles[:3])
                    if profiles
                    else (0, 0, 0),
                    tuple(offset for _, offset in profiles[3:])
                    if profiles
                    else (0, 0, 0),
                ],
                dtype=np.int32,
            )
        )
    specs, scatter = _local_monitors(
        (
            spec
            for spec in program.monitors
            if spec.dft_enabled and spec.freq_count and spec.dft_point_count
        ),
        logical,
        owned,
        axis,
        *(ownership or (lo, hi)),
        origin=lo,
    )
    local_program = replace(program, monitors=specs)
    state = initial._replace(
        **{
            name: np.broadcast_to(np.float32(0), shape) for name, shape in owned.items()
        },
        **relative,
        **execute.monitor_runtime.empty_monitor_values(
            local_program, num_steps=program.config.num_steps, virtual_dft=True
        ),
    )
    if tensor_coefficients is not None:
        program = replace(program, coefficients=tensor_coefficients)
    coefficients = _local_coefficients(
        program,
        coefficient_table or module.electric_coefficient_table,
        axis,
        lo,
        stops,
        logical,
        pack_lossless=tensor_coefficients is None,
    )
    args, values, aliases = rt._graph_io(state, context, coefficients)
    graph_count = len(args)
    if ownership and not fixed_cpml:
        values_in = list(args)
        for phase, start in (("h", 12), ("e", abi.CPML_E_PSI_INPUT_OFFSET - 19)):
            if phase in local_profiles:
                values_in[start : start + 19] = local_profiles[phase]
        for phase, names in enumerate((_FIELDS[:3], _FIELDS[3:])):
            index = graph_count - 6 + 3 * phase + axis
            metric = np.asarray(values_in[index])
            if metric.size > 1:
                # One metric per target Yee coordinate. Components in a phase
                # share the largest derivative support on the cut axis.
                stop = max(stops[name] for name in names)
                values_in[index] = metric[lo:stop]
        args = tuple(values_in)
    if (
        fixed_cpml
        and cpml.enabled
        and getattr(module, "compiled_transparent_halos", False)
    ):
        # A cut is an internal Yee interface. Keep packed CPML12 storage for
        # the CUDA kernels and give only the artificial sides identity response.
        values_in = list(args)
        for start, terms in (
            (abi.CPML_H_PSI_INPUT_OFFSET - 18, cpml.h_terms),
            (abi.CPML_E_PSI_INPUT_OFFSET - 18, cpml.e_terms),
        ):
            for j, term in enumerate(terms):
                if term.axis != axis:
                    continue
                for kind in range(3):
                    idx = start + 3 * j + kind
                    profile = np.array(args[idx], dtype=np.float32, copy=True)
                    samples = profile.reshape(-1)
                    if lo > 0:
                        samples[:12] = float(kind == 2)
                    if hi < minimum:
                        samples[12:] = float(kind == 2)
                    values_in[idx] = profile
        args = tuple(values_in)
    if ownership:
        virtual = tuple(
            np.broadcast_to(np.float32(0), shape) for shape in owned.values()
        )
        virtual += (np.empty(0, np.float32),) * 6
        for buffer in virtual:
            aliases[len(args)] = len(values)
            args, values = (*args, buffer), (*values, buffer)
    sources, coincident, disjoint = _local_sources(groups, axis, lo, stops, rt)
    packed = (
        rt.pack_dft_monitors(
            specs,
            host=True,
            ragged=getattr(module, "compiled_ragged_dft_gather", False),
        )
        if specs
        else None
    )
    args, _, _, _, count = rt._monitor_io(
        state,
        packed,
        (*args, *sources),
        values,
        aliases,
        observation_origin=state.t,
        observation_step_offset=np.int32(0),
        host=True,
    )
    if ownership:
        args = (*args, *((np.empty(0, np.int32),) * 5))
    schedule = rt.NativeSchedulePlan(
        layout=abi.PROGRAM_LAYOUT_YEE_IN_PLACE,
        flags=rt._native_schedule_flags(
            context, coefficients, sources=True, monitors=count
        )
        if ownership
        else 0,
        temporal_steps=2 if ownership else 1,
        monitor_count=count,
        coincident_source_group_mask=coincident,
        disjoint_source_group_mask=disjoint,
    )
    attributes = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in rt._program_attributes(
            context, program.config.num_steps, schedule
        ).items()
    }
    if ownership and not fixed_cpml:
        # Local metadata is authoritative. Internal Yee faces are transparent.
        boundary_code = attributes["boundary_code"] & 0x3F
        if lo:
            boundary_code &= ~(1 << (2 * axis))
        if hi < minimum:
            boundary_code &= ~(2 << (2 * axis))
        attributes["boundary_code"] = boundary_code
    zeroable = set(range(len(_FIELDS)))
    if ownership:
        zeroable.update(range(graph_count, graph_count + 6))
        graph_count += len(virtual)
    if cpml.enabled:
        for offset in (abi.CPML_H_PSI_INPUT_OFFSET, abi.CPML_E_PSI_INPUT_OFFSET):
            zeroable.update(range(offset, offset + abi.CPML_TERM_COUNT))
    monitor_offset = (
        graph_count + abi.SOURCE_GROUP_COUNT * abi.SOURCE_GROUP_BUFFER_COUNT
    )
    zeroable.update(range(monitor_offset + 7, monitor_offset + 10))
    buffers = []
    for index, value in enumerate(args):
        array = np.asarray(value)
        if index in zeroable and array.ndim:
            buffers.append(module.ZeroBuffer(list(array.shape), array.dtype.name))
        else:
            buffers.append(np.ascontiguousarray(array) if array.ndim else array)
    if ownership:
        layout = _CPML if cpml.enabled else _YEE
        payload = _resident_packet_bytes(buffers, module, layout)
        return tuple(buffers), attributes, tuple(geometry), specs, scatter, payload
    surface_bytes = 0
    for name in _FIELDS:
        if "xyz".index(name[1]) == 2 - axis:
            continue
        incoming = int(
            name[0] == "e"
        )  # E arrives from the upper neighbor, H from the lower.
        copies = int(staged[incoming]) + int(axis != 0 and staged[1 - incoming])
        surface_bytes += (
            4 * prod(n for d, n in enumerate(owned[name]) if d != axis) * copies
        )
    payload = sum(a.nbytes for a in buffers) + surface_bytes
    payload += 4 + sum(x.nbytes for x in geometry)
    return tuple(buffers), attributes, tuple(geometry), specs, scatter, payload


def _download_dft(session, specs, scatter, cpml, target, temporal=False):
    from beamz.simulation import _cuda_abi as abi

    if not specs:
        return
    value_count = sum(6 * spec.freq_count * spec.dft_point_count for spec in specs)
    weight_count = sum(spec.freq_count for spec in specs)
    parts = tuple(
        np.empty(n, dtype=np.float32) for n in (value_count, value_count, weight_count)
    )
    offset = abi.CPML_GRAPH_OUTPUT_COUNT if cpml else abi.YEE_GRAPH_OUTPUT_COUNT
    if temporal:
        offset += 12
    for i, part in enumerate(parts):
        session.download_into(offset + i, part)
    for local, (full, points) in zip(specs, scatter, strict=True):
        length = 6 * full.freq_count
        for dst, src in zip(target[:2], parts[:2], strict=True):
            local_offset = local.dft_value_offset
            full_offset = full.dft_value_offset
            row = dst[full_offset : full_offset + length * full.dft_point_count]
            row.reshape(6, full.freq_count, full.dft_point_count)[:, :, points] += src[
                local_offset : local_offset + length * len(points)
            ].reshape(6, full.freq_count, len(points))
        target[2][full.dft_weight_offset : full.dft_weight_offset + full.freq_count] = (
            parts[2][
                local.dft_weight_offset : local.dft_weight_offset + full.freq_count
            ]
        )


@dataclass
class _ShardStepper:
    module: Any
    program: CompiledProgram
    initial: SimulationState
    shards: tuple
    domain_measure: float
    edges: tuple
    payloads: tuple
    temporal: tuple
    recordings: tuple
    regions: tuple
    key: tuple
    device_uuids: tuple
    release: weakref.finalize
    lock: Any = field(default_factory=threading.RLock)
    completed: int = 0
    temporal_windows: int = 0
    prefix: SimulationState | None = None
    observation_step: int = 0

    def _close(self):
        with self.lock:
            self.release()

    def reset(self):
        for ordinal, owner, _, _ in self.shards:
            with _on_device(self.module, ordinal):
                owner.reset()
        self.completed = 0
        self.temporal_windows = 0
        self.prefix = None
        self.observation_step = 0

    def _checkpoint_slices(self, axis, lo, hi, *, staggered_high=True):
        def stop(shape):
            return hi + int(staggered_high and shape[axis] > self.edges[-1])

        for name in _FIELDS:
            yield name, _slice(axis, lo, stop(getattr(self.initial, name).shape))
        cpml = self.program.boundary.cpml
        for phase in ("h", "e"):
            for i, term in enumerate(getattr(cpml, phase + "_terms")):
                shape = getattr(self.initial, term.component.lower()).shape
                if term.axis == axis:
                    packed, offset = _compact_psi(term, axis, lo, stop(shape), shape)
                    selection = _slice(axis, offset, offset + packed[axis])
                else:
                    selection = _slice(axis, lo, stop(shape))
                yield (phase, i), selection

    def restore_checkpoint(self, simulation, state, contract):
        from beamz.simulation.execute import runtime_inputs

        self.prefix = runtime_inputs(
            self.program,
            state,
            monitor_steps=simulation.num_steps - contract.initial_step,
        )
        step = contract.initial_step
        origin = np.float32(simulation.time[0])
        spacing = np.float32(self.program.config.dt)
        time = np.float32(state.t)
        if not np.isfinite(time):
            raise _Unsupported(
                "Native sharded continuation requires a finite state time"
            )
        on_grid = np.abs(time - (origin + spacing * np.float32(step))) <= np.spacing(
            abs(time)
        )
        if not on_grid:
            origin = time
            self.observation_step = step
        axis = self.temporal[1]
        for (ordinal, owner, specs, _), (lo, hi) in zip(
            self.shards, self.regions, strict=True
        ):
            packet = []
            for name, selection in self._checkpoint_slices(axis, lo, hi):
                if isinstance(name, tuple):
                    phase, index = name
                    value = getattr(self.prefix, f"cpml_psi_{phase}_terms")[index]
                else:
                    value = getattr(self.prefix, name)
                packet.append(np.ascontiguousarray(np.asarray(value)[selection]))
            lengths = (
                sum(6 * spec.freq_count * spec.dft_point_count for spec in specs),
                sum(spec.freq_count for spec in specs),
            )
            packet.extend(np.zeros(n, np.float32) for n in (lengths[0], *lengths))
            with _on_device(self.module, ordinal):
                owner.restore_state(
                    packet, float(origin), step, step - self.observation_step
                )
        self.prefix = self.prefix._replace(
            **{name: getattr(self.initial, name) for name in _FIELDS},
            cpml_psi_h_terms=(),
            cpml_psi_e_terms=(),
        )
        self.completed = step

    def checkpoint(self, state):
        axis = self.temporal[1]
        leaves: dict[str | tuple[str, int], np.ndarray] = {
            name: np.empty(getattr(self.initial, name).shape, np.float32)
            for name in _FIELDS
        }
        cpml = self.program.boundary.cpml
        for phase in ("h", "e"):
            leaves.update(
                {
                    (phase, i): np.empty(term.slab.shape, dtype=np.float32)
                    for i, term in enumerate(getattr(cpml, phase + "_terms"))
                }
            )
        for rank, ((ordinal, owner, _, _), (lo, hi)) in enumerate(
            zip(self.shards, self.regions, strict=True)
        ):
            begin, end = self.edges[rank : rank + 2]
            # Internal cuts assign the staggered interface sample to the upper
            # owner. The final rank owns the actual staggered exterior sample.
            local = self._checkpoint_slices(axis, lo, hi)
            owned = self._checkpoint_slices(
                axis, begin, end, staggered_high=end == self.edges[-1]
            )
            with _on_device(self.module, ordinal):
                for index, ((name, region), (_, target)) in enumerate(
                    zip(local, owned, strict=True)
                ):
                    region, target = region[axis], target[axis]
                    if target.start == target.stop:
                        continue
                    array = leaves[name]
                    shape = list(array.shape)
                    shape[axis] = region.stop - region.start
                    values = np.empty(shape, dtype=array.dtype)
                    owner.download_into(index, values)
                    array[_slice(axis, target.start, target.stop)] = values[
                        _slice(
                            axis,
                            target.start - region.start,
                            target.stop - region.start,
                        )
                    ]
        return state._replace(
            **{name: leaves[name] for name in _FIELDS},
            cpml_psi_h_terms=tuple(leaves["h", i] for i in range(len(cpml.h_terms))),
            cpml_psi_e_terms=tuple(leaves["e", i] for i in range(len(cpml.e_terms))),
        )

    def _advance(self, steps, *, snapshot=True):
        owners = [owner for _, owner, _, _ in self.shards]
        if self.temporal:
            spans, axis, window, depth, slots, *packet_links = self.temporal
            elapsed = self.module.NativeSession._run_temporal_partitions(
                owners, spans, axis, steps, window, depth, slots, True, *packet_links
            )
            self.temporal_windows += steps // window
        else:
            elapsed = self.module.NativeSession._run_shards(owners, steps)
        self.completed += steps
        return self._snapshot(include_recorders=False) if snapshot else None, elapsed

    def _recording_result_options(self):
        return {"owned_recordings": True}

    def _final_observations(self):
        return self._snapshot()

    def _snapshot(self, *, include_recorders=True):
        dft = tuple(
            np.zeros(getattr(self.initial, name).shape, np.float32) for name in _DFT
        )
        for ordinal, owner, specs, scatter in self.shards:
            with _on_device(self.module, ordinal):
                clocks = owner.clocks()
                if (
                    clocks["current_step"] != self.completed
                    or clocks["elapsed_steps"] != self.completed - self.observation_step
                ):
                    raise RuntimeError(
                        "native shards diverged in their physical clocks"
                    )
                _download_dft(
                    owner,
                    specs,
                    scatter,
                    self.program.boundary.cpml.enabled,
                    dft,
                    bool(self.temporal),
                )
        if self.prefix is not None:
            for name, values in zip(_DFT, dft, strict=True):
                values += np.asarray(getattr(self.prefix, name))
        recordings = {}
        if include_recorders and self.initial.recorded_fields:
            specs = tuple(
                spec for spec in self.program.monitors if spec.recorder_index >= 0
            )
            counts = tuple(self.completed // spec.record_interval for spec in specs)
            indices = tuple(
                np.arange(1, count + 1, dtype=np.int32) * spec.record_interval
                for count, spec in zip(counts, specs, strict=True)
            )
            histories = tuple(
                np.zeros_like(history) for history in self.initial.recorded_fields
            )
            previous_steps = 0 if self.prefix is None else int(self.prefix.current_step)
            if self.prefix is not None and previous_steps:
                for spec in specs:
                    count = previous_steps // spec.record_interval
                    for index in spec.field_buffer_indices:
                        histories[index][:count] = self.prefix.recorded_fields[index][
                            :count
                        ]
            for shard in self.recordings:
                for index, rows, history, interval in shard:
                    count = self.completed // interval
                    start = previous_steps // interval
                    histories[index].reshape(history.shape[0], -1)[
                        start:count, rows
                    ] += history[start:count]
            times = tuple(
                np.float32(clocks["observation_origin"])
                + (steps - self.observation_step).astype(np.float32)
                * np.float32(self.program.config.dt)
                for steps in indices
            )
            if self.prefix is not None and previous_steps:
                for target, original, spec in zip(
                    times, self.prefix.recorded_times, specs, strict=True
                ):
                    count = previous_steps // spec.record_interval
                    target[:count] = original[:count]
            recordings = dict(
                recorded_fields=histories,
                recorded_steps=indices,
                recorded_times=times,
                recorded_counts=tuple(np.asarray(n, np.int32) for n in counts),
            )
        return self.initial._replace(
            **dict(zip(_DFT, dft, strict=True)),
            **recordings,
            current_step=np.asarray(self.completed, np.int32),
            t=np.asarray(clocks["time"], np.float32),
        )

    def _field_diagnostics(self):
        from beamz.simulation.cuda.engine._runtime import _summarize_diagnostics

        reports = []
        for ordinal, owner, _, _ in self.shards:
            with _on_device(self.module, ordinal):
                reports.append(owner.field_diagnostics())
        return _summarize_diagnostics(
            {
                "sums": np.sum([report["sums"] for report in reports], axis=0),
                "max_fields": [report["max_fields"] for report in reports],
                "finite": [report["finite"] for report in reports],
            },
            self.domain_measure,
        )


class _RollingUnavailable(Exception):
    """The short exchange window cannot use rolling on every local partition."""


def _admit_peer_packets(devices, remaining, payloads, module):
    """Admit both ends of each non-peer cut after every mandatory rank has been priced."""
    cost = module.temporal_peer_staging_bytes
    links = []
    for rank, (left, right) in enumerate(zip(devices[:-1], devices[1:], strict=True)):
        pack = (
            left != right
            and not module._has_direct_exchange(left, right)
            and min(remaining[left], remaining[right]) >= cost
        )
        links.append(pack)
        if pack:
            for index, dev in ((rank, left), (rank + 1, right)):
                remaining[dev] -= cost
                payloads[index] += cost
    return tuple(links)


def _shard_diagnostics(program, axis, lo, hi, begin, end, ownership, tensor):
    from beamz.simulation.cuda.engine._prepare import _diagnostic_inputs

    if tensor:
        return _diagnostic_inputs(program, (axis, lo, hi), owned=ownership)
    return _diagnostic_inputs(
        program, (axis, begin, ownership[1] if ownership else end)
    )


def _prepare_shards(
    simulation,
    module,
    devices,
    axis,
    adaptive,
    budget,
    resources,
    temporal=(),
    reference=(),
    minimum_band=0,
    checkpoint=False,
):
    import jax

    from beamz.simulation import execute
    from beamz.simulation.compile import _PROGRAM_CACHE, compile_program
    from beamz.simulation.cuda.engine._prepare import (
        _coefficient_table,
        _source_scatter,
    )

    cached_programs = set(_PROGRAM_CACHE)
    program = compile_program(
        simulation,
        num_steps=min(256, simulation.num_steps),
        backend="cuda_streamed",
        native_preparation=True,
        backend_resolver=lambda _: "cuda_streamed",
        setup_context_factory=lambda _: jax.default_device(jax.devices("cpu")[0]),
    )
    for key, value in tuple(_PROGRAM_CACHE.items()):
        if key.native_preparation and key not in cached_programs and value is program:
            del _PROGRAM_CACHE[key]
    _validate(program, native_effects=bool(temporal))
    context = execute.build_step_context(program, low_memory=True)
    _validate(program, context, native_effects=bool(temporal))
    tensor_payload, tensor_coefficients = (), None
    tensor = bool(program.coefficients.e_inverse_offdiagonal.size)
    if tensor:
        if not temporal or not getattr(module, "compiled_temporal_tensor", False):
            raise _Unsupported(
                "Sharded tensor CUDA requires a tensor-capable temporal component"
            )
        from beamz.simulation.cuda.storage import tensor as tensors

        tensor_payload = tensors.payload(
            context,
            program.coefficients,
            scratch_factory=lambda shape, dtype: np.broadcast_to(
                np.zeros((), dtype), shape
            ),
        )
        tensor_coefficients = tensors.coefficients(context, program.coefficients)
    groups = tuple(
        context.source_batches[(phase, component)][0]
        for phase, components in execute.SOURCE_PHASE_COMPONENTS.items()
        for component in components
    )
    logical = {
        name: tuple(getattr(program.grid, name.capitalize()).shape) for name in _FIELDS
    }
    scatter_sources = []
    if temporal:
        for timing, (phase, components) in enumerate(
            execute.SOURCE_PHASE_COMPONENTS.items()
        ):
            for component, name in enumerate(components):
                _, rest = context.source_batches[(phase, name)]
                for spec in rest:
                    packet = _source_scatter(
                        spec,
                        logical[name.lower()],
                        timing,
                        component,
                        index_dtype=np.int64,
                    )
                    if packet is not None:
                        scatter_sources.append(packet)
    length = max(shape[axis] for shape in logical.values())
    window = depth = slots = exchange = guard = 0
    fast_cpml = False
    if temporal:
        from beamz.simulation.cuda.runtime import _uniform_cpml_thickness

        window, depth, slots = temporal
        thickness = _uniform_cpml_thickness(context)
        length = min(shape[axis] for shape in logical.values())
        fast_cpml = (
            not tensor
            and not checkpoint
            and program.config.metric_kind == "isotropic_uniform"
            and thickness == 12
            and not context.boundary.cpml.metallic_edges
            and length // len(devices) >= window + 12
        )
        guard = _halo_guard(module) if fast_cpml else 2
        if tensor:
            window, depth = 8, 4
            temporal = window, depth, slots
        exchange = 2 * window if tensor else window
        if length // len(devices) < exchange + guard:
            raise _Unsupported("Temporal shards require sufficient halo depth")
    spans = []
    edges = [rank * length // len(devices) for rank in range(len(devices) + 1)]
    initial = SimulationState(
        **{
            name: np.broadcast_to(np.float32(0), shape)
            for name, shape in logical.items()
        },
        cpml_psi_h_terms=(),
        cpml_psi_e_terms=(),
        **execute.monitor_runtime.empty_monitor_values(
            program,
            num_steps=simulation.num_steps,
            host_recorders=True,
            virtual_dft=True,
        ),
        t=np.asarray(simulation.time[0], np.float32),
        current_step=np.asarray(0, np.int32),
    )
    shards, payloads = [], []
    domain_measure = 0.0
    reserve = 512 << 20
    remaining = {}
    shard_recordings, shard_regions = [], []
    per_link_packets = (
        temporal
        and axis == 0
        and getattr(
            module,
            "compiled_temporal_packet_links",
            False,
        )
    )

    def close(owner, device):
        with _on_device(module, device):
            owner.close()

    for rank, ordinal in enumerate(devices):
        with _on_device(module, ordinal):
            begin, end = edges[rank : rank + 2]
            lo, hi = begin, end
            ownership = None
            if temporal:
                halo = exchange + guard
                lo = begin - halo if rank else 0
                hi = end + halo if rank + 1 < len(devices) else length
                ownership = (begin, end + int(rank == len(devices) - 1))
                spans.append((lo, begin, end))
            shard_regions.append((lo, hi))
            neighbor_storage = not temporal and getattr(
                module, "compiled_shard_rows", False
            )
            neighbors = (
                devices[rank - 1] if rank else -1,
                devices[rank + 1] if rank + 1 < len(devices) else -1,
            )
            staged = (
                tuple(
                    peer >= 0 and not module._has_direct_exchange(ordinal, peer)
                    for peer in neighbors
                )
                if neighbor_storage
                else (True, True)
            )
            args, attrs, geometry, specs, scatter, payload = _local_packet(
                program,
                context,
                groups,
                initial,
                module,
                axis,
                lo,
                hi,
                logical,
                ownership,
                staged,
                coefficient_table=_coefficient_table(simulation, module),
                tensor_coefficients=tensor_coefficients,
                fixed_cpml=bool(temporal and fast_cpml),
            )
            local_tensor, scatters, recorders, outputs, effect_bytes = _local_effects(
                module,
                program,
                initial,
                logical,
                args,
                axis,
                lo,
                hi,
                ownership,
                temporal,
                tensor_payload,
                scatter_sources,
            )
            shard_recordings.append(outputs)
            payload += effect_bytes
            diagnostics = (
                _shard_diagnostics(program, axis, lo, hi, begin, end, ownership, tensor)
                if adaptive
                else None
            )
            if diagnostics is not None:
                payload += diagnostics[3]
            info = module.device_info()
            available = remaining.setdefault(
                ordinal,
                min(
                    budget or int(info["total_memory_bytes"] * 0.9),
                    info["free_memory_bytes"],
                )
                - reserve,
            )
            if temporal:
                if axis != 0:
                    from beamz.simulation import _cuda_abi as abi

                    staged = sum(
                        peer >= 0 and not module._has_direct_exchange(ordinal, peer)
                        for peer in neighbors
                    )
                    if staged:
                        fields = list(args[:6])
                        cpml = program.boundary.cpml
                        for start, terms in (
                            (abi.CPML_H_PSI_INPUT_OFFSET, cpml.h_terms),
                            (abi.CPML_E_PSI_INPUT_OFFSET, cpml.e_terms),
                        ):
                            fields.extend(
                                args[start + j]
                                for j, t in enumerate(terms)
                                if t.axis != axis or not fast_cpml
                            )
                        face_bytes = sum(
                            4 * prod(n for k, n in enumerate(f.shape) if k != axis)
                            for f in fields
                        )
                        payload += staged * 2 * (exchange + 1) * face_bytes
                cut = {}
                if getattr(module, "compiled_shard_resource_cuts", False):
                    cut = {
                        "partition_cut": (
                            axis,
                            begin - lo if rank else -1,
                            end - lo if rank + 1 < len(devices) else -1,
                        ),
                        "boundary_code": attrs["boundary_code"],
                    }
                local_shape = tuple(min(f.shape[k] for f in args[:6]) for k in range(3))
                band = 0
                if (
                    fast_cpml
                    and axis == 0
                    and cut
                    and depth % 8 == 0
                    and getattr(module, "compiled_rolling_tiles", False)
                ):
                    band = _rolling_band(local_shape, info, simulation.num_steps)
                    if band:
                        cut["rolling_band"] = band
                resource_plan = module.moving_resource_plan(
                    args,
                    depth=depth,
                    raw_slots=slots,
                    macro_slab_count=window // depth,
                    recorders=recorders,
                    scatters=scatters,
                    tensor=tensor,
                    **cut,
                )
                available_extra = (
                    min(available, info["free_memory_bytes"] - reserve) - payload
                )
                envelope = None
                if reference and resource_plan.get("rolling_extra_bytes") is not None:
                    # Shortening the exchange removes entire resident Yee planes.
                    # Price the old workspace on the smaller input conservatively.
                    old_window, old_depth, old_slots = reference
                    old_plan = module.moving_resource_plan(
                        args,
                        depth=old_depth,
                        raw_slots=old_slots,
                        macro_slab_count=old_window // old_depth,
                        recorders=recorders,
                        scatters=scatters,
                        **{
                            key: value
                            for key, value in cut.items()
                            if key != "rolling_band"
                        },
                    )
                    neighbors = int(rank > 0) + int(rank + 1 < len(devices))
                    fields_per_plane = sum(prod(field.shape[1:]) for field in args[:6])
                    envelope = (
                        old_plan["extra_gpu_bytes"]
                        + old_plan.get("field_frontier_bytes", 0)
                        + 4 * fields_per_plane * neighbors * (old_window - window)
                    )
                width, rolling_bytes = (
                    _admit_rolling(
                        resource_plan,
                        prod(local_shape),
                        band,
                        available_extra,
                        envelope=envelope,
                    )
                    if band
                    else (0, None)
                )
                if reference and (not width or width < minimum_band):
                    raise _RollingUnavailable
                if width:
                    attrs = dict(attrs, rolling_band=width)
                payload += int(
                    resource_plan["extra_gpu_bytes"]
                    if rolling_bytes is None
                    else rolling_bytes
                )
                if axis == 0 and not per_link_packets:
                    # Older binaries can still exchange dense faces without scratch.
                    attrs = dict(attrs, allow_peer_packets=0)
            if payload > min(available, info["free_memory_bytes"] - reserve):
                raise MemoryError(
                    f"CUDA shard {rank} needs {payload} bytes plus {reserve} reserve"
                )
            remaining[ordinal] = (
                min(available, info["free_memory_bytes"] - reserve) - payload
            )
            kind = "mstep" if temporal else "generated"
            if neighbor_storage and (info.get("major"), info.get("minor")) in (
                (8, 0),
                (8, 6),
                (9, 0),
                (12, 0),
            ):
                kind = "sharded"
            owner = module.NativeSession(
                args,
                attrs,
                preserve_inputs=False,
                kind=kind,
                initialize_scratch=False,
            )
            resources.callback(close, owner, ordinal)
            if local_tensor or recorders or scatters:
                owner.configure_effects(local_tensor, scatters, recorders, window)
            if not temporal:
                if neighbor_storage:
                    owner._configure_shard(geometry, neighbors)
                else:
                    owner._configure_shard(geometry)
            if diagnostics is not None:
                domain_measure = diagnostics[2]
                selection = (
                    axis,
                    begin - lo,
                    end - lo if rank + 1 < len(devices) else -1,
                )
                if len(diagnostics) == 5:
                    owner.configure_diagnostics(
                        *diagnostics[:3], interpolation=diagnostics[4]
                    )
                else:
                    owner.configure_diagnostics(
                        *diagnostics[:3], *((selection,) if ownership else ())
                    )
            shards.append((ordinal, owner, specs, scatter))
            payloads.append(
                payload
                if temporal
                else int(owner.metadata()["allocated_payload_bytes"])
            )
            del args, geometry
    packet_links = (
        (_admit_peer_packets(devices, remaining, payloads, module),)
        if per_link_packets
        else ()
    )
    return (
        program,
        initial,
        tuple(shards),
        domain_measure,
        tuple(edges),
        tuple(payloads),
        (tuple(spans), axis, *temporal, *packet_links) if temporal else (),
        tuple(shard_recordings),
        tuple(shard_regions),
    )


def _run_shards(
    simulation,
    module,
    devices,
    axis,
    *,
    progress=False,
    store_full_materials=False,
    performance=True,
    termination=None,
    wall_start=None,
    temporal=(),
    minimum_band=0,
    segment=None,
    state=None,
):
    """Execute spatial partitions on their assigned CUDA devices."""
    import jax

    from beamz._helpers import _finish_inline_progress, _print_inline_progress
    from beamz.simulation import backend as backends
    from beamz.simulation import execute
    from beamz.simulation.cuda.engine._runtime import (
        _DEVICE_LOCKS,
        _LOCK,
        _SESSIONS,
        _clear,
        _retire_overlapping,
    )
    from beamz.simulation.results import SimulationResults, SimulationRun

    checkpoint = segment is not None
    steps_requested = segment.steps if checkpoint else simulation.num_steps
    infos = {ordinal: dict(module.device_info(ordinal)) for ordinal in set(devices)}
    uuids = sorted({str(info["uuid"]) for info in infos.values()})
    with _LOCK:
        locks = tuple(
            _DEVICE_LOCKS.setdefault(uuid, threading.RLock()) for uuid in uuids
        )
    with ExitStack() as stack, jax.default_device(jax.devices("cpu")[0]):
        for lock in locks:
            stack.enter_context(lock)
        budget = int(getenv("BEAMZ_CUDA_MEMORY_BUDGET_BYTES", "0"))
        if budget < 0:
            raise ValueError("CUDA memory budget must be nonnegative")
        graph_options = (
            backends.cuda_flags_from_env() & CUDA_GRAPH_CACHE,
            backends.cuda_graph_cache_capacity_from_env(),
        )
        key = (
            "sharded",
            checkpoint,
            tuple(devices),
            axis,
            budget,
            termination is not None,
            temporal,
            minimum_band,
            graph_options,
        )
        with _LOCK:
            _retire_overlapping(simulation, set(uuids))
            stepper = cast("_ShardStepper | None", _SESSIONS.get(simulation))
            if stepper is not None and (
                stepper.key != key or stepper.device_uuids != tuple(uuids)
            ):
                _clear(simulation)
                stepper = None
            reused = stepper is not None
            if stepper is None:
                with ExitStack() as resources:
                    preparation = (
                        simulation,
                        module,
                        devices,
                        axis,
                        termination is not None,
                        budget,
                    )
                    parts = None
                    if (
                        not checkpoint
                        and temporal
                        and axis == 0
                        and getattr(module, "compiled_rolling_tiles", False)
                    ):
                        shape = simulation.grid.shape_zyx
                        owned = (shape[0] // len(devices), *shape[1:])
                        if temporal[0] > 16 and _rolling_band(
                            owned, infos[devices[0]], simulation.num_steps
                        ):
                            with ExitStack() as trial:
                                try:
                                    parts = _prepare_shards(
                                        *preparation,
                                        trial,
                                        (16, 8, 6),
                                        temporal,
                                        minimum_band,
                                    )
                                except _RollingUnavailable:
                                    pass
                                else:
                                    resources.enter_context(trial.pop_all())
                    if parts is None:
                        if minimum_band:
                            raise _Unsupported(
                                "The admitted rolling band underfills this GPU"
                            )
                        parts = _prepare_shards(
                            *preparation, resources, temporal, checkpoint=checkpoint
                        )
                    release = weakref.finalize(simulation, resources.pop_all().close)
                    stepper = _ShardStepper(
                        module,
                        *parts,
                        key=key,
                        device_uuids=tuple(uuids),
                        release=release,
                    )
                _SESSIONS[simulation] = stepper

            # Drop the rank locks before taking the registry lock on failure.
            # A partial advance cannot become the next cached starting state.
            def discard_failed(exc_type, _error, _traceback):
                if exc_type is not None:
                    with _LOCK:
                        if _SESSIONS.get(simulation) is stepper:
                            _clear(simulation)

            stack.push(discard_failed)
            stack.enter_context(stepper.lock)
        if reused:
            stepper.reset()
        if checkpoint and state is not None:
            stepper.restore_checkpoint(simulation, state, segment)
        program = stepper.program
        payloads, edges = stepper.payloads, stepper.edges
        reserve = 512 << 20
        if termination is not None:
            result = execute.run_until_terminated(
                simulation,
                termination,
                backend="cuda",
                progress=progress,
                store_full_materials=store_full_materials,
                performance=performance,
                sharding=None,
                _native=stepper,
                _wall_start=wall_start,
            )
        else:
            batches = (
                execute._progress_chunk_lengths(steps_requested)
                if progress
                else (steps_requested,)
            )
            elapsed = 0.0
            try:
                for steps in batches:
                    _, duration = stepper._advance(steps, snapshot=False)
                    elapsed += duration
                    if progress:
                        _print_inline_progress(
                            stepper.completed,
                            steps_requested,
                            label="Running simulation",
                        )
            finally:
                if progress:
                    _finish_inline_progress()
            stats = (
                execute._performance_for(
                    program, runtime_s=elapsed, steps=steps_requested
                )
                if performance
                else None
            )
            snapshot = stepper._snapshot()
            if checkpoint:
                snapshot = stepper.checkpoint(snapshot)
            result = SimulationResults.from_run(
                simulation,
                runtime_fields=program.grid,
                monitor_results=execute._decode_monitor_results(
                    simulation,
                    program,
                    snapshot,
                    owned_recordings=True,
                ),
                completed_steps=stepper.completed,
                store_full_materials=store_full_materials,
                source_launch_powers=execute._compiled_source_launch_powers(
                    program, len(simulation.sources)
                ),
                performance=stats,
            )
            if stats is not None:
                execute._print_performance(stats)
        object.__setattr__(
            result,
            "_cuda_plan",
            MappingProxyType(
                dict(
                    backend="cuda",
                    binding="native",
                    schedule="mstep" if temporal else "ordinary",
                    **(
                        dict(
                            exchange_window=stepper.temporal[2],
                            temporal_windows=stepper.temporal_windows,
                            moving_depth=stepper.temporal[3],
                            overlap_slots=stepper.temporal[4],
                            rolling_bands=tuple(
                                owner.metadata().get("rolling_band", 0)
                                for _, owner, _, _ in stepper.shards
                            ),
                        )
                        if temporal
                        else {}
                    ),
                    state_ownership=segment.ownership if checkpoint else "private",
                    initial_step=segment.initial_step if checkpoint else 0,
                    final_step=stepper.completed,
                    payload_bytes=sum(payloads),
                    shard_payload_bytes=tuple(payloads),
                    reserve_bytes=reserve,
                    shard_axis="zyx"[axis],
                    shard_devices=tuple(dict.fromkeys(devices)),
                    partition_devices=tuple(devices),
                    shard_edges=tuple(edges),
                    reason=(
                        "temporal wavefronts with overlapped boundary exchange"
                        if temporal
                        else "native slab partitions with peer halo exchange"
                    ),
                )
            ),
        )
        return SimulationRun(results=result, state=snapshot) if checkpoint else result


def _native_shard_axis(request, shape, count):
    if request != "auto":
        return "zyx".index(_select_sharding_axis(request, shape))
    # A face normal to x reads one FP32 value from each strided row. Each
    # such load costs at least one 32-byte sector. The y and z faces read rows.
    eligible = [axis for axis in range(3) if shape[axis] >= count]
    if not eligible:
        return "zyx".index(_select_sharding_axis("auto", shape))
    return max(eligible, key=lambda axis: shape[axis] / (8 if axis == 2 else 1))


def _temporal_shard_choice(
    shape, axis, count, steps, device, *, automatic, tile=(), guard=14
):
    """Amortize the halo and wavefront over the cells owned by each GPU."""
    name = device.get("name", "")
    local_3090 = count == 1 and name == "NVIDIA GeForce RTX 3090"
    rolling_a100 = (
        name == "NVIDIA A100-SXM4-80GB"
        and axis == 0
        and bool(_rolling_band((shape[0] // count, *shape[1:]), device, steps))
    )
    if automatic and (
        (
            not local_3090
            and not rolling_a100
            and name not in ("NVIDIA H100 80GB HBM3", "NVIDIA GeForce RTX 5090")
        )
        or prod(shape) // count < 60_000_000
        or steps < 128
    ):
        return ()
    depth = 32 if name == "NVIDIA GeForce RTX 5090" else 16
    extent = shape[axis] // count
    if extent < 2 * depth + guard:
        return ()
    window = 2 * depth
    if local_3090:
        return window, depth, 4
    for width in (64, 128):
        if extent >= 10 * width and steps >= 2 * width:
            window = max(window, width)
    if (
        name in ("NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB")
        and window >= 64
        and tile
        and device.get("multiprocessors", 0) > 0
        and _frontier_tiles(shape, axis, tile) >= 6 * device["multiprocessors"]
    ):
        # Broad slabs already fill the GPU. Further widening the halo increases
        # redundant work and storage without exposing more concurrent tiles.
        return 64, depth, 8
    slots = 4 if depth == 32 else (8 if window == 128 else 6)
    return window, depth, slots


def _frontier_tiles(shape, axis, tile):
    return prod(
        (n + t - 1) // t
        for k, (n, t) in enumerate(zip(shape, tile, strict=True))
        if k != axis
    )


def _halo_guard(module):
    return 2 if getattr(module, "compiled_transparent_halos", False) else 14


def _minimum_temporal_slabs(shape, axis, count, window, guard=14):
    """Fit staggered local fields, including both exchange halos, in int32."""
    width = shape[axis]
    face = prod(n + 1 for k, n in enumerate(shape) if k != axis)
    span = np.iinfo(np.int32).max // face - 1
    halo = window + guard
    if (width + count - 1) // count + min(count - 1, 2) * halo <= span:
        return 1
    owned = span - 2 * halo
    if owned < halo:
        raise _Unsupported("CUDA partition cross-section exceeds the local index range")
    return max(2, (width + count * owned - 1) // (count * owned))


def _split_underfilled_frontier(shape, axis, count, depth, device, tile):
    """A second resident slab supplies independent tiles during narrow wavefronts."""
    if (
        device.get("name") not in ("NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 3090")
        or not tile
    ):
        return False
    return (
        (count == 1 or device["name"] == "NVIDIA GeForce RTX 5090")
        and _frontier_tiles(shape, axis, tile) <= device.get("multiprocessors", 0)
        and shape[axis] // (2 * count) >= 16 * depth
        and prod(shape) // count >= 60_000_000
    )


def _try_sharded_run(
    simulation,
    sharding,
    *,
    backend,
    schedule,
    termination,
    wall_start,
    progress,
    store_full_materials,
    performance,
    segment=None,
    state=None,
):
    """Admit physical multi-GPU runs while preserving explicit legacy policies."""
    from beamz.simulation import backend as backends
    from beamz.simulation.cuda.mstep.schedule import normalize_schedule_request

    config = normalize_sharding_config(sharding)
    name = backends.normalize_backend(backend)
    if not config.enabled or config.backend not in (None, "gpu", "cuda"):
        return None, config.enabled
    if (
        name == "jax"
        or (name == "auto" and getenv("JAX_PLATFORMS") == "cpu")
        or not simulation.is_3d
    ):
        return None, True
    try:
        module = backends._load_extension()
    except (ImportError, OSError, RuntimeError):
        return None, config.num_devices != 1
    if not hasattr(module, "_device_count"):
        return None, config.num_devices != 1
    try:
        backends._require_cuda_abi(module)
    except backends.CudaBackendUnavailable:
        if name == "auto":
            return None, True
        raise
    try:
        device_count = module._device_count()
    except RuntimeError:
        return None, config.num_devices != 1
    count = device_count if config.num_devices is None else int(config.num_devices)
    if count < 1 or (count == 1 and device_count == 0):
        return None, False
    if count > device_count:
        raise ValueError(
            f"Requested {count} sharding devices, but only {device_count} CUDA devices are available"
        )
    normalized, _ = normalize_schedule_request(name, schedule, 1)
    if backends._has_cuda_legacy_options(native_flags=CUDA_GRAPH_CACHE):
        return None, count > 1
    capacity = getenv("BEAMZ_CUDA_MEMORY_POLICY", "auto").strip().lower() == "capacity"
    if capacity and normalized == "mstep":
        raise backends.CudaBackendUnavailable(
            "Temporal CUDA requires extra workspace; memory policy 'capacity' requires ordinary execution"
        )
    shape = simulation.grid.shape_zyx
    ordinary_axis = _native_shard_axis(config.axis, shape, count)
    temporal_axis = (
        max(range(3), key=shape.__getitem__) if config.axis == "auto" else ordinary_axis
    )
    temporal = ()
    device = {}
    if (
        normalized != "streamed"
        and not capacity
        and hasattr(module, "_has_direct_exchange")
        and hasattr(getattr(module, "NativeSession", None), "_run_temporal_partitions")
    ):
        device = module.device_info(0)
        temporal = _temporal_shard_choice(
            shape,
            temporal_axis,
            count,
            simulation.num_steps,
            device,
            automatic=normalized == "auto",
            tile=getattr(module, "compiled_owned_zyx", ()),
            guard=_halo_guard(module),
        )
        if temporal and termination is not None:
            window, depth, slots = temporal
            # Checks occur at the requested physical steps. Fit a complete
            # two-macro exchange between checks whenever the cadence allows it.
            while depth > 4 and 2 * depth > termination.chunk_steps:
                depth //= 2
            window = min(
                window,
                max(2 * depth, termination.chunk_steps // (2 * depth) * (2 * depth)),
            )
            temporal = window, depth, slots
    devices = tuple(range(count))
    options: dict[str, Any] = dict(
        progress=progress,
        store_full_materials=store_full_materials,
        performance=performance,
        termination=termination,
        wall_start=wall_start,
    )
    if temporal:
        window, depth, slots = temporal
        per_gpu = _minimum_temporal_slabs(
            shape, temporal_axis, count, window, _halo_guard(module)
        )
        minimum = tuple(ordinal for ordinal in devices for _ in range(per_gpu))
        placements: list[tuple[tuple[int, ...], tuple[int, int, int]]] = (
            [(minimum, temporal)] if len(minimum) > 1 else []
        )
        if per_gpu == 1 and _split_underfilled_frontier(
            shape,
            temporal_axis,
            count,
            depth,
            device,
            getattr(module, "compiled_owned_zyx", ()),
        ):
            paired = tuple(ordinal for ordinal in devices for _ in range(2))
            placements.insert(0, (paired, (2 * depth, depth, slots)))
        for placement, choice in placements:
            try:
                return _run_shards(
                    simulation,
                    module,
                    placement,
                    temporal_axis,
                    temporal=choice,
                    minimum_band=(
                        64
                        if normalized == "auto"
                        and device.get("name") == "NVIDIA A100-SXM4-80GB"
                        else 0
                    ),
                    segment=segment,
                    state=state,
                    **options,
                ), False
            except _Unsupported:
                break
            except backends.CudaBackendUnavailable:
                if name != "auto" or normalized != "auto":
                    raise
                return None, True
            except MemoryError:
                if placement == minimum and (per_gpu > 1 or normalized == "mstep"):
                    raise
    if count == 1:
        return None, False
    if normalized == "mstep" or segment is not None:
        return None, True
    try:
        result = _run_shards(simulation, module, devices, ordinary_axis, **options)
    except (backends.CudaBackendUnavailable, _Unsupported):
        return None, True
    return result, False
