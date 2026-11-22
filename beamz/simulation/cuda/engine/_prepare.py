"""CPU preparation and typed native-packet boundary for the CUDA backend."""
from __future__ import annotations

from math import prod
from typing import Any

import numpy as np

from ._capabilities import _METRICS, _Unsupported, _validate
from ._contracts import _CPML, _FIELDS, _YEE

_CHUNK_STEPS = 256
_RECORDING_STAGING_BYTES = 64 << 20


def _encode_diagnostic_material(material, shape=None):
    from beamz._region_array import RegionArray, flat_tiles

    if not isinstance(material, RegionArray):
        material = np.asarray(material)
        if material.size == 1 or not any(material.strides):
            return {"values": np.asarray([material.flat[0]], np.float32), "ends": np.empty(0, np.int32)}
    if shape is not None and material.shape != shape:
        material = np.broadcast_to(np.asarray(material), shape)
    if material.size > np.iinfo(np.int32).max:
        raise ValueError("Native energy diagnostics require int32 field indices")
    starts, values = [], []
    cursor = run_count = 0
    last = None
    for region in flat_tiles(material.shape, material.dtype, budget=4 << 20):
        flat = np.asarray(material[region]).reshape(-1)
        changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        if last is None or last != flat[0]:
            changes = np.r_[0, changes]
        run_count += changes.size
        if run_count * 2 >= material.size:
            starts.clear()
            values.clear()
            return {
                "values": np.asarray(material, dtype=np.float32, order="C").ravel(),
                "ends": np.empty(0, np.int32),
            }
        if changes.size:
            starts.append(changes.astype(np.int32) + cursor)
            values.append(np.asarray(flat[changes], np.float32))
        last = flat[-1]
        cursor += flat.size
    coefficients = np.concatenate(values)
    ends = (
        np.r_[np.concatenate(starts)[1:], np.int32(cursor)]
        if run_count > 1 else np.empty(0, np.int32)
    )
    return {"values": coefficients, "ends": ends}


def _diagnostic_inputs(program, partition=None, *, owned=None):
    """Lossless materials and separable Yee measures; no volume-sized weights."""
    from beamz.lattice import component_axis_offsets_3d
    from beamz.simulation.execute import _axis_integration_weights

    materials, axes = [], []
    if owned is not None and partition is None:
        raise ValueError("Owned diagnostic coordinates require a local partition")

    def region(shape):
        if owned is None:
            return selection
        # The local Yee cell interval uses the staggered high sample only
        # when the corresponding global component has it.
        stop = hi + int(shape[dimension] > program.grid.geometry.shape_zyx[dimension])
        return tuple(slice(lo, stop) if a == dimension else slice(None) for a in range(3))

    def measure(edges, offset, count, axis, crop):
        widths = _axis_integration_weights(edges, offset, count)[crop[axis]]
        if owned is not None and axis == dimension:
            widths = np.array(widths, copy=True)
            widths[:owned[0] - lo] = 0
            widths[owned[1] - lo:] = 0
        return widths

    if partition is None:
        selection = (slice(None),) * 3
    else:
        dimension, lo, hi = partition
        selection = tuple(slice(lo, hi) if i == dimension else slice(None) for i in range(3))
    for name in _FIELDS:
        component = name.capitalize()
        shape = getattr(program.grid, component).shape
        crop = region(shape)
        material = getattr(program.grid, ("eps_" if name[0] == "e" else "mu_") + name)
        if partition is not None and material.size != 1:
            if material.shape != shape:
                material = np.broadcast_to(np.asarray(material), shape)
            material = material[crop]
        materials.append(_encode_diagnostic_material(material, shape if partition is None else None))
        offsets = component_axis_offsets_3d(component)
        axes.append(tuple(
            measure(getattr(program.grid.geometry, axis + "_edges"), offsets[axis], count, index, crop)
            for index, (axis, count) in enumerate(zip("zyx", shape, strict=True))
        ))
    if getattr(program.grid.material_grid, "uses_full_permittivity", False):
        if partition is not None and owned is None:
            raise _Unsupported("Native tensor diagnostics require complete local Yee supports")
        tensor = program.grid.material_grid.yee_tensors["eps_node"]
        crop = region(tensor.shape[1:])
        node_widths = tuple(
            measure(getattr(program.grid.geometry, axis + "_edges"), 0.0, count, index, crop)
            for index, (axis, count) in enumerate(zip("zyx", tensor.shape[1:], strict=True))
        )
        for entry in tensor[3:6]:
            materials.append(_encode_diagnostic_material(np.asarray(entry[crop])))
            axes.append(node_widths)
    if owned is not None and len(materials) == 9:
        interpolation = tuple(
            _axis_integration_weights(
                getattr(program.grid.geometry, "zyx"[2 - component] + "_edges"),
                0.5,
                getattr(program.grid, "E" + "xyz"[component]).shape[2 - component],
            )[selection[2 - component]]
            for component in range(3)
        )
    else:
        interpolation = ()
    domain_measure = float(np.prod(program.grid.geometry.extent))
    # Bound device partials and kernel bookkeeping before selecting a schedule.
    required = sum(a.nbytes for spec in materials for a in spec.values()) + sum(a.nbytes for weights in axes for a in weights) + (1 << 20)
    result = materials, axes, domain_measure, required + sum(a.nbytes for a in interpolation)
    return (*result, interpolation) if owned is not None else result


def _coefficient_table(simulation, module):
    """Match the resolved setup device's FP32 division on each lossless material table."""
    if getattr(simulation, "setup_device_resolved", None) == "cpu":
        from beamz.simulation.region_materials import lossless_electric_source

        return lossless_electric_source
    return module.electric_coefficient_table


def _compile(simulation: Any = "", module: Any = "", *, steps: int = 0, memory_budget: int = -1) -> Any:
    import jax

    from beamz.simulation.backend import CudaBackendUnavailable, _require_cuda_abi
    from beamz.simulation.compile import compile_program

    _require_cuda_abi(module)
    if (
        not simulation.is_3d
        or simulation.grid.metric_kind
        not in _METRICS
        or (
            simulation.grid.metric_kind != "isotropic_uniform"
            and not getattr(module, "compiled_rectilinear", False)
        )
    ):
        raise _Unsupported("A metric-aware module is required for nonuniform 3D grids")
    # The module's native constructor validates CUDA availability and the full
    # packet before execution. This is preparation only, with no FFI execution.
    exact_coefficients = getattr(module, "compiled_electric_coefficient_math", "") == "cuda_fast_divide_f32"
    try:
        program = compile_program(
            simulation,
            num_steps=steps,
            backend="cuda_streamed",
            progress=False,
            backend_resolver=lambda requested: "cuda_streamed",
            native_preparation=exact_coefficients,
            setup_context_factory=lambda resolved: jax.default_device(
                jax.devices("cpu")[0]
            ),
        )
    except CudaBackendUnavailable as error:
        # Capability rejection occurs before native allocation or launch. Let the
        # established frontend preserve auto/JAX and explicit-CUDA error policy.
        raise _Unsupported(str(error)) from error
    _validate(program, native_effects=hasattr(getattr(module, "NativeSession", None), "configure_effects"))
    if 0 <= memory_budget <= (512 << 20):
        raise MemoryError("CUDA memory budget cannot fit preparation and context reserve")
    if exact_coefficients and not program.coefficients.e_inverse_offdiagonal.size:
        program = _lower_coefficients(program, module, _coefficient_table(simulation, module))
    return program


def _pack(arguments: tuple[Any, ...] = (), fields: tuple[Any, ...] = (), *, cpml: bool = True) -> tuple[Any, ...]:
    from beamz.simulation.cuda.storage.material_blocks import pack_material_blocks

    indices = (_CPML if cpml else _YEE).inputs
    values = list(arguments)
    sources = [np.asarray(values[indices["e_source_x"] + c]) for c in range(3)]
    if all(v.dtype == np.int32 and v.ndim in (1, 2) for v in sources):
        return arguments
    lossless = True
    for c, source in enumerate(sources):
        decay = np.asarray(values[indices["e_decay_x"] + c])
        if (
            source.dtype != np.float32
            or not np.isfinite(source).all()
            or decay.dtype != np.float32
            or not np.isfinite(decay).all()
        ):
            raise ValueError(
                "Native CUDA requires finite FP32 electric coefficients"
            )
        lossless &= bool(np.all(decay == np.float32(1)))
    if not lossless:
        return arguments
    if all(v.ndim == 0 for v in sources):
        pool = np.asarray([v.item() for v in sources], np.float32)
        tables = (pool,) * 3
        codes = tuple(
            np.full((1, (int(np.prod(fields[len(_FIELDS)//2 + c].shape)) + 31) // 32), ~c, np.int32)
            for c in range(3)
        )
    else:
        dense = tuple(
            np.broadcast_to(v, fields[len(_FIELDS)//2 + c].shape) for c, v in enumerate(sources)
        )
        packed = pack_material_blocks(dense, max_bytes=2**63 - 1)
        if packed is None:
            raise ValueError(
                "Cannot losslessly encode rectilinear electric coefficients"
            )
        tables, codes = packed
    values[indices["e_decay_x"]:indices["e_source_x"]] = tables
    values[indices["e_source_x"]:indices["e_term_metadata"]] = codes
    return tuple(values)


def _source_scatter(spec, shape, timing, component, *, index_dtype: type[np.int32] | type[np.int64] = np.int32):
    """Flatten NumPy-style indices without constructing a dense coordinate grid."""
    if prod(shape) > np.iinfo(index_dtype).max:
        raise _Unsupported("Native sparse sources exceed the field index range")
    flat = np.asarray(0, dtype=index_dtype)
    for axis, extent in enumerate(shape):
        dims = [1, 1, 1]
        dims[axis] = extent
        coords = np.broadcast_to(np.arange(extent, dtype=index_dtype).reshape(dims), shape)
        try:
            local = coords[spec.index]
        except (IndexError, TypeError) as error:
            raise _Unsupported("Source index requires the general scatter binding") from error
        flat = flat + local * index_dtype(prod(shape[axis + 1 :]))
    coefficients = np.broadcast_to(np.asarray(spec.coeff, np.float32), flat.shape)
    flat, coefficients = flat.reshape(-1), coefficients.reshape(-1)
    if not flat.size:
        return None
    if flat.size > np.iinfo(np.int32).max:
        raise _Unsupported("Native sparse sources require int32 contribution offsets")
    order = np.argsort(flat, kind="stable")
    flat = flat[order]
    changed = np.r_[True, flat[1:] != flat[:-1]]
    ends = np.r_[np.flatnonzero(changed)[1:], flat.size].astype(np.int32) if not np.all(changed) else np.empty(0, np.int32)
    return (
        timing, component, np.ascontiguousarray(flat[changed]), ends,
        np.ascontiguousarray(coefficients[order]), np.ascontiguousarray(spec.waveform, dtype=np.float32),
    )


def _recorder_inputs(program, state, *, minimum_chunk=1):
    recorders = []
    for spec in program.monitors:
        if spec.recorder_index < 0:
            continue
        for component, sign, index, indices, weights in zip(
            spec.canonical_components, spec.component_signs, spec.field_buffer_indices,
            spec.field_interp_flat_idx, spec.field_interp_weights, strict=True,
        ):
            history = state.recorded_fields[index]
            indices = np.asarray(indices)
            if np.max(indices, initial=0) > np.iinfo(np.int32).max:
                raise _Unsupported("Native recorder gather requires int32 field indices")
            recorders.append((
                _FIELDS.index(component.lower()), int(spec.record_interval), sign,
                np.ascontiguousarray(indices, dtype=np.int32),
                np.ascontiguousarray(weights, dtype=np.float32),
                history.reshape(history.shape[0], prod(history.shape[1:])),
            ))

    def scratch_bytes(steps):
        return sum(
            min(history.shape[0], (steps + interval - 1) // interval)
            * history.shape[1] * history.itemsize
            for _, interval, _, _, _, history in recorders
        )

    chunk = _CHUNK_STEPS
    cap = max(_RECORDING_STAGING_BYTES, scratch_bytes(minimum_chunk))
    while scratch_bytes(chunk) > cap:
        chunk //= 2
    bytes_ = scratch_bytes(chunk) + sum(i.nbytes + w.nbytes for _, _, _, i, w, _ in recorders)
    return tuple(recorders), chunk, bytes_


def _capture(
    program: Any = "", state: Any = "", steps: int = 0, *, native_effects: bool = False,
    clamp_source_tails: bool = False, ragged_gather: bool = False,
    minimum_recorder_chunk: int = 1,
) -> dict[str, Any]:
    from beamz.simulation import _cuda_abi as abi
    from beamz.simulation.cuda import pack_dft_monitors
    from beamz.simulation.cuda import runtime as rt
    from beamz.simulation.execute import SOURCE_PHASE_COMPONENTS, build_step_context

    ctx = build_step_context(program, low_memory=True)
    layout = _CPML if ctx.boundary.cpml.enabled else _YEE
    _validate(program, ctx, native_effects=native_effects)
    if state.polarization: raise _Unsupported("Native compiler does not support dispersive state")
    groups, scatters = [], []
    for timing, (phase, components) in enumerate(SOURCE_PHASE_COMPONENTS.items()):
        for component, name in enumerate(components):
            group, rest = ctx.source_batches[(phase, name)]
            groups.append(group)
            for source in rest:
                scatter = _source_scatter(source, getattr(state, name.lower()).shape, timing, component)
                if scatter is not None:
                    scatters.append(scatter)
    spectral = tuple(
        m for m in program.monitors if m.dft_enabled and m.freq_count and m.dft_point_count
    )
    if spectral and state.dft_vec_re.dtype != np.float32:
        raise _Unsupported("Native vector DFT acquisition requires FP32 accumulators")
    monitors = pack_dft_monitors(spectral, host=True, ragged=ragged_gather) if spectral else None
    coeffs, tensor = program.coefficients, ()
    if coeffs.e_inverse_offdiagonal.size:
        from beamz.simulation.cuda.storage.tensor import coefficients, payload

        tensor = payload(
            ctx, coeffs,
            scratch_factory=lambda shape, dtype: np.broadcast_to(np.zeros((), dtype), shape),
        )
        coeffs = coefficients(ctx, coeffs)
    arguments, values, aliases = rt._graph_io(state, ctx, coeffs)
    fields = rt._fields(state)
    if program.config.metric_kind != "isotropic_uniform" and not tensor:
        arguments = _pack(arguments, fields, cpml=ctx.boundary.cpml.enabled)
    virtual = tuple(np.broadcast_to(np.zeros((), f.dtype), f.shape) for f in fields)
    virtual += tuple(np.empty((0,), np.float32) for _ in fields)
    for value in virtual:
        aliases[len(arguments)] = len(values)
        arguments, values = (*arguments, value), (*values, value)
    source_arguments = list(rt._source_group_arguments(groups))
    if clamp_source_tails:
        for index in range(1, len(source_arguments), 3):
            waveforms = np.ascontiguousarray(source_arguments[index])
            if waveforms.size and waveforms.dtype == np.float32:
                bits = waveforms.view(np.uint32)
                changes = np.flatnonzero(np.any(bits != bits[:, -1:], axis=0))
                length = int(changes[-1]) + 2 if changes.size else 1
                source_arguments[index] = np.ascontiguousarray(waveforms[:, :length])
    arguments = (*arguments, *source_arguments)
    arguments, values, aliases, _, count = rt._monitor_io(
        state,
        monitors,
        arguments,
        values,
        aliases,
        observation_origin=state.t,
        observation_step_offset=np.int32(0),
        host=True,
    )
    # Moving reconstructs its own ownership plan. Generated global execution
    # needs no plan. The five reserved ABI inputs are never consumed by either.
    arguments = (*arguments, *(np.empty((0,), np.int32) for _ in range(5)))
    flags = rt._native_schedule_flags(
        ctx, coeffs, sources=True, monitors=count
    )
    if np.asarray(arguments[layout.inputs["e_source_x"]]).dtype == np.int32:
        flags |= abi.NATIVE_SCHEDULE_PACKED_MATERIAL
    schedule = rt.NativeSchedulePlan(
        layout=abi.PROGRAM_LAYOUT_YEE_IN_PLACE,
        flags=flags,
        monitor_count=count,
        coincident_source_group_mask=rt._coincident_source_group_mask(groups),
        disjoint_source_group_mask=rt._disjoint_source_group_mask(tuple(groups)),
        temporal_steps=2,
    )
    if len(arguments) != layout.input_count or len(values) != layout.output_count:
        raise ValueError("Unexpected shared native packet ABI")
    recorders, chunk, recording_bytes = _recorder_inputs(
        program, state, minimum_chunk=minimum_recorder_chunk,
    )
    effects_bytes = (
        recording_bytes + sum(v.nbytes for v in tensor)
        + sum(a.nbytes for _, _, *parts in scatters for a in parts)
    )
    attributes = rt._program_attributes(ctx, steps, schedule)
    if clamp_source_tails:
        attributes["clamp_source_tails"] = True
    return dict(
        arguments=arguments,
        result_values=values,
        aliases=aliases,
        attributes=attributes,
        legacy_mstep_plan=False,
        effects=(tensor, tuple(scatters), recorders),
        effects_bytes=effects_bytes,
        chunk_steps=chunk,
    )


def _lower_coefficients(program, module, coefficient_table):
    from dataclasses import replace

    import numpy as np

    from beamz.const import EPS_0
    from beamz.design.discretization import _allclose_bounded
    from beamz.simulation.compile import _elide_uniform_grid
    from beamz.simulation.cuda.storage.material_blocks import pack_material_blocks

    epsilon = []; lossless = True
    for component, axis in zip(("Ex", "Ey", "Ez"), "xyz", strict=True):
        shape = getattr(program.grid, component).shape
        region = getattr(program.grid, "region_" + axis)
        if any(s.indices(n) != (0, n, 1) for s, n in zip(region, shape, strict=True)):
            raise _Unsupported("CUDA coefficient lowering requires complete Yee supports")
        sigma = getattr(program.grid, "sig_" + axis)
        lossless &= _allclose_bounded(sigma, 0, rtol=0, atol=0)
        values = getattr(program.grid, "eps_" + axis)
        if values.dtype != np.float32:
            raise _Unsupported(
                "CUDA coefficient lowering requires explicit FP32 Yee permittivity"
            )
        epsilon.append(values if values.shape == shape else np.broadcast_to(values, shape))
    if not lossless:
        if not getattr(module,"compiled_general_ordinary",False):
            raise _Unsupported("Installed coefficient lowering requires lossless materials")
        changes = {f"e_{kind}_{axis}": _elide_uniform_grid(getattr(program.coefficients, f"e_{kind}_{axis}")) for kind in ("decay", "source") for axis in "xyz"}
        return replace(program, coefficients=program.coefficients._replace(**changes))
    packed = pack_material_blocks(tuple(epsilon), max_bytes=2**63 - 1, host=True)
    if packed is None:
        raise ValueError("Cannot losslessly encode the epsilon tables")
    tables, headers = packed
    lowered = {}
    changes = {}
    for axis, table, codes in zip("xyz", tables, headers, strict=True):
        key = id(table)
        if key not in lowered:
            lowered[key] = coefficient_table(
                np.ascontiguousarray(table),
                np.float32(program.config.dt),
                np.float32(EPS_0),
            )
            if not np.isfinite(lowered[key]).all():
                raise ValueError("Electric coefficients must remain finite FP32")
            lowered[key].setflags(write=False)
        changes["e_decay_" + axis] = lowered[key]
        changes["e_source_" + axis] = codes
    return replace(program, coefficients=program.coefficients._replace(**changes))
