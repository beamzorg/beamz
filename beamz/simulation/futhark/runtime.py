"""Lower complete 3D runs to the optional Futhark-generated JAX FFI target.

The native library is built by ``futhark/build.py``. One FFI call advances the
whole run: Futhark wraps XLA's device buffers without copying, loops over every
timestep on the GPU, and the handler copies the final state back into the
aliased XLA outputs.
"""

from __future__ import annotations

import ctypes
import os
from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from beamz.simulation.cuda.runtime import (
    _metallic_edge_mask,
    _metric_kind_code,
    _phase_metrics,
)
from beamz.simulation.model import SimulationState

TARGET = "beamz_futhark_program"
_SYMBOL = "BeamzFutharkProgram"
_LIBRARY_ENV = "BEAMZ_FUTHARK_LIBRARY"
_CPU_ENV = "BEAMZ_FUTHARK_CPU"
_NATIVE = Path(__file__).resolve().parent / "_native"
_SOURCE_GROUP_COUNT = 9
_TERM_COUNT = 6
# Derivative axis of each CPML term, in the fixed 3D curl order shared with
# cuda/src/yee_primitives.cuh (CpmlAxis).
_TERM_AXES = (1, 0, 0, 2, 2, 1)


class FutharkBackendUnavailable(RuntimeError):
    """The explicitly requested Futhark backend cannot run in this process."""


def gpu_platform() -> str | None:
    """Return "cuda" or "hip" for JAX's default GPU, or None without one."""
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        return None
    version = str(getattr(devices[0].client, "platform_version", "")).lower()
    return "hip" if "rocm" in version or "hip" in version else "cuda"


def futhark_platform() -> str:
    """The native build for JAX's default device; on CPU, BEAMZ_FUTHARK_CPU
    picks "multicore" (default), "ispc" or sequential "c"."""
    return gpu_platform() or os.environ.get(_CPU_ENV, "multicore")


def library_path(platform: str = "cuda") -> Path:
    return Path(
        os.environ.get(_LIBRARY_ENV) or _NATIVE / f"libbeamz_futhark_{platform}.so"
    )


@cache
def _register() -> str:
    platform = futhark_platform()
    path = library_path(platform)
    if not path.exists():
        raise FutharkBackendUnavailable(
            f"{path} does not exist; build it with "
            f"`python futhark/build.py --platform {platform}`"
        )
    library = ctypes.cdll.LoadLibrary(str(path))
    jax.ffi.register_ffi_target(
        TARGET,
        jax.ffi.pycapsule(getattr(library, _SYMBOL)),
        platform={"hip": "ROCM", "cuda": "CUDA"}.get(platform, "cpu"),
    )
    return str(path)


def futhark_backend_status() -> str | None:
    """Register the target once; return None when usable, else the reason."""
    try:
        _register()
    except (OSError, AttributeError, RuntimeError) as exc:
        return str(exc)
    return None


def _volume(value) -> jax.Array:
    """Coefficients may be elided to a scalar; Futhark broadcasts unit axes."""
    value = jnp.asarray(value, dtype=jnp.float32)
    if value.ndim == 0:
        return value.reshape((1, 1, 1))
    if value.ndim != 3 or 0 in value.shape:
        raise FutharkBackendUnavailable(
            "Futhark requires dense or scalar 3D update coefficients"
        )
    return value


def _vector(value) -> jax.Array:
    return jnp.asarray(value, dtype=jnp.float32).reshape((-1,))


def _constrained(shape, component: int, phase: int, edges: int, z, y, x):
    """Vectorised SourceCellConstrained: cells a PEC wall pins to zero."""
    coordinates = (z, y, x)
    normal = 2 - component

    def hit(axis):
        coordinate = coordinates[axis]
        low = bool(edges & (1 << (2 * axis))) & (coordinate == 0)
        high = bool(edges & (1 << (2 * axis + 1))) & (coordinate == shape[axis] - 1)
        return low | high

    if phase == 0:
        return hit(normal)
    return np.logical_or.reduce([hit(axis) for axis in range(3) if axis != normal])


def storage_shape(field_shapes) -> tuple[int, int, int]:
    """The zero-padded bounding shape every component occupies during a run."""
    return tuple(int(max(shape[axis] for shape in field_shapes)) for axis in range(3))


def source_table(groups, field_shapes, edges: int, storage=None):
    """Flatten the nine (timing, component) slab groups into one scatter table.

    Targets index ``storage`` (default: each component's own shape). Returns
    per-cell flat targets (-1 when dropped), amplitudes, waveform
    offsets and lengths, group indices and the concatenated waveforms. Out-of-field cells are dropped and post-update sources skip
    PEC-constrained cells, exactly as the CUDA source kernels do.
    """
    if len(groups) != _SOURCE_GROUP_COUNT:
        raise ValueError("Futhark source table requires nine phase/component groups")
    targets, amplitudes, offsets, lengths, group_ids, waves = [], [], [], [], [], []
    wave_offset = 0
    for index, group in enumerate(groups):
        timing, component = divmod(index, 3)
        if group is not None:
            # Timing 1 targets H; timings 0 and 2 target E.
            shape = field_shapes[(3 if timing == 1 else 0) + component]
            coeffs = np.asarray(group.coeffs, dtype=np.float32)
            waveforms = np.asarray(group.waveforms, dtype=np.float32)
            starts = np.asarray(group.starts, dtype=np.int64)
            local = np.indices(coeffs.shape[1:]).reshape(3, -1)
            for source in range(coeffs.shape[0]):
                z, y, x = (local[axis] + starts[source, axis] for axis in range(3))
                inside = (
                    (z >= 0)
                    & (y >= 0)
                    & (x >= 0)
                    & (z < shape[0])
                    & (y < shape[1])
                    & (x < shape[2])
                )
                if timing:
                    inside &= ~_constrained(
                        shape, component, 0 if timing == 1 else 1, edges, z, y, x
                    )
                stride = shape if storage is None else storage
                flat = np.where(inside, (z * stride[1] + y) * stride[2] + x, -1)
                targets.append(flat)
                amplitudes.append(coeffs[source].reshape(-1))
                offsets.append(np.full(flat.shape, wave_offset))
                lengths.append(np.full(flat.shape, waveforms.shape[1]))
                group_ids.append(np.full(flat.shape, index))
                waves.append(waveforms[source])
                wave_offset += waveforms.shape[1]
    if wave_offset >= np.iinfo(np.int32).max or any(
        int(np.prod(shape)) >= np.iinfo(np.int32).max
        for shape in (*field_shapes, *(() if storage is None else (storage,)))
    ):
        raise FutharkBackendUnavailable("Futhark source indices exceed int32")

    def stack(values, dtype):
        return np.concatenate(values).astype(dtype) if values else np.zeros(0, dtype)

    return (
        stack(targets, np.int32),
        stack(amplitudes, np.float32),
        stack(offsets, np.int32),
        stack(lengths, np.int32),
        stack(group_ids, np.int32),
        stack(waves, np.float32),
    )


def _cpml_arguments(state, ctx):
    """Slab bounds, padded 1D profiles and the twelve recurrence buffers."""
    cpml = ctx.boundary.cpml
    if not cpml.enabled:
        slabs = np.zeros((2 * _TERM_COUNT, 2), dtype=np.int32)
        profile = jnp.zeros((2 * _TERM_COUNT, 1), dtype=jnp.float32)
        empty = jnp.zeros((0, 0, 0), dtype=jnp.float32)
        return (slabs, profile, profile, profile), (empty,) * (2 * _TERM_COUNT)
    terms = (*cpml.h_terms, *cpml.e_terms)
    psi = (*state.cpml_psi_h_terms, *state.cpml_psi_e_terms)
    if len(terms) != 2 * _TERM_COUNT or len(psi) != 2 * _TERM_COUNT:
        raise FutharkBackendUnavailable("3D Futhark CPML requires six terms per phase")
    for index, term in enumerate(terms):
        canonical = index % _TERM_COUNT
        if (
            term.component[1] != "xyz"[canonical // 2]
            or int(term.axis) != _TERM_AXES[canonical]
            or (float(term.sign) > 0) != (canonical % 2 == 0)
        ):
            raise FutharkBackendUnavailable(
                "CPML terms are not in the canonical 3D curl order"
            )
    if any(value.dtype != jnp.float32 for value in psi):
        raise FutharkBackendUnavailable("Futhark CPML recurrences must be float32")
    length = max(1, max(int(np.size(term.a)) for term in terms))

    def profiles(name):
        return jnp.stack(
            [
                jnp.pad(_vector(getattr(term, name)), (0, length - np.size(term.a)))
                for term in terms
            ]
        )

    slabs = np.asarray(
        [(int(term.slab.low), int(term.slab.high)) for term in terms], dtype=np.int32
    )
    return (slabs, profiles("a"), profiles("b"), profiles("inv_kappa")), psi


def _storage_indices(indices, shapes, storage):
    """Rebase [M][6][P][N] plan offsets from component shapes onto storage."""
    rebased = []
    for component, shape in enumerate(shapes):
        flat = indices[:, component]
        z, rest = jnp.divmod(flat, shape[1] * shape[2])
        y, x = jnp.divmod(rest, shape[2])
        valid = (flat >= 0) & (flat < int(np.prod(shape)))
        rebased.append(
            jnp.where(valid, (z * storage[1] + y) * storage[2] + x, -1).astype(
                jnp.int32
            )
        )
    return jnp.stack(rebased, axis=1)


def _empty_monitors():
    return (
        jnp.zeros((0, 6, 1, 1), dtype=jnp.int32),
        jnp.zeros((0, 6, 1, 1), dtype=jnp.float32),
        jnp.zeros((0, 1), dtype=jnp.float32),
        jnp.zeros((0, 6), dtype=jnp.float32),
        jnp.zeros((0, 5), dtype=jnp.int32),
        jnp.zeros((0, 2), dtype=jnp.int32),
        jnp.zeros((0, 3), dtype=jnp.float32),
    )


def run_program(
    state,
    ctx,
    coeffs,
    groups,
    packed_monitors,
    nsteps: int,
    *,
    observation_origin=None,
    observation_step_offset: int | jax.Array = 0,
) -> SimulationState:
    """Advance sources, fields, CPML memory and vector DFTs in one native call."""
    if nsteps < 1:
        raise ValueError("Futhark step count must be positive")
    _register()
    fields = (state.hx, state.hy, state.hz, state.ex, state.ey, state.ez)
    if any(value.dtype != jnp.float32 or value.ndim != 3 for value in fields):
        raise FutharkBackendUnavailable("Futhark requires float32 3D fields")
    edges = _metallic_edge_mask(ctx.boundary.cpml.metallic_edges)
    e_fields = (state.ex, state.ey, state.ez)
    h_fields = (state.hx, state.hy, state.hz)
    shapes = tuple(value.shape for value in (*e_fields, *h_fields))
    storage = storage_shape(shapes)
    sources = source_table(groups, shapes, edges, storage)
    (slabs, cpml_a, cpml_b, cpml_k), psi = _cpml_arguments(state, ctx)
    if packed_monitors is None:
        monitors = _empty_monitors()
    else:
        # Neighbor-major gather plans give coalesced reads in the DFT kernel.
        indices, weights, *rest = packed_monitors
        monitors = (
            jnp.swapaxes(_storage_indices(indices, shapes, storage), 2, 3),
            jnp.swapaxes(weights, 2, 3),
            *rest,
        )
    accumulators = tuple(
        jnp.asarray(value, dtype=jnp.float32).reshape((-1,))
        for value in (state.dft_vec_re, state.dft_vec_im, state.dft_weight_sum)
    )
    if packed_monitors is None:
        accumulators = tuple(jnp.zeros((0,), jnp.float32) for _ in accumulators)
    e_values = [
        getattr(coeffs, f"e_{kind}_{axis}")
        for kind in ("decay", "source")
        for axis in "xyz"
    ]
    if all(jnp.ndim(value) == 1 for value in e_values) and all(
        jnp.asarray(value).dtype == jnp.int32 for value in e_values[3:]
    ):
        # compile.py packed a lossless E update into codebooks (in the decay
        # slots) and four 8-bit cell codes per int32 word (in the source slots).
        tables = tuple(jnp.asarray(value, jnp.float32) for value in e_values[:3])
        codes = tuple(jnp.asarray(value, jnp.int32) for value in e_values[3:])
        e_values = [jnp.ones((1, 1, 1), jnp.float32)] * 3 + [
            jnp.zeros((1, 1, 1), jnp.float32)
        ] * 3
    else:
        tables = (jnp.zeros((0,), jnp.float32),) * 3
        codes = (jnp.zeros((0,), jnp.int32),) * 3
    coefficients = (
        *(
            _volume(getattr(coeffs, name))
            for name in (
                "h_decay_x", "h_decay_y", "h_decay_z",
                "h_source_x", "h_source_y", "h_source_z",
            )
        ),
        *(_volume(value) for value in e_values),
        *tables,
        *codes,
    )  # fmt: skip
    metrics = tuple(
        _vector(value) for phase in (0, 1) for value in _phase_metrics(ctx, phase)
    )
    origin = state.t if observation_origin is None else observation_origin
    clocks = (
        jnp.asarray(state.current_step, dtype=jnp.int32).reshape((1,)),
        jnp.asarray(origin, dtype=jnp.float32).reshape((1,)),
        jnp.asarray(observation_step_offset, dtype=jnp.int32).reshape((1,)),
    )
    arguments = (
        *e_fields,
        *h_fields,
        *coefficients,
        *metrics,
        jnp.asarray(slabs),
        cpml_a,
        cpml_b,
        cpml_k,
        *psi,
        *(jnp.asarray(value) for value in sources),
        *monitors,
        *accumulators,
        *clocks,
    )
    state_values = (*e_fields, *h_fields, *psi, *accumulators)
    psi_start = 6 + len(coefficients) + len(metrics) + 4
    dft_start = psi_start + len(psi) + len(sources) + len(monitors)
    aliases = {index: index for index in range(6)}
    aliases.update({psi_start + index: 6 + index for index in range(len(psi))})
    aliases.update({dft_start + index: 6 + len(psi) + index for index in range(3)})
    call = jax.ffi.ffi_call(
        TARGET,
        tuple(jax.ShapeDtypeStruct(v.shape, v.dtype) for v in state_values),
        input_output_aliases=aliases,
        vmap_method="sequential",
    )
    outputs = call(
        *arguments,
        nsteps=np.int64(nsteps),
        dt=np.float32(ctx.dt),
        inv_resolution=np.float32(1) / np.float32(ctx.resolution),
        edges=np.int64(edges),
        metric_kind=np.int64(_metric_kind_code(ctx)),
    )
    ex, ey, ez, hx, hy, hz = outputs[:6]
    next_state = state._replace(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz)
    if ctx.boundary.cpml.enabled:
        next_state = next_state._replace(
            cpml_psi_h_terms=tuple(outputs[6 : 6 + _TERM_COUNT]),
            cpml_psi_e_terms=tuple(outputs[6 + _TERM_COUNT : 6 + 2 * _TERM_COUNT]),
        )
    if packed_monitors is not None:
        re, im, weight = outputs[6 + len(psi) :]
        next_state = next_state._replace(
            dft_vec_re=re.reshape(state.dft_vec_re.shape),
            dft_vec_im=im.reshape(state.dft_vec_im.shape),
            dft_weight_sum=weight.reshape(state.dft_weight_sum.shape),
        )
    return next_state
