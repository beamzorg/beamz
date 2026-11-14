"""Shared halo exchange and scan-local CPML ownership for distributed backends."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from beamz.simulation.model import (
    ShardingConfig,
    ShardingLayout,
    ShardingPlan,
    ShardingToken,
)
from beamz.simulation.results.observe import is_vector_dft_monitor

_MESH_AXIS = "fdtd"


def _shard_map():
    try:
        return jax.shard_map
    except AttributeError:
        from jax.experimental.shard_map import shard_map

        return shard_map


def exchange_faces(value, *, axis, num_devices, lower=True, upper=True):
    """Exchange adjacent one-cell faces inside a manual ``fdtd`` mesh.

    The outer faces are zero, never periodic. Only source buffers have halos;
    the native phase returns owned cells in the original partition layout.
    """
    low = jax.lax.slice_in_dim(value, 0, 1, axis=axis)
    high = jax.lax.slice_in_dim(
        value, value.shape[axis] - 1, value.shape[axis], axis=axis
    )
    from_low = (
        jax.lax.ppermute(high, _MESH_AXIS, [(i, i + 1) for i in range(num_devices - 1)])
        if lower
        else jnp.zeros_like(high)
    )
    from_high = (
        jax.lax.ppermute(low, _MESH_AXIS, [(i + 1, i) for i in range(num_devices - 1)])
        if upper
        else jnp.zeros_like(low)
    )
    return from_low, from_high


def exchange_halos(value, *, axis, num_devices, lower=True, upper=True):
    """Attach exchanged faces for the pure-JAX stencil representation."""
    low, high = exchange_faces(
        value, axis=axis, num_devices=num_devices, lower=lower, upper=upper
    )
    return jnp.concatenate((low, value, high), axis=axis)


def _owned_psi(value, term, *, axis, origin, extent, logical_shape):
    """Keep only this rank's physical recurrence entries; neutralize padding."""
    mask = jnp.asarray(True)
    for d, size in enumerate(value.shape):
        positions = jnp.arange(size, dtype=jnp.int32)
        if d == term.axis:
            positions = cast(
                jax.Array,
                jnp.where(
                    positions < term.slab.low,
                    positions,
                    positions - term.slab.low + logical_shape[d] - term.slab.high,
                ),
            )
        elif d == axis:
            positions = positions + origin
        valid = (positions >= 0) & (positions < logical_shape[d])
        if d == axis and d == term.axis:
            valid &= (positions >= origin) & (positions < origin + extent)
        shape = [1, 1, 1]
        shape[d] = size
        mask = mask & valid.reshape(shape)
    return cast(jax.Array, jnp.where(mask, value, jnp.zeros_like(value)))


def scan_local_cpml(state, program, *, assemble=False):
    """Keep normal CPML recurrences on their owners between scan boundaries.

    A leading mesh axis stores one packed slab per rank. The physical entries
    are disjoint; only the scan output reduces them into public continuation
    arrays. This is internal storage, never a change to SimulationState's API.
    """
    plan = program.sharding
    axis = plan.layout.axis
    shard_map = _shard_map()

    updates = {}
    for phase in ("h", "e"):
        name = f"cpml_psi_{phase}_terms"
        values = []
        for value, term in zip(
            getattr(state, name),
            getattr(program.boundary.cpml, f"{phase}_terms"),
            strict=True,
        ):
            if term.axis != axis:
                values.append(value)
                continue
            extent = (
                plan.layout.padded_shapes[term.component][axis]
                // plan.layout.num_devices
            )
            logical_shape = plan.layout.logical_shapes[term.component]

            def distribute(
                value, term=term, extent=extent, logical_shape=logical_shape
            ):
                owned = _owned_psi(
                    value,
                    term,
                    axis=axis,
                    origin=jax.lax.axis_index(_MESH_AXIS) * extent,
                    extent=extent,
                    logical_shape=logical_shape,
                )
                return owned[None, ...]

            def collect(value):
                return jax.lax.psum(value[0], _MESH_AXIS)

            convert = shard_map(
                collect if assemble else distribute,
                mesh=plan.mesh,
                in_specs=P(_MESH_AXIS) if assemble else P(),
                out_specs=P() if assemble else P(_MESH_AXIS),
            )
            values.append(convert(value))
        updates[name] = tuple(values)
    return state._replace(**updates)


def _clamped_starts(value, group):
    return tuple(
        tuple(
            min(
                max(start if start >= 0 else start + value.shape[d], 0),
                value.shape[d] - group.max_sizes[d],
            )
            for d, start in enumerate(source_start)
        )
        for source_start in group.starts_tuple
    )


def requires_local_injection(value, group, plan):
    """Static, single-owner scatters already lower well; localize crossing slabs.

    Larger source batches otherwise use dynamic global read/modify/write. Both
    those and patches crossing a partition can introduce non-stencil gathers.
    """
    if group.n > 2:
        return True
    axis = plan.layout.axis
    extent = value.shape[axis] // plan.layout.num_devices
    width = group.max_sizes[axis]
    return any(
        start[axis] // extent != (start[axis] + width - 1) // extent
        for start in _clamped_starts(value, group)
    )


def sources_are_interior(state, batches, plan):
    """Prove that every possible source write avoids physical boundary cells.

    Use the padded batch extent and the actual clamped start, not just the
    original source bounds. Irregular or boundary-touching sources keep masks.
    """
    for (_, component), (group, rest) in batches.items():
        if rest:
            return False
        if group is None:
            continue
        shape = plan.layout.logical_shapes[component]
        value = getattr(state, component.lower())
        for start in _clamped_starts(value, group):
            if any(
                first < 1 or first + width > size - 1
                for first, width, size in zip(
                    start, group.max_sizes, shape, strict=True
                )
            ):
                return False
    return True


def apply_batched_slabs(value, abs_step, group, plan):
    axis, count = plan.layout.axis, plan.layout.num_devices
    spec = P(*("fdtd" if d == axis else None for d in range(value.ndim)))
    extent = value.shape[axis] // count
    starts = _clamped_starts(value, group)
    shard_map = _shard_map()

    @partial(shard_map, mesh=plan.mesh, in_specs=(spec, P(), P(), P()), out_specs=spec)
    def inject(local, step, coefficients, waveforms):
        safe_step = jnp.clip(step, 0, waveforms.shape[1] - 1)
        origin = jax.lax.axis_index("fdtd") * extent
        width = min(group.max_sizes[axis], extent)
        patch_shape = list(group.max_sizes)
        patch_shape[axis] = width
        broadcast = [1] * local.ndim
        broadcast[axis] = width
        for source, start in enumerate(starts):
            local_start = jnp.clip(start[axis] - origin, 0, extent - width)
            positions = origin + local_start + jnp.arange(width) - start[axis]
            valid = (positions >= 0) & (positions < group.max_sizes[axis])
            patch = jnp.take(coefficients[source], positions, axis=axis, mode="clip")
            addition = cast(
                jax.Array,
                jnp.where(
                    valid.reshape(broadcast), patch * waveforms[source, safe_step], 0
                ),
            )
            target = list(start)
            target[axis] = local_start
            target = tuple(jnp.asarray(index, dtype=jnp.int32) for index in target)
            if local.size > 2**31 - 1:
                # Keep multidimensional indexing when a flat FP32-mode JAX
                # index would overflow its default signed 32-bit integer.
                previous = jax.lax.dynamic_slice(local, target, tuple(patch_shape))
                local = jax.lax.dynamic_update_slice(
                    local, previous + addition.astype(local.dtype), target
                )
                continue
            # A multidimensional read/modify/write slice can make XLA transpose
            # entire field shards to favor a thin source plane. Native stepping
            # then needs the original layout too. Linear indices keep the patch
            # update in the field's existing row-major storage without those
            # full-volume layout buffers.
            flat_indices = jnp.asarray(0, dtype=jnp.int32)
            stride = 1
            for dim in reversed(range(local.ndim)):
                index_shape = [1] * local.ndim
                index_shape[dim] = patch_shape[dim]
                flat_indices = (
                    flat_indices
                    + (
                        target[dim]
                        + jnp.arange(patch_shape[dim], dtype=jnp.int32).reshape(
                            index_shape
                        )
                    )
                    * stride
                )
                stride *= local.shape[dim]
            local = (
                local.reshape(-1)
                .at[flat_indices.reshape(-1)]
                .add(addition.astype(local.dtype).reshape(-1), unique_indices=True)
                .reshape(local.shape)
            )
        return local

    return inject(value, abs_step, group.coeffs, group.waveforms)


_AXIS = "fdtd"


def supported(program):
    return (
        program.config.is_3d
        and program.sharding.layout.enabled
        and bool(program.monitors)
        and all(is_vector_dft_monitor(mon) for mon in program.monitors)
    )


def scan_local_dft(state, program, *, assemble=False):
    """Keep public one-dimensional arenas unchanged outside compiled scans.

    A continuation's prior integral is placed on rank zero exactly once. New
    contributions belong to the ranks owning their interpolation neighbors.
    Weights are replicated and count time samples, not devices.
    """

    def distribute(value):
        return cast(jax.Array, jnp.where(jax.lax.axis_index(_AXIS) == 0, value, 0))[
            None, :
        ]

    def collect(value):
        return jax.lax.psum(value[0], _AXIS)

    convert = _shard_map()(
        collect if assemble else distribute,
        mesh=program.sharding.mesh,
        in_specs=P(_AXIS) if assemble else P(),
        out_specs=P() if assemble else P(_AXIS),
    )
    return state._replace(
        dft_vec_re=convert(state.dft_vec_re),
        dft_vec_im=convert(state.dft_vec_im),
    )


def accumulate_dft(program, mon, carry, fields, time, dt):
    from beamz.simulation.results.observe import _accumulate_dft

    axis = program.sharding.layout.axis
    spec = P(*(_AXIS if i == axis else None for i in range(3)))
    global_shapes = tuple(field.shape for field in fields)

    @partial(
        _shard_map(),
        mesh=program.sharding.mesh,
        in_specs=((spec,) * 6, P(_AXIS), P(_AXIS), P(), P(), P()),
        out_specs=(P(_AXIS), P(_AXIS), P()),
    )
    def local(local_fields, re, im, weight, time, dt):
        vectors = mon.dft_sampling.sample(
            local_fields,
            active_mask=np.asarray(mon.dft_component_mask),
            local_axis=axis,
            shard_index=jax.lax.axis_index(_AXIS),
            shapes=global_shapes,
        )
        re, im, weight = _accumulate_dft(
            mon,
            (re[0], im[0], weight),
            None,
            time,
            dt,
            sampled_vectors=vectors,
        )
        return re[None, :], im[None, :], weight

    return local(fields, *carry, time, dt)


_COMPONENT_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_AXIS_TO_INDEX = {"z": 0, "y": 1, "x": 2}
_INDEX_TO_AXIS = ("z", "y", "x")
_MESH_AXIS = "fdtd"


def normalize_sharding_config(value) -> ShardingConfig:
    """Normalize public sharding input into a stable configuration."""

    # Resolve device layout once on the host so execution receives concrete placement
    # objects.
    if value is None or value is False:
        return ShardingConfig()
    if isinstance(value, tuple):
        return ShardingConfig(*value)
    if isinstance(value, ShardingConfig):
        return value
    if value is True:
        return ShardingConfig(enabled=True)
    if isinstance(value, Mapping):
        return ShardingConfig(**{"enabled": True, **value})
    raise TypeError(
        "sharding must be None, bool, dict, tuple, or beamz.simulation.ShardingConfig"
    )


def sharding_cache_token(value) -> ShardingToken:
    """Return a stable scalar cache token for a sharding selection."""
    # Key cached work by semantic execution inputs so equivalent states reuse
    # compilation safely.
    cfg = normalize_sharding_config(value)
    return (cfg.enabled, cfg.axis, cfg.num_devices, cfg.backend)


def _select_sharding_axis(axis: str, grid_shape: tuple[int, int, int]) -> str:
    # Resolve device layout once on the host so execution receives concrete placement
    # objects.
    if axis != "auto":
        return axis
    lengths = tuple(int(v) for v in grid_shape)
    best = max(range(3), key=lambda idx: (lengths[idx], -idx))
    return _INDEX_TO_AXIS[best]


def _jax_devices_for_config(cfg: ShardingConfig) -> tuple[object, ...]:
    # Choose devices deterministically to keep cache keys and array placement
    # reproducible.
    if not cfg.enabled:
        return ()
    try:
        devices = (
            tuple(jax.devices(cfg.backend)) if cfg.backend else tuple(jax.devices())
        )
    except Exception as exc:
        backend = cfg.backend or "default"
        raise ValueError(
            f"No JAX devices are available for backend {backend!r}"
        ) from exc
    if not devices:
        backend = cfg.backend or "default"
        raise ValueError(f"No JAX devices are available for backend {backend!r}")
    num_devices = len(devices) if cfg.num_devices is None else int(cfg.num_devices)
    if num_devices <= 1:
        return ()
    if num_devices > len(devices):
        raise ValueError(
            f"Requested {num_devices} sharding devices, but only {len(devices)} "
            f"are available for backend {cfg.backend or 'default'}"
        )
    return devices[:num_devices]


def _pad_shape_for_devices(
    shape: tuple[int, ...], axis: int, num_devices: int
) -> tuple[int, ...]:
    # Read dimensions from the authoritative layout so all allocations share one
    # static contract.
    out = list(int(v) for v in shape)
    size = out[int(axis)]
    remainder = size % int(num_devices)
    if remainder:
        out[int(axis)] = size + (int(num_devices) - remainder)
    return tuple(out)


def build_sharding_plan(
    fields,
    cfg: ShardingConfig,
    *,
    is_3d: bool,
    aligned_components: bool = False,
) -> ShardingPlan:
    # 1. Normalize the material-grid rank to canonical z-y-x order so every later shape
    # calculation starts from the same physical domain contract.
    logical_base_values = tuple(int(v) for v in fields.permittivity.shape)
    if len(logical_base_values) == 2:
        logical_base_shape = (
            1,
            logical_base_values[0],
            logical_base_values[1],
        )
    elif len(logical_base_values) == 3:
        logical_base_shape = (
            logical_base_values[0],
            logical_base_values[1],
            logical_base_values[2],
        )
    else:
        raise ValueError(
            f"Compiled storage requires a rank-2 or rank-3 material grid, got "
            f"{logical_base_values!r}."
        )
    # 2. Build the unsharded layout directly from the canonical physical supports.
    if not cfg.enabled:
        logical_shapes = {
            name: tuple(int(v) for v in getattr(fields, name).shape)
            for name in _COMPONENT_NAMES
        }
        return ShardingPlan(
            ShardingLayout(
                enabled=False,
                axis_name="z",
                axis=0,
                num_devices=1,
                backend=cfg.backend,
                logical_shapes=logical_shapes,
                padded_shapes=dict(logical_shapes),
            ),
            None,
        )
    # 3. Validate the sharded path early because its partitioning assumptions only hold
    # for three-dimensional component grids.
    if not is_3d:
        raise NotImplementedError("compiled sharding currently supports 3D runs only")
    if len(logical_base_shape) != 3:
        raise ValueError(
            f"3D sharding requires a 3-axis grid, got {logical_base_shape}"
        )

    # 4. Resolve eligible devices, falling back through this same planner when the
    # requested backend supplies none.
    from beamz.simulation.execute import sharding as sharding_runtime

    devices = sharding_runtime._jax_devices_for_config(cfg)
    if not devices:
        return build_sharding_plan(
            fields,
            ShardingConfig(enabled=False, backend=cfg.backend),
            is_3d=is_3d,
        )
    # 5. Select the partition axis and pad every staggered component independently so
    # each device receives an equal static extent.
    axis_name = _select_sharding_axis(cfg.axis, logical_base_shape)
    axis = _AXIS_TO_INDEX[axis_name]
    num_devices = len(devices)
    logical_shapes = {
        name: tuple(int(v) for v in getattr(fields, name).shape)
        for name in _COMPONENT_NAMES
    }
    padded_shapes = {
        name: _pad_shape_for_devices(shape, axis, num_devices)
        for name, shape in logical_shapes.items()
    }
    if aligned_components:
        # Explicit local stencils use the same coordinate for every component.
        # Independent rounding can otherwise put their interfaces on different
        # global cells (for example 16 versus 18 cells split across two GPUs).
        extent = max(shape[axis] for shape in padded_shapes.values())
        padded_shapes = {
            name: tuple(extent if i == axis else size for i, size in enumerate(shape))
            for name, shape in padded_shapes.items()
        }
    # 6. Resolve the mesh once; every later placement reuses this exact object.
    return ShardingPlan(
        ShardingLayout(
            enabled=True,
            axis_name=axis_name,
            axis=axis,
            num_devices=num_devices,
            backend=cfg.backend,
            logical_shapes=logical_shapes,
            padded_shapes=padded_shapes,
        ),
        jax.sharding.Mesh(np.asarray(devices, dtype=object), (_MESH_AXIS,)),
    )
