"""Optional multi-device lowering, padding, placement, and result cropping."""

from __future__ import annotations

import json
import os

# Retain the names available from this module before its implementation was split.
from collections.abc import Mapping as Mapping
from dataclasses import replace
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from beamz._region_array import RegionArray, place_region_array
from beamz.constitutive import _fit_array_to_shape
from beamz.simulation.compile.backend import trace_preparation

# Sharding may pad the high side for equal device partitions. Padding is storage-only;
# logical component shapes remain authoritative for curls, monitors, and result crops.
from beamz.simulation.execute.distributed import (
    _AXIS_TO_INDEX,  # noqa: F401
    _COMPONENT_NAMES,
    _INDEX_TO_AXIS,  # noqa: F401
    _MESH_AXIS,
    _jax_devices_for_config,  # noqa: F401
    _pad_shape_for_devices,
    _select_sharding_axis,  # noqa: F401
    build_sharding_plan,  # noqa: F401
    normalize_sharding_config,  # noqa: F401
    sharding_cache_token,  # noqa: F401
)
from beamz.simulation.model import (
    CpmlPackedSlabSpec,
    DerivativeMetricPlan,
    ShardingLayout,
)
from beamz.simulation.model import ShardingConfig as ShardingConfig
from beamz.simulation.model import ShardingPlan as ShardingPlan
from beamz.simulation.model import ShardingToken as ShardingToken


def _array_shape(arr) -> tuple[int, ...]:
    # Read dimensions from the authoritative layout so all allocations share one
    # static contract.
    return tuple(int(v) for v in getattr(arr, "shape", np.shape(arr)))


def _array_ndim(arr) -> int:
    # Make conversion explicit to keep dtype and device placement from changing
    # tracing signatures.
    ndim = getattr(arr, "ndim", None)
    if ndim is not None:
        return int(ndim)
    return len(_array_shape(arr))


def _pad_high_to_shape(arr, shape: tuple[int, ...], *, pad_value=0.0):
    # Preserve NumPy setup arrays on the host and use the same crop/pad recipe for JAX.
    target_shape = tuple(int(v) for v in shape)
    if isinstance(arr, RegionArray):
        return arr.padded(target_shape, pad_value)
    xp = np if _is_host_array(arr) else jnp
    return _fit_array_to_shape(arr, target_shape, pad_value=pad_value, xp=xp)


def _crop_high_to_shape(arr, shape: tuple[int, ...]) -> jnp.ndarray:
    # Read dimensions from the authoritative layout so all allocations share one
    # static contract.
    slices = tuple(slice(0, int(v)) for v in shape)
    return jnp.asarray(arr)[slices]


_COEFFICIENT_COMPONENTS = {
    **{
        f"h_{kind}_{axis}": f"H{axis}"
        for axis in "xyz"
        for kind in ("decay", "source", "sigma_m")
    },
    **{
        f"e_{kind}_{axis}": f"E{axis}"
        for axis in "xyz"
        for kind in (
            "decay",
            "source",
            "conductivity",
            "permittivity",
            "inverse_diagonal",
        )
    },
    "e_inverse_offdiagonal": "Node",
}


def _lower_coefficients(coefficients, layout: ShardingLayout):
    """Pad compiled material coefficients without recreating a fields object."""
    if not layout.enabled:
        return coefficients
    updates = {}
    for name, component in _COEFFICIENT_COMPONENTS.items():
        value = getattr(coefficients, name)
        if _array_ndim(value) == 0 or not int(np.prod(_array_shape(value))):
            continue
        is_tensor = name == "e_inverse_offdiagonal"
        neutral = (
            1.0
            if "_decay_" in name or ("_permittivity_" in name and not is_tensor)
            else 0.0
        )
        target_shape = (
            _pad_shape_for_devices(
                tuple(
                    max(shape[axis] for shape in layout.logical_shapes.values())
                    for axis in range(len(next(iter(layout.logical_shapes.values()))))
                ),
                layout.axis,
                layout.num_devices,
            )
            if component == "Node"
            else layout.padded_shapes[component]
        )
        if is_tensor:
            target_shape = (*target_shape, int(value.shape[-1]))
        updates[name] = _pad_high_to_shape(value, target_shape, pad_value=neutral)
    return coefficients._replace(**updates)


def _lower_cpml_term(term, layout: ShardingLayout):
    """Extend one packed CPML slab across padded transverse dimensions."""
    dimensions = list(layout.padded_shapes[term.component])
    dimensions[term.axis] = term.slab.low + term.slab.high
    target_shape = tuple(dimensions)
    slab = CpmlPackedSlabSpec(
        term.axis,
        term.slab.low,
        term.slab.high,
        target_shape,
        term.slab.logical_stop,
    )

    def profile(value, neutral):
        # Separable profiles broadcast across transverse dimensions. Expanding
        # a singleton with zero padding would turn off CPML everywhere except
        # the first plane. Only pad dimensions that already carry spatial data.
        shape = tuple(
            1 if size == 1 and axis != term.axis else target
            for axis, (size, target) in enumerate(
                zip(value.shape, target_shape, strict=True)
            )
        )
        return _pad_high_to_shape(value, shape, pad_value=neutral)

    return replace(
        term,
        a=profile(term.a, 0.0),
        b=profile(term.b, 1.0),
        inv_kappa=profile(term.inv_kappa, 1.0),
        slab=slab,
    )


def lower_compiled_arrays(coefficients, boundary, layout: ShardingLayout):
    """Lower logical coefficients and boundary arrays to a sharded storage layout."""
    if not layout.enabled:
        return coefficients, boundary
    coefficients = _lower_coefficients(coefficients, layout)
    cpml = replace(
        boundary.cpml,
        h_terms=tuple(_lower_cpml_term(term, layout) for term in boundary.cpml.h_terms),
        e_terms=tuple(_lower_cpml_term(term, layout) for term in boundary.cpml.e_terms),
    )
    masks = {
        f"{component.lower()}_mask": _pad_high_to_shape(
            getattr(boundary.metallic, f"{component.lower()}_mask"),
            layout.padded_shapes[component],
            # Storage-only cells behave like PEC cells and remain identically zero.
            pad_value=True,
        )
        for component in _COMPONENT_NAMES
    }
    return coefficients, replace(
        boundary,
        cpml=cpml,
        metallic=replace(boundary.metallic, **masks),
    )


_METRIC_SOURCE_COMPONENTS = {
    "e_to_h_x": "Ey",
    "e_to_h_y": "Ex",
    "e_to_h_z": "Ex",
    "h_to_e_x": "Hy",
    "h_to_e_y": "Hx",
    "h_to_e_z": "Hx",
}


def lower_derivative_metrics(
    metrics: DerivativeMetricPlan, layout: ShardingLayout
) -> DerivativeMetricPlan:
    """Extend separable metrics across storage-only cells on the sharded axis."""
    if not layout.enabled:
        return metrics

    updates = {}
    for name, component in _METRIC_SOURCE_COMPONENTS.items():
        if not name.endswith(f"_{layout.axis_name}"):
            continue
        metric = jnp.asarray(getattr(metrics, name))
        if metric.ndim == 0 or not metric.size:
            continue
        logical = layout.logical_shapes[component][layout.axis]
        padded = layout.padded_shapes[component][layout.axis]
        extra = int(padded) - int(logical)
        if extra > 0:
            updates[name] = jnp.concatenate(
                (metric, jnp.full((extra,), metric[-1], dtype=metric.dtype))
            )
    return metrics._replace(**updates)


def _replicated_sharding(mesh):
    return (
        None
        if mesh is None
        else jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    )


def _array_sharding(program, arr, mesh):
    shape, axis = _array_shape(arr), int(program.sharding.layout.axis)
    if (
        len(shape) > axis
        and int(shape[axis]) > 0
        and int(shape[axis]) % int(program.sharding.layout.num_devices) == 0
    ):
        spec: list[str | None] = [None] * len(shape)
        spec[axis] = _MESH_AXIS
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))
    return _replicated_sharding(mesh)


@trace_preparation("placement")
def place_tree(program, tree, *, shard_arrays: bool = True):
    """Place one complete pytree according to the program's backend plan."""
    if not program.sharding.layout.enabled:
        try:
            devices = jax.devices(jax.default_backend())
        except Exception:
            devices = jax.devices()
        return jax.tree_util.tree_map(
            lambda value: (
                jax.device_put(jnp.asarray(value), devices[0])
                if devices
                else jnp.asarray(value)
            ),
            tree,
        )
    mesh = program.sharding.mesh
    trace_path = os.getenv("BEAMZ_TRACE_PLACEMENT")
    if trace_path:

        def traced_place(path, value):
            target = (
                _array_sharding(program, value, mesh)
                if shard_arrays
                else _replicated_sharding(mesh)
            )
            return _trace_placement(
                jax.tree_util.keystr(path),
                value,
                target,
                lambda: _place_array(value, target),
            )

        return jax.tree_util.tree_map_with_path(traced_place, tree)
    return jax.tree_util.tree_map(
        lambda value: _place_array(
            value,
            _array_sharding(program, value, mesh)
            if shard_arrays
            else _replicated_sharding(mesh),
        ),
        tree,
    )


def _trace_placement(path, value, target, operation):
    trace_path = os.getenv("BEAMZ_TRACE_PLACEMENT")
    if not trace_path:
        return operation()
    devices = sorted(target.addressable_devices, key=lambda d: d.id)
    before = [dict(id=d.id, stats=d.memory_stats()) for d in devices]
    placed = operation()
    placed.block_until_ready()
    record = dict(
        path=path,
        shape=list(getattr(value, "shape", ())),
        output_shape=list(placed.shape),
        dtype=str(getattr(value, "dtype", type(value).__name__)),
        source_devices=[str(d) for d in value.devices()]
        if isinstance(value, jax.Array)
        else ["host"],
        target=str(target),
        before=before,
        after=[dict(id=d.id, stats=d.memory_stats()) for d in devices],
    )
    with open(trace_path, "a") as stream:
        stream.write(json.dumps(record) + "\n")
    return placed


def _place_padded_host_array(value, shape, target):
    """Crop/pad only destination shards, never a full global field copy."""
    host = np.asarray(value)

    def shard(index):
        bounds = [part.indices(size) for part, size in zip(index, shape, strict=True)]
        if any(step != 1 for _, _, step in bounds):
            raise ValueError("Host padding requires contiguous shard indices")
        slices = tuple(
            slice(min(lo, size), min(hi, size))
            for (lo, hi, _), size in zip(bounds, host.shape, strict=True)
        )
        local = host[slices]
        padding = tuple(
            (0, hi - lo - size)
            for (lo, hi, _), size in zip(bounds, local.shape, strict=True)
        )
        return np.pad(local, padding) if any(high for _, high in padding) else local

    return jax.make_array_from_callback(shape, target, shard)


def _prepare_component(program, component, value):
    layout = program.sharding.layout
    if layout.enabled and _is_host_array(value):
        shape = layout.padded_shapes[component]
        target = _array_sharding(
            program,
            jax.ShapeDtypeStruct(shape, value.dtype),
            program.sharding.mesh,
        )
        return _trace_placement(
            f"padded.{component}",
            value,
            target,
            lambda: _place_padded_host_array(value, shape, target),
        )
    return pad_component(program, component, value)


@lru_cache(maxsize=128)
def _device_reshard(sharding):
    # Eager device_put can assemble a differently partitioned array on the host.
    # Let compiled collectives convert public and solver layouts on-device.
    return jax.jit(lambda value: value, out_shardings=sharding)


def _is_host_array(value):
    return isinstance(value, np.ndarray) or (
        isinstance(value, jax.Array)
        and value.is_fully_addressable
        and len(value.devices()) == 1
        and all(device.platform == "cpu" for device in value.devices())
    )


def _place_array(value, target):
    if isinstance(value, RegionArray):
        return place_region_array(value, target)
    if (
        isinstance(value, jax.Array)
        and len(value.sharding.device_set) > 1
        and value.sharding.device_set == target.device_set
    ):
        if value.sharding.is_equivalent_to(target, value.ndim):
            return value
        return _device_reshard(target)(value)
    # A CPU JAX array can be uncommitted. Eager device_put's multi-slice may
    # execute on the default accelerator after the CPU setup context has ended.
    # Slice host storage explicitly, so no global array is staged on GPU 0.
    if _is_host_array(value):
        host = np.asarray(value)
        return jax.make_array_from_callback(
            host.shape, target, lambda index: host[index]
        )
    return jax.device_put(value, target)


def pad_component(program, component: str, value):
    """Pad one logical component to its optional device-partition shape."""
    return _pad_high_to_shape(
        value,
        program.sharding.layout.padded_shapes[component],
        pad_value=0.0,
    )


def crop_component(program, component: str, value):
    """Crop one backend component back to its logical Yee support."""
    layout = program.sharding.layout
    shape = layout.logical_shapes[component]
    if layout.enabled:
        # A staggered logical extent need not divide the device count. Letting
        # SPMD infer the crop layout can replicate the entire field on every GPU.
        # Prefer the simulation axis, then another divisible physical axis. The
        # canonical public shape is unchanged, including during continuation.
        axes = (
            layout.axis,
            *(axis for axis in range(len(shape)) if axis != layout.axis),
        )
        for axis in axes:
            if shape[axis] > 0 and shape[axis] % layout.num_devices == 0:
                spec: list[str | None] = [None] * len(shape)
                spec[axis] = _MESH_AXIS
                sharding = jax.sharding.NamedSharding(
                    program.sharding.mesh, jax.sharding.PartitionSpec(*spec)
                )
                return _component_crop(sharding)(value, shape)
    return _crop_high_to_shape(value, shape)


@lru_cache(maxsize=64)
def _component_crop(sharding):
    # Repeated public calls reuse compilation; the cache owns no field buffers.
    return jax.jit(_crop_high_to_shape, static_argnums=(1,), out_shardings=sharding)


def prepare_state(program, state, *, replicated_fields):
    """Pad and place a logical runtime state for execution."""
    state = state._replace(
        **{
            name.lower(): _prepare_component(
                program, name, getattr(state, name.lower())
            )
            for name in _COMPONENT_NAMES
        }
    )
    placed = place_tree(program, state)
    from beamz.simulation.execute import jax_sharding

    if program.sharding.layout.enabled and (
        program.config.backend == "cuda_streamed" or jax_sharding.supported(program)
    ):
        # Public scan inputs replicate normal slabs; the scan distributes their
        # recurrence entries internally. Transverse slabs follow field ownership.
        updates = {}
        for phase in ("h", "e"):
            name = f"cpml_psi_{phase}_terms"
            terms = getattr(program.boundary.cpml, f"{phase}_terms")
            updates[name] = tuple(
                place_tree(program, value, shard_arrays=False)
                if term.axis == program.sharding.layout.axis
                else value
                for value, term in zip(getattr(placed, name), terms, strict=True)
            )
        placed = placed._replace(**updates)
    return placed._replace(
        **{
            name: place_tree(program, getattr(state, name), shard_arrays=False)
            for name in replicated_fields
        }
    )


def crop_state(program, state):
    """Remove backend padding before publishing a continuation state."""
    if not program.sharding.layout.enabled:
        return state
    updates: dict[str, jax.Array | tuple[jax.Array, ...]] = {
        name.lower(): crop_component(program, name, getattr(state, name.lower()))
        for name in _COMPONENT_NAMES
    }
    for phase in ("h", "e"):
        terms = getattr(program.boundary.cpml, f"{phase}_terms")
        updates[f"cpml_psi_{phase}_terms"] = tuple(
            _crop_high_to_shape(
                value, logical_cpml_shape(program.sharding.layout, term)
            )
            for value, term in zip(
                getattr(state, f"cpml_psi_{phase}_terms"), terms, strict=True
            )
        )
    return state._replace(**updates)


def logical_cpml_shape(layout, term):
    """Physical packed-slab shape independent of the device partition layout."""
    shape = list(layout.logical_shapes[term.component])
    shape[term.axis] = term.slab.low + term.slab.high
    return tuple(shape)
