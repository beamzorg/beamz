"""Sharding selection and static device-layout planning."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import numpy as np

from beamz.simulation.model import (
    ShardingConfig,
    ShardingLayout,
    ShardingPlan,
    ShardingToken,
)

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
    from beamz.simulation import sharding as sharding_runtime

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
