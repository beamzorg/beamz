"""Whole-column multi-step scheduling, derived from the tiling in PR #308.

The numerical operations remain in the CUDA Yee primitives. This module only
proves which cells can advance S times per pass (S = 2 or 4) and which
intermediate cells must exist. Tiles are 16x32; eligibility and scratch
allocation use 8x16 quadrants.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from beamz.simulation.backend import CudaBackendUnavailable

TILE_Y, TILE_X = 16, 32
QUAD_Y, QUAD_X = TILE_Y // 2, TILE_X // 2
_PLAN_CACHE = OrderedDict()
_PLAN_LOCK = threading.RLock()
_PLAN_CACHE_BYTES = 256 << 20


@dataclass(frozen=True)
class MstepGeometry:
    shape: tuple[int, int, int]
    bounds: np.ndarray
    core_masks: np.ndarray
    core_cells: int

    def summary(self):
        return {
            "core_fraction": self.core_cells / int(np.prod(self.shape)),
            "active_tiles": int(np.count_nonzero(self.core_masks)),
            "column_depth": max(0, int(self.bounds[1] - self.bounds[0])),
            "owned_quadrants": sum(
                int(np.count_nonzero(self.core_masks & (1 << bit))) for bit in range(4)
            ),
        }


@dataclass(frozen=True)
class MstepPlan(MstepGeometry):
    slots: np.ndarray
    # Intermediate levels 1..S-1, one row each, padded with -1.
    first_cells: np.ndarray
    edge_cells: np.ndarray
    scratch_cells: int
    steps: int = 2

    def summary(self):
        return {
            **super().summary(),
            "schedule": "cuda_mstep",
            "tile": (TILE_Y, TILE_X),
            "temporal_steps": self.steps,
            "first_step_cells": int(np.count_nonzero(self.first_cells >= 0)),
            "edge_cells": self.edge_cells.size,
            "scratch_bytes": 6 * self.scratch_cells * 4,
            "plan_bytes": sum(
                a.nbytes
                for a in (
                    self.bounds,
                    self.core_masks,
                    self.slots,
                    self.first_cells,
                    self.edge_cells,
                )
            ),
        }


def build_mstep_geometry(field_shapes, cpml, groups, steps=2):
    """Prove column eligibility before allocating any intermediate queues.

    An owned quadrant's dependency cone reaches `steps` cells in every
    direction, so CPML, sources and the domain edge stay that far away.
    """
    shape = tuple(max(s[a] for s in field_shapes) for a in range(3))
    nz, ny, nx = shape
    if int(np.prod(shape)) >= np.iinfo(np.int32).max:
        raise CudaBackendUnavailable("cuda_mstep cell indices exceed int32")
    nq_y = (ny + QUAD_Y - 1) // QUAD_Y
    nq_x = (nx + QUAD_X - 1) // QUAD_X
    # Include padding quadrants so each tile always has four mask bits.
    nq_y += nq_y % 2
    nq_x += nq_x % 2
    low = np.ones(3, dtype=int)
    high = np.asarray(shape) - 3
    for term in (*cpml.h_terms, *cpml.e_terms):
        axis = int(term.axis)
        c = "xyz".index(term.component[1]) + (0 if term.component[0] == "H" else 3)
        low[axis] = max(low[axis], int(term.slab.low))
        if term.slab.high:
            high[axis] = min(
                high[axis], field_shapes[c][axis] - int(term.slab.high) - 1
            )
    zb, zt = int(low[0] + steps), int(high[0]) - (steps - 2)
    qy = np.arange(nq_y)[:, None] * QUAD_Y
    qx = np.arange(nq_x)[None, :] * QUAD_X
    core = (qy - steps >= low[1]) & (qy + QUAD_Y + steps - 1 <= high[1])
    core = core & (qx - steps >= low[2]) & (qx + QUAD_X + steps - 1 <= high[2])
    if zt - zb < 2 * steps:
        core[:] = False
    for group in groups:
        if group is None:
            continue
        extent = np.asarray(group.coeffs.shape[1:])
        for start in np.asarray(group.starts):
            end = start + extent
            if np.any(end <= 0) or np.any(start >= shape):
                continue
            intersects = (qy - steps < end[1]) & (qy + QUAD_Y + steps > start[1])
            intersects = (
                intersects & (qx - steps < end[2]) & (qx + QUAD_X + steps > start[2])
            )
            core[intersects] = False
    masks = sum(
        core[dy::2, dx::2].astype(np.int32) << (2 * dy + dx)
        for dy in range(2)
        for dx in range(2)
    )
    return MstepGeometry(
        shape,
        np.asarray((zb, zt), np.int32),
        np.asarray(masks, dtype=np.int32),
        int(core.sum()) * QUAD_Y * QUAD_X * max(0, zt - zb),
    )


def build_mstep_plan(field_shapes, cpml, groups, monitor_indices=None, steps=2):
    """Reuse dependency plans when only execution settings or values change."""
    digest = hashlib.sha256()
    structure = (
        steps,
        tuple(field_shapes),
        tuple(
            (t.axis, t.component, t.slab.low, t.slab.high)
            for t in (*cpml.h_terms, *cpml.e_terms)
        ),
        tuple(
            None if g is None else (g.coeffs.shape, np.asarray(g.starts).shape)
            for g in groups
        ),
    )
    digest.update(repr(structure).encode())
    for group in groups:
        if group is not None:
            digest.update(np.ascontiguousarray(group.starts, dtype=np.int64).tobytes())
    if monitor_indices is not None:
        indices = np.ascontiguousarray(monitor_indices)
        digest.update(repr((indices.shape, indices.dtype.str)).encode())
        digest.update(indices.tobytes())
    key = digest.digest()
    with _PLAN_LOCK:
        if key in _PLAN_CACHE:
            _PLAN_CACHE.move_to_end(key)
            return _PLAN_CACHE[key]
    plan = _build_mstep_plan(field_shapes, cpml, groups, monitor_indices, steps)
    for array in _plan_arrays(plan):
        array.setflags(write=False)
    with _PLAN_LOCK:
        if plan.summary()["plan_bytes"] <= _PLAN_CACHE_BYTES:
            _PLAN_CACHE[key] = plan
            while (
                len(_PLAN_CACHE) > 2
                or sum(p.summary()["plan_bytes"] for p in _PLAN_CACHE.values())
                > _PLAN_CACHE_BYTES
            ):
                _PLAN_CACHE.popitem(last=False)
    return plan


def _build_mstep_plan(field_shapes, cpml, groups, monitor_indices=None, steps=2):
    """Build static dependency queues without allocating a volume-sized mask.

    Every level-k edge cell needs level k-1 E at its positive neighbours;
    those E values need level k-1 H at their negative neighbours. A one-cell
    Chebyshev margin per level covers both, including curl cross-axis
    dependencies, so level k is the edge region dilated by steps - k cells.
    Monitors additionally request their sample cells and negative neighbours
    at every level. Source intersections exclude an entire column quadrant
    and its `steps`-cell halo, so the tiled kernel never injects a source.
    """
    geometry = build_mstep_geometry(field_shapes, cpml, groups, steps)
    shape = geometry.shape
    nz, ny, nx = shape
    zb, zt = map(int, geometry.bounds)
    masks = geometry.core_masks
    nq_y, nq_x = masks.shape[0] * 2, masks.shape[1] * 2
    core = np.zeros((nq_y, nq_x), bool)
    for dy in range(2):
        for dx in range(2):
            core[dy::2, dx::2] = (masks & (1 << (2 * dy + dx))) != 0
    edge_xy = ~np.repeat(np.repeat(core, QUAD_Y, axis=0), QUAD_X, axis=1)[:ny, :nx]

    def dilate(mask):
        padded = np.pad(mask, 1, constant_values=True)
        return np.logical_or.reduce(
            [padded[dy : dy + ny, dx : dx + nx] for dy in range(3) for dx in range(3)]
        )

    # level_xy[k - 1]: cells level k computes inside the core's z range.
    level_xy = [edge_xy]
    for _ in range(steps - 1):
        level_xy.insert(0, dilate(level_xy[0]))
    requests = []
    if monitor_indices is not None:
        # DFT plans use E,H order; native field banks use H,E order.
        for c, dims in enumerate((*field_shapes[3:], *field_shapes[:3])):
            indices = np.asarray(monitor_indices)[:, c].reshape(-1)
            indices = np.unique(indices[(indices >= 0) & (indices < np.prod(dims))])
            zz, rest = np.divmod(indices, dims[1] * dims[2])
            yy, xx = np.divmod(rest, dims[2])
            for dz, dy, dx in ((0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1)):
                z, y, x = zz + dz, yy + dy, xx + dx
                valid = (z >= 0) & (z < nz) & (y >= 0) & (y < ny) & (x >= 0) & (x < nx)
                requests.append((z[valid] * ny + y[valid]) * nx + x[valid])
    watched = {}
    if requests:
        # Collapse repeated frequencies, components and neighbours in bulk.
        # Large monitor planes otherwise spend most preparation time inserting
        # individual Python integers into per-plane lists.
        rows = np.unique(np.concatenate(requests))
        z, xy = np.divmod(rows, ny * nx)
        planes, starts = np.unique(z, return_index=True)
        for plane, start, end in zip(
            planes, starts, (*starts[1:], rows.size), strict=True
        ):
            watched[int(plane)] = xy[start:end]

    def watched_mask(z, radius):
        """Monitor rows near plane z, dilated by `radius` cells in 3D."""
        mask = np.zeros((ny, nx), bool)
        for plane in range(z - radius, z + radius + 1):
            if plane in watched:
                mask.reshape(-1)[watched[plane]] = True
        for _ in range(radius):
            if mask.any():
                padded = np.pad(mask, 1)
                mask = np.logical_or.reduce(
                    [
                        padded[dy : dy + ny, dx : dx + nx]
                        for dy in range(3)
                        for dx in range(3)
                    ]
                )
        return mask

    slots = np.full((nz, nq_y, nq_x), -1, dtype=np.int32)
    levels = [[] for _ in range(steps - 1)]
    edge = []
    all_xy = np.arange(ny * nx, dtype=np.int32)
    edge_rows = np.flatnonzero(edge_xy).astype(np.int32)
    slot_count = 0
    for z in range(nz):
        is_core_z = zb <= z < zt
        edge.append((edge_rows if is_core_z else all_xy) + z * ny * nx)
        for level in range(1, steps):
            margin = steps - level
            if zb + margin <= z < zt - margin:
                mask = level_xy[level - 1].copy()
            else:
                mask = np.ones((ny, nx), bool)
            # Monitors observe every level. Level k computes them from level
            # k - 1, so earlier levels need them dilated one cell per level.
            mask |= watched_mask(z, steps - 1 - level)
            rows = np.flatnonzero(mask).astype(np.int32)
            levels[level - 1].append(rows + z * ny * nx)
            if level == 1:
                # Level 1 contains every later level; it sizes the scratch.
                first_rows = rows
        yy, xx = np.divmod(first_rows, nx)
        occupied = np.unique((yy // QUAD_Y) * nq_x + xx // QUAD_X)
        slots[z].reshape(-1)[occupied] = np.arange(
            slot_count, slot_count + occupied.size
        )
        slot_count += occupied.size
    # Missing intermediate cells map to one initialized spare quadrant. Values
    # from it may feed discarded edge calculations, never an owned result.
    slots[slots < 0] = slot_count
    # Beyond two steps, consecutive levels alternate between two banks.
    banks = 1 if steps == 2 else 2
    if banks * (slot_count + 1) * QUAD_Y * QUAD_X >= np.iinfo(np.int32).max:
        raise CudaBackendUnavailable("cuda_mstep scratch offsets exceed int32")
    rows = [np.concatenate(level) for level in levels]
    first = np.full((len(rows), max(r.size for r in rows)), -1, dtype=np.int32)
    for level, r in enumerate(rows):
        first[level, : r.size] = r
    return MstepPlan(
        shape,
        geometry.bounds,
        masks,
        geometry.core_cells,
        slots,
        first,
        np.concatenate(edge),
        banks * (slot_count + 1) * QUAD_Y * QUAD_X,
        steps,
    )


def run_mstep(
    state,
    ctx,
    coeffs,
    groups,
    packed_monitors,
    plan,
    nsteps,
    *,
    observation_origin,
    observation_step_offset,
    plan_buffers=None,
):
    """Lower the shared CUDA program with compact intermediate field storage."""
    import jax.numpy as jnp

    from beamz.simulation import _cuda_abi as abi
    from beamz.simulation.cuda import runtime as rt

    if nsteps < plan.steps or not plan.core_cells:
        # Ordinary execution has no pair workspace, including a one-step tail
        # after native chunks. Force the shared in-place program here.
        from dataclasses import replace

        return rt.run_native_steps(
            state,
            replace(ctx, low_memory=True),
            coeffs,
            groups,
            packed_monitors,
            nsteps,
            observation_origin=observation_origin,
            observation_step_offset=observation_step_offset,
        )
    fields = rt._fields(state)
    if any(f.dtype != jnp.float32 for f in fields):
        raise CudaBackendUnavailable("cuda_mstep requires float32 fields")
    arguments, values, aliases = rt._graph_io(state, ctx, coeffs)
    state_count = len(values)
    final = tuple(jnp.empty_like(f) for f in fields)
    scratch = tuple(jnp.zeros((plan.scratch_cells,), jnp.float32) for _ in fields)
    for value in (*final, *scratch):
        aliases[len(arguments)] = len(values)
        arguments = (*arguments, value)
        values = (*values, value)
    arguments = (*arguments, *rt._source_group_arguments(groups))
    arguments, values, aliases, monitor_start, count = rt._monitor_io(
        state,
        packed_monitors,
        arguments,
        values,
        aliases,
        observation_origin=observation_origin,
        observation_step_offset=observation_step_offset,
    )
    if plan_buffers is None:
        plan_buffers = tuple(jnp.asarray(a) for a in _plan_arrays(plan))
    arguments = (*arguments, *plan_buffers)
    flags = rt._native_schedule_flags(ctx, coeffs, sources=True, monitors=count)
    schedule = rt.NativeSchedulePlan(
        layout=abi.PROGRAM_LAYOUT_YEE_IN_PLACE,
        flags=flags,
        monitor_count=count,
        coincident_source_group_mask=rt._coincident_source_group_mask(groups),
        disjoint_source_group_mask=rt._disjoint_source_group_mask(groups),
        temporal_steps=plan.steps,
    )
    outputs = rt._ffi_call(abi.CUDA_MSTEP_TARGET, values, aliases)(
        *arguments, **rt._program_attributes(ctx, nsteps, schedule)
    )
    bank = state_count if (nsteps // plan.steps) % 2 else 0
    result = rt._replace_fields(state, outputs, bank)
    if ctx.boundary.cpml.enabled:
        result = result._replace(
            cpml_psi_h_terms=outputs[6:12], cpml_psi_e_terms=outputs[12:18]
        )
    if count:
        result = rt._replace_dft(result, outputs, monitor_start)
    return result


def _plan_arrays(plan):
    return (plan.bounds, plan.core_masks, plan.slots, plan.first_cells, plan.edge_cells)


class _BoundPlan:
    """Keep plan values out of compiled code while preserving the two-input API."""

    def __init__(self, executable, buffers):
        self.executable, self.buffers = executable, buffers

    def __call__(self, state, coefficients):
        return self.executable(state, coefficients, None, self.buffers)

    def lower(self, state, coefficients):
        return _BoundPlanLowered(
            self.executable.lower(state, coefficients, None, self.buffers), self.buffers
        )

    def __getattr__(self, name):
        return getattr(self.executable, name)


class _BoundPlanLowered:
    def __init__(self, lowered, buffers):
        self.lowered, self.buffers = lowered, buffers

    def compile(self, *args, **kwargs):
        return _BoundPlan(self.lowered.compile(*args, **kwargs), self.buffers)

    def __getattr__(self, name):
        return getattr(self.lowered, name)


def bind_plan_buffers(scan, plan):
    import jax

    buffers = jax.device_put(_plan_arrays(plan))
    # Include the one-time upload in preparation, not the first execution.
    jax.block_until_ready(buffers)
    return _BoundPlan(scan, buffers)
