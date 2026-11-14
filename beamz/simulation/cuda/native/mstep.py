"""Whole-column two-step scheduling, derived from the tiling in PR #308.

The numerical operations remain in the CUDA Yee primitives. This module only
proves which cells can advance twice and which intermediate cells must exist.
Tiles are 16x32; eligibility and scratch allocation use 8x16 quadrants.
"""

from __future__ import annotations

import hashlib
import operator
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from beamz.simulation.compile.backend import CudaBackendUnavailable
from beamz.simulation.cuda.native import abi

TILE_Y, TILE_X = 16, 32
QUAD_Y, QUAD_X = TILE_Y // 2, TILE_X // 2
_PLAN_CACHE: OrderedDict[Any, Any] = OrderedDict()
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
    first_cells: np.ndarray
    edge_cells: np.ndarray
    scratch_cells: int

    def summary(self):
        return {
            **super().summary(),
            "schedule": "cuda_mstep",
            "tile": (TILE_Y, TILE_X),
            "temporal_steps": 2,
            "first_step_cells": self.first_cells.size,
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


def build_mstep_geometry(field_shapes, cpml, groups):
    """Prove column eligibility before allocating any intermediate queues."""
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
    zb, zt = int(low[0] + 2), int(high[0])
    qy = np.arange(nq_y)[:, None] * QUAD_Y
    qx = np.arange(nq_x)[None, :] * QUAD_X
    core = (qy - 2 >= low[1]) & (qy + QUAD_Y + 1 <= high[1])
    core = core & (qx - 2 >= low[2]) & (qx + QUAD_X + 1 <= high[2])
    if zt - zb < 4:
        core[:] = False
    for group in groups:
        if group is None:
            continue
        extent = np.asarray(group.coeffs.shape[1:])
        for start in np.asarray(group.starts):
            end = start + extent
            if np.any(end <= 0) or np.any(start >= shape):
                continue
            intersects = (qy - 2 < end[1]) & (qy + QUAD_Y + 2 > start[1])
            intersects = intersects & (qx - 2 < end[2]) & (qx + QUAD_X + 2 > start[2])
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


def build_mstep_plan(field_shapes, cpml, groups, monitor_indices=None):
    """Reuse dependency plans when only execution settings or values change."""
    digest = hashlib.sha256()
    structure = (
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
    plan = _build_mstep_plan(field_shapes, cpml, groups, monitor_indices)
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


def _build_mstep_plan(field_shapes, cpml, groups, monitor_indices=None):
    """Build static dependency queues without allocating a volume-sized mask.

    Every second-step edge cell needs first-step E at its positive neighbours;
    those E values need first-step H at their negative neighbours. A one-cell
    Chebyshev margin covers both, including curl cross-axis dependencies.
    Monitors additionally request their sample cells and negative neighbours.
    Source intersections exclude an entire column quadrant and its two-cell
    halo, so the tiled kernel never needs to inject a source.
    """
    geometry = build_mstep_geometry(field_shapes, cpml, groups)
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
    padded = np.pad(edge_xy, 1, constant_values=True)
    first_xy = np.logical_or.reduce(
        [padded[dy : dy + ny, dx : dx + nx] for dy in range(3) for dx in range(3)]
    )
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
    slots = np.full((nz, nq_y, nq_x), -1, dtype=np.int32)
    first, edge = [], []
    all_xy = np.arange(ny * nx, dtype=np.int32)
    edge_rows = np.flatnonzero(edge_xy).astype(np.int32)
    slot_count = 0
    for z in range(nz):
        is_core_z = zb <= z < zt
        edge.append((edge_rows if is_core_z else all_xy) + z * ny * nx)
        mask = first_xy.copy() if zb + 1 <= z < zt - 1 else np.ones((ny, nx), bool)
        if z in watched:
            mask.reshape(-1)[watched[z]] = True
        rows = np.flatnonzero(mask).astype(np.int32)
        first.append(rows + z * ny * nx)
        yy, xx = np.divmod(rows, nx)
        occupied = np.unique((yy // QUAD_Y) * nq_x + xx // QUAD_X)
        slots[z].reshape(-1)[occupied] = np.arange(
            slot_count, slot_count + occupied.size
        )
        slot_count += occupied.size
    # Missing intermediate cells map to one initialized spare quadrant. Values
    # from it may feed discarded edge calculations, never an owned result.
    slots[slots < 0] = slot_count
    if (slot_count + 1) * QUAD_Y * QUAD_X >= np.iinfo(np.int32).max:
        raise CudaBackendUnavailable("cuda_mstep scratch offsets exceed int32")
    return MstepPlan(
        shape,
        geometry.bounds,
        masks,
        geometry.core_cells,
        slots,
        np.concatenate(first),
        np.concatenate(edge),
        (slot_count + 1) * QUAD_Y * QUAD_X,
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

    from beamz.simulation.cuda import runtime as rt
    from beamz.simulation.cuda.native import abi

    if nsteps < 2 or not plan.core_cells:
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
        temporal_steps=2,
    )
    outputs = rt._ffi_call(abi.CUDA_MSTEP_TARGET, values, aliases)(
        *arguments, **rt._program_attributes(ctx, nsteps, schedule)
    )
    bank = state_count if (nsteps // 2) % 2 else 0
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


TILES = {
    "16x32": 0,
    "8x32": abi.CUDA_MSTEP_TILE8X32,
    "8x16": abi.CUDA_MSTEP_TILE8X16,
    "16x16": abi.CUDA_MSTEP_TILE16X16,
}
SEGMENTS = {
    "whole": 0,
    "32": abi.CUDA_MSTEP_SEGMENT32,
    "64": abi.CUDA_MSTEP_SEGMENT64,
    "128": abi.CUDA_MSTEP_SEGMENT128,
    "256": abi.CUDA_MSTEP_SEGMENT256,
}
VARIANT_FLAGS = sum(TILES.values()) | sum(SEGMENTS.values())
ALLOWED_FLAGS = abi.CUDA_GRAPH_CACHE | abi.CUDA_BF16_PSI | VARIANT_FLAGS


def variant_flags_from_env():
    tile = os.environ.get("BEAMZ_CUDA_MSTEP_TILE", "16x32").strip().lower()
    segment = os.environ.get("BEAMZ_CUDA_MSTEP_SEGMENT", "whole").strip().lower()
    if tile not in TILES:
        raise ValueError("BEAMZ_CUDA_MSTEP_TILE must be 16x32, 8x32, 8x16 or 16x16")
    if segment not in SEGMENTS:
        raise ValueError("BEAMZ_CUDA_MSTEP_SEGMENT must be whole, 32, 64, 128 or 256")
    return TILES[tile] | SEGMENTS[segment]


def variant_summary(flags):
    tiles = [name for name, bit in TILES.items() if bit and flags & bit]
    segments = [name for name, bit in SEGMENTS.items() if bit and flags & bit]
    if len(tiles) > 1 or len(segments) > 1:
        raise ValueError("Conflicting mstep tile or segment flags")
    tile = tiles[0] if tiles else "16x32"
    segment = int(segments[0]) if segments else 0
    return {"tile": tuple(map(int, tile.split("x"))), "column_segment": segment}


_AXES = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
_GEOMETRY_MODEL = "cpml12-queue-v1"
_MINIMUM_WORK_REDUCTION = 0.25
# (physical cells, canonical GCUPS, rotated GCUPS, additional startup seconds).
# Compact monitor-block ordering; two fresh rounds at 256 steps, checked at
# 4096 steps. Source: docs/reviews/cuda-schedule-space.json.gz, main-anchors
# after restoring streamed field addressing; horizons are earlier profiles.
# Archive at commit 5c389a04, linked from
# docs/reviews/pr334-post-merge-validation-2026-10-17.md.
_LAYOUT_COSTS = (
    (8388608, 5.2982, 6.9365, 1.1399),
    (16777216, 5.3617, 6.8345, 1.8993),
)


def _queue_work(shape):
    """Count 64x4 queue thread slots per phase, including inactive lanes.

    Mirrors LaunchCpmlQueueTile in cuda/src/update.cu for CPML12. Across
    either Yee triplet, the per-axis maximum extent is n+1 and minimum is n.
    The queue partitions z faces, then y faces, then x faces, then the core.
    This is a geometric work proxy, not a prediction of memory traffic or time;
    native fusion can replace the core with a different kernel.
    """
    z, y, x = shape
    iz, iy, ix = z - 24, y - 24, x - 24

    def tiles(n, width):
        return (n + width - 1) // width

    z_blocks = tiles(x + 1, 64) * tiles(y + 1, 4) * 25
    y_blocks = tiles(x + 1, 64) * tiles(25, 4) * iz
    x_blocks = tiles(25, 64) * tiles(iy, 4) * iz
    core_blocks = tiles(ix, 64) * tiles(iy, 4) * iz
    shell_slots = 256 * (z_blocks + y_blocks + x_blocks)
    core_slots = 256 * core_blocks
    return dict(
        shell_slots=shell_slots,
        core_slots=core_slots,
        total_slots=shell_slots + core_slots,
        slots_per_cell=(shell_slots + core_slots) / (z * y * x),
    )


def predict_layout(shape) -> dict[str, Any]:
    """Estimate an RTX3090 CPML12 storage order from logical (z,y,x) extents.

    Pure host arithmetic: no device queries, trials or persistent profiles.
    Only right-handed cyclic permutations are considered. Require a 25% queue
    work reduction to change layout: smaller proxy wins have not reliably
    predicted timing wins in the recorded shape study. This margin is a policy
    choice, not a measured speedup or statistical confidence bound. Keep 64x4:
    32x8 has no established benefit after repairing the narrow-x layout.

    Runtime selection separately checks device and supported workload scope.
    See docs/reviews/cuda-layout-prediction-2026-09-18.md for evidence and limits.
    """
    try:
        shape = tuple(operator.index(n) for n in shape)
    except TypeError as exc:
        raise ValueError("Expected three integer domain extents") from exc
    if len(shape) != 3 or min(shape) <= 24:
        raise ValueError("CPML12 layout prediction needs three extents above 24")
    candidates: dict[str, dict[str, Any]] = {
        "".join(map(str, axes)): dict(
            stored_shape=tuple(shape[a] for a in axes),
            **_queue_work(tuple(shape[a] for a in axes)),
        )
        for axes in _AXES
    }
    baseline = candidates["012"]["total_slots"]
    best = min(candidates, key=lambda name: candidates[name]["total_slots"])
    reduction = 1 - candidates[best]["total_slots"] / baseline
    selected = best if reduction >= _MINIMUM_WORK_REDUCTION else "012"
    return dict(
        model=_GEOMETRY_MODEL,
        choice=dict(axes=tuple(map(int, selected)), shell_tile="64x4"),
        logical_shape=shape,
        candidate_work=candidates,
        minimum_work_reduction=_MINIMUM_WORK_REDUCTION,
        best_work_reduction=reduction,
        reason=(
            "Substantial reduction in CPML queue thread slots"
            if selected != "012"
            else "Canonical layout retained; no substantial proxy improvement"
        ),
    )


def predict_layout_time_to_result(shape, steps, expected_runs=1):
    """Repay measured rotation startup; keep canonical outside this cost scope.

    Geometry remains a candidate generator. The measured cost family is the
    narrow-x, CPML12, packed-E case; runtime additionally checks the workload.
    """
    try:
        work_steps, runs = operator.index(steps), operator.index(expected_runs)
    except TypeError as exc:
        raise ValueError("steps and expected_runs must be positive integers") from exc
    if (
        isinstance(steps, bool)
        or isinstance(expected_runs, bool)
        or min(work_steps, runs) < 1
    ):
        raise ValueError("steps and expected_runs must be positive integers")
    report = predict_layout(shape)
    candidate = report["choice"]
    report = {
        **report,
        "candidate_choice": candidate,
        "choice": dict(axes=(0, 1, 2), shell_tile="64x4"),
    }
    z, y, x = report["logical_shape"]
    if candidate["axes"] == (0, 1, 2):
        return report
    if not (
        candidate["axes"] == (1, 2, 0)
        and 512 <= z <= 1024
        and y == 256
        and 63 <= x <= 65
    ):
        return {**report, "reason": "Rotated layout has no measured startup model"}
    cells = z * y * x
    small, large = _LAYOUT_COSTS
    fraction = np.clip(np.log2(cells / small[0]), 0, 1)
    canonical, rotated, extra = (
        float(a + fraction * (b - a)) for a, b in zip(small[1:], large[1:], strict=True)
    )
    saved = cells * work_steps * runs * (1 / canonical - 1 / rotated) / 1e9
    admitted = saved > 1.25 * extra
    return {
        **report,
        "cost_model": "rtx3090-layout-startup-v2",
        "choice": candidate if admitted else report["choice"],
        "expected_runs": expected_runs,
        "predicted_saved_execution_s": saved,
        "extra_startup_s": extra,
        "reason": "Predicted rotation savings repay startup"
        if admitted
        else "Predicted rotation savings do not repay startup",
    }
