"""Experimental traced dependency tightening for the existing mstep planner.

Only remove unnecessary first-phase queue entries. Existing ownership, compact
slots, scratch allocation, source exclusions and native ABI stay unchanged.
This is injected by a research worker, not selected by Simulation defaults.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

try:
    from scripts.cuda_stencil_trace import trace_yee_step
except ModuleNotFoundError:
    from cuda_stencil_trace import trace_yee_step


def intermediate_offsets():
    """Derive H1/E1 publication support needed for arbitrary H2/E2 edge cells.

    H2 reads E1 at forward curl neighbours and old H1 at center. Those E1
    samples require H1 at backward curl neighbours. Union component supports
    because the existing native queue advances all three components together.
    """
    first = {update.output.field: update for update in trace_yee_step(0)}
    offsets = {(0, 0, 0)}
    for update in trace_yee_step(1)[:3]:
        for read in update.reads:
            offsets.add(read.offset)
            if read.field.startswith("E"):
                for parent in first[read.field].reads:
                    if parent.field.startswith("H"):
                        offsets.add(
                            tuple(
                                a + b
                                for a, b in zip(read.offset, parent.offset, strict=True)
                            )
                        )
    return tuple(sorted(offsets))


def _shift_into(target, source, dy, dx):
    ny, nx = target.shape
    target[max(0, dy) : min(ny, ny + dy), max(0, dx) : min(nx, nx + dx)] |= source[
        max(0, -dy) : min(ny, ny - dy), max(0, -dx) : min(nx, nx - dx)
    ]


def _monitor_rows(field_shapes, monitor_indices, shape):
    if monitor_indices is None:
        return {}
    nz, ny, nx = shape
    requests = []
    for c, dims in enumerate((*field_shapes[3:], *field_shapes[:3])):
        rows = np.asarray(monitor_indices)[:, c].reshape(-1)
        rows = np.unique(rows[(rows >= 0) & (rows < np.prod(dims))])
        z, rest = np.divmod(rows, dims[1] * dims[2])
        y, x = np.divmod(rest, dims[2])
        for dz, dy, dx in ((0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1)):
            zz, yy, xx = z + dz, y + dy, x + dx
            valid = (
                (zz >= 0) & (zz < nz) & (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
            )
            requests.append((zz[valid] * ny + yy[valid]) * nx + xx[valid])
    if not requests:
        return {}
    rows = np.unique(np.concatenate(requests))
    z, xy = np.divmod(rows, ny * nx)
    planes, starts = np.unique(z, return_index=True)
    return {
        int(plane): xy[start:end]
        for plane, start, end in zip(
            planes, starts, (*starts[1:], rows.size), strict=True
        )
    }


def tighten_first_queue(plan, field_shapes, monitor_indices=None):
    """Return a new plan with traced publication support, preserving allocations.

    Every CPML/source cell remains an edge cell, so all auxiliary recurrences
    still advance. Monitoring requests are preserved exactly, including their
    backward-neighbour H1 dependencies. Unneeded computed halo values can be
    discarded; their dependencies need not recursively expand the queue.
    """
    nz, ny, nx = plan.shape
    masks = plan.core_masks
    core = np.zeros((masks.shape[0] * 2, masks.shape[1] * 2), bool)
    for dy in range(2):
        for dx in range(2):
            core[dy::2, dx::2] = (masks & (1 << (2 * dy + dx))) != 0
    edge = ~np.repeat(np.repeat(core, 8, axis=0), 16, axis=1)[:ny, :nx]
    all_xy = np.ones((ny, nx), bool)
    zb, zt = map(int, plan.bounds)
    watched = _monitor_rows(field_shapes, monitor_indices, plan.shape)
    offsets = intermediate_offsets()
    cached = {}
    first = []
    plane_size = ny * nx
    for z in range(nz):
        key = tuple((0 <= z - dz < nz, zb <= z - dz < zt) for dz in (-1, 0, 1))
        if key not in cached:
            needed = np.zeros((ny, nx), bool)
            for dz, dy, dx in offsets:
                valid, is_core = key[dz + 1]
                if valid:
                    _shift_into(needed, edge if is_core else all_xy, dy, dx)
            cached[key] = needed
        needed = cached[key]
        if z in watched:
            needed = needed.copy()
            needed.reshape(-1)[watched[z]] = True
        begin, end = np.searchsorted(
            plan.first_cells, (z * plane_size, (z + 1) * plane_size)
        )
        rows = plan.first_cells[begin:end]
        first.append(rows[needed.reshape(-1)[rows - z * plane_size]])
    cells = np.concatenate(first).astype(np.int32, copy=False)
    cells.setflags(write=False)
    return replace(plan, first_cells=cells)
