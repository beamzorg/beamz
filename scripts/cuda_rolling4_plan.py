"""Conservative four-step interior/chronological-edge dependency envelopes.

Pure NumPy planning; no CUDA or JAX initialization. The private rolling-D4
candidate advances source/CPML-free columns in one kernel, while first three
edge/observation envelopes evolve in place and the final edge writes bank B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rolling4Plan:
    shape: tuple[int, int, int]
    bounds: np.ndarray
    core_masks: np.ndarray
    first_cells: np.ndarray
    edge_cells: np.ndarray
    core_cells: int
    observed_planes: int

    def summary(self):
        return {
            "depth": 4,
            "shape": list(self.shape),
            "bounds": self.bounds.tolist(),
            "core_cells": self.core_cells,
            "core_fraction": self.core_cells / int(np.prod(self.shape)),
            "first_envelope_cells": self.first_cells.size,
            "final_edge_cells": self.edge_cells.size,
            "observed_planes": self.observed_planes,
            "first_envelope_radius": 3,
            "monitor_seed_radius": 2,
            "plan_bytes": sum(
                x.nbytes
                for x in (
                    self.bounds,
                    self.core_masks,
                    self.first_cells,
                    self.edge_cells,
                )
            ),
        }


def dilate_xy(mask, radius):
    padded = np.pad(mask, ((0, 0), (radius, radius)), constant_values=False)
    horizontal = np.logical_or.reduce(
        [padded[:, dx : dx + mask.shape[1]] for dx in range(2 * radius + 1)]
    )
    padded = np.pad(horizontal, ((radius, radius), (0, 0)), constant_values=False)
    return np.logical_or.reduce(
        [padded[dy : dy + mask.shape[0]] for dy in range(2 * radius + 1)]
    )


def observed_envelope(field_shapes, indices, shape):
    nz, ny, nx = shape
    requests = []
    if indices is None:
        return {}
    # Packed monitor component order E,H; native packet order H,E.
    for component, dims in enumerate((*field_shapes[3:], *field_shapes[:3])):
        ids = np.asarray(indices)[:, component].reshape(-1)
        ids = np.unique(ids[(ids >= 0) & (ids < np.prod(dims))])
        z, rest = np.divmod(ids, dims[1] * dims[2])
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
    planes, xy = np.divmod(rows, ny * nx)
    zs, starts = np.unique(planes, return_index=True)
    watched = {}
    for z, start, end in zip(zs, starts, (*starts[1:], rows.size), strict=True):
        mask = np.zeros((ny, nx), bool)
        mask.reshape(-1)[xy[start:end]] = True
        expanded = dilate_xy(mask, 2)
        for target in range(max(0, int(z) - 2), min(nz, int(z) + 3)):
            if target not in watched:
                watched[target] = expanded.copy()
            else:
                watched[target] |= expanded
    return watched


def build_rolling4_plan(field_shapes, thickness, groups, monitor_indices=None):
    """groups contains (coefficient_shape[S,Z,Y,X], starts[S,3]) or None."""
    if len(field_shapes) != 6 or any(len(s) != 3 or min(s) < 1 for s in field_shapes):
        raise ValueError("Six positive staggered field shapes required")
    if thickness < 1:
        raise ValueError("Positive uniform CPML thickness required")
    shape = tuple(int(max(s[a] for s in field_shapes)) for a in range(3))
    nz, ny, nx = shape
    if np.prod(shape) >= np.iinfo(np.int32).max:
        raise ValueError("Rolling4 cell offsets exceed int32")
    minimum = np.min(field_shapes, axis=0)
    low = np.full(3, thickness, dtype=int)
    high = minimum - thickness - 1
    zb, zt = int(low[0] + 4), int(high[0] - 2)
    nq_y = (ny + 7) // 8
    nq_x = (nx + 15) // 16
    nq_y += nq_y % 2
    nq_x += nq_x % 2
    qy = np.arange(nq_y)[:, None] * 8
    qx = np.arange(nq_x)[None, :] * 16
    core = (qy - 4 >= low[1]) & (qy + 8 + 3 <= high[1])
    core = core & (qx - 4 >= low[2]) & (qx + 16 + 3 <= high[2])
    if zt <= zb:
        core[:] = False
    for group in groups:
        if group is None:
            continue
        coeff_shape, starts = group
        extent = np.asarray(coeff_shape[1:])
        for start in np.asarray(starts):
            end = start + extent
            if np.any(end <= 0) or np.any(start >= shape):
                continue
            intersects = (qy - 4 < end[1]) & (qy + 8 + 4 > start[1])
            intersects = intersects & (qx - 4 < end[2]) & (qx + 16 + 4 > start[2])
            core[intersects] = False
    masks = sum(
        core[dy::2, dx::2].astype(np.int32) << (2 * dy + dx)
        for dy in range(2)
        for dx in range(2)
    )
    edge_xy = ~np.repeat(np.repeat(core, 8, axis=0), 16, axis=1)[:ny, :nx]
    first_xy = dilate_xy(edge_xy, 3)
    # Q includes all z outside owned columns, hence any missing XY-domain
    # neighbours are also in Q; outside XY is immaterial to valid outputs.
    watched = observed_envelope(field_shapes, monitor_indices, shape)
    all_xy = np.arange(ny * nx, dtype=np.int32)
    edge_rows = np.flatnonzero(edge_xy).astype(np.int32)
    first = []
    edge = []
    for z in range(nz):
        edge.append((edge_rows if zb <= z < zt else all_xy) + z * ny * nx)
        mask = first_xy.copy() if zb + 3 <= z < zt - 3 else np.ones((ny, nx), bool)
        if z in watched:
            mask |= watched[z]
        first.append(np.flatnonzero(mask).astype(np.int32) + z * ny * nx)
    return Rolling4Plan(
        shape,
        np.asarray([zb, zt], np.int32),
        np.asarray(masks, np.int32),
        np.concatenate(first),
        np.concatenate(edge),
        int(core.sum()) * 128 * max(0, zt - zb),
        len(watched),
    )
