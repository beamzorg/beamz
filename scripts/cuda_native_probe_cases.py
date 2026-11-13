"""Explicit complete-physics reliability packets, separate from benchmark physics."""

from __future__ import annotations

import numpy as np


def point_probe_packet(inputs, attributes, point, steps):
    """Keep seeded fields/psi, inject at a chosen site and gather all six fields.

    The source packet and direct observation are intentionally changed. A fresh
    ordinary Program reference is required; persisted scene references are invalid.
    Every source is multi-source coincident, matching the native arithmetic contract.
    """
    if len(inputs) != 134 or steps < 1:
        raise ValueError("Expected the complete134-input packet and positive steps")
    point = tuple(int(x) for x in point)
    if len(point) != 3 or any(
        not 0 <= point[a] < min(inputs[i].shape[a] for i in (2, 4)) for a in range(3)
    ):
        raise ValueError("Probe point must lie in both source supports (Hz/Ey)")
    changed = list(inputs)
    attrs = dict(attributes)
    attrs.update(nsteps=steps, coincident_source_group_mask=160,
                 disjoint_source_group_mask=0)
    for group in range(9):
        offset = 86 + 3 * group
        active = group in (5, 7)  # Hz and Ey, after their canonical phase.
        count = 2 if active else 0
        changed[offset] = np.full((count, 1, 1, 1), .5, np.float32)
        waveform = np.empty((count, steps), np.float32)
        if active:
            t = np.arange(steps, dtype=np.int32)
            waveform[0] = 1 + (t % 3).astype(np.float32) * np.float32(.125)
            waveform[1] = -.5 + (t % 5).astype(np.float32) * np.float32(.0625)
        changed[offset + 1] = waveform
        changed[offset + 2] = np.broadcast_to(
            np.asarray(point, np.int32), (count, 3)
        ).copy()
    indices = np.full_like(inputs[113], -1)
    weights = np.zeros_like(inputs[114])
    for component in range(6):
        native = (component + 3) % 6  # monitor Ex..Hz, packet Hx..Ez.
        if any(point[a] >= inputs[native].shape[a] for a in range(3)):
            continue  # Unsupported Yee components contribute exact zero.
        linear = np.ravel_multi_index(point, inputs[native].shape)
        indices[:, component, 0, 0] = linear
        weights[:, component, 0, 0] = 1
    changed[113], changed[114] = indices, weights
    return changed, attrs


def probe_points(inputs):
    """Include cache16/owned15 transitions, XY warp seams and high-side CPML."""
    high = [min(v.shape[a] for v in inputs[:6]) - 1 for a in range(3)]
    proposed = [(0, 15, 31), (14, 16, 32), (15, 15, 31),
                (29, 16, 32), (30, 16, 32), (31, 15, 31), tuple(high)]
    points = list(dict.fromkeys(tuple(min(max(0, p[a]), high[a]) for a in range(3))
                                for p in proposed))
    points.append((min(inputs[i].shape[0] for i in (2, 4)) - 1,
                   min(15, high[1]), min(31, high[2])))
    return list(dict.fromkeys(points))


def branch_witness(metadata, result):
    """Do not confuse a valid tail-only/fallback replay with positive coverage."""
    macros = int(result.get("moving_macro_count", 0))
    uniform = int(metadata.get("moving_uniform_bulk_tiles", 0))
    noio = int(metadata.get("moving_uniform_noio_tiles", 0))
    return {"actual_macro_count": macros, "certified_bulk_tiles": uniform,
            "positive_macro": macros > 0, "positive_uniform_branch": macros > 0 and uniform > 0,
            "certified_noio_tiles": noio, "positive_noio_branch": macros > 0 and noio > 0}
