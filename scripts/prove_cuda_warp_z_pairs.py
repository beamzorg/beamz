"""Exhaustive lane routing for a FP32 paper-sized, paired-Z register cache.

Cache16x16x32;1024threads;48 retained field scalars per thread. A warp holds
16 X positions at two adjacent Z parities. Shared X/Z frames exchange Y/Z/X
neighbors; a512-float Y edge frame supplies only cross-warp X-half neighbors.
This proves operand routing, not compiler resources or device synchronization.
"""

from __future__ import annotations

import json

CACHE = (16, 16, 32)
THREADS = 1024
SLOTS = 8


def owner(z, y, x):
    z, y, x = z % 16, y % 16, x % 32
    return ((2 * y + x // 16) * 32 + (z % 2) * 16 + x % 16, z // 2)


def position(thread, j):
    warp, lane = divmod(thread, 32)
    return (2 * j + lane // 16, warp // 2, 16 * (warp % 2) + lane % 16)


def prove(*, wrong_parity=False, wrong_x_block=False, full_width_shuffle=False):
    occupied = {owner(z, y, x) for z in range(16) for y in range(16) for x in range(32)}
    assert len(occupied) == 16 * 16 * 32
    assert all(
        position(t, j) == (z, y, x)
        for z in range(16)
        for y in range(16)
        for x in range(32)
        for t, j in [owner(z, y, x)]
    )
    checked = cross_half = parity = 0
    for phase in (0, 1):
        shift = 1 if phase == 0 else -1
        for thread in range(THREADS):
            warp, lane = divmod(thread, 32)
            for j in range(SLOTS):
                z, y, x = position(thread, j)
                for axis in (0, 2):
                    expected = (
                        ((z + shift) % 16, y, x)
                        if axis == 0
                        else (z, y, (x + shift) % 32)
                    )
                    if axis == 0:
                        source_thread = thread ^ 16
                        source_parity = (source_thread % 32) // 16
                        # A shuffle reads the operand computed BY THE SOURCE lane.
                        if phase == 0:
                            source_j = j if source_parity else (j + 1) % 8
                        else:
                            source_j = (j - 1) % 8 if source_parity else j
                        if wrong_parity:
                            source_j = (j + shift) % 8 if source_parity else j
                        actual = position(source_thread, source_j)
                        parity += 1
                    elif (lane % 16 == 15 and shift == 1) or (
                        lane % 16 == 0 and shift == -1
                    ):
                        # E frame publishes first columns; H frame last columns.
                        source_block = ((x + shift) % 32) // 16
                        if wrong_x_block:
                            source_block ^= 1
                        source_x = source_block * 16 + (0 if shift == 1 else 15)
                        actual = (z, y, source_x)
                        cross_half += 1
                    else:
                        width = 32 if full_width_shuffle else 16
                        source_lane = (lane // width) * width + (lane + shift) % width
                        actual = position(warp * 32 + source_lane, j)
                    assert actual == expected, (
                        phase,
                        thread,
                        j,
                        axis,
                        actual,
                        expected,
                    )
                    checked += 1
    return {
        "cache": CACHE,
        "threads": THREADS,
        "private_field_scalars_per_thread": 48,
        "neighbor_routes_verified": checked,
        "cross_half_Y_routes": cross_half,
        "paired_Z_routes": parity,
        "shared_Y_edge_floats": 512,
    }


if __name__ == "__main__":
    print(json.dumps(prove(), indent=2))
