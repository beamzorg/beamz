"""Independent transfer/ownership proof for the register1024 pure-X packets.

Token values use physical point/component identities. Exhausts all ring heads,
thread groups, and fixed private slots; checks support clipping, packed surface
ownership and disjoint terminal writes. Memory-order validity additionally
requires the documented B1/B2/B3/B4 protocol and CUDA sanitizer gates.
"""

from __future__ import annotations

import json


def prove(steps=32, origin=(14, 30, 62), shape=(37, 39, 73)):
    cache = (8, 16, 32)
    owned = (7, 15, 31)
    points = owned[0] * owned[1]
    shapes = (
        (shape[0], shape[1], shape[2] + 1),
        (shape[0], shape[1] + 1, shape[2]),
        (shape[0] + 1, shape[1], shape[2]),
        (shape[0] + 1, shape[1] + 1, shape[2]),
        (shape[0] + 1, shape[1], shape[2] + 1),
        (shape[0], shape[1] + 1, shape[2] + 1),
    )
    reads = writes = 0

    def surface(z, y, x):
        if z == 6:
            return y * 31 + x
        if y == 14:
            return 15 * 31 + z * 31 + x
        if x == 30:
            return 15 * 31 + 6 * 31 + z * 14 + y
        return -1

    def token(field, n, p):
        return (
            (field, n, p)
            if all(0 <= q < extent for q, extent in zip(p, shapes[field], strict=True))
            else ("zero",)
        )

    for n in range(1, steps + 1):
        hz, hy, hx = ((-n - 1) % extent for extent in cache)
        low = tuple(q - n for q in origin)
        staged_low = {}
        for i in range(points):
            dz, dy = 1 + i // 15, 1 + i % 15
            p = (origin[0] - 1 + dz - n, origin[1] - 1 + dy - n, origin[2] - 1 - n)
            assert surface(dz - 1, dy - 1, 30) >= 0
            for c in range(6):
                staged_low[c * points + i] = token(c, n, p)
        staged_high = {}
        normal_terminal = set()
        for thread in range(1024):
            sx, sy = thread % 32, (thread // 32) % 16
            dy, dx = (sy - hy) % 16, (sx - hx) % 32
            for j in range(4):
                zslot = thread // 512 + j * 2
                dz = (zslot - hz) % 8
                p = (
                    origin[0] - 1 + dz - n,
                    origin[1] - 1 + dy - n,
                    origin[2] - 1 + dx - n,
                )
                if sx == hx and sy != hy and zslot != hz:
                    key = (dz - 1) * 15 + dy - 1
                    assert 0 <= key < points
                    for c in range(6):
                        assert staged_low[c * points + key] == token(c, n, p)
                        reads += 1
                if sx == (hx + 31) % 32 and 0 < dy < 15 and 0 < dz < 7:
                    key = (dz - 1) * 15 + dy - 1
                    for c in range(6):
                        assert c * points + key not in staged_high
                        staged_high[c * points + key] = token(c, n, p)
                if dz and dy and dx:
                    out_surface = surface(dz - 1, dy - 1, dx - 1)
                    if out_surface < 15 * 31 + 6 * 31:
                        for c in range(6):
                            if token(c, n, p) != ("zero",):
                                normal_terminal.add((c, p))
        high_terminal = set()
        for i in range(6 * 14):
            z, y = i // 14, i % 14
            key = z * 15 + y
            p = (origin[0] + z - n, origin[1] + y - n, origin[2] + 30 - n)
            assert surface(z, y, 30) == 15 * 31 + 6 * 31 + i
            for c in range(6):
                assert staged_high[c * points + key] == token(c, n, p)
                if token(c, n, p) != ("zero",):
                    high_terminal.add((c, p))
                writes += 1
        assert normal_terminal.isdisjoint(high_terminal)
        expected = {
            (c, (z, y, x))
            for z in range(low[0], low[0] + 7)
            for y in range(low[1], low[1] + 15)
            for x in range(low[2], low[2] + 31)
            for c in range(6)
            if token(c, n, (z, y, x)) != ("zero",)
        }
        assert normal_terminal | high_terminal == expected
    return {
        "steps": steps,
        "origin": origin,
        "shape": shape,
        "low_scalar_reads_verified": reads,
        "high_scalar_writes_verified": writes,
        "low_high_stage_ranges_disjoint": True,
        "terminal_writers_disjoint": True,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            [prove(origin=o) for o in ((0, 0, 0), (7, 15, 31), (14, 30, 62))], indent=2
        )
    )
