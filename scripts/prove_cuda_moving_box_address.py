"""Independent integer proof for relative moving-box trail addressing.

Enumerates all local cells and eight nonpositive neighbor offsets. Global
division/remainder is the oracle; the proposed lowering uses local masks and a
linear lower-neighbor tile delta. No floating point or CUDA execution.
"""

from __future__ import annotations

import itertools
import json


def surface(z, y, x):
    cells = [(a, b, c) for a in range(4) for b in range(8) for c in range(32)
             if a == 3 or b == 7 or c == 31]
    assert len(cells) == 373 and len(set(cells)) == 373
    if z == 3:
        return y * 32 + x
    if y == 7:
        return 8 * 32 + z * 32 + x
    if x == 31:
        return 8 * 32 + 3 * 32 + z * 7 + y
    return -1


def main():
    checked = 0
    surface_indices = {surface(z, y, x) for z, y, x in itertools.product(range(4), range(8), range(32))}
    assert surface_indices == set(range(-1, 373))
    for bz, by, bx in itertools.product((0, 1, 2), repeat=3):
        nx, ny = 5, 7
        owner = (bz * ny + by) * nx + bx
        for lz, ly, lx in itertools.product(range(4), range(8), range(32)):
            u = (4 * bz + lz, 8 * by + ly, 32 * bx + lx)
            for delta in itertools.product((-1, 0), repeat=3):
                q = tuple(a + b for a, b in zip(u, delta, strict=True))
                if min(q) < 0:
                    continue
                source_box = tuple(a // b for a, b in zip(q, (4, 8, 32), strict=True))
                oracle_tile = (source_box[0] * ny + source_box[1]) * nx + source_box[2]
                oracle_local = tuple(a % b for a, b in zip(q, (4, 8, 32), strict=True))
                shifted = tuple(a + b for a, b in zip((lz, ly, lx), delta, strict=True))
                lowered_tile = owner - (shifted[0] < 0) * nx * ny - (shifted[1] < 0) * nx - (shifted[2] < 0)
                lowered_local = tuple(a & b for a, b in zip(shifted, (3, 7, 31), strict=True))
                assert (lowered_tile, lowered_local) == (oracle_tile, oracle_local)
                if oracle_tile != owner:
                    assert surface(*oracle_local) >= 0
                    assert sum(source_box) < bz + by + bx
                checked += 1
    print(json.dumps({"scope": __doc__, "address_cases_checked": checked,
                      "surface_indices": 373, "result": "pass"}, indent=2))


if __name__ == "__main__":
    main()
