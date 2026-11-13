"""Independent bounds/ownership proof for an optional full-interior D4 lowering.

No CUDA execution. This proves admission and geometric access safety; arithmetic,
warp convergence, and complete-state parity remain separate GPU requirements.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def partition(limit, tile=(64, 8, 25), margin=16):
    low = (margin,) * 3
    high = tuple(n - margin for n in limit)
    cut = tuple(lo + ((hi - lo) // t) * t for lo, hi, t in zip(low, high, tile, strict=False))
    boxes = [("full", low, cut)]
    # Disjoint first-non-full-axis partition, in X,Y,Z order.
    boxes += [
        ("checked_x", (low[0], low[1], cut[2]), high),
        ("checked_y", (low[0], cut[1], low[2]), (high[0], high[1], cut[2])),
        ("checked_z", (cut[0], low[1], low[2]), (high[0], cut[1], cut[2])),
    ]
    return [b for b in boxes if all(a < c for a, c in zip(b[1], b[2], strict=False))]


def prove_access(origin, tile, limit, *, corrupt_margin=False):
    depth = 4
    if corrupt_margin:
        origin = (0, origin[1], origin[2])
    for k in range(8):
        lo = (k + 1) // 2
        width, height = tile[2] + 7 - k, tile[1] + 7 - k
        # Extremal active coordinates suffice for affine rectangular accesses.
        for wave, qy, qx in itertools.product(
            (k, tile[0] + 6), (lo, lo + height - 1), (lo, lo + width - 1)
        ):
            cell = (origin[0] + wave - depth - k // 2,
                    origin[1] - depth + qy, origin[2] - depth + qx)
            # Every computed output and coefficient access must be strictly
            # inside the common physical support, away from every PEC face.
            assert all(0 < a < n - 1 for a, n in zip(cell, limit, strict=False)), (k, cell, limit)
            if k == 0:
                # H0 reads old E at center and +1 on each derivative axis.
                for axis in range(3):
                    adjacent = list(cell)
                    adjacent[axis] += 1
                    assert all(0 <= a < n for a, n in zip(adjacent, limit, strict=False))
    return 8 * 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = []
    for limit in ((35, 49, 65), (64, 96, 128), (136, 136, 136),
                  (256, 256, 256), (272, 272, 272), (137, 139, 149)):
        tile = (64, 8, 25)
        boxes = partition(limit)
        expected_volume = 1
        volume = 0
        for n in limit:
            expected_volume *= n - 32
        for _, low, high in boxes:
            box_volume = 1
            for a, b in zip(low, high, strict=False):
                box_volume *= b - a
            volume += box_volume
        assert volume == expected_volume
        for (_, al, ah), (_, bl, bh) in itertools.combinations(boxes, 2):
            assert any(a <= b or c <= d for a, b, c, d in zip(ah, bl, bh, al, strict=False))
        full = next((b for b in boxes if b[0] == "full"), None)
        count = 0
        if full:
            for origin in itertools.product(*(range(a, b, t) for a, b, t in zip(full[1], full[2], tile, strict=False))):
                count += prove_access(origin, tile, limit)
        cases.append({"limit_zyx": limit, "interior_volume": volume,
                      "full_tiles": count // 64, "affine_extrema_checked": count,
                      "boxes": boxes})
    try:
        prove_access((16, 16, 16), (64, 8, 25), (136, 136, 136), corrupt_margin=True)
    except AssertionError:
        negative_control = "rejected"
    else:
        raise AssertionError("Missing-margin negative control was accepted")
    result = {"scope": "CPU affine bounds and disjoint ownership only", "cases": cases,
              "missing_margin_negative_control": negative_control}
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
