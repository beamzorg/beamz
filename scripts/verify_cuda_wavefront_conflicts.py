"""Exhaustively check physical read/write conflicts between unordered DAG tasks.

Writes the fixed-fixture results to docs/reviews/rtx3090-wavefront-conflict-oracle.json.
This checks task footprints, not device memory ordering or numerical arithmetic.
"""

import itertools
import json
import math
from pathlib import Path

import numpy as np

results = []
for tile, depth, unit in [
    ((8, 8, 8), 8, False),
    ((16, 8, 16), 8, False),
    ((16, 8, 16), 8, True),
    ((8, 4, 8), 2, True),
    ((8, 4, 8), 4, True),
]:
    shape = (29, 31, 37)
    passes = 3
    delta = (1, 1, 1) if unit else tuple(v // depth for v in tile)
    grid = tuple(math.ceil(n / t) + 1 for n, t in zip(shape, tile, strict=True))
    nodes = sorted(
        itertools.product(range(passes), *(range(n) for n in grid)),
        key=lambda n: 4 * n[0] + sum(n[1:]),
    )
    ids = {n: i for i, n in enumerate(nodes)}
    anc = []
    writes = []
    reads = []
    for i, (p, *b) in enumerate(nodes):
        parents = []
        for axis in range(3):
            if b[axis]:
                prev = b.copy()
                prev[axis] -= 1
                parents.append((p, *prev))
        if p and all(b[a] + 1 < grid[a] for a in range(3)):
            parents.append((p - 1, *(v + 1 for v in b)))
        bits = 0
        for parent in parents:
            j = ids[parent]
            assert j < i
            bits |= anc[j] | (1 << j)
        anc.append(bits)
        w = np.zeros(shape, bool)
        for t in range(depth):
            lo = [max(0, b[a] * tile[a] - t * delta[a]) for a in range(3)]
            hi = [min(shape[a], (b[a] + 1) * tile[a] - t * delta[a]) for a in range(3)]
            if all(lower < upper for lower, upper in zip(lo, hi, strict=True)):
                w[
                    tuple(
                        slice(lower, upper) for lower, upper in zip(lo, hi, strict=True)
                    )
                ] = True
        r = w.copy()
        for axis in range(3):
            low = [slice(None)] * 3
            high = low.copy()
            low[axis] = slice(None, -1)
            high[axis] = slice(1, None)
            r[tuple(low)] |= w[tuple(high)]
            r[tuple(high)] |= w[tuple(low)]
        writes.append(int.from_bytes(np.packbits(w).tobytes(), "little"))
        reads.append(int.from_bytes(np.packbits(r).tobytes(), "little"))
    pairs = 0
    for i in range(len(nodes)):
        for j in range(i):
            if anc[i] & (1 << j):
                continue
            assert not (writes[i] & reads[j] or writes[j] & reads[i]), (
                tile,
                unit,
                nodes[i],
                nodes[j],
            )
            pairs += 1
    results.append(
        dict(
            shape=shape,
            tile=tile,
            depth=depth,
            unit_shift=unit,
            tasks=len(nodes),
            unordered_pairs_checked=pairs,
        )
    )
print(results)
Path("docs/reviews/rtx3090-wavefront-conflict-oracle.json").write_text(
    json.dumps(
        dict(
            note="Conservative combined H/E read/write footprint check for all incomparable DAG task pairs. No CUDA memory-order proof.",
            cases=results,
        ),
        indent=2,
    )
    + "\n"
)
