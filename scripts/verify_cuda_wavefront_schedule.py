"""Check tiled in-place Yee scheduling with integer field-version oracles.

No floating-point solver or performance claim: every field/CPML update carries
its timestep, and every stencil read must observe the required version. Tile
windows slide by tile_size/depth, generalizing the CPU's cubic group schedule.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np


def verify(shape, tile, depth, passes, seed, *, unit_shift=False, omit_temporal=False):
    assert all(t % depth == 0 for t in tile)
    shift = (1, 1, 1) if unit_shift else tuple(t // depth for t in tile)
    grid = tuple(math.ceil(n / t) + 1 for n, t in zip(shape, tile, strict=True))
    nodes = list(itertools.product(range(passes), *(range(n) for n in grid)))
    indegree = {}
    successors = {node: [] for node in nodes}
    for node in nodes:
        p, *b = node
        parents = []
        for axis in range(3):
            if b[axis]:
                parent = b.copy()
                parent[axis] -= 1
                parents.append((p, *parent))
        if p and not omit_temporal and all(b[a] + 1 < grid[a] for a in range(3)):
            parents.append((p - 1, *(v + 1 for v in b)))
        indegree[node] = len(parents)
        for parent in parents:
            successors[parent].append(node)
    ready = [node for node in nodes if indegree[node] == 0]
    rng = random.Random(seed)
    h = np.zeros(shape, np.int32)
    e = np.zeros(shape, np.int32)
    psi_h = np.zeros(shape, np.int32)
    psi_e = np.zeros(shape, np.int32)
    coordinates = np.indices(shape)
    pml = np.logical_or.reduce(
        [(coordinates[a] < 12) | (coordinates[a] >= shape[a] - 12) for a in range(3)]
    )
    source = (shape[0] // 2, 12, shape[2] // 2)
    monitor = (shape[0] // 2, shape[1] - 13, shape[2] // 2)
    source_times, monitor_times = [], []
    visits = 0
    completed = 0

    def require(actual, expected, what, node, local):
        if not np.array_equal(actual, expected):
            mismatch = np.argwhere(actual != expected)[0].tolist()
            raise AssertionError(
                f"{what}: node={node}, local={local}, slice index={mismatch}"
            )

    while ready:
        index = rng.randrange(len(ready))
        node = ready[index]
        ready[index] = ready[-1]
        ready.pop()
        _, *b = node
        for local in range(depth):
            lo = tuple(max(0, b[a] * tile[a] - local * shift[a]) for a in range(3))
            hi = tuple(
                min(shape[a], (b[a] + 1) * tile[a] - local * shift[a]) for a in range(3)
            )
            if any(lower >= upper for lower, upper in zip(lo, hi, strict=True)):
                continue
            region = tuple(
                slice(lower, upper) for lower, upper in zip(lo, hi, strict=True)
            )
            old = h[region].copy()
            assert np.all(old == node[0] * depth + local), (node, local)
            require(e[region], old, "H center E read", node, local)
            for axis in range(3):
                begin, end = list(lo), list(hi)
                end[axis] = min(end[axis], shape[axis] - 1)
                own = tuple(
                    slice(lower, upper) for lower, upper in zip(begin, end, strict=True)
                )
                begin[axis] += 1
                end[axis] += 1
                other = tuple(
                    slice(lower, upper) for lower, upper in zip(begin, end, strict=True)
                )
                require(e[other], h[own], "H forward E read", node, local)
            require(psi_h[region][pml[region]], old[pml[region]], "H psi", node, local)
            h[region] += 1
            psi_h[region] += pml[region]
            # All H writes in the tile are visible before E starts.
            require(h[region], e[region] + 1, "E center H read", node, local)
            for axis in range(3):
                begin, end = list(lo), list(hi)
                begin[axis] = max(begin[axis], 1)
                own = tuple(
                    slice(lower, upper) for lower, upper in zip(begin, end, strict=True)
                )
                begin[axis] -= 1
                end[axis] -= 1
                other = tuple(
                    slice(lower, upper) for lower, upper in zip(begin, end, strict=True)
                )
                require(h[other], e[own] + 1, "E backward H read", node, local)
            require(psi_e[region][pml[region]], old[pml[region]], "E psi", node, local)
            e[region] += 1
            psi_e[region] += pml[region]
            visits += math.prod(
                upper - lower for lower, upper in zip(lo, hi, strict=True)
            )
            for point, times in ((source, source_times), (monitor, monitor_times)):
                if all(lo[a] <= point[a] < hi[a] for a in range(3)):
                    times.append(int(e[point]))
        completed += 1
        for successor in successors[node]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    expected = passes * depth
    assert completed == len(nodes)
    assert np.all(h == expected) and np.all(e == expected)
    assert np.all(psi_h[pml] == expected) and np.all(psi_e[pml] == expected)
    assert np.all(psi_h[~pml] == 0) and np.all(psi_e[~pml] == 0)
    assert source_times == list(range(1, expected + 1))
    assert monitor_times == list(range(1, expected + 1))
    assert visits == math.prod(shape) * expected
    return dict(
        shape=shape,
        tile=tile,
        depth=depth,
        passes=passes,
        seed=seed,
        shift=shift,
        nodes=completed,
        cell_timesteps=visits,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for shape in ((29, 31, 37), (37, 29, 31), (31, 37, 29)):
        for tile, depth in (((8, 8, 8), 8), ((16, 8, 16), 8), ((16, 8, 16), 4)):
            for seed in range(4):
                results.append(verify(shape, tile, depth, 3, seed))
    # Rectangular windows with unit shift also pass: the CPU edge set can
    # overconstrain execution without becoming incorrect. Keep both choices.
    for seed in range(4):
        results.append(verify((29, 31, 37), (16, 8, 16), 8, 3, seed, unit_shift=True))
    for shape in ((29, 31, 37), (37, 29, 31), (31, 37, 29)):
        for depth in (2, 4):
            for seed in range(4):
                results.append(
                    verify(shape, (8, 4, 8), depth, 3, seed, unit_shift=True)
                )
    try:
        verify((29, 31, 37), (16, 8, 16), 8, 3, 0, omit_temporal=True)
    except AssertionError as error:
        negative_control = str(error)
    else:
        raise AssertionError("Missing-temporal-edge control unexpectedly passed")
    args.output.write_text(
        json.dumps(
            dict(
                note="Integer version oracle, not numerical or GPU race validation.",
                cases=results,
                rejected_missing_temporal_edge=negative_control,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"{len(results)} schedules passed; missing-temporal-edge control rejected")


if __name__ == "__main__":
    main()
