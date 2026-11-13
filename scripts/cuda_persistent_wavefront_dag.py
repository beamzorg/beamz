"""Trace moving-tile RAW and trail-slot WAR dependencies for a ready queue.

This is a CPU scheduling/proof model. It does not run a persistent CUDA kernel
or predict hardware throughput. In particular, dropping rank joins requires
slot-reuse anti-dependencies even when all field RAW dependencies are retained.
"""
from __future__ import annotations

import heapq
import itertools
from collections import defaultdict, deque


def tile_rank(tile, counts):
    _, ny, nx = counts
    return tile // (ny * nx) + (tile // nx) % ny + tile % nx


def raw_successors(counts):
    nz, ny, nx = counts
    result = [set() for _ in range(nz * ny * nx)]
    for z, y, x in itertools.product(range(nz), range(ny), range(nx)):
        tile = (z * ny + y) * nx + x
        for mask in range(1, 8):
            zz, yy, xx = z + (mask & 1), y + ((mask >> 1) & 1), x + ((mask >> 2) & 1)
            if zz < nz and yy < ny and xx < nx:
                result[tile].add((zz * ny + yy) * nx + xx)
    return result


def rank_slots(counts):
    ranks = defaultdict(list)
    for tile in range(counts[0] * counts[1] * counts[2]):
        ranks[tile_rank(tile, counts)].append(tile)
    slots = [0] * sum(map(len, ranks.values()))
    for nodes in ranks.values():
        for slot, tile in enumerate(nodes):
            slots[tile] = slot
    return slots, max(map(len, ranks.values()))


def build(counts, slots, frontier, epochs=4, *, retain_reuse_edges=True):
    counts = tuple(int(n) for n in counts)
    if len(counts) != 3 or min(counts) < 1 or epochs < 4 or frontier < 1:
        raise ValueError("Three positive dimensions and at least4 trail epochs required")
    n = counts[0] * counts[1] * counts[2]
    if len(slots) != n:
        raise ValueError("Actual per-tile slot mapping required")
    slots = tuple(int(v) for v in slots)
    ranks = tuple(tile_rank(t, counts) for t in range(n))
    for rank in set(ranks):
        values = [slots[t] for t in range(n) if ranks[t] == rank]
        if len(set(values)) != len(values) or min(values) < 0 or max(values) >= frontier:
            raise ValueError("Trail slots must be unique within each rank and in range")
    raw = raw_successors(counts)
    successors = [set(x) for x in raw]
    holders = defaultdict(list)
    for tile in range(n):
        holders[(ranks[tile] % epochs) * frontier + slots[tile]].append(tile)
    previous = {}
    for nodes in holders.values():
        nodes.sort(key=lambda t: ranks[t])
        for old, new in zip(nodes, nodes[1:]):
            previous[new] = old
            if retain_reuse_edges:
                successors[old].add(new)
                for consumer in raw[old]:
                    successors[consumer].add(new)
    predecessors = [set() for _ in range(n)]
    for src, targets in enumerate(successors):
        for target in targets:
            if ranks[src] >= ranks[target]:
                raise ValueError("Resource dependency is not strictly forward in rank")
            predecessors[target].add(src)
    return {"counts": counts, "slots": slots, "frontier": frontier, "epochs": epochs,
            "ranks": ranks, "raw_successors": raw, "successors": successors,
            "predecessors": predecessors, "previous_slot_holder": previous,
            "raw_edges": sum(map(len, raw)), "all_edges": sum(map(len, successors)),
            "retains_slot_reuse_antidependencies": retain_reuse_edges}


def simulate(dag, durations, workers, *, check_slot_lifetimes=True):
    """Async completion model; ready jobs use deterministic rank/tile priority."""
    n = len(dag["ranks"])
    if len(durations) != n or workers < 1 or min(durations) <= 0:
        raise ValueError("Positive durations and worker count required")
    pending = [len(x) for x in dag["predecessors"]]
    ready = [(dag["ranks"][t], t) for t, d in enumerate(pending) if not d]
    heapq.heapify(ready)
    running, done = [], set()
    started = set()
    clock = 0
    starts, ends = [None] * n, [None] * n
    max_running = 0
    while ready or running:
        while ready and len(running) < workers:
            _, tile = heapq.heappop(ready)
            if tile in started or not dag["predecessors"][tile] <= done:
                raise AssertionError("Ready-queue ownership or RAW completion failed")
            old = dag["previous_slot_holder"].get(tile)
            if check_slot_lifetimes and old is not None:
                if old not in done or not dag["raw_successors"][old] <= done:
                    raise AssertionError("Trail slot overwritten before its old readers retire")
            started.add(tile)
            starts[tile] = clock
            heapq.heappush(running, (clock + durations[tile], tile))
            max_running = max(max_running, len(running))
        if not running:
            raise AssertionError("Ready queue deadlocked with incomplete work")
        clock = running[0][0]
        finished = []
        while running and running[0][0] == clock:
            _, tile = heapq.heappop(running)
            done.add(tile)
            ends[tile] = clock
            finished.append(tile)
        for tile in finished:
            for child in dag["successors"][tile]:
                pending[child] -= 1
                if pending[child] == 0:
                    heapq.heappush(ready, (dag["ranks"][child], child))
    if len(done) != n or len(started) != n or any(pending):
        raise AssertionError("Incomplete or duplicated queue coverage")
    # Unlimited-worker lower bound under these fixed input durations.
    longest = [0] * n
    for tile in sorted(range(n), key=lambda t: (dag["ranks"][t], t)):
        longest[tile] = durations[tile] + max((longest[p] for p in dag["predecessors"][tile]), default=0)
    return {"tiles": n, "workers": workers, "makespan": clock,
            "maximum_running": max_running, "work_lower_bound": sum(durations) / workers,
            "critical_path_lower_bound": max(longest), "starts": starts, "ends": ends,
            "scope": "CPU fixed-duration scheduling model; not measured CUDA execution or speedup"}


def csr(dag):
    offsets, edges = [0], []
    for children in dag["successors"]:
        edges.extend(sorted(children))
        offsets.append(len(edges))
    return {"offsets": offsets, "edges": edges,
            "initial_indegrees": [len(x) for x in dag["predecessors"]],
            "initial_ready": [i for i, x in enumerate(dag["predecessors"]) if not x]}
