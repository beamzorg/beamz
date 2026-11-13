#!/usr/bin/env python3
"""Independent Yee dependency closure proof for rolling-D4 static envelopes."""

from __future__ import annotations

import argparse
import itertools
import json
from functools import lru_cache
from pathlib import Path

FIRST = ((2, 1), (0, 0), (1, 2))
SECOND = ((1, 0), (2, 2), (0, 1))


def shifted(point, axis, delta):
    q = list(point)
    q[axis] += delta
    return tuple(q)


@lru_cache(None)
def closure(family, version, component, point):
    node = (family, version, component, point)
    if version == 0:
        return frozenset([node])
    dependencies = [(family, version - 1, component, point)]
    for source, axis in (FIRST[component], SECOND[component]):
        if family == 0:
            dependencies.extend(
                [
                    (1, version - 1, source, point),
                    (1, version - 1, source, shifted(point, axis, 1)),
                ]
            )
        else:
            dependencies.extend(
                [
                    (0, version, source, point),
                    (0, version, source, shifted(point, axis, -1)),
                ]
            )
    result = {node}
    for dep in dependencies:
        result.update(closure(*dep))
    return frozenset(result)


def prove():
    origin = (0, 0, 0)
    core = set().union(
        *(closure(family, 4, c, origin) for family in [0, 1] for c in range(3))
    )
    updates = {n for n in core if n[1] > 0}
    points = [n[3] for n in updates]
    minimum = [min(p[a] for p in points) for a in range(3)]
    maximum = [max(p[a] for p in points) for a in range(3)]
    assert minimum == [-4] * 3 and maximum == [3] * 3
    # Final E4 reads complete H4 from bank B. Its H4 neighbours are either
    # produced by the bulk or belong to Q themselves; they are not A3 ghosts.
    # For every Q cell it is enough to have E3 old fields and H4's dependencies.
    edge = set().union(
        *(closure(0, 4, c, origin) for c in range(3)),
        *(closure(1, 3, c, origin) for c in range(3)),
    )
    required = {n for n in edge if 0 < n[1] < 4}
    assert all(max(map(abs, n[3])) <= 3 for n in required)
    seeds = [origin, (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    envelope = {
        tuple(seed[a] + delta[a] for a in range(3))
        for seed in seeds
        for delta in itertools.product(range(-2, 3), repeat=3)
    }
    observation = set().union(
        *(
            closure(family, v, c, origin)
            for family in [0, 1]
            for v in [1, 2, 3]
            for c in range(3)
        )
    )
    assert all(n[3] in envelope for n in observation if n[1] > 0)
    # A radius2 Q margin is not enough: retain a concrete missing dependency.
    negative = next(n for n in required if max(map(abs, n[3])) == 3)
    return {
        "passed": True,
        "scope": "Exact relative interior Yee dependency DAG; physical boundary guards can only remove dependencies",
        "core_updated_support_min": minimum,
        "core_updated_support_max": maximum,
        "core_update_nodes": len(updates),
        "edge_intermediate_nodes": len(required),
        "monitor_intermediate_nodes": sum(n[1] > 0 for n in observation),
        "edge_envelope_radius": 3,
        "monitor_seed_radius": 2,
        "negative_radius2_missing_node": negative,
        "chronology": "H1/E1, H2/E2, H3/E3 advance S inA withsources+DFT aftereach; H4/E4 advanceQ intoB reading completeH4, thenDFT4; sourcefreebulk suppliesC4.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = prove()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
