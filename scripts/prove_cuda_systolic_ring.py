"""Check conservative four-frontier trail retirement against traced consumers.

CPU model only. Boxes in one topological rank may execute in any order. At the
start of rank r, its epoch is reclaimed as a whole; all readers of its previous
occupant must already have completed. No spinning producer/consumer CTAs or
CUDA memory-order guarantee is modeled here. Version-zero reads use initial
global state and consume no trail capacity.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

try:
    from scripts.cuda_stencil_trace import trace_cpml_step, trace_yee_step
    from scripts.prove_cuda_systolic_schedule import (
        _producer_publications,
        component_shapes,
    )
except ModuleNotFoundError:
    from cuda_stencil_trace import trace_cpml_step, trace_yee_step
    from prove_cuda_systolic_schedule import _producer_publications, component_shapes


def surface_slot(point, tile):
    z, y, x = point
    nz, ny, nx = tile
    if z == nz - 1:
        return y * nx + x
    if y == ny - 1:
        return ny * nx + z * nx + x
    if x == nx - 1:
        return ny * nx + (nz - 1) * nx + z * (ny - 1) + y
    raise AssertionError("Cross-box source is not on an outgoing surface")


def prove_ring(
    shape=(5, 6, 7),
    tile=(3, 4, 5),
    steps=4,
    *,
    cpml=False,
    epochs=4,
    wrong_version=False,
):
    if min(*shape, *tile, steps, epochs) < 1:
        raise ValueError("Positive dimensions, steps and epoch count required")
    shapes = component_shapes(shape)
    equations = trace_cpml_step() if cpml else trace_yee_step()
    fields = {u.output.field for u in equations}
    values = {
        (f, n, p)
        for f in fields
        for n in range(1, steps + 1)
        for p in itertools.product(*(range(s) for s in shapes[f]))
    }

    def box(point, version):
        return tuple((v + version) // b for v, b in zip(point, tile, strict=True))

    tiles = {box(p, n) for _, n, p in values}
    frontiers = defaultdict(list)
    for owner in sorted(tiles):
        frontiers[sum(owner)].append(owner)
    slot = {owner: i for group in frontiers.values() for i, owner in enumerate(group)}
    incoming, outgoing = defaultdict(set), defaultdict(set)
    for source, consumer in _producer_publications(
        equations, values, shapes, box, steps
    ):
        producer = box(source[2], source[1])
        assert 1 <= sum(consumer) - sum(producer) <= 3
        incoming[consumer].add(source)
        outgoing[producer].add(source)

    def address(source, *, read=False):
        field, version, point = source
        producer = box(point, version)
        local = tuple((v + version) % b for v, b in zip(point, tile, strict=True))
        return (
            sum(producer) % epochs,
            slot[producer],
            version - (1 if read and wrong_version else 0),
            field,
            surface_slot(local, tile),
        )

    live, reads, peak = {}, 0, 0
    for rank in sorted(frontiers):
        # Whole-epoch reuse is intentionally conservative. With four epochs,
        # all consumers (at most rank+3) finish before that epoch is reclaimed.
        live = {key: value for key, value in live.items() if key[0] != rank % epochs}
        for owner in frontiers[rank]:
            for source in incoming[owner]:
                if live.get(address(source, read=True)) != source:
                    raise AssertionError("Retired or wrong-version trail consumed")
                reads += 1
            for source in outgoing[owner]:
                key = address(source)
                if key in live:
                    raise AssertionError("Aliased publication address")
                live[key] = source
            peak = max(peak, len(live))
    max_frontier = max(map(len, frontiers.values()))
    surface = math.prod(tile) - math.prod(v - 1 for v in tile)
    return {
        "shape": shape,
        "tile": tile,
        "steps": steps,
        "cpml": cpml,
        "epochs": epochs,
        "tiles": len(tiles),
        "max_frontier": max_frontier,
        "checked_consumer_values": reads,
        "peak_live_tokens": peak,
        "surface_cells": surface,
        "ring_payload_bound_bytes": epochs
        * max_frontier
        * steps
        * len(fields)
        * surface
        * 4,
        "all_tiles_payload_bytes": len(tiles) * steps * len(fields) * surface * 4,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [
        prove_ring(shape, tile, n, cpml=cpml)
        for shape, tile in (((5, 6, 7), (3, 4, 5)), ((4, 5, 6), (2, 3, 4)))
        for n in (2, 4, 8)
        for cpml in (False, True)
    ]
    negatives = {}
    for name, options in (
        ("early_epoch_reuse", {"epochs": 3}),
        ("wrong_version", {"wrong_version": True}),
    ):
        try:
            prove_ring(cpml=True, **options)
        except AssertionError as error:
            negatives[name] = str(error)
        else:
            raise AssertionError(f"Negative control did not fail: {name}")
    report = {
        "schema": "beamz.systolic-trail-ring-proof.v1",
        "passed": True,
        "scope": "CPU logical-address/lifetime model, not CUDA ordering or full CPML admission",
        "cases": cases,
        "negative_controls": negatives,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
