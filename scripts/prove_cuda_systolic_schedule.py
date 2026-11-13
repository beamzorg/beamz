"""Dependency/ownership proof for a moving-subdomain Yee schedule.

This is a CPU scheduling model, not a CUDA implementation or speed benchmark.
At field version n, transformed coordinates are u=(z+n,y+n,x+n). H_n reads
E_(n-1) at the same/positive physical neighbor; E_n reads H_n at the
same/negative neighbor. Every transformed dependency is componentwise <= u:
H<-E reads include u-(1,1,0) and its permutations, E<-H uses u or u-e_axis,
and old local fields/psi are at u-(1,1,1). Fixed boxes in u move
upwind in physical space. Their inter-box graph is acyclic without overlap.

Real arithmetic dependencies come from the existing cell/CPML tracer. Boundary
reads outside each component's physical extent model the reference zero wall.
Sources, material migration, CPML classification and chronological interpolated
DFT execution still require a production scheduler; this proof does not supply
those mechanisms. It intentionally retains all versions on CPU to check a
future compact-face CUDA implementation against an independent oracle. An
inverse-stencil producer enumeration is checked against consumption-side needs.
It is not a packed CUDA surface layout. Version-zero entries are initial reads.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path

try:
    from scripts.cuda_stencil_trace import trace_cpml_step, trace_yee_step
except ModuleNotFoundError:
    from cuda_stencil_trace import trace_cpml_step, trace_yee_step


def component_shapes(shape):
    """Different extents exercise physical walls and staggered ownership."""
    z, y, x = shape
    result = {
        "Ex": (z + 1, y + 1, x),
        "Ey": (z + 1, y, x + 1),
        "Ez": (z, y + 1, x + 1),
        "Hx": (z, y, x + 1),
        "Hy": (z, y + 1, x),
        "Hz": (z + 1, y, x),
    }
    for family in "HE":
        for term in range(6):
            result[f"psi_{family}{term}"] = result[family + "xyz"[term // 2]]
    return result


def _positions(shape):
    return itertools.product(*(range(n) for n in shape))


def _producer_publications(equations, values, shapes, box, steps):
    """Enumerate outgoing targets by inverting the one-step stencil.

    This traverses produced values and inverse offsets, separately from the
    consumer-order checker. The eventual surface encoder must match this set.
    """
    inverse = {}
    for equation in equations:
        for read in equation.reads:
            inverse.setdefault(read.field, set()).add(
                (
                    equation.output.field,
                    1 - read.version,
                    tuple(-v for v in read.offset),
                )
            )
    publications = set()
    for source in values:
        field, version, position = source
        source_owner = box(position, version)
        for target_field, dt, offset in inverse[field]:
            target_version = version + dt
            if not 1 <= target_version <= steps:
                continue
            target = tuple(a + b for a, b in zip(position, offset, strict=True))
            if any(
                p < 0 or p >= n
                for p, n in zip(target, shapes[target_field], strict=True)
            ):
                continue
            target_owner = box(target, target_version)
            if target_owner != source_owner:
                publications.add((source, target_owner))
    return publications


def prove(
    shape=(5, 6, 7),
    tile=(3, 4, 5),
    steps=4,
    *,
    cpml=False,
    shear=True,
    drop_boundary=False,
):
    if min(*shape, *tile, steps) < 1:
        raise ValueError("Positive dimensions and step count required")
    shapes = component_shapes(shape)
    tracer = trace_cpml_step if cpml else trace_yee_step
    updates = {n: tracer(n - 1) for n in range(1, steps + 1)}
    fields = {u.output.field for u in updates[1]}

    def box(position, version):
        shift = version if shear else 0
        return tuple(
            (p + shift) // width for p, width in zip(position, tile, strict=True)
        )

    tiles = {
        box(p, n)
        for n in updates
        for field in fields
        for p in _positions(shapes[field])
    }
    produced, required_boundaries = set(), set()
    stats = Counter()
    # Sum is a strict topological rank for every cross-box dependency. Equal
    # sum boxes have no dependencies on one another; tuple order is arbitrary.
    for owner in sorted(tiles, key=lambda t: (sum(t), t)):
        for n, equations in updates.items():
            for equation in equations:
                for p in _positions(shapes[equation.output.field]):
                    if box(p, n) != owner:
                        continue
                    destination = (equation.output.field, n, p)
                    if destination in produced:
                        raise AssertionError("Duplicate physical update")
                    for read in equation.reads:
                        q = tuple(a + b for a, b in zip(p, read.offset, strict=True))
                        if any(
                            v < 0 or v >= size
                            for v, size in zip(q, shapes[read.field], strict=True)
                        ):
                            stats["physical_wall_reads"] += 1
                            continue
                        source = (read.field, read.version, q)
                        source_box = box(q, read.version)
                        if any(a > b for a, b in zip(source_box, owner, strict=True)):
                            raise AssertionError(
                                "Dependency points upwind: cyclic box graph"
                            )
                        if read.version and source not in produced:
                            raise AssertionError("Dependency consumed before producer")
                        if source_box != owner:
                            gap = sum(owner) - sum(source_box)
                            assert 1 <= gap <= 3
                            stats["max_cross_box_rank_gap"] = max(
                                stats["max_cross_box_rank_gap"], gap
                            )
                            required_boundaries.add((source, owner))
                            stats["cross_box_read_operands"] += 1
                        stats["valid_read_operands"] += 1
                    produced.add(destination)
                    stats["updates"] += 1
    expected = steps * sum(math.prod(shapes[f]) for f in fields)
    assert len(produced) == expected
    initial = {(f, 0, p) for f in fields for p in _positions(shapes[f])}
    published = _producer_publications(
        updates[1], produced | initial, shapes, box, steps
    )
    if drop_boundary:
        published.remove(next(iter(published)))
    if required_boundaries - published:
        raise AssertionError("Missing face-trail publication")
    assert published == required_boundaries, "Unnecessary outgoing publication"
    boundary_values = {source for source, _ in required_boundaries}
    return {
        "shape": shape,
        "tile": tile,
        "steps": steps,
        "cpml": cpml,
        "tiles": len(tiles),
        "unique_updates": len(produced),
        "executed_over_useful_updates": 1.0,
        "unique_cross_box_values": len(boundary_values),
        "producer_consumer_publications": len(published),
        "initial_cross_box_values": sum(source[1] == 0 for source in boundary_values),
        **stats,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [
        prove(shape, tile, n, cpml=cpml)
        for shape, tile in (((5, 6, 7), (3, 4, 5)), ((3, 4, 5), (2, 3, 4)))
        for n in (2, 4, 8)
        for cpml in (False, True)
    ]
    negatives = {}
    for name, options in (
        ("unskewed_boxes", {"shear": False}),
        ("missing_face_trail", {"drop_boundary": True}),
    ):
        try:
            prove(**options)
        except AssertionError as error:
            negatives[name] = str(error)
        else:
            raise AssertionError(f"Negative control did not fail: {name}")
    report = {
        "schema": "beamz.systolic-dependency-proof.v1",
        "passed": True,
        "scope": "CPU dependency/unique-ownership proof only; no CUDA or performance claim",
        "cases": cases,
        "negative_controls": negatives,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
