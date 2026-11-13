"""Color mutable E-CPML storage from actual guards and affine owned lifetimes.

For a fixed physical point, validity and CPML activation are time independent.
An owned circular-cache slot keeps that point until a ghost/recycle transition;
the first-owned rule reloads before any new point can use its old color. Thus
same-point guard intersections determine which logical states need distinct
private colors. All intersections use integer versions, not swept bounding boxes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA, derive_geometry
from scripts.cuda_space_time_ir import (
    FIELDS,
    IllegalSchedule,
    validate_stream_signature,
    yee_operator,
)

FIXED_E_COLORS = (0, 0, 1, 0, 1, 1)


@dataclass(frozen=True)
class OwnedRegion:
    minimum: tuple[int, int, int]
    origin: tuple[int, int, int]
    owned: tuple[int, int, int]
    depth: int

    def __post_init__(self):
        if any(len(v) != 3 for v in (self.minimum, self.origin, self.owned)):
            raise IllegalSchedule("Integer ZYX geometry required")
        if (any(type(v) is not int or v < 24 for v in self.minimum)
                or any(type(v) is not int or v < 0 for v in self.origin)
                or any(type(v) is not int or v < 2 for v in self.owned)
                or type(self.depth) is not int or self.depth < 1):
            raise IllegalSchedule("CPML12 grid, positive owned extents and depth required")


def terms_from_ir(operator, region):
    validate_stream_signature(operator)
    geometry = derive_geometry(operator)
    result = []
    for term in geometry["cpml_terms"]:
        if term["phase"] != 1:
            continue
        if operator.specs[term["term"]].precision != "float32":
            raise IllegalSchedule("Private F32 state storage requires canonical FP32 psi")
        number = int(term["term"][-1])
        owner = FIELDS.index(term["output"])
        shape = tuple(n + d for n, d in zip(region.minimum, SUPPORT_DELTA[owner], strict=True))
        axis = term["axis_zyx"]
        if shape[axis] != region.minimum[axis] + 1:
            raise IllegalSchedule("E CPML differentiated extent must be N+1")
        result.append({"term": number, "name": term["term"], "owner": owner, "axis": axis, "shape": shape})
    result.sort(key=lambda term: term["term"])
    if [term["term"] for term in result] != list(range(6)):
        raise IllegalSchedule("All six canonical E recurrences required")
    return result, geometry["operator_sha256"]


def active_boxes(term):
    for side in (0, 1):
        lower, upper = [0, 0, 0], list(term["shape"])
        axis = term["axis"]
        if side:
            lower[axis] = upper[axis] - 12
        else:
            upper[axis] = 12
        yield side, tuple(lower), tuple(upper)


def trajectory_witness(region, lower, upper, *, previous_owned=False):
    """Exact integer solution p in box AND p in [origin-n,origin+B-n)."""
    if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
        return None
    shift = int(previous_owned)
    origin = tuple(o + shift for o in region.origin)
    owned = tuple(b - shift for b in region.owned)
    first = max(1 + shift, *(o - hi + 1 for o, hi in zip(origin, upper, strict=True)))
    last = min(region.depth, *(o + b - lo - 1 for o, b, lo in zip(origin, owned, lower, strict=True)))
    if first > last:
        return None
    point = tuple(max(lo, o - first) for lo, o in zip(lower, origin, strict=True))
    assert all(p < min(hi, o + b - first) for p, hi, o, b in zip(point, upper, origin, owned, strict=True))
    return {"version": first, "point_zyx": point, "feasible_versions": (first, last)}


def intersection_witness(region, a, b):
    for side_a, low_a, high_a in active_boxes(a):
        for side_b, low_b, high_b in active_boxes(b):
            lower = tuple(max(x, y) for x, y in zip(low_a, low_b, strict=True))
            upper = tuple(min(x, y) for x, y in zip(high_a, high_b, strict=True))
            witness = trajectory_witness(region, lower, upper)
            if witness is not None:
                return {"states": (a["term"], b["term"]), "sides": (side_a, side_b), **witness}
    return None


def optimal_coloring(edges):
    """Exact six-node graph coloring; not a resource or performance estimate."""
    adjacent = [set() for _ in range(6)]
    for a, b in edges:
        adjacent[a].add(b)
        adjacent[b].add(a)
    order = sorted(range(6), key=lambda state: (-len(adjacent[state]), state))
    colors = [-1] * 6

    def solve(index, count):
        if index == 6:
            return tuple(colors)
        state = order[index]
        used = {colors[other] for other in adjacent[state] if colors[other] >= 0}
        for color in range(count):
            if color in used:
                continue
            colors[state] = color
            found = solve(index + 1, count)
            if found is not None:
                return found
        colors[state] = -1
        return None

    for count in range(1, 7):
        result = solve(0, count)
        if result is not None:
            return result
    raise AssertionError("Six states must admit six colors")


def certify(region, operator=None, *, threads=1024):
    operator = yee_operator(cpml_axes=(0, 1, 2)) if operator is None else operator
    terms, operator_hash = terms_from_ir(operator, region)
    witnesses = [w for a, b in itertools.combinations(terms, 2) if (w := intersection_witness(region, a, b)) is not None]
    edges = [tuple(w["states"]) for w in witnesses]
    collisions = [w for w in witnesses if FIXED_E_COLORS[w["states"][0]] == FIXED_E_COLORS[w["states"][1]]]
    warm = None
    for term in terms:
        for side, lower, upper in active_boxes(term):
            found = trajectory_witness(region, lower, upper, previous_owned=True)
            if found:
                warm = {"term": term["term"], "side": side, "prior_version": found["version"] - 1, **found}
                break
        if warm:
            break
    cache = tuple(b + 1 for b in region.owned)
    plane = cache[1] * cache[2]
    if type(threads) is not int or threads < plane or threads % plane or cache[0] % (threads // plane):
        raise IllegalSchedule("Fixed XY owners require an integral Z stride and slot count")
    step = threads // plane
    mapping = optimal_coloring(edges)
    evidence = {"schema": "beamz.guarded-private-Epsi-storage.v1", "region": asdict(region),
                "operator_sha256": operator_hash, "terms": terms, "edges": edges, "edge_witnesses": witnesses,
                "fixed_colors": FIXED_E_COLORS, "fixed_two_safe": not collisions,
                "fixed_two_counterexample": collisions[0] if collisions else None,
                "optimal_colors": mapping, "optimal_color_count": max(mapping) + 1,
                "warm_private_use_witness": warm,
                "mapping": {"cache_zyx": cache, "threads": threads, "z_stride": step, "slots": cache[0] // step,
                            "fixed_two_private_floats_per_thread": 2 * cache[0] // step},
                "lifetime": {"stationary": "cacheIndex=(physicalPoint-origin) mod(cacheDims), hence thread/private-slot identity is constant while owned",
                             "first_owned": "n==window.first OR any relative-owned-coordinate==1 reloads global previous psi before cache use",
                             "warm": "all relative coordinates>1 implies same physical point was owned at n-1; validity/activation are time invariant",
                             "publication": "any relative coordinate==B OR n==Depth flushes current psi; interior stays owned at n+1",
                             "late_end": "If the clipped valid window ends beforeDepth, any valid last point must leave the owned box next step, hence lies on an outgoing high face",
                             "guard_exclusivity": "Nonadjacent states are never both active for the same physical point in any owned version; recycled slots reload before a different point uses the color"},
                "scope": "CPU exact guard/storage certificate, not generated CUDA, register feasibility, synchronization admission or speed"}
    evidence["proof_sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", type=int, nargs=3, required=True)
    p.add_argument("--origin", type=int, nargs=3, required=True)
    p.add_argument("--owned", type=int, nargs=3, default=(7, 15, 31))
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--threads", type=int, default=1024)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = certify(OwnedRegion(tuple(a.shape), tuple(a.origin), tuple(a.owned), a.depth), threads=a.threads)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("fixed_two_safe", "optimal_color_count", "warm_private_use_witness", "mapping")}))


if __name__ == "__main__":
    main()
