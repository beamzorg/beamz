"""Allocate persistent local-state registers under verified guard exclusivity.

The admitted native tile has one immutable active CPML axis. Its logical ψ
recurrences remain distinct, but guards on different axes are mutually exclusive
for the complete owned lifetime of one physical cache point. All four states
on the same axis interfere, including H/E states across phase boundaries.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_geometry_ir import derive_geometry
from scripts.cuda_space_time_codegen import mutable_groups
from scripts.cuda_space_time_ir import IllegalSchedule, yee_operator
from scripts.cuda_trace_cpml_lower import lower_group


def admitted_flags(flags):
    return flags in (2, 4, 8, 34, 36, 40)


def allocate(operator=None, *, exclusive_axis=True):
    operator = yee_operator(cpml_axes=(0, 1, 2)) if operator is None else operator
    geometry = derive_geometry(operator)
    # Reuse the actual control-flow recognizer: arbitrary cross-state local
    # equations do not inherit canonical CPML guard exclusivity merely because
    # their names or derivative offsets resemble CPML.
    for phase in (0, 1):
        for field, group in mutable_groups(operator, phase):
            lower_group(field, group, operator.specs)
    terms = sorted(geometry["cpml_terms"], key=lambda t: (t["phase"], int(t["term"][-1])))
    if len(terms) != 12 or any(t.precision != "float32" for t in operator.tensors if t.role == "local-state"):
        raise IllegalSchedule("Native predicated allocator currently admits all twelve FP32 CPML states")
    edges = {(a["term"], b["term"]) for a, b in itertools.combinations(terms, 2)
             if not exclusive_axis or a["axis_zyx"] == b["axis_zyx"]}
    colors = {}
    for term in terms:
        name = term["term"]
        forbidden = {colors[other] for a, b in edges for other in (a if b == name else b if a == name else "",)
                     if other in colors}
        color = 0
        while color in forbidden:
            color += 1
        colors[name] = color
    count = max(colors.values()) + 1
    report = {"schema": "beamz.predicated-local-state-storage.v1",
              "operator_sha256": geometry["operator_sha256"], "geometry_sha256": geometry["geometry_sha256"],
              "class_predicate": "NoIO, flags&14 in {2,4,8}, immutable for this CTA/macro",
              "lifetime": "entire consecutive owned run of one physical point; release after high/final publication",
              "guards": {t["term"]: {"axis": t["axis_zyx"], "phase": t["phase"]} for t in terms},
              "interference_edges": sorted(edges), "colors": colors, "colors_per_point": count,
              "optimal_lower_bound": 4 if exclusive_axis else 12,
              "requires": ["actual canonical six-field supports", "immutable single-axis tile fact",
                           "first-owned initialization before use", "distinct same-axis H/E histories",
                           "ordinary global loads; unique high/final global publication"],
              "unchanged": ["logical state/coefficient IDs", "active predicates", "rounding/arithmetic nodes", "physical ψ arrays"]}
    verify(report)
    report["allocation_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    return report


def verify(report, *, allow_multiple_axes=False):
    masks = (2, 4, 8, 6, 10, 12, 14) if allow_multiple_axes else (2, 4, 8)
    guards, colors = report["guards"], report["colors"]
    for mask in masks:
        active = [name for name, guard in guards.items() if mask & (2 << guard["axis"])]
        assert len({colors[name] for name in active}) == len(active), "Simultaneously live H/E recurrences share one register"
    assert report["colors_per_point"] == max(colors.values()) + 1


def verify_guard_lifetime(masks):
    """A register cannot change axis meaning while the physical point survives."""
    assert masks and all(mask in (2, 4, 8) for mask in masks)
    assert len(set(masks)) == 1, "A persistent point changed its active-axis interpretation without first-owned reload"


def main():
    report = allocate()
    assert list(report["colors"].values()) == [0, 0, 1, 0, 1, 1, 2, 2, 3, 2, 3, 3]
    verify_guard_lifetime([2] * 16)
    rejected = []
    for name, callback in (("multiple active axes", lambda: verify(report, allow_multiple_axes=True)),
                           ("axis changes during lifetime", lambda: verify_guard_lifetime([2, 4]))):
        try:
            callback()
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError(f"Negative control escaped: {name}")
    print(json.dumps({"allocation": report, "negative_controls_rejected": rejected}, indent=2))


if __name__ == "__main__":
    main()
