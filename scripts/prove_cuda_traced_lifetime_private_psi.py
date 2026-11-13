"""Bind both-family16-float private psi cache to canonical CPML ownership.

Independent integer recurrence oracle verifies exact version handoffs and packed
slabs. Actual CPML graph terms certify two colors per single phase/axis. Resource,
GPU arithmetic and performance gates remain separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cuda_native_geometry_ir import derive_geometry
from scripts.prove_cuda_stationary_psi_cache import prove, prove_owner_boundary_classes


def prove_colors(*, axes=(0,), same_color=False):
    terms = derive_geometry()["cpml_terms"]
    owners, checked = {}, 0
    for phase in (0, 1):
        active = [term for term in terms if term["phase"] == phase and term["axis_zyx"] in axes]
        for tid in range(1024):
            for j in range(4):
                for term in active:
                    index = int(term["term"][-1])
                    color = 0 if same_color else int(index in (2, 4, 5))
                    key = phase, color, tid, j
                    state = term["term"], tid, j
                    if key in owners and owners[key] != state:
                        raise AssertionError((axes, key, owners[key], state, "Private psi colors alias distinct logical recurrences"))
                    owners[key] = state
                    checked += 1
    return {"axes_zyx": axes, "independent_phases": 2, "point_slots": 4096,
            "checked_logical_state_owners": checked, "floats_per_thread": 16,
            "result": "pass"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [prove(minimum, depth, owned=(7, 15, 31), mode="single-axis", load_policy="first-owned")
             for minimum, depth in (((32, 35, 37), 4), ((64, 96, 128), 16), ((65, 97, 129), 16))]
    negatives = []
    for option in ("omit_h_publication", "omit_e_publication", "omit_low_refresh", "wrong_e_extent", "future_publication"):
        try:
            prove((64, 96, 128), 16, owned=(7, 15, 31), mode="all-cache", load_policy="first-owned", **{option: True})
        except AssertionError as exc:
            negatives.append({"mutation": option, "rejected": True, "witness": str(exc)})
        else:
            raise AssertionError(f"Unsafe mutation escaped: {option}")
    for name, options in (("two-active-axes-use-two-colors", {"axes": (0, 1)}),
                          ("both-terms-use-one-color", {"same_color": True})):
        try:
            prove_colors(**options)
        except AssertionError as exc:
            negatives.append({"mutation": name, "rejected": True, "witness": str(exc)})
        else:
            raise AssertionError(f"Unsafe color mutation escaped: {name}")
    report = {"schema": "beamz.both-family-private-psi-ownership.v1",
              "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "canonical_terms": derive_geometry()["cpml_terms"],
              "single_axis_colors": [prove_colors(axes=(axis,)) for axis in range(3)],
              "owner_boundary": prove_owner_boundary_classes((7, 15, 31)),
              "packed_recurrence_cases": cases, "negative_controls": negatives,
              "scope": "Integer recurrence/storage/version proofs; actualC++adapterbitpattern tests separate; no GPU/FP numerical/resource/performance claim"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"positive_packed_cases": len(cases), "single_axis_color_cases": 3,
                      "negative_rejections": len(negatives), "output": str(args.output)}))


if __name__ == "__main__":
    main()
