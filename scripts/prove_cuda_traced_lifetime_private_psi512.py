"""Rebind the actual all-shared field oracle to512x8 owner mapping.

The known oracle's read/phase/barrier logic stays identical. Only the owner
function, number of Z slots and warp count change. Packed CPML handoffs are
physical-point properties and are independently repeated for7x15x31 ownership.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import itertools
import json
import sys
from functools import lru_cache
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import prove_cuda_traced_lifetime_shared_fields as original
from scripts.cuda_native_geometry_ir import derive_geometry
from scripts.prove_cuda_stationary_psi_cache import prove, prove_owner_boundary_classes


@dataclass(frozen=True)
class Owner512:
    shape: tuple[int, int, int] = (8, 16, 32)
    threads: int = 512

    def point(self, tid, j):
        return j, (tid // 32) % 16, tid % 32

    def tid(self, point):
        return point[1] * 32 + point[2]


@lru_cache(None)
def rebound_oracle():
    namespace = dict(vars(original))
    namespace["Owner"] = Owner512
    code = inspect.getsource(original.simulate)
    slots = code.count("range(4)")
    if slots != 5 or code.count("list(range(32))") != 2:
        raise ValueError("Original oracle owner loops changed; explicit review required")
    code = code.replace("range(4)", "range(8)").replace("list(range(32))", "list(range(16))")
    exec(compile(code, "<same-field-oracle512-owner-only>", "exec"), namespace)
    exec(compile(inspect.getsource(original.prove_hazards), "<same-hazard-oracle512-owner-only>", "exec"), namespace)
    return namespace["simulate"], namespace["prove_hazards"]


def prove_owner_mapping(*, zstep=1):
    g = Owner512()
    inverse = {}
    for tid, j in itertools.product(range(512), range(8)):
        p = (j * zstep) % 8, tid // 32, tid % 32
        if p in inverse:
            raise AssertionError("512owner mapping duplicates a physical cache slot")
        inverse[p] = tid, j
        if g.point(tid, j) != p or g.tid(p) != tid:
            raise AssertionError("512point/thread inverse disagrees with actual shader")
    if len(inverse) != 4096:
        raise AssertionError("512owner mapping omits a physical cache slot")
    terms = derive_geometry()["cpml_terms"]
    counts = []
    for axis in range(3):
        state_owners = {}
        for term in terms:
            if term["axis_zyx"] != axis:
                continue
            phase, index = term["phase"], int(term["term"][-1])
            color = int(index in (2, 4, 5))
            for point, (tid, j) in inverse.items():
                key = phase, color, tid, j
                identity = term["term"], point
                if key in state_owners and state_owners[key] != identity:
                    raise AssertionError("512history aliases different logical CPML states")
                state_owners[key] = identity
        counts.append({"axis_zyx": axis, "actual_history_state_owners": len(state_owners)})
    return {"threads": 512, "slots": 8, "point_owner_bijection": len(inverse),
            "private_history_floats_per_thread": 32, "history_state_owners_per_axis": counts,
            "total_CTA_history_floats": 16384}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    simulate, hazards = rebound_oracle()
    cases = [simulate(first=first, steps=steps, seed=seed)
             for first, steps, seed in itertools.product((1, 4, 13), (1, 2) if args.quick else (1, 2, 8, 16), (7, 29))]
    negatives = []
    for name in original.NEGATIVES:
        try:
            simulate(first=1, steps=2, seed=7, negative=name)
        except original.Hazard as exc:
            negatives.append({"mutation": name, "rejected": True, "witness": str(exc)})
        else:
            raise AssertionError(f"Unsafe512 schedule escaped: {name}")
    packed = [prove(minimum, depth, owned=(7, 15, 31), mode="single-axis", load_policy="first-owned")
              for minimum, depth in (((32, 35, 37), 4), ((64, 96, 128), 16), ((65, 97, 129), 16))]
    result = {"schema": "beamz.private-psi512-field-and-state-owner.v1",
              "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "base_field_oracle_sha256": hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
              "owner_mapping": prove_owner_mapping(), "same_phase_hazards": hazards(),
              "positive_field_cases": cases, "negative_controls": negatives,
              "total_actual_field_checks": sum(c["actual_field_and_clock_checks"] for c in cases),
              "packed_CPML_owner_cases": packed, "physical_owner_boundary": prove_owner_boundary_classes((7, 15, 31)),
              "scope": "512field/version/visibility and packedstateowner proofs; actualCUDA numerical/resource/sanitizer/performance gates separate"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"positive_field_cases": len(cases), "negative_rejections": len(negatives),
                      "total_actual_field_checks": result["total_actual_field_checks"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
