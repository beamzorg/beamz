"""Last-use proof for fusing only the final two partial-Ey Z publications.

CPU token model only. The native proposal enables fusion only at T1024 on the
8x32x32 cache; smaller-thread mappings are checked as conservative logical
controls and keep the original native schedule. Every field FMA remains local
and ordered in the original per-point component order; no arithmetic is modeled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prove_cuda_partial_ey_lifetimes import (
    Hazard,
    Layout,
    collective_proof,
    simulate,
    traced_reads,
)


def report():
    layouts = [Layout(8, 32, threads) for threads in (1024, 512, 256)]
    cases = [simulate(layout, version=n, seed=seed, tail_pair=True)
             for layout in layouts for n in range(1, 33) for seed in (13, 37)]
    negative_controls = {}
    for name in ("tail-no-second-publication", "tail-save-after-H-publish",
                 "tail-restore-before-RL", "tail-overwrite-first-result"):
        try:
            simulate(layouts[0], version=2, seed=13, tail_pair=True, negative=name)
        except Hazard as error:
            negative_controls[name] = repr(error)
        else:
            raise AssertionError(f"Unsafe tail schedule passed: {name}")
    reads = traced_reads()
    return {"scope": __doc__, "cases": cases, "valid_cases": len(cases),
            "checked_reads": sum(c["checked_field_reads"] for c in cases),
            "negative_controls": negative_controls, "collectives": collective_proof(),
            "trace_reader_matrix": {str(f): [(r.field, r.version, r.offset) for r in rs] for f, rs in reads.items()},
            "schedule": ["R0: retire all H old-E reads/import H",
                         "unchanged P0..P5 and E0..E5; fast warp may next save/publish both H6/H7",
                         "P67: publish ALL Y groups of BOTH planes and retire E5 H5 readers",
                         "restore pending E5; compute E6 and E7, retaining 4 Ex/Ez values per Y group",
                         "RL: retire ALL H6/H7 XY readers; restore BOTH final E planes",
                         "RE: publish full E and next phase counter"],
            "last_use": {"H_Z_neighbors": "Private H throughout E phase; no E7 reader needs shared H6",
                         "shared_H6_H7": "Cross-warp Y readers remain through E6/E7, so retain BOTH until RL",
                         "old_ExEz6_7": "4 distinct old-own values; overwritten individually only after their original component FMA",
                         "E6_outputs": "Remain in those same two saved slots while E7 consumes its separate old-own pair",
                         "H_private": "Remains live for next H own-state FMA; not reusable for tail scratch"},
            "native_scope": "T==1024 only; original generic512/256 branch byte-identical",
            "resource_bound": {"baseline_ExEz_saved_floats_per_Ygroup": 2,
                               "tail_pair_saved_and_pending_floats_per_Ygroup": 4,
                               "additional_T1024_field_floats": 2,
                               "shared_bytes_change": 0, "barriers_11_to": 10,
                               "warning": "Source-level liveness, not ptxas register prediction"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = report()
    result["producer_oracle_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    base = Path(__file__).with_name("prove_cuda_partial_ey_lifetimes.py")
    result["shared_oracle_sha256"] = hashlib.sha256(base.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("valid_cases", "checked_reads", "negative_controls")}))


if __name__ == "__main__":
    main()
