"""Trace-driven lifetime proof for small all-shared Ey/private pending E.

Integer fields only: no FP arithmetic, native packed addressing or hardware
convergence claim. Direct Hx/Hy boundary snapshots have unique opposite-chunk
same-XY consumers and never alias private Ex/Ez pending storage.
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
    prove_late_bootstrap,
    simulate,
    traced_reads,
)


def report():
    cases = [
        simulate(Layout(8, 16, t), version=n, seed=seed, shared_ey_private_pending=True)
        for t in (1024, 512)
        for n in range(1, 33)
        for seed in (13, 37)
    ]
    negatives = {}
    for name in (
        "missing-snapshot-publication",
        "previous-version-snapshot",
        "wrong-snapshot-slot",
        "missing-old-H-retirement",
        "early-Ey-ghost-import",
        "missing-plane-retirement",
        "missing-last-retirement",
    ):
        try:
            simulate(
                Layout(8, 16, 1024),
                version=2,
                seed=37,
                shared_ey_private_pending=True,
                negative=name,
            )
        except Hazard as error:
            negatives[name] = repr(error)
        else:
            raise AssertionError(f"Unsafe schedule passed: {name}")
    late = [
        prove_late_bootstrap(Layout(8, 16, t), origin=origin)
        for t in (1024, 512)
        for origin in ((70, 40, 40), (40, 70, 40), (40, 40, 70), (70, 70, 70))
    ]
    read_matrix = {
        str(f): [(r.field, r.version, r.offset) for r in rs]
        for f, rs in traced_reads().items()
    }
    return {
        "scope": __doc__,
        "cases": cases,
        "valid_cases": len(cases),
        "checked_reads": sum(c["checked_field_reads"] for c in cases),
        "negative_controls": negatives,
        "late_bootstrap_cases": late,
        "trace_reads": read_matrix,
        "required_barriers": "R0 retires ALL old Ey/Ex/Ez H reads and publishes coalesced H stage; P0 publishes direct Hx/Hy snapshots and current Hxy; each Pj retires previous Hxy; RL retires last Hxy; RE publishes full E/phasecounter",
        "snapshot_ownership": "Producer physicalZ3/7 stores h0/h1 at producerChunk*512+y*32+x; oppositeChunk E0 consumer reads exactly this coordinate; private pending never writes snapshots",
        "Ey_lifetime": "All Ey permanently shared, H sees n-1 until R0; current ghost Ey imported after R0; only old-own Ey read in E, then next H sees n after RE",
        "private_pending_lifetime": "Every thread writes pending after E j0, even a ghost; j>0 only restores after Pj retired neighbors. Pending fields are not read by another thread.",
        "source_field_counts": {
            "1024": {"H": 12, "own_ExEz": 2, "pending_ExEz": 2, "total": 16},
            "512": {"H": 24, "own_ExEz": 2, "pending_ExEz": 2, "total": 28},
        },
        "shared_bytes": 58680,
        "barriers_per_step": {"1024": 7, "512": 11},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report()
    result["oracle_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["shared_oracle_sha256"] = hashlib.sha256(
        Path(__file__).with_name("prove_cuda_partial_ey_lifetimes.py").read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "valid_cases",
                    "checked_reads",
                    "shared_bytes",
                    "barriers_per_step",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
