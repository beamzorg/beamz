"""Actual-read oracle for 4-plane source chunks published in pairs.

No kernel arithmetic is simulated. Source-level peak is H12 + current Ex/Ez4 +
previous pending4=20 field floats. Private snapshots remain separate/immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prove_cuda_partial_ey_lifetimes import Hazard, Layout, simulate


def policy5_superset(flags, low, high, minimum):
    """Bind the ordinary Policy3/4 bounds fact before using generic Policy5."""
    if type(flags) is not int or flags not in range(256) or flags & 63 not in (16, 17):
        raise ValueError(
            "Only actual source-only/source-observer tile classes are admitted"
        )
    if not (len(low) == len(high) == len(minimum) == 3) or any(
        not (0 < a <= b and b + 1 < n)
        for a, b, n in zip(low, high, minimum, strict=True)
    ):
        raise ValueError("Existing no-native-boundary trajectory fact required")
    assert not flags & 14 and not flags & 32
    return {
        "extra_Valid_checks": "true for every component: all coordinates lie strictly inside minimum supports",
        "extra_LocalActive": "false: all three actual axis flags absent",
        "extra_E_boundary_guards": "false: coordinate never0 orN",
        "source_guard": bool(flags & 16),
        "observe_guard": bool(flags & 1),
    }


def report():
    cases = [
        simulate(
            Layout(8, 16, 1024),
            version=n,
            seed=seed,
            shared_ey_private_pending=True,
            paired_planes=True,
        )
        for n in range(1, 33)
        for seed in (13, 37)
    ]
    negatives = {}
    for name in (
        "paired-early-restore",
        "paired-forget-first-pending",
        "paired-overwrite-second-old",
        "paired-missing-H3",
        "missing-snapshot-publication",
        "previous-version-snapshot",
    ):
        try:
            simulate(
                Layout(8, 16, 1024),
                version=2,
                seed=37,
                shared_ey_private_pending=True,
                paired_planes=True,
                negative=name,
            )
        except Hazard as error:
            negatives[name] = repr(error)
        else:
            raise AssertionError(f"Unsafe paired schedule passed: {name}")
    flags = [f for f in range(256) if f & 63 in (16, 17)]
    for f in flags:
        policy5_superset(f, (5, 14, 46), (26, 43, 91), (64, 96, 128))
    return {
        "scope": __doc__,
        "cases": cases,
        "valid_cases": len(cases),
        "checked_reads": sum(c["checked_field_reads"] for c in cases),
        "negative_controls": negatives,
        "policy5_superset_flag_classes": flags,
        "source_float_bound": {
            "H": 12,
            "current_old_or_new_ExEz": 4,
            "previous_pending_ExEz": 4,
            "total": 20,
        },
        "shared_bytes": 58680,
        "barriers_per_step": 5,
        "schedule": [
            "R0: retire all old E Hreaders and publish Hstage",
            "P01: publish both H0/H1 and immutable H3/H7 snapshots",
            "compute E0/E1 into separate own slots; copy four results to pending",
            "saveoldE2/E3 and publishH2/H3, while pending4 remainslive",
            "P23: retire ALL H0/H1 readers, then restore E0/E1 pending",
            "compute E2/E3 in original point order; copy four results to pending",
            "RL: retire BOTH final H planes, restore E2/E3",
            "RE: publish full E/currentphase",
        ],
        "collectives": "Every Hy(-X) shuffle remains unconditional fullmask beforeOwned/Valid; full Ey remainsshared",
        "snapshot": "Separate producer-indexed Hx/Hy Z3/Z7 arrays neveralias pending; oppositechunk sameXY E0consumer only",
        "admission": "Native source mask20+Layout1+lossless isotropicH retained; Policy5 reuse on3/4IDs additionally uses actual validated flags and old no-boundary bounds fact",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    r = report()
    r["oracle_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    r["shared_oracle_sha256"] = hashlib.sha256(
        Path(__file__).with_name("prove_cuda_partial_ey_lifetimes.py").read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(r, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: r[k]
                for k in (
                    "valid_cases",
                    "checked_reads",
                    "barriers_per_step",
                    "shared_bytes",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
