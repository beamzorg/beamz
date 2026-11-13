#!/usr/bin/env python3
"""Inject traced mstep queue tightening into a fresh benchmark worker.

For nanoparticle measurements pass --runner mie --worker and one schedule and
resolution, so the worker itself (not an unmodified child) installs the proof.
All other arguments are forwarded to the selected retained benchmark runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--runner", choices=("revision", "mie"), default="revision")
    parser.add_argument("--conservative", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.runner == "mie" and "--worker" not in remaining:
        parser.error("Mie dependency experiments must run directly with --worker")
    sys.path.insert(0, str(ROOT))
    from beamz.simulation.cuda import mstep
    from scripts.cuda_mstep_dependencies import (
        intermediate_offsets,
        tighten_first_queue,
    )

    original = mstep.build_mstep_plan
    trials = []

    def tightened(field_shapes, cpml, groups, monitor_indices=None):
        plan = original(field_shapes, cpml, groups, monitor_indices)
        changed = tighten_first_queue(plan, field_shapes, monitor_indices)
        trials.append(
            {
                "original_first_cells": int(plan.first_cells.size),
                "traced_first_cells": int(changed.first_cells.size),
                "scratch_bytes_unchanged": int(plan.summary()["scratch_bytes"]),
            }
        )
        return changed

    if not args.conservative:
        mstep.build_mstep_plan = tightened
    runner = (
        ROOT
        / "scripts"
        / (
            "benchmark_cuda_revision.py"
            if args.runner == "revision"
            else "benchmark_cuda_mie.py"
        )
    )
    sys.argv = [str(runner), *remaining]
    code = 0
    try:
        runpy.run_path(str(runner), run_name="__main__")
    except SystemExit as error:
        code = error.code
    if "--output" in remaining:
        output = Path(remaining[remaining.index("--output") + 1])
        if output.exists():
            report = json.loads(output.read_text())
            report["experimental_dependency_plan"] = {
                "policy": "conservative" if args.conservative else "traced_yee",
                "offsets": intermediate_offsets(),
                "builds": trials,
                "planner_sha256": hashlib.sha256(
                    (ROOT / "scripts/cuda_mstep_dependencies.py").read_bytes()
                ).hexdigest(),
            }
            output.write_text(json.dumps(report, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
