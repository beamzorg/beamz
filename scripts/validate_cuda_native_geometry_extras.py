#!/usr/bin/env python3
"""Gate actual Y ownership/warp-row seams and shifted depth16 clocks with a pinned oracle.

Uses NumPy only. An explicitly separate ordinary CUDA module runs in a child
process, exits, and saves all30 reference leaves before candidate GPU work starts.
These seeded, changed-source packets are correctness witnesses, not benchmarks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cuda_fixture_store import digest_array
from scripts.export_fdtdx_comparison import file_digest
from scripts.validate_cuda_native_moving import load_fixture
from scripts.validate_cuda_native_observer_dispatch import separated_probe_packet
from scripts.validate_cuda_plane_retirement import module_at, moving_repeats, ordinary


def packet(fixture, inputs, case):
    if case.startswith("point-"):
        source = [32, 32, 14]
        source[{"Z": 0, "Y": 1, "X": 2}[case[6]]] = int(case[7:])
        source = tuple(source)
        observer = (source[0], source[1], source[2] + 1)
        return separated_probe_packet(inputs, fixture["attributes"], source, observer, 35)
    changed = list(inputs)
    changed[126] = np.asarray(np.float32(3.125) * np.float32(fixture["attributes"]["dt"]))
    changed[127] = np.asarray(5, dtype=np.int32)
    changed[128] = np.asarray(3, dtype=np.int32)
    for g in (5, 7):
        if changed[87 + 3 * g].shape[1] < 40:
            raise ValueError("Shifted clock witness requires an explicit >=40-sample waveform")
    return changed, dict(fixture["attributes"], nsteps=35)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--oracle-module", type=Path)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("point-Z3", "point-Z4", "point-Z7", "point-Z8",
                                               "point-Y15", "point-Y16", "point-Y31", "point-Y32",
                                               "point-Y46", "point-Y47", "point-X31", "point-X32",
                                               "shifted-clock"))
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--require-observer-branch", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "beamz.native-geometry-extras.v1", "passed": False,
              "module": str(args.module), "module_sha256": file_digest(args.module),
              "fixture": str(args.fixture), "fixture_sha256": file_digest(args.fixture),
              "worker_sha256": file_digest(__file__), "depth": 16, "steps": 35,
              "scope": "Changed correctness packets; independent pinned ordinary Program",
              "cases": []}
    try:
        fixture, data, inputs = load_fixture(args.fixture)
        cases = (args.case,) if args.case else ("point-Z3", "point-Z4", "point-Z7", "point-Z8",
                                                  "point-Y15", "point-Y16", "point-Y31", "point-Y32",
                                                  "point-Y46", "point-Y47", "point-X31", "point-X32",
                                                  "shifted-clock")
        if args.reference_only:
            if not args.case:
                raise ValueError("Reference child requires one explicit case")
            module = module_at(args.module)
            changed, attributes = packet(fixture, inputs, args.case)
            values = ordinary(module, fixture, data, changed, attributes, 35)
            np.savez(args.output, **values)
            return 0
        if args.oracle_module is None:
            raise ValueError("An explicitly pinned independent --oracle-module is required")
        report["oracle_module"] = str(args.oracle_module)
        report["oracle_module_sha256"] = file_digest(args.oracle_module)
        # The child owns GPU first; no candidate session/context exists yet.
        references = {}
        for case in cases:
            path = args.output.with_name(args.output.stem + "." + case + ".oracle.npz")
            command = [sys.executable, str(Path(__file__).resolve()), "--reference-only",
                       "--module", str(args.oracle_module), "--fixture", str(args.fixture),
                       "--case", case, "--output", str(path)]
            if file_digest(args.oracle_module) != report["oracle_module_sha256"]:
                raise RuntimeError("Pinned oracle binary changed")
            subprocess.run(command, check=True, timeout=120)
            if file_digest(args.oracle_module) != report["oracle_module_sha256"]:
                raise RuntimeError("Pinned oracle binary changed during reference")
            references[case] = (path, command)
        module = module_at(args.module)
        report["compiled_cache_zyx"] = getattr(module, "partial_ey_cache", None)
        report["observer1024_compiled"] = bool(getattr(module, "moving_observer_uniform_1024", False))
        for case in cases:
            changed, attributes = packet(fixture, inputs, case)
            before = [digest_array(value) for value in changed]
            reference, command = references[case]
            with np.load(reference, allow_pickle=False) as stored:
                expected = {key: stored[key] for key in stored.files}
            if len(expected) != 30:
                raise AssertionError("Independent oracle omitted state leaves")
            for preserve in (False, True):
                rows = moving_repeats(module, fixture, data, changed, attributes,
                                      expected, SimpleNamespace(steps=35, depth=16, repeats=2),
                                      preserve)
                for row in rows:
                    observer_count = int(row["metadata"].get("moving_uniform_observer_tiles", 0))
                    positive_observer = (observer_count > 0
                                         if args.require_observer_branch and case.startswith("point-Y")
                                         else True)
                    row.update(case=case, oracle_command=command,
                               oracle_sha256=file_digest(reference),
                               positive_observer=positive_observer,
                               certified_observer_tiles=observer_count)
                    row["passed"] = (row["parity"]["all30_bitwise"]
                                     and row["branch_witness"]["positive_macro"]
                                     and row["source_observation_counters_correct"]
                                     and positive_observer)
                    report["cases"].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    if not row["passed"]:
                        raise AssertionError(f"Geometry seam/clock failed: {case}")
            if before != [digest_array(value) for value in changed]:
                raise AssertionError("Candidate modified caller packet")
        report["jax_imported"] = "jax" in sys.modules
        report["passed"] = len(report["cases"]) == 4 * len(cases) and all(
            row["passed"] for row in report["cases"])
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"])}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
