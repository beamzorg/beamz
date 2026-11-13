#!/usr/bin/env python3
"""Validate source-presence specialization using a separate pinned CUDA oracle.

All packets are changed correctness witnesses, never benchmark replacements.
Reference children finish before candidate GPU work starts. SourceOnly uses a
separated seeded detector; the coincident cases prove source-to-DFT sensitivity.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cuda_fixture_store import digest_array
from scripts.cuda_native_probe_cases import branch_witness
from scripts.export_fdtdx_comparison import file_digest
from scripts.validate_cuda_native_moving import exact, load_fixture
from scripts.validate_cuda_native_observer_dispatch import separated_probe_packet
from scripts.validate_cuda_plane_retirement import module_at, ordinary, state_of

CASES = (
    ("source-only", 20, (32, 32, 32), (80, 32, 32), 0, None),
    ("source-observer", 20, (32, 32, 32), (32, 32, 32), 1, None),
    ("source-cpml-observer", 20, (6, 32, 32), (6, 32, 32), 2, None),
    ("source-cpml-observer-Z4", 20, (4, 32, 32), (4, 32, 32), 2, None),
    ("zero-values20", 20, (32, 32, 32), (32, 32, 32), 1, None),
    *((f"fallback-{mask}", mask, (32, 32, 32), (32, 32, 32), None, None)
      for mask in (0, 1, 4, 16, 21, 63)),
    ("capture20-to0", 20, (32, 32, 32), (32, 32, 32), 1, 0),
    ("capture0-to20", 0, (32, 32, 32), (32, 32, 32), None, 20),
)


def packet(inputs, attributes, mask, source, observer, case=None):
    base, attrs = separated_probe_packet(inputs, attributes, source, observer, 35)
    templates = (base[101:104], base[107:110])
    changed = list(base)
    groups = []
    for g in range(9):
        changed[86 + 3*g:89 + 3*g] = [np.zeros((0, 1, 1, 1), np.float32),
                                      np.zeros((0, 35), np.float32),
                                      np.zeros((0, 3), np.int32)]
    for bit in range(6):
        if mask & (1 << bit):
            g = 3 + bit
            changed[86 + 3*g:89 + 3*g] = [value.copy() for value in templates[bit // 3]]
            # Deliberately non-dyadic products test sequential RN source adds.
            values = ((.17320507764816284, -.3141592741012573),
                      (23227684., -265949.3125))[bit // 3]
            for row, value in enumerate(values):
                changed[86+3*g][row, ...] = np.float32(value)
            if case == "source-cpml-observer-Z4":
                changed[86+3*g] = np.repeat(changed[86+3*g], 2, axis=1)
                changed[86+3*g].view(np.uint32)[:, 1, ...] += np.uint32(2)
                changed[88+3*g][:, 0] -= 1  # Source covers physical Z3 and Z4.
            if case == "zero-values20":
                changed[86+3*g].fill(0)  # Counts stay two, class stays20.
            groups.append(g)
    attrs.update(coincident_source_group_mask=sum(1 << g for g in groups),
                 disjoint_source_group_mask=0, nsteps=35)
    return changed, attrs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--oracle-module", type=Path)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=tuple(row[0] for row in CASES))
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--case-group", choices=("all", "positive-kernels"), default="all")
    parser.add_argument("--ownership", choices=("both", "owned", "preserve"), default="both")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--schedule", choices=("both", "whole", "split"), default="both")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "beamz.native-source-masks.v1", "passed": False, "cases": [],
              "module": str(args.module), "module_sha256": file_digest(args.module),
              "worker_sha256": file_digest(__file__), "fixture": str(args.fixture),
              "fixture_sha256": file_digest(args.fixture), "depth": 16, "steps": 35}
    try:
        fixture, data, inputs = load_fixture(args.fixture)
        cases = [row for row in CASES if args.case is None or row[0] == args.case]
        if args.case is None and args.case_group == "positive-kernels":
            cases = [row for row in cases if row[0] in ("source-only", "source-observer", "source-cpml-observer-Z4")]
        if args.repeats < 1:
            raise ValueError("Positive repeat count required")
        if args.reference_only:
            if len(cases) != 1:
                raise ValueError("Reference child requires one explicit case")
            name, mask, source, observer, _, _ = cases[0]
            changed, attrs = packet(inputs, fixture["attributes"], mask, source, observer, case=name)
            module = module_at(args.module)
            expected = ordinary(module, fixture, data, changed, attrs, 35)
            quiet = list(changed)
            for g in range(3, 9):
                quiet[86 + 3*g] = np.zeros_like(changed[86 + 3*g])
            disabled = ordinary(module, fixture, data, quiet, attrs, 35)
            np.savez(args.output, **{"expected__"+k: v for k, v in expected.items()},
                     **{"quiet__"+k: v for k, v in disabled.items()})
            return 0
        if args.oracle_module is None:
            raise ValueError("A separate pinned --oracle-module is mandatory")
        report["oracle_module"] = str(args.oracle_module)
        report["oracle_module_sha256"] = file_digest(args.oracle_module)
        references = {}
        for name, *_ in cases:
            path = args.output.with_name(args.output.stem + "." + name + ".oracle.npz")
            command = [sys.executable, str(Path(__file__).resolve()), "--reference-only",
                       "--module", str(args.oracle_module), "--fixture", str(args.fixture),
                       "--case", name, "--output", str(path)]
            if file_digest(args.oracle_module) != report["oracle_module_sha256"]:
                raise RuntimeError("Pinned oracle changed")
            subprocess.run(command, check=True, timeout=120)
            if file_digest(args.oracle_module) != report["oracle_module_sha256"]:
                raise RuntimeError("Pinned oracle changed during reference")
            references[name] = path
        module = module_at(args.module)
        if not (getattr(module, "moving_source_mask20_512", False)
                or getattr(module, "moving_source_pending_mask20_1024", False)):
            raise ValueError("Candidate lacks the source-mask20 specialization")
        for name, mask, source, observer, required_policy, replacement in cases:
            reference = references[name]
            with np.load(reference, allow_pickle=False) as values:
                expected = {k[10:]: values[k] for k in values.files if k.startswith("expected__")}
                quiet = {k[7:]: values[k] for k in values.files if k.startswith("quiet__")}
            field_live = any(not np.array_equal(expected[k], quiet[k])
                             for k in ("ex", "ey", "ez", "hx", "hy", "hz"))
            dft_live = any(not np.array_equal(expected[k], quiet[k])
                           for k in expected if "dft" in k)
            if mask and name != "zero-values20" and not field_live:
                raise AssertionError("Source witness has no field effect")
            if mask and name != "zero-values20" and source == observer and not dft_live:
                raise AssertionError("Coincident source/DFT witness is not live")
            ownerships = (False, True) if args.ownership == "both" else (args.ownership == "preserve",)
            schedules = ([35], [17, 18]) if args.schedule == "both" else (([35],) if args.schedule == "whole" else ([17, 18],))
            for preserve in ownerships:
                for chunks in schedules:
                    changed, attrs = packet(inputs, fixture["attributes"], mask, source, observer, case=name)
                    session = module.NativeSession(changed, attrs, preserve_inputs=preserve)
                    try:
                        if replacement is not None:
                            for g in range(3, 9):
                                changed[86+3*g][...] *= np.float32(1.5)
                                changed[87+3*g][...] *= np.float32(.5)
                                changed[88+3*g][..., 1] += 1
                            alternate, other_attrs = packet(inputs, fixture["attributes"], replacement, source, observer)
                            changed[86:113] = alternate[86:113]
                            attrs.update(other_attrs)
                        before = [digest_array(value) for value in changed]
                        for repeat in range(args.repeats):
                            if repeat and not preserve:
                                session.reset()
                            result = session.run_moving(chunks, 16)
                            metadata = session.metadata()
                            witness = branch_witness(metadata, result)
                            parity = exact(state_of(fixture, data, session, result), expected)
                            counts = list(metadata.get("moving_source_mask20_policy_tiles", ()))
                            active = bool(metadata.get("moving_source_mask20_dispatch_active", False))
                            class_ok = (int(metadata.get("moving_source_presence_mask", -1)) == mask
                                        and active == (mask == 20)
                                        and (required_policy is None or counts[required_policy] > 0))
                            row = {"case": name, "initial_mask": mask,
                                   "caller_replacement_mask": replacement, "preserve": preserve,
                                   "repeat": repeat, "chunks": chunks, "parity": parity,
                                   "metadata": metadata, "branch_witness": witness,
                                   "source_disabled_field_sensitivity": field_live,
                                   "source_disabled_DFT_sensitivity": dft_live,
                                   "DFT_sensitivity_required": bool(mask and name != "zero-values20" and source == observer),
                                   "source_values_zero": name == "zero-values20",
                                   "oracle_sha256": file_digest(reference),
                                   "passed": parity["all30_bitwise"] and class_ok
                                   and witness["positive_macro"]}
                            report["cases"].append(row)
                            args.output.write_text(json.dumps(report, indent=2)+"\n")
                            if not row["passed"]:
                                raise AssertionError(f"Source mask failed: {name}")
                        if before != [digest_array(value) for value in changed]:
                            raise AssertionError("Solve modified caller data after intentional mutation")
                    finally:
                        session.close()
        report["jax_imported"] = "jax" in sys.modules
        expected_cases = len(cases) * len(ownerships) * len(schedules) * args.repeats
        report["passed"] = len(report["cases"]) == expected_cases and all(
            row["passed"] for row in report["cases"])
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"])}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
