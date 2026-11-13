#!/usr/bin/env python3
"""Validate guarded shared E-psi caching against separate ordinary Program oracles.

The numerical cases deliberately change profiles and source/observation sites.
Their fresh references are mandatory; persisted scene references are not valid.
Use the native-only interpreter and an exclusive GPU grant for execution.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cuda_fixture_store import digest_array
from scripts.cuda_native_probe_cases import branch_witness, point_probe_packet
from scripts.export_fdtdx_comparison import file_digest
from scripts.validate_cuda_native_moving import exact, load_fixture
from scripts.validate_cuda_native_profile_facts import PROFILE_INDICES
from scripts.validate_cuda_plane_retirement import module_at, ordinary, state_of

CASES = ("canonical", "unequal-H", "unequal-E", "signed-zero-E",
         "distinct-H-E", "caller-profile-mutation", "caller-unequal-to-equal")
PAIRS = ((0, 5), (1, 2), (3, 4))


def make_packet(inputs, attrs, case, steps):
    packet, attrs = point_probe_packet(inputs, attrs, (14, 16, 32), steps)
    packet = list(packet)
    for index in PROFILE_INDICES:
        packet[index] = np.array(packet[index], copy=True, order="C")
    if case == "unequal-H":
        packet[13 + 3 * 5] *= np.float32(.75)
    elif case in ("unequal-E", "caller-unequal-to-equal"):
        packet[44 + 3 * 5] *= np.float32(.75)
    elif case == "signed-zero-E":
        packet[44].reshape(-1)[0] = np.float32(0.)
        packet[44 + 3 * 5].reshape(-1)[0] = np.float32(-0.)
    elif case == "distinct-H-E":
        for term in range(6):
            packet[44 + 3 * term] *= np.float32(.875)
            packet[45 + 3 * term] *= np.float32(.9375)
    return packet, attrs


def mutate_after_upload(packet, case):
    if case == "caller-profile-mutation":
        for index in PROFILE_INDICES:
            packet[index][...] *= np.float32(.5)
    elif case == "caller-unequal-to-equal":
        for base in (13, 44):
            for first, second in PAIRS:
                for kind in range(3):
                    packet[base + 3 * second + kind][...] = packet[base + 3 * first + kind]


def cache_witness(metadata, result, require, packet=None):
    names = ("selector_enabled", "state_independent", "dispatch_active",
             "single_axis_tiles_zyx", "active_tiles", "retained_tiles", "retained_witness")
    facts = {name: metadata.get("moving_shared_epsi_" + name) for name in names}
    positive = (bool(facts["selector_enabled"]) and bool(facts["state_independent"])
                and bool(facts["dispatch_active"]) and int(facts["retained_tiles"] or 0) > 0
                and int(result.get("moving_macro_count", 0)) > 0)
    witness = facts["retained_witness"]
    counts = facts["single_axis_tiles_zyx"]
    active = int(facts["active_tiles"] or 0)
    retained = int(facts["retained_tiles"] or 0)
    positive = (positive and isinstance(counts, (tuple, list)) and len(counts) == 3
                and all(int(value) >= 0 for value in counts)
                and sum(int(value) for value in counts) >= active >= retained > 0)
    if positive:
        positive = (isinstance(witness, (tuple, list)) and len(witness) == 8
                    and 1 <= int(witness[1]) <= 16 and int(witness[1]) > int(witness[7])
                    and 1 <= int(witness[7]) < int(witness[1])
                    and 6 <= int(witness[2]) < 12 and int(witness[0]) >= 0
                    and 0 <= int(witness[6]) < 3
                    and int(counts[int(witness[6])]) > 0)
    if positive and packet is not None:
        term = int(witness[2]) - 6
        axis = int(witness[6])
        point = tuple(int(value) for value in witness[3:6])
        shape = packet[3 + term // 2].shape
        positive = (axis == (1, 0, 0, 2, 2, 1)[term]
                    and all(0 <= point[a] < shape[a] for a in range(3))
                    and (point[axis] < 12 or point[axis] >= shape[axis] - 12))
    facts["all_three_single_axes_present"] = bool(
        isinstance(counts, (tuple, list)) and len(counts) == 3 and all(int(v) > 0 for v in counts))
    facts["positive_retained_path"] = positive
    if require and not positive:
        raise AssertionError(f"No nonvacuous retained shared E-psi witness: {facts}")
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--oracle-module", type=Path)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--witness-kind", choices=("shared-epsi", "noio"), default="shared-epsi")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--ownership", choices=("both", "owned", "preserve"), default="both")
    parser.add_argument("--schedule", choices=("both", "whole", "split"), default="both")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "beamz.shared-epsi-validation.v1", "passed": False,
              "module": str(args.module), "module_sha256": file_digest(args.module),
              "worker_sha256": file_digest(__file__), "fixture": str(args.fixture),
              "fixture_sha256": file_digest(args.fixture), "cases": [], "steps": args.steps,
              "scope": "Canonical per-term profiles; no equal-profile assumption; depth16"}
    try:
        if importlib.util.find_spec("jax") is not None:
            raise RuntimeError("Use the NumPy-only interpreter")
        if args.steps != 35 or args.repeats < 1:
            raise ValueError("This frozen probe requires35 steps and a positive repeat count")
        fixture, data, inputs = load_fixture(args.fixture)
        cases = [case for case in CASES if args.case is None or case == args.case]
        if args.reference_only:
            if len(cases) != 1:
                raise ValueError("Reference child needs one case")
            packet, attrs = make_packet(inputs, fixture["attributes"], cases[0], args.steps)
            module = module_at(args.module)
            expected = ordinary(module, fixture, data, packet, attrs, args.steps)
            tail = ordinary(module, fixture, data, packet, attrs, 3)
            np.savez(args.output, **{"expected__" + k: v for k, v in expected.items()},
                     **{"tail__" + k: v for k, v in tail.items()})
            return 0
        if args.oracle_module is None:
            raise ValueError("A separately pinned --oracle-module is mandatory")
        oracle_sha = file_digest(args.oracle_module)
        report.update(oracle_module=str(args.oracle_module), oracle_module_sha256=oracle_sha)
        references = {}
        for case in cases:
            target = args.output.with_name(args.output.stem + "." + case + ".oracle.npz")
            command = [sys.executable, str(Path(__file__).resolve()), "--reference-only",
                       "--module", str(args.oracle_module), "--fixture", str(args.fixture),
                       "--case", case, "--output", str(target)]
            if file_digest(args.oracle_module) != oracle_sha:
                raise RuntimeError("Pinned ordinary oracle changed")
            subprocess.run(command, check=True, timeout=120)
            references[case] = target
        module = module_at(args.module)

        def kernel_witness(metadata, result, packet):
            if args.witness_kind == "shared-epsi":
                return cache_witness(metadata, result, True, packet)
            facts = branch_witness(metadata, result)
            if not facts["positive_noio_branch"]:
                raise AssertionError("No productive compiled NoIO macro")
            return facts

        ownerships = (False, True) if args.ownership == "both" else (args.ownership == "preserve",)
        schedules = ([35], [17, 18]) if args.schedule == "both" else (
            ([35],) if args.schedule == "whole" else ([17, 18],))
        for case in cases:
            with np.load(references[case], allow_pickle=False) as ref:
                expected = {k[10:]: np.array(ref[k]) for k in ref.files if k.startswith("expected__")}
                tail_expected = {k[6:]: np.array(ref[k]) for k in ref.files if k.startswith("tail__")}
            for preserve in ownerships:
                for chunks in schedules:
                    packet, attrs = make_packet(inputs, fixture["attributes"], case, args.steps)
                    session = module.NativeSession(packet, attrs, preserve_inputs=preserve)
                    try:
                        mutate_after_upload(packet, case)
                        before = [digest_array(value) for value in packet]
                        for repeat in range(args.repeats):
                            if repeat and not preserve:
                                session.reset()
                            result = session.run_moving(chunks, 16)
                            metadata = session.metadata()
                            facts = kernel_witness(metadata, result, packet)
                            parity = exact(state_of(fixture, data, session, result), expected)
                            row = {"case": case, "preserve": preserve, "chunks": chunks,
                                   "repeat": repeat, "parity": parity, "cache_witness": facts,
                                   "branch_witness": branch_witness(metadata, result),
                                   "metadata": metadata, "oracle_sha256": file_digest(references[case]),
                                   "passed": parity["all30_bitwise"]}
                            report["cases"].append(row)
                            args.output.write_text(json.dumps(report, indent=2) + "\n")
                            if not row["passed"]:
                                raise AssertionError(f"Shared E-psi state mismatch:{case}")
                        if before != [digest_array(value) for value in packet]:
                            raise AssertionError("Solve changed caller arrays after intentional mutation")
                        session.reset()
                        result = session.run_moving([3], 16)
                        metadata = session.metadata()
                        parity = exact(state_of(fixture, data, session, result), tail_expected)
                        row = {"case": case, "preserve": preserve, "chunks": [3],
                               "kind": "reset-tail-only", "parity": parity,
                               "actual_macros": int(result.get("moving_macro_count", 0)),
                               "metadata": metadata,
                               "passed": parity["all30_bitwise"]
                               and int(result.get("moving_macro_count", 0)) == 0}
                        report["cases"].append(row)
                        if not row["passed"]:
                            raise AssertionError("Tail-only/reset changed physics or executed a macro")
                        if not preserve and chunks == [35]:
                            session.reset()
                            session.run_moving([17], 16)
                            result = session.run_moving([18], 16)
                            metadata = session.metadata()
                            facts = kernel_witness(metadata, result, packet)
                            parity = exact(state_of(fixture, data, session, result), expected)
                            report["cases"].append({
                                "case": case, "kind": "owned-call-continuation-17-18",
                                "parity": parity, "cache_witness": facts, "metadata": metadata,
                                "passed": parity["all30_bitwise"]})
                            if not parity["all30_bitwise"]:
                                raise AssertionError("Cross-invocation continuation mismatch")
                    finally:
                        session.close()
        report["jax_imported"] = "jax" in sys.modules
        report["passed"] = bool(report["cases"]) and all(row["passed"] for row in report["cases"])
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "states": len(report["cases"])}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
