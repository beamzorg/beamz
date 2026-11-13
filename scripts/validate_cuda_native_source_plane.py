#!/usr/bin/env python3
"""Exercise full-plane source specialization and physical fallback packets."""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cuda_fixture_store import digest_array
from scripts.cuda_native_block_cases import rank2_e_packet
from scripts.cuda_native_probe_cases import branch_witness
from scripts.export_fdtdx_comparison import file_digest
from scripts.validate_cuda_native_moving import exact, load_fixture
from scripts.validate_cuda_plane_retirement import module_at, ordinary, state_of

CASES = ("normal-plane", "direct-source-DFT", "count3", "nx2", "cropped-ZY",
         "shifted-ZY", "caller-shift-after-true", "caller-equalize-after-false",
         "rank2", "nonisotropic-H")


def make_packet(inputs, attrs, case, steps=35, synthetic=False):
    packet, attrs = list(inputs), dict(attrs)
    attrs["nsteps"] = steps
    if synthetic:
        nz, ny, nx = (min(field.shape[axis] for field in inputs[:6]) for axis in range(3))
        attrs.update(coincident_source_group_mask=160, disjoint_source_group_mask=0)
        for group in range(9):
            base, count = 86 + 3 * group, 2 if group in (5, 7) else 0
            packet[base] = np.zeros((count, nz - 24, ny - 24, 1), np.float32)
            packet[base + 1] = np.zeros((count, steps), np.float32)
            x = min(31, nx - 32) + (group == 7)
            packet[base + 2] = np.repeat(np.array([[12, 12, x]], np.int32), count, axis=0)
            if count:
                for row, value in enumerate((.1732050776, -.3141592741) if group == 5
                                            else (23227684., -265949.3125)):
                    packet[base][row] = np.float32(value)
                    packet[base][row, -1, -1, 0] = np.nextafter(
                        np.float32(value), np.float32(np.inf))
    for group in (5, 7):
        base = 86 + 3 * group
        packet[base] = np.array(packet[base], copy=True)
        packet[base + 2] = np.array(packet[base + 2], copy=True)
        count = 3 if case == "count3" else 2
        time = np.arange(steps, dtype=np.int32)
        packet[base + 1] = np.stack([
            np.float32((1 if row % 2 == 0 else -.5))
            + ((time + row) % 7).astype(np.float32) * np.float32(.09375)
            for row in range(count)])
        if case == "count3":
            packet[base] = np.concatenate((packet[base], packet[base][:1] * np.float32(.625)))
            packet[base + 2] = np.repeat(packet[base + 2][:1], 3, axis=0)
        if case == "nx2":
            packet[base] = np.concatenate((packet[base], packet[base] * np.float32(.75)), axis=3)
        if case == "cropped-ZY":
            packet[base] = np.ascontiguousarray(packet[base][:, :-1, :-1, :])
        if case in ("shifted-ZY", "caller-equalize-after-false"):
            packet[base + 2][:, :2] += 1
    if case == "rank2":
        packet = rank2_e_packet(packet)
    if case == "nonisotropic-H":
        packet[9] = np.asarray(np.array(packet[9], copy=True) * np.float32(1.0001), dtype=np.float32)
    if case != "normal-plane":
        point = (32, 32, int(packet[109][0, 2]))  # E-y group7 starts.
        indices, weights = np.full_like(packet[113], -1), np.zeros_like(packet[114])
        for component in range(6):
            native = (component + 3) % 6
            indices[:, component, 0, 0] = np.ravel_multi_index(point, packet[native].shape)
            weights[:, component, 0, 0] = 1
        packet[113], packet[114] = indices, weights
    return packet, attrs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--oracle-module", type=Path)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--collector", type=Path, help="Optional bounded actual-kernel CUPTI witness")
    parser.add_argument("--steps", type=int, choices=(17, 35), default=35)
    parser.add_argument("--ownership", choices=("both", "owned", "preserve"), default="both")
    parser.add_argument("--schedule", choices=("both", "whole", "split"), default="both")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--synthetic-full-plane", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "beamz.source-plane-validation.v1", "passed": False, "cases": [],
              "module": str(args.module), "module_sha256": file_digest(args.module),
              "fixture_sha256": file_digest(args.fixture), "worker_sha256": file_digest(__file__)}
    try:
        fixture, data, inputs = load_fixture(args.fixture)
        cases = [case for case in CASES if args.case is None or case == args.case]
        if args.reference_only:
            if len(cases) != 1:
                raise ValueError("Reference requires one case")
            packet, attrs = make_packet(inputs, fixture["attributes"], cases[0], args.steps, args.synthetic_full_plane)
            module = module_at(args.module)
            expected = ordinary(module, fixture, data, packet, attrs, args.steps)
            quiet = list(packet)
            for group in (5, 7):
                quiet[86 + 3 * group] = np.zeros_like(packet[86 + 3 * group])
            disabled = ordinary(module, fixture, data, quiet, attrs, args.steps)
            np.savez(args.output, **{"expected__" + k: v for k, v in expected.items()},
                     **{"quiet__" + k: v for k, v in disabled.items()})
            return 0
        if args.oracle_module is None:
            raise ValueError("A separate pinned ordinary Program oracle is mandatory")
        oracle_sha = file_digest(args.oracle_module)
        references = {}
        for case in cases:
            target = args.output.with_name(args.output.stem + "." + case + ".oracle.npz")
            if file_digest(args.oracle_module) != oracle_sha:
                raise RuntimeError("Pinned oracle changed")
            command = [sys.executable, str(Path(__file__).resolve()), "--reference-only",
                       "--module", str(args.oracle_module), "--fixture", str(args.fixture),
                       "--case", case, "--steps", str(args.steps), "--output", str(target)]
            if args.synthetic_full_plane:
                command.append("--synthetic-full-plane")
            subprocess.run(command, check=True, timeout=120)
            references[case] = target
        module = module_at(args.module)
        ownerships = (False, True) if args.ownership == "both" else (args.ownership == "preserve",)
        split = [17, 18] if args.steps == 35 else [16, args.steps - 16]
        schedules = ([args.steps], split) if args.schedule == "both" else (
            ([args.steps],) if args.schedule == "whole" else (split,))
        for case in cases:
            with np.load(references[case], allow_pickle=False) as ref:
                expected = {k[10:]: np.array(ref[k]) for k in ref.files if k.startswith("expected__")}
                quiet = {k[7:]: np.array(ref[k]) for k in ref.files if k.startswith("quiet__")}
            live = any(not np.array_equal(expected[k], quiet[k]) for k in expected)
            if not live:
                raise AssertionError("Source waveform has no physical sensitivity")
            for preserve in ownerships:
                for chunks in schedules:
                    packet, attrs = make_packet(inputs, fixture["attributes"], case, args.steps, args.synthetic_full_plane)
                    session = module.NativeSession(packet, attrs, preserve_inputs=preserve)
                    try:
                        if case in ("caller-shift-after-true", "caller-equalize-after-false"):
                            for group in (5, 7):
                                packet[88 + 3 * group][:, :2] += (
                                    1 if case == "caller-shift-after-true" else -1)
                        before = [digest_array(value) for value in packet]
                        for repeat in range(args.repeats):
                            if repeat and not preserve:
                                session.reset()
                            library = None
                            if args.collector:
                                if len(cases) != 1 or len(ownerships) != 1 or len(schedules) != 1 or args.repeats != 1:
                                    raise ValueError("CUPTI witness requires one case, ownership, schedule and repeat")
                                library = ctypes.CDLL(str(args.collector.resolve()))
                                library.beamz_profile_start.argtypes = [ctypes.c_char_p]
                                library.beamz_profile_error.restype = ctypes.c_char_p
                                trace = args.output.with_suffix(".activity.jsonl")
                                if library.beamz_profile_start(str(trace.resolve()).encode()):
                                    raise RuntimeError(str(library.beamz_profile_error()))
                            try:
                                result = session.run_moving(chunks, 16)
                            finally:
                                if library and library.beamz_profile_stop():
                                    raise RuntimeError(str(library.beamz_profile_error()))
                            if args.collector:
                                report["activity"] = {"path": str(trace), "sha256": file_digest(trace),
                                                      "collector_sha256": file_digest(args.collector)}
                            metadata = session.metadata()
                            parity = exact(state_of(fixture, data, session, result), expected)
                            want_class = case not in ("count3", "nx2", "cropped-ZY", "shifted-ZY",
                                                      "caller-equalize-after-false")
                            want_dispatch = want_class and case not in ("rank2", "nonisotropic-H")
                            class_ok = bool(metadata.get("moving_source_plane_pair_class")) == want_class
                            dispatch_ok = bool(metadata.get("moving_source_plane_pair_dispatch_active")) == want_dispatch
                            tiles = metadata.get("moving_source_plane_pair_policy_tiles", ())
                            p4_positive = case != "direct-source-DFT" or (len(tiles) == 2 and tiles[1] > 0)
                            row = {"case": case, "preserve": preserve, "chunks": chunks,
                                   "repeat": repeat, "metadata": metadata, "parity": parity,
                                   "source_sensitivity": live, "expected_class": want_class,
                                   "expected_dispatch": want_dispatch, "positive_policy4": p4_positive,
                                   "passed": parity["all30_bitwise"] and class_ok and dispatch_ok
                                   and p4_positive and branch_witness(metadata, result)["positive_macro"]}
                            report["cases"].append(row)
                            if not row["passed"]:
                                raise AssertionError(f"Source-plane guard/state failure:{case}")
                        if before != [digest_array(value) for value in packet]:
                            raise AssertionError("Solve changed caller arrays")
                    finally:
                        session.close()
        report["passed"] = len(report["cases"]) == len(cases) * len(ownerships) * len(schedules) * args.repeats
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "states": len(report["cases"])}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
