#!/usr/bin/env python3
"""Check ordered coincident-source sums with3/7 rows against a pinned Program."""
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
from scripts.validate_cuda_native_source_masks import packet
from scripts.validate_cuda_plane_retirement import module_at, ordinary, state_of

COUNTS = ((3, 3), (7, 7), (3, 7), (7, 3))


def source_packet(inputs, attrs, counts):
    changed, attrs = packet(inputs, attrs, 20, (4, 32, 32), (4, 32, 32),
                            case="source-cpml-observer-Z4")
    for group, count in zip((5, 7), counts, strict=True):
        base = 86 + 3 * group
        coefficients, waveform, starts = changed[base:base + 3]
        extra = []
        waves = []
        for row in range(2, count):
            value = np.float32((row + 1) * (.375 if group == 5 else 2_000_000))
            if row % 2:
                value = -value
            spatial = np.full(coefficients.shape[1:], value, np.float32)
            spatial.reshape(-1)[1] = np.nextafter(value, np.float32(np.inf))
            extra.append(spatial)
            time = np.arange(waveform.shape[1], dtype=np.int32)
            waves.append(np.float32(.625) + ((time + row) % 7).astype(np.float32) * np.float32(.09375))
        changed[base] = np.concatenate((coefficients, np.stack(extra)), axis=0)
        changed[base + 1] = np.concatenate((waveform, np.stack(waves)), axis=0)
        changed[base + 2] = np.repeat(starts[:1], count, axis=0)
    return changed, attrs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--oracle-module", type=Path)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs=2)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "beamz.source-row-counts.v1", "passed": False, "cases": [],
              "module": str(args.module), "module_sha256": file_digest(args.module),
              "worker_sha256": file_digest(__file__), "fixture_sha256": file_digest(args.fixture)}
    try:
        fixture, data, inputs = load_fixture(args.fixture)
        cases = (tuple(args.counts),) if args.counts else COUNTS
        if args.reference_only:
            if len(cases) != 1:
                raise ValueError("Reference child needs one count pair")
            changed, attrs = source_packet(inputs, fixture["attributes"], cases[0])
            module = module_at(args.module)
            expected = ordinary(module, fixture, data, changed, attrs, 35)
            np.savez(args.output, **expected)
            return 0
        if args.oracle_module is None:
            raise ValueError("Separate pinned ordinary Program oracle is required")
        oracle_sha = file_digest(args.oracle_module)
        references = {}
        for counts in cases:
            target = args.output.with_name(args.output.stem + f".{counts[0]}-{counts[1]}.oracle.npz")
            command = [sys.executable, str(Path(__file__).resolve()), "--reference-only",
                       "--module", str(args.oracle_module), "--fixture", str(args.fixture),
                       "--counts", *map(str, counts), "--output", str(target)]
            if file_digest(args.oracle_module) != oracle_sha:
                raise RuntimeError("Oracle changed")
            subprocess.run(command, check=True, timeout=120)
            references[counts] = target
        module = module_at(args.module)
        for counts in cases:
            with np.load(references[counts], allow_pickle=False) as reference:
                expected = {key: np.array(reference[key]) for key in reference.files}
            for preserve in (False, True):
                for chunks in ([35], [17, 18]):
                    changed, attrs = source_packet(inputs, fixture["attributes"], counts)
                    before = [digest_array(value) for value in changed]
                    session = module.NativeSession(changed, attrs, preserve_inputs=preserve)
                    try:
                        for repeat in range(2):
                            if repeat and not preserve:
                                session.reset()
                            result = session.run_moving(chunks, 16)
                            parity = exact(state_of(fixture, data, session, result), expected)
                            metadata = session.metadata()
                            source_class = (metadata.get("moving_source_presence_mask") == 20
                                            and metadata.get("moving_source_mask20_dispatch_active")
                                            and sum(metadata.get("moving_source_mask20_policy_tiles", ())) > 0)
                            row = {"source_counts_H_E": counts, "preserve": preserve,
                                   "chunks": chunks, "repeat": repeat, "metadata": metadata,
                                   "parity": parity, "oracle_sha256": file_digest(references[counts]),
                                   "passed": parity["all30_bitwise"] and bool(source_class)
                                   and branch_witness(metadata, result)["positive_macro"]}
                            report["cases"].append(row)
                            if not row["passed"]:
                                raise AssertionError("Ordered multi-source sum mismatch")
                        if before != [digest_array(value) for value in changed]:
                            raise AssertionError("Candidate modified caller inputs")
                    finally:
                        session.close()
        report["passed"] = len(report["cases"]) == len(cases) * 8
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "states": len(report["cases"])}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
