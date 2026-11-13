#!/usr/bin/env python3
"""Validate complete hybrid schedules, resets, repeats and unsupported sources."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_cuda_native_runtime import compare_state, logical_native_state
from scripts.cuda_fixture_store import ArrayReader, compare_arrays
from scripts.export_cuda_native_fixture import buffer_digest
from scripts.export_fdtdx_comparison import file_digest


def exact_state(actual, expected):
    parity = compare_state(actual, expected)
    parity["all30_bitwise"] = all(
        compare_arrays(actual[name], value)["bitwise_exact"]
        for name, value in expected.items()
    )
    return parity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--negative-fixture", type=Path)
    p.add_argument(
        "--method", choices=("hybrid", "mstep", "grouped", "rolling4"), default="hybrid"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if importlib.util.find_spec("jax") is not None:
        raise RuntimeError("Requires JAX absent")
    spec = importlib.util.spec_from_file_location("_cuda_native", a.module)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    f = json.loads(a.fixture.read_text())
    d = (
        ArrayReader(a.fixture.parent / f["data_directory"])
        if f.get("data_format") == "npy_directory_v1"
        else np.load(a.fixture.parent / f["data_file"], allow_pickle=False)
    )
    inputs = [d[x["key"]] for x in f["inputs"]]
    before = [buffer_digest(x) for x in inputs]
    expected = {x["name"]: d[x["reference_key"]] for x in f["state_leaves"]}
    records = []
    for preserve in [False, True]:
        for chunks in [[7], [3, 4], [3, 3, 1], [1, 1, 5], [2, 5]]:
            session = m.NativeSession(inputs, f["attributes"], preserve_inputs=preserve)
            try:
                for repeat in range(2):
                    if repeat and not preserve:
                        session.reset()
                    run = getattr(
                        session,
                        {
                            "hybrid": "run_hybrid",
                            "grouped": "run_grouped",
                            "rolling4": "run_rolling4",
                            "mstep": "run_schedule",
                        }[a.method],
                    )
                    result = run(chunks)
                    actual = logical_native_state(
                        f,
                        d,
                        session.download(),
                        session.clocks(),
                        result["output_bank"],
                    )
                    parity = exact_state(actual, expected)
                    records.append(
                        {
                            "preserve": preserve,
                            "chunks": chunks,
                            "repeat": repeat,
                            "parity": parity,
                            "clocks": session.clocks(),
                            "metadata": session.metadata(),
                        }
                    )
            finally:
                session.close()
    session = m.NativeSession(inputs, f["attributes"], preserve_inputs=False)
    try:
        run = getattr(
            session,
            {
                "hybrid": "run_hybrid",
                "grouped": "run_grouped",
                "rolling4": "run_rolling4",
                "mstep": "run_schedule",
            }[a.method],
        )
        run([3])
        result = run([4])
        actual = logical_native_state(
            f, d, session.download(), session.clocks(), result["output_bank"]
        )
        records.append(
            {
                "case": "cross-call3+4",
                "parity": exact_state(actual, expected),
                "clocks": session.clocks(),
            }
        )
    finally:
        session.close()
    negative = None
    if a.negative_fixture:
        if a.method != "hybrid":
            p.error("The source-overlap rejection is specific to --method hybrid")
        nf = json.loads(a.negative_fixture.read_text())
        nd = np.load(a.negative_fixture.parent / nf["data_file"], allow_pickle=False)
        ni = [nd[x["key"]] for x in nf["inputs"]]
        session = m.NativeSession(ni, nf["attributes"], preserve_inputs=False)
        try:
            payload = session.metadata()["allocated_payload_bytes"]
            try:
                session.run_hybrid([7])
            except ValueError as error:
                negative = {
                    "error": str(error),
                    "passed": "source overlaps" in str(error),
                    "no_device_allocations": session.metadata()[
                        "allocated_payload_bytes"
                    ]
                    == payload,
                    "ready": session.metadata()["hybrid_ready"],
                }
            else:
                negative = {"passed": False, "error": "Unsupported source accepted"}
        finally:
            session.close()
    host = before == [buffer_digest(x) for x in inputs]
    report = {
        "schema": "beamz.hybrid-complete-gates.v1",
        "module": str(a.module),
        "module_sha256": file_digest(a.module),
        "method": a.method,
        "fixture": str(a.fixture),
        "fixture_sha256": file_digest(a.fixture),
        "worker_sha256": file_digest(__file__),
        "cases": records,
        "negative": negative,
        "host_inputs_unchanged": host,
        "jax_imported": "jax" in sys.modules,
        "passed": host
        and all(x["parity"]["all30_bitwise"] for x in records)
        and (
            negative is None or negative["passed"] and negative["no_device_allocations"]
        ),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "cases": len(records),
                "negative": negative,
                "host_inputs_unchanged": host,
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
