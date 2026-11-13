#!/usr/bin/env python3
"""Gate actual complete moving macros against independent ordinary schedules.

Runs only with JAX absent. Includes immutable host inputs, physical state resets,
chunk/odd-tail continuation, same-shaped changed scenes, and strict all30 leaves.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_cuda_native_runtime import logical_native_state
from scripts.cuda_fixture_store import ArrayReader, compare_arrays, digest_array
from scripts.export_fdtdx_comparison import file_digest


def load_fixture(path):
    fixture = json.loads(path.read_text())
    data = (
        ArrayReader(path.parent / fixture["data_directory"])
        if fixture.get("data_format") == "npy_directory_v1"
        else np.load(path.parent / fixture["data_file"], allow_pickle=False)
    )
    return fixture, data, [data[item["key"]] for item in fixture["inputs"]]


def exact(actual, expected):
    leaves = {
        name: compare_arrays(actual[name], value) for name, value in expected.items()
    }
    return {
        "all30_bitwise": len(leaves) == 30
        and all(item["bitwise_exact"] for item in leaves.values()),
        "leaves": leaves,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--depth", type=int, choices=(4, 8, 16, 32), required=True)
    p.add_argument("--steps", type=int)
    p.add_argument("--oracle-kind", choices=("program", "mstep"), default="program",
                   help="Fresh independent ordinary Program by default")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if importlib.util.find_spec("jax") is not None:
        p.error("This proof requires a NumPy-only environment with JAX absent")
    spec = importlib.util.spec_from_file_location("_cuda_native", a.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture, data, inputs = load_fixture(a.fixture)
    before = [digest_array(value) for value in inputs]
    steps = a.steps or fixture["attributes"]["nsteps"]
    if steps < a.depth:
        p.error("Positive gates must execute an actual full moving macro")
    records = []

    def state(session, result):
        clocks = session.clocks()
        if clocks["elapsed_steps"] - clocks["current_step"] != int(inputs[128]) - int(
            inputs[127]
        ):
            raise RuntimeError(
                "Elapsed/current-step clock relation differs from the uploaded origin"
            )
        if np.float32(clocks["observation_origin"]).view(np.uint32) != inputs[126].view(
            np.uint32
        ):
            raise RuntimeError("Observation origin changed")
        return logical_native_state(
            fixture, data, session.download(), clocks, result["output_bank"]
        )

    def ordinary(values, chunks, program=a.oracle_kind == "program"):
        session = module.NativeSession(
            values,
            fixture["attributes"],
            preserve_inputs=False,
            kind="program" if program else "mstep",
        )
        try:
            result = session.run_schedule(chunks)
            return state(session, result)
        finally:
            session.close()

    expected = (
        ordinary(
            inputs,
            [min(256, steps)] * (steps // 256) + ([steps % 256] if steps % 256 else []),
        )
        if steps >= 256
        else ordinary(inputs, [steps])
    )
    schedules = [[steps]]
    if steps > a.depth:
        schedules.extend(
            [[a.depth, steps - a.depth], [a.depth + 1, steps - a.depth - 1]]
            if steps > a.depth + 1
            else [[1, steps - 1]]
        )
    if steps >= 2 * a.depth + 3:
        schedules.append([a.depth + 3, a.depth - 1, steps - 2 * a.depth - 2])
    if steps >= 256:
        schedules.append(
            [256] * (steps // 256) + ([steps % 256] if steps % 256 else [])
        )
    for preserve in (False, True):
        for chunks in schedules:
            if any(count <= 0 for count in chunks):
                continue
            session = module.NativeSession(
                inputs, fixture["attributes"], preserve_inputs=preserve
            )
            try:
                for repeat in range(2):
                    if repeat and not preserve:
                        session.reset()
                    result = session.run_moving(chunks, a.depth)
                    records.append(
                        {
                            "case": "schedule-reset-repeat",
                            "preserve": preserve,
                            "chunks": chunks,
                            "repeat": repeat,
                            "actual_macro_count": sum(n // a.depth for n in chunks),
                            "parity": exact(state(session, result), expected),
                            "clocks": session.clocks(),
                            "metadata": session.metadata(),
                        }
                    )
            finally:
                session.close()
    session = module.NativeSession(inputs, fixture["attributes"], preserve_inputs=False)
    try:
        split = a.depth + 1 if steps > a.depth + 1 else 1
        session.run_moving([split], a.depth)
        result = session.run_moving([steps - split], a.depth)
        records.append(
            {
                "case": "cross-call-continuation",
                "chunks": [split, steps - split],
                "actual_macro_count": split // a.depth + (steps - split) // a.depth,
                "parity": exact(state(session, result), expected),
            }
        )
    finally:
        session.close()
    # Fresh sessions can reuse the same CUDA addresses. Host-selected tile facts
    # must follow the new source starts / raw-observation locations, not an old graph.
    for kind in ("source-start", "monitor-index"):
        changed = list(inputs)
        if kind == "source-start":
            for g in range(3, 9):
                coeff = changed[86 + 3 * g]
                if coeff.shape[0]:
                    index = 88 + 3 * g
                    starts = changed[index].copy()
                    axis = next(
                        (
                            axis
                            for axis in range(3)
                            if coeff.shape[axis + 1] + int(starts[0, axis]) + 1
                            < min(value.shape[axis] for value in changed[:6])
                        ),
                        None,
                    )
                    if axis is not None:
                        starts[:, axis] += 1
                        changed[index] = starts
                        break
            else:
                raise RuntimeError("No source available for changed-scene gate")
        else:
            indices = changed[113].copy()
            for c in range(6):
                limit = inputs[(c + 3) % 6].size
                active = indices[:, c] >= 0
                indices[:, c][active] = (indices[:, c][active] + 1) % limit
            changed[113] = indices
        ref = ordinary(changed, [steps], program=True)
        session = module.NativeSession(
            changed, fixture["attributes"], preserve_inputs=True
        )
        try:
            result = session.run_moving([steps], a.depth)
            records.append(
                {
                    "case": "same-shape-changed-" + kind,
                    "actual_macro_count": steps // a.depth,
                    "parity": exact(state(session, result), ref),
                }
            )
        finally:
            session.close()
    # Host-selected facts must describe the immutable upload, even if callers
    # mutate their original tiny coefficient/start/index arrays afterward.
    changed = list(inputs)
    captured = list(range(6, 12)) + [113] + [88 + 3 * g for g in range(9)]
    for index in captured:
        changed[index] = inputs[index].copy()
    session = module.NativeSession(
        changed, fixture["attributes"], preserve_inputs=False
    )
    try:
        for index in range(9, 12):
            changed[index][...] *= np.float32(2)
        for g in range(9):
            if changed[88 + 3 * g].size:
                changed[88 + 3 * g][...] += np.int32(1)
        for c in range(6):
            active = changed[113][:, c] >= 0
            changed[113][:, c][active] = (changed[113][:, c][active] + 1) % inputs[
                c
            ].size
        result = session.run_moving([steps], a.depth)
        records.append(
            {
                "case": "captured-upload-facts-ignore-later-host-mutation",
                "actual_macro_count": steps // a.depth,
                "parity": exact(state(session, result), expected),
            }
        )
    finally:
        session.close()
    negatives = []
    for kind in ("pec", "dense-H", "single-source", "pre-E"):
        changed = list(inputs)
        attrs = dict(fixture["attributes"])
        if kind == "pec":
            attrs["boundary_code"] |= 1
        elif kind == "dense-H":
            changed[9] = np.full(inputs[0].shape, float(inputs[9]), dtype=np.float32)
        elif kind == "single-source":
            g = next(g for g in range(3, 9) if inputs[86 + 3 * g].shape[0])
            for index in range(86 + 3 * g, 89 + 3 * g):
                changed[index] = inputs[index][:1].copy()
        else:
            g = next(g for g in range(6, 9) if inputs[86 + 3 * g].shape[0])
            target = g % 3
            for offset in range(3):
                changed[86 + 3 * target + offset] = inputs[86 + 3 * g + offset]
            attrs["coincident_source_group_mask"] |= 1 << target
        session = module.NativeSession(changed, attrs, preserve_inputs=False)
        try:
            payload = session.metadata()["allocated_payload_bytes"]
            try:
                session.run_moving([a.depth], a.depth)
            except ValueError as error:
                negatives.append(
                    {
                        "case": kind,
                        "error": str(error),
                        "passed": not session.metadata()["moving_ready"]
                        and session.metadata()["allocated_payload_bytes"] == payload,
                    }
                )
            else:
                negatives.append({"case": kind, "passed": False})
        finally:
            session.close()
    witnesses = {
        name: {
            "nonzero_count": int(np.count_nonzero(value)),
            "max_abs": float(np.max(np.abs(value))) if value.size else 0.0,
        }
        for name, value in expected.items()
        if "dft" in name.lower()
    }
    fixture_parity = None
    if steps == fixture["attributes"]["nsteps"]:
        fixture_parity = exact(
            expected,
            {
                item["name"]: data[item["reference_key"]]
                for item in fixture["state_leaves"]
            },
        )
    immutable = before == [digest_array(value) for value in inputs]
    report = {
        "schema": "beamz.native-moving-complete-gates.v1",
        "module": str(a.module),
        "module_sha256": file_digest(a.module),
        "fixture": str(a.fixture),
        "fixture_sha256": file_digest(a.fixture),
        "worker_sha256": file_digest(__file__),
        "depth": a.depth,
        "oracle_kind": a.oracle_kind,
        "steps": steps,
        "resource_plan": module.moving_resource_plan(inputs, a.depth),
        "cases": records,
        "negative_cases": negatives,
        "dft_witnesses": witnesses,
        "ordinary_vs_independent_fixture": fixture_parity,
        "host_inputs_unchanged": immutable,
        "jax_imported": "jax" in sys.modules,
        "passed": immutable
        and all(item["passed"] for item in negatives)
        and all(item["parity"]["all30_bitwise"] for item in records)
        and (fixture_parity is None or fixture_parity["all30_bitwise"]),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "cases": len(records),
                "actual_macro_counts": [item["actual_macro_count"] for item in records],
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
