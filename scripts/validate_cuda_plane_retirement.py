#!/usr/bin/env python3
"""Bounded complete native geometry/material/probe checks, with explicit coverage.

Use the NumPy-only interpreter. Probe packets change source/monitor locations and
always receive a fresh ordinary CUDA Program oracle; they are not benchmark cases.
A sanitizer can select --case-index to instrument one positive macro cheaply.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_cuda_native_runtime import logical_native_state
from scripts.cuda_fixture_store import digest_array
from scripts.cuda_native_block_cases import rank2_e_packet
from scripts.cuda_native_const_block_cases import constant_rank2_e_packet
from scripts.cuda_native_probe_cases import (
    branch_witness,
    point_probe_packet,
    probe_points,
)
from scripts.export_fdtdx_comparison import file_digest
from scripts.validate_cuda_native_moving import exact, load_fixture


def module_at(path):
    if importlib.util.find_spec("jax") is not None:
        raise RuntimeError("Validation requires JAX absent")
    spec = importlib.util.spec_from_file_location("_cuda_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_of(fixture, data, session, result):
    return logical_native_state(fixture, data, session.download(), session.clocks(),
                                result["output_bank"])


def ordinary(module, fixture, data, inputs, attrs, steps):
    session = module.NativeSession(inputs, attrs, preserve_inputs=False, kind="program")
    try:
        result = session.run_schedule([steps])
        return state_of(fixture, data, session, result)
    finally:
        session.close()


def moving_repeats(module, fixture, data, inputs, attrs, expected, args, preserve):
    session = module.NativeSession(inputs, attrs, preserve_inputs=preserve)
    rows = []
    try:
        for repeat in range(args.repeats):
            if repeat and not preserve:
                session.reset()
            result = session.run_moving([args.steps], args.depth)
            metadata = session.metadata()
            clocks = session.clocks()
            counters_ok = (clocks["current_step"] == int(inputs[127]) + args.steps
                           and clocks["elapsed_steps"] == int(inputs[128]) + args.steps)
            rows.append({"repeat": repeat, "preserve": preserve,
                         "parity": exact(state_of(fixture, data, session, result), expected),
                         "branch_witness": branch_witness(metadata, result),
                         "source_observation_counters_correct": counters_ok,
                         "metadata": metadata})
    finally:
        session.close()
    return rows


def probe_cases(module, fixture, data, inputs, args):
    rows = []
    points = probe_points(inputs)
    if args.case_index is not None:
        points = [points[args.case_index]]
    for point in points:
        changed, attrs = point_probe_packet(inputs, fixture["attributes"], point, args.steps)
        before = [digest_array(x) for x in changed]
        expected = ordinary(module, fixture, data, changed, attrs, args.steps)
        quiet = list(changed)
        for group in (5, 7):
            quiet[86 + 3 * group] = np.zeros_like(changed[86 + 3 * group])
        no_source = ordinary(module, fixture, data, quiet, attrs, args.steps)
        field_delta = any(not np.array_equal(expected[n], no_source[n])
                          for n in ("ex", "ey", "ez", "hx", "hy", "hz"))
        dft_delta = any(not np.array_equal(expected[n], no_source[n])
                        for n in expected if "dft" in n)
        results = moving_repeats(module, fixture, data, changed, attrs, expected,
                                 args, preserve=False)
        rows.append({"case": "point-source-and-direct-six-component-observer",
                     "point_zyx": point, "changed_packet": True,
                     "source_disabled_sensitivity": {"fields": field_delta, "DFT": dft_delta},
                     "source_and_observation_live": field_delta and dft_delta,
                     "host_inputs_unchanged": before == [digest_array(x) for x in changed],
                     "results": results})
        args.progress(rows)
    return rows


def material_cases(module, fixture, data, inputs, args):
    attrs = dict(fixture["attributes"], nsteps=args.steps)
    expected = ordinary(module, fixture, data, inputs, attrs, args.steps)
    rows = []
    for preserve in (False, True):
        changed = list(inputs)
        for i in range(37, 43):
            changed[i] = inputs[i].copy()
        session = module.NativeSession(changed, attrs, preserve_inputs=preserve)
        try:
            for i in range(37, 40):
                changed[i][...] *= np.float32(2)
            for i in range(40, 43):
                changed[i].view(np.uint32)[...] ^= np.uint32(0x01010101)
            for repeat in range(args.repeats):
                if repeat and not preserve:
                    session.reset()
                result = session.run_moving([args.steps], args.depth)
                rows.append({"case": "immutable-material-upload-capture", "preserve": preserve,
                             "repeat": repeat,
                             "parity": exact(state_of(fixture, data, session, result), expected),
                             "branch_witness": branch_witness(session.metadata(), result)})
                args.progress(rows)
        finally:
            session.close()
    for label in ("new-uniform-weights", "new-nonuniform-weights"):
        changed = list(inputs)
        for i in range(37, 43):
            changed[i] = inputs[i].copy()
        for i in range(37, 40):
            changed[i][...] = (np.float32(.125 * (i - 36)) if label == "new-uniform-weights"
                              else changed[i] * np.float32(1.5))
        expected = ordinary(module, fixture, data, changed, attrs, args.steps)
        for result in moving_repeats(module, fixture, data, changed, attrs, expected,
                                     args, preserve=False):
            rows.append({"case": label, **result})
            args.progress(rows)
    return rows


def block_cases(module, fixture, data, inputs, args):
    attrs = dict(fixture["attributes"], nsteps=args.steps)
    expected = ordinary(module, fixture, data, inputs, attrs, args.steps)
    changed = constant_rank2_e_packet(inputs) if args.constant_blocks else rank2_e_packet(inputs)
    # Both independent decoders must agree before testing the new schedule.
    decoded = ordinary(module, fixture, data, changed, attrs, args.steps)
    encoding_parity = exact(decoded, expected)
    if not encoding_parity["all30_bitwise"]:
        raise RuntimeError("Representation-only conversion changed ordinary physical results")
    rows = []
    for preserve in (False, True):
        for result in moving_repeats(module, fixture, data, changed, attrs, expected,
                                     args, preserve):
            rows.append({"case": "lossless-rank2-packed-E", "representation_only": True,
                         "ordinary_decoder_all30": encoding_parity["all30_bitwise"], **result})
            args.progress(rows)
    return rows


def nonisotropic_cases(module, fixture, data, inputs, args):
    attrs = dict(fixture["attributes"], nsteps=args.steps)
    changed = list(inputs)
    changed[9] = inputs[9].copy()
    changed[9][...] *= np.float32(1.25)
    expected = ordinary(module, fixture, data, changed, attrs, args.steps)
    rows = []
    for preserve in (False, True):
        for result in moving_repeats(module, fixture, data, changed, attrs, expected,
                                     args, preserve):
            result["noio_fallback_correct"] = result["branch_witness"]["certified_noio_tiles"] == 0
            rows.append({"case": "nonisotropic-H-NoIO-fallback", **result})
            args.progress(rows)
    return rows


def clock_cases(module, fixture, data, inputs, args):
    if args.steps <= 6 + args.depth:
        raise ValueError("Clock continuation needs steps>6+depth")
    changed = list(inputs)
    dt = np.float32(fixture["attributes"]["dt"])
    origin = np.float32(float(dt) * 3.125)
    for index, value in [(126, origin), (127, np.int32(5)), (128, np.int32(3))]:
        changed[index] = np.asarray(value)
    attrs = dict(fixture["attributes"], nsteps=args.steps)
    expected = ordinary(module, fixture, data, changed, attrs, args.steps)
    rows = []
    for preserve in (False, True):
        for result in moving_repeats(module, fixture, data, changed, attrs, expected,
                                     args, preserve):
            rows.append({"case": "shifted-origin-actual-macro", **result})
            args.progress(rows)
    session = module.NativeSession(changed, attrs, preserve_inputs=False)
    try:
        checkpoint = session.run_moving([3, 3], args.depth)
        checkpoint_expected = ordinary(module, fixture, data, changed, attrs, 6)
        checkpoint_parity = exact(state_of(fixture, data, session, checkpoint), checkpoint_expected)
        elapsed = int(changed[128]) + 6
        split_time = np.float32(origin + np.float32(dt * np.float32(elapsed)))
        time_exact = (np.float32(session.clocks()["time"]).view(np.uint32)
                      == split_time.view(np.uint32))
        result = session.run_moving([args.depth, args.steps - 6 - args.depth], args.depth)
        rows.append({"case": "elapsed-nine-checkpoint-then-actual-macro-continuation",
                     "checkpoint": checkpoint_parity,
                     "checkpoint_correct": checkpoint_parity["all30_bitwise"] and bool(time_exact),
                     "source_observation_counters_correct": (session.clocks()["current_step"] == 5 + args.steps
                                                              and session.clocks()["elapsed_steps"] == 3 + args.steps),
                     "parity": exact(state_of(fixture, data, session, result), expected),
                     "branch_witness": branch_witness(session.metadata(), result)})
        args.progress(rows)
    finally:
        session.close()
    return rows


def epsi_hit_cases(module, fixture, data, inputs, args):
    attrs = dict(fixture["attributes"], nsteps=args.steps)
    expected = ordinary(module, fixture, data, inputs, attrs, args.steps)
    tail_expected = ordinary(module, fixture, data, inputs, attrs, 1)
    rows = []
    for preserve in (False, True):
        session = module.NativeSession(inputs, attrs, preserve_inputs=preserve)
        try:
            for repeat in range(2):
                if repeat:
                    session.reset()
                result = session.run_moving([args.steps], args.depth)
                metadata = session.metadata()
                hit = (metadata.get("moving_private_epsi_debug_hit") is True
                       and metadata.get("moving_private_epsi_debug_raw") == 8
                       and metadata.get("moving_private_epsi_debug_real_errors") == 0)
                row = {"case": "private-Epsi-retained-hit", "preserve": preserve,
                       "repeat": repeat, "runtime_cache_hit_correct": hit,
                       "metadata": metadata, "branch_witness": branch_witness(metadata, result),
                       "parity": exact(state_of(fixture, data, session, result), expected)}
                if not repeat:
                    session.reset()
                    tail = session.run_moving([1], args.depth)
                    meta = session.metadata()
                    tail_parity = exact(state_of(fixture, data, session, tail), tail_expected)
                    row["tail_after_reset"] = {"metadata": meta, "parity": tail_parity,
                                               "actual_macro_count": tail["moving_macro_count"]}
                    row["tail_clear_correct"] = (not meta.get("moving_private_epsi_debug_hit", True)
                                                  and meta.get("moving_private_epsi_debug_raw") == 0
                                                  and meta.get("moving_private_epsi_debug_real_errors") == 0
                                                  and tail["moving_macro_count"] == 0
                                                  and tail_parity["all30_bitwise"])
                rows.append(row)
                args.progress(rows)
        finally:
            session.close()
    return rows


def flattened_results(rows):
    return [result for row in rows for result in row.get("results", [row])]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("materials", "probes", "blocks", "nonisotropic", "clocks", "epsi-hit"), required=True)
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--case-index", type=int)
    p.add_argument("--require-uniform-branch", action="store_true")
    p.add_argument("--require-noio-branch", action="store_true")
    p.add_argument("--constant-blocks", action="store_true")
    return p


def main():
    args = parser().parse_args()
    report = {"schema": "beamz.plane-retirement-complete-gates.v1",
              "mode": args.mode, "module": str(args.module),
              "module_sha256": file_digest(args.module),
              "fixture": str(args.fixture), "fixture_sha256": file_digest(args.fixture),
              "worker_sha256": file_digest(__file__), "steps": args.steps,
              "depth": args.depth, "oracle": "fresh ordinary CUDA Program",
              "passed": False}
    def progress(rows):
        report["cases"] = rows
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    args.progress = progress
    try:
        if args.steps < args.depth or args.repeats < 1:
            raise ValueError("Positive gates need an actual macro and >=1repeat")
        module = module_at(args.module)
        fixture, data, inputs = load_fixture(args.fixture)
        before = [digest_array(x) for x in inputs]
        worker = {"probes": probe_cases, "materials": material_cases, "blocks": block_cases, "nonisotropic": nonisotropic_cases, "clocks": clock_cases, "epsi-hit": epsi_hit_cases}[args.mode]
        rows = worker(module, fixture, data, inputs, args)
        results = flattened_results(rows)
        report.update(cases=rows, positive_macro=all(r["branch_witness"]["positive_macro"]
                                                   for r in results),
                      positive_uniform_branch=any(r["branch_witness"]["positive_uniform_branch"]
                                                  for r in results),
                      positive_noio_branch=any(r["branch_witness"]["positive_noio_branch"]
                                               for r in results),
                      host_inputs_unchanged=before == [digest_array(x) for x in inputs],
                      jax_imported="jax" in sys.modules, support_status="accepted")
        report["passed"] = (report["positive_macro"] and report["host_inputs_unchanged"]
                            and all(r["parity"]["all30_bitwise"] for r in results)
                            and all(r.get("source_observation_counters_correct", True)
                                    for r in results)
                            and all(r.get("source_and_observation_live", True) for r in rows)
                            and all(r.get("host_inputs_unchanged", True) for r in rows)
                            and all(r.get("runtime_cache_hit_correct", True) for r in results)
                            and all(r.get("tail_clear_correct", True) for r in results)
                            and all(r.get("checkpoint_correct", True) for r in results)
                            and all(r.get("noio_fallback_correct", True) for r in results)
                            and (not args.require_uniform_branch or report["positive_uniform_branch"])
                            and (not args.require_noio_branch or report["positive_noio_branch"]))
    except Exception as error:
        report.update(support_status="rejected-or-failed", error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "support_status": report["support_status"],
                      "positive_uniform_branch": report.get("positive_uniform_branch", False)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
