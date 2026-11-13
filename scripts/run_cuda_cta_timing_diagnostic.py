#!/usr/bin/env python3
"""NumPy-only complete-state gate and last-macro CTA diagnostic capture.

DIAGNOSTIC ONLY: no normal benchmark or speedup metric is emitted. Timerstores
and terminal barriers perturb execution. Fresh and graph-replay invocations
must both match all30 independent fixture leaves before interval analysis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_cuda_cta_intervals import summarize
from scripts.benchmark_cuda_large_runtime import exact_admission
from scripts.benchmark_cuda_native_runtime import chunks_for
from scripts.cuda_fixture_store import ArrayReader, compare_arrays, digest_array, drop_pages
from scripts.export_fdtdx_comparison import file_digest


def validate_state(session, result, fixture, store, directory):
    directory.mkdir(parents=True, exist_ok=True)
    clocks, rows = session.clocks(), []
    for index, leaf in enumerate(fixture["state_leaves"]):
        mapping = leaf["mapping"]
        raw = ((18 if result["output_bank"] else 0) + mapping["component"] if mapping["kind"] == "field"
               else mapping["index"] if mapping["kind"] == "output" else None)
        expected = store[leaf["reference_key"]]
        path = directory / f"leaf-{index:02d}.npy"
        if raw is not None:
            actual = np.lib.format.open_memmap(path, mode="w+", dtype=np.dtype(leaf["dtype"]), shape=tuple(leaf["shape"]))
            session.download_into(raw, actual)
        elif mapping["kind"] == "clock":
            actual = np.asarray(clocks["time" if mapping["name"] == "t" else "current_step"], dtype=np.dtype(leaf["dtype"]))
        else:
            actual = store[leaf["initial_key"]]
        row = compare_arrays(actual, expected)
        row["name"] = leaf["name"]
        rows.append(row)
        if raw is not None:
            actual.flush();drop_pages(actual)
            if row["passed"] and row["bitwise_exact"]:
                del actual
                path.unlink()
            else:
                row["failed_readback"] = str(path)
        drop_pages(expected)
    return {"passed": len(rows) == 30 and all(r["passed"] and r["bitwise_exact"] for r in rows), "leaves": rows, "clocks": clocks}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--expected-module-sha256", required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--moving-depth", type=int, choices=(4, 16), default=16)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--ownership", choices=("preserve", "owned"), default="preserve")
    a = p.parse_args()
    a.backend, a.method = "native", "moving"
    if a.repeats < 1:
        p.error("At least one diagnostic invocation required")
    if importlib.util.find_spec("jax") is not None:
        p.error("Use the NumPy-only environment; JAX must be absent")
    if file_digest(a.module) != a.expected_module_sha256:
        raise ValueError("Diagnostic binary differs from the explicit expected hash")
    f = json.loads(a.fixture.read_text())
    if f.get("schema") != "beamz.cuda-native-fixture.v1" or f.get("data_format") != "npy_directory_v1":
        raise ValueError("Complete immutable mmap-backed native fixture required")
    if file_digest(a.fixture.parent / f["data_file"]) != f["data_sha256"]:
        raise ValueError("Independent fixture manifest differs")
    store = ArrayReader(a.fixture.parent / f["data_directory"])
    inputs = []
    for item in f["inputs"]:
        store.verify(item["key"]);inputs.append(store[item["key"]])
    for leaf in f["state_leaves"]:
        store.verify(leaf["reference_key"])
    spec = importlib.util.spec_from_file_location("_cuda_native", a.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not getattr(module, "moving_cta_timing_diagnostic_only", False):
        raise ValueError("Refusing a module without the explicit diagnostic-only marker")
    chunks = chunks_for(f["attributes"]["nsteps"], "chunk256")
    if not any(count >= a.moving_depth for count in chunks):
        raise ValueError("A positive complete moving macro is required")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "beamz.complete-state-CTA-diagnostic.v1", "diagnostic_only": True,
              "not_a_performance_candidate": True, "status": "incomplete",
              "module": str(a.module.resolve()), "module_sha256": file_digest(a.module),
              "fixture": str(a.fixture.resolve()), "fixture_sha256": file_digest(a.fixture),
              "reference_component_sha256": f.get("native_component_sha256"),
              "worker_sha256": file_digest(__file__), "chunks": chunks, "ownership": a.ownership,
              "moving_depth": a.moving_depth, "runs": []}
    session = None
    try:
        report["admission"] = exact_admission(f, store, a, module)
        session = module.NativeSession(inputs, f["attributes"], preserve_inputs=a.ownership == "preserve")
        for value in inputs:
            drop_pages(value)
        physical = tuple(min(int(v.shape[axis]) for v in inputs[:6]) for axis in range(3))
        for repeat in range(a.repeats):
            session.reset()
            result = session.run_moving(chunks, a.moving_depth)
            state = validate_state(session, result, f, store, a.output.parent / (a.output.stem + f".readback-{repeat}"))
            if not state["passed"]:
                report["runs"].append({"repeat": repeat, "state": state})
                raise AssertionError("Diagnostic instrumentation changed independent all30 physics")
            diagnostic = session.moving_tile_diagnostics()
            arrays = {k: v for k, v in diagnostic.items() if isinstance(v, np.ndarray)}
            metadata = {k: v for k, v in diagnostic.items() if k not in arrays}
            arrays_path = a.output.with_name(a.output.stem + f"-run{repeat}.npz")
            metadata_path = a.output.with_name(a.output.stem + f"-run{repeat}-metadata.json")
            np.savez(arrays_path, **arrays)
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            analysis = summarize(arrays, metadata, physical)
            analysis_path = a.output.with_name(a.output.stem + f"-run{repeat}-analysis.json")
            analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
            # Deliberately omit instrumented wall/CUDA timing as benchmark data.
            execution = {k: result[k] for k in ("nsteps", "moving_macro_count", "ordinary_tail_steps", "output_bank", "moving_depth")}
            row = {"repeat": repeat, "state": state, "execution_counters": execution,
                   "arrays": str(arrays_path), "arrays_sha256": file_digest(arrays_path),
                   "metadata": str(metadata_path), "metadata_sha256": file_digest(metadata_path),
                   "analysis": str(analysis_path), "analysis_sha256": file_digest(analysis_path),
                   "rank_order_violations": analysis["rank_order_violations"],
                   "dependency_timestamp_violations": analysis["conservative_dependency_timestamp_violations"]}
            report["runs"].append(row)
            if row["rank_order_violations"] or row["dependency_timestamp_violations"]:
                raise AssertionError("Recorded intervals violate actual rank/slot-dependency ordering")
        report["metadata"] = session.metadata()
        report["host_inputs_unchanged"] = all(digest_array(v) == item["sha256"] for v, item in zip(inputs, f["inputs"], strict=True))
        if not report["host_inputs_unchanged"]:
            raise AssertionError("Diagnostic invocation changed caller inputs")
        report["status"] = "ok_diagnostic_only"
    except Exception as error:
        report.update(status="error", error=str(error), traceback=traceback.format_exc())
    finally:
        if session is not None:
            session.close()
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "independent_all30_runs": len(report["runs"]), "output": str(a.output)}))
    return 0 if report["status"] == "ok_diagnostic_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
