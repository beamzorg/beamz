#!/usr/bin/env python3
"""Profile unchanged native CUDA graphs with optional CUPTI activity records.

Warm event/wall timings remain separate from the instrumented extra invocation.
Every logical output leaf is checked against the prepared independent reference.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_cuda_native_runtime import chunks_for
from scripts.cuda_fixture_store import (
    ArrayReader,
    compare_arrays,
    digest_array,
    drop_pages,
)
from scripts.export_fdtdx_comparison import file_digest


def summarize(records):
    groups = collections.defaultdict(lambda: {"duration_ns": 0, "count": 0})
    kernels = {}
    dropped = sum(r.get("dropped_records", 0) for r in records)
    events = sorted(
        (r for r in records if r["kind"] == "kernel"), key=lambda r: r["start_ns"]
    )
    for r in events:
        name = r["name"]
        policy_match = re.search(
            r"AsymmetricILi\d+ELi\d+ELb[01]ELb[01]ELi([0-5])E", name
        )
        mode_source_match = re.search(
            r"AsymmetricModeSource54SixILi\d+ELi\d+ELi([3-5])E", name
        )
        unified_source_match = re.search(
            r"Temporal(?:Sources|Effects)ILi\d+ELi\d+ELb[01]ELi([1-5])E", name
        )
        policy_labels = (
            "bulk",
            "NoIO CPML/boundary",
            "interior observation",
            "interior source",
            "interior source+observation",
            "full general fallback",
        )
        specialization = (
            "moving policy 1 NoIO CPML/boundary (uniform)"
            if any(kernel in name for kernel in (
                "AsymmetricNoIOUniform1024", "AsymmetricNoIOFixedCube", "AsymmetricNoIORectCached"))
            else "moving policy 2 interior observation (uniform)"
            if "AsymmetricObserverUniform1024" in name
            else "moving policy " + mode_source_match[1] + " "
                 + policy_labels[int(mode_source_match[1])] + " (traced mode source)"
            if mode_source_match
            else "moving policy " + unified_source_match[1] + " "
                 + policy_labels[int(unified_source_match[1])] + " (shared source/effects)"
            if unified_source_match
            else None
        )
        group = specialization or (
            "moving policy "
            + policy_match[1]
            + " "
            + policy_labels[int(policy_match[1])]
            if policy_match
            else "two-step core"
            if any(
                core in name
                for core in (
                    "MstepPackedColumnCore",
                    "MstepRegisterColumnCore",
                    "MstepColumnCore",
                )
            )
            else "four-step rolling interior"
            if "beamz_rolling_d4" in name
            else "two-step temporal CPML shell"
            if "TemporalCpmlPair" in name
            else "full-edge H/E"
            if any(
                edge in name
                for edge in (
                    "MstepFullEdgePhase",
                    "MstepGroupedFullEdge",
                    "MstepCompactFullEdge",
                )
            )
            else "compact-edge H/E"
            if "MstepEdgePhase" in name
            else "moving source/CPML/observation/boundary kernel"
            if "moving_native" in name and "Asymmetric" in name and "ELb0ELb" in name
            else "moving bulk kernel"
            if "moving_native" in name and "Asymmetric" in name and "ELb1ELb" in name
            else "moving asymmetric complete kernel (unclassified)"
            if "moving_native" in name and "Asymmetric" in name
            else "ownership epoch"
            if "moving_native" in name and "BeginEpoch" in name
            else "DFT"
            if "Dft" in name
            else "sources"
            if "Source" in name
            else "device copies implemented as kernels"
            if name.startswith("memcpy")
            else "clocks"
            if "AdvanceClocks" in name
            else "other"
        )
        duration = r["end_ns"] - r["start_ns"]
        if duration < 0:
            raise ValueError("CUPTI kernel has invalid interval")
        for target in (
            groups[group],
            kernels.setdefault(
                name, {"duration_ns": 0, "count": 0, "resource_example": r}
            ),
        ):
            target["duration_ns"] += duration
            target["count"] += 1
    total = sum(x["duration_ns"] for x in groups.values())
    if not total or dropped:
        raise RuntimeError(
            f"Incomplete activity trace: total={total}, dropped={dropped}"
        )
    union_ns = 0
    merged_end = None
    merged_start = None
    edges = []
    for record in events:
        start, end = record["start_ns"], record["end_ns"]
        if end == start:
            continue
        edges.extend(((start, 1), (end, -1)))
        if merged_end is None:
            merged_start, merged_end = start, end
        elif start <= merged_end:
            merged_end = max(merged_end, end)
        else:
            union_ns += merged_end - merged_start
            merged_start, merged_end = start, end
    if merged_end is not None:
        union_ns += merged_end - merged_start
    active = max_active = 0
    for _, change in sorted(edges):
        active += change
        max_active = max(max_active, active)
    return {
        "scope": "Instrumented summed GPU kernel durations, including graph copies implemented as kernels; excludes DMA copy activities and CPU gaps; not unprofiled wall-time shares",
        "dropped_records": dropped,
        "kernel_total_s": total / 1e9,
        "kernel_interval_union_s": union_ns / 1e9,
        "summed_over_union_concurrency_factor": total / union_ns if union_ns else 0,
        "maximum_simultaneous_kernels": max_active,
        "kernel_union_scope": "Union of instrumented kernel intervals only; excludes DMA operations and gaps. Group shares use summed durations, not critical-path wall shares.",
        "groups": [
            dict(group=k, **v, share=v["duration_ns"] / total)
            for k, v in sorted(groups.items(), key=lambda x: -x[1]["duration_ns"])
        ],
        "kernels": [
            dict(name=k, **v)
            for k, v in sorted(kernels.items(), key=lambda x: -x[1]["duration_ns"])
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--collector", type=Path, required=True)
    p.add_argument(
        "--setting", choices=("fixture", "whole", "seg64"), default="fixture"
    )
    p.add_argument(
        "--method",
        choices=("mstep", "hybrid", "grouped", "rolling4", "moving"),
        default="mstep",
    )
    p.add_argument("--moving-depth", type=int, choices=(4, 16), default=16)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--samples", type=int, default=2)
    a = p.parse_args()
    if importlib.util.find_spec("jax") is not None:
        raise RuntimeError("Use the NumPy-only interpreter")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    f = json.loads(a.fixture.read_text())
    if f.get("data_format") == "npy_directory_v1":
        data = ArrayReader(a.fixture.parent / f["data_directory"])
    else:
        data = np.load(a.fixture.parent / f["data_file"], allow_pickle=False)
    inputs = []
    for item in f["inputs"]:
        value = data[item["key"]]
        if digest_array(value) != item["sha256"]:
            raise ValueError(f"Input hash differs: {item['key']}")
        inputs.append(value)
        drop_pages(value)
    attrs = dict(f["attributes"])
    if a.setting != "fixture":
        attrs["cuda_flags"] &= ~(
            (1 << 21)
            | (1 << 22)
            | (1 << 20)
            | (1 << 23)
            | (1 << 24)
            | (1 << 25)
            | (1 << 26)
        )
        if a.setting == "seg64":
            attrs["cuda_flags"] |= 1 << 24
    spec = importlib.util.spec_from_file_location("_cuda_native", a.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    setup_started = time.perf_counter()
    session = module.NativeSession(inputs, attrs, preserve_inputs=True)
    setup_s = time.perf_counter() - setup_started
    if a.method == "moving":

        def run(chunks):
            return session.run_moving(chunks, depth=a.moving_depth)
    else:
        run = getattr(
            session,
            {
                "hybrid": "run_hybrid",
                "grouped": "run_grouped",
                "rolling4": "run_rolling4",
                "mstep": "run_schedule",
            }[a.method],
        )
    for value in inputs:
        drop_pages(value)
    chunks = chunks_for(attrs["nsteps"], "chunk256")
    try:
        samples = []
        warmups = []
        for i in range(a.warmups + a.samples):
            started = time.perf_counter()
            result = run(chunks)
            wall = time.perf_counter() - started
            if i >= a.warmups:
                samples.append({"wall_s": wall, **result})
            else:
                warmups.append({"wall_s": wall, **result})
        library = ctypes.CDLL(str(a.collector.resolve()))
        library.beamz_profile_start.argtypes = [ctypes.c_char_p]
        library.beamz_profile_error.restype = ctypes.c_char_p
        trace = a.output.with_suffix(".activity.jsonl")
        error = library.beamz_profile_start(str(trace.resolve()).encode())
        if error:
            raise RuntimeError(f"CUPTI start{error}: {library.beamz_profile_error()}")
        started = time.perf_counter()
        profile_result = run(chunks)
        profile_wall = time.perf_counter() - started
        error = library.beamz_profile_stop()
        if error:
            raise RuntimeError(f"CUPTI stop{error}: {library.beamz_profile_error()}")
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        clocks = session.clocks()
        leaves = []
        for index, leaf in enumerate(f["state_leaves"]):
            m = leaf["mapping"]
            raw = None
            if m["kind"] == "field":
                raw = (18 if profile_result["output_bank"] else 0) + m["component"]
            elif m["kind"] == "output":
                raw = m["index"]
            expected = data[leaf["reference_key"]]
            temporary = a.output.with_suffix(f".leaf-{index}.npy")
            if raw is not None:
                value = np.lib.format.open_memmap(
                    temporary,
                    mode="w+",
                    dtype=np.dtype(leaf["dtype"]),
                    shape=tuple(leaf["shape"]),
                )
                session.download_into(raw, value)
            elif m["kind"] == "clock":
                value = np.asarray(
                    clocks["time" if m["name"] == "t" else "current_step"],
                    dtype=leaf["dtype"],
                )
            else:
                value = data[leaf["initial_key"]]
            row = {"name": leaf["name"], **compare_arrays(value, expected)}
            leaves.append(row)
            drop_pages(expected)
            if raw is not None:
                value.flush()
                drop_pages(value)
                del value
                if row["passed"]:
                    temporary.unlink()
        report = {
            "schema": "beamz.native-phase-profile.v1",
            "fixture": str(a.fixture),
            "fixture_sha256": file_digest(a.fixture),
            "module": str(a.module),
            "module_sha256": file_digest(a.module),
            "worker_sha256": file_digest(__file__),
            "collector_sha256": file_digest(a.collector),
            "setting": a.setting,
            "method": a.method,
            "moving_depth": a.moving_depth if a.method == "moving" else None,
            "attributes": attrs,
            "ownership": "preserve",
            "session_setup_s": setup_s,
            "setup_scope": "NativeSession construction, private persistent host copies, device allocation and prepared-input upload; excludes fixture read/hash audits",
            "warmup_calls": warmups,
            "chunks": chunks,
            "warm_samples": samples,
            "median_wall_s": float(np.median([x["wall_s"] for x in samples])),
            "profile_wall_s": profile_wall,
            "profile_timing": profile_result,
            "session_metadata": session.metadata(),
            "parity": {
                "passed": all(x["passed"] for x in leaves),
                "all30_bitwise": all(x["bitwise_exact"] for x in leaves),
                "leaves": leaves,
            },
            "activity": summarize(records),
            "trace": str(trace),
        }
        report["host_inputs_unchanged"] = all(
            digest_array(value) == item["sha256"]
            for value, item in zip(inputs, f["inputs"], strict=True)
        )
        report["status"] = (
            "ok"
            if report["parity"]["all30_bitwise"] and report["host_inputs_unchanged"]
            else "failed_parity_or_input_mutation"
        )
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in ("setting", "median_wall_s", "parity", "activity")
                },
                indent=2,
            )
        )
        return 0 if report["status"] == "ok" else 1
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
