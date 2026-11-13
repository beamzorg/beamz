#!/usr/bin/env python3
"""Profile one extra warmed CUDA call, outside benchmark timing samples.

Both fixtures use twelve CPML cells per face. The summary groups instrumented
GPU kernel durations; it does not estimate unprofiled wall-time shares.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def summarize_trace(path, steps=None):
    with gzip.open(path, "rt") as stream:
        payload = json.load(stream)
    groups = collections.defaultdict(lambda: {"duration_us": 0.0, "count": 0})
    kernels = collections.defaultdict(lambda: {"duration_us": 0.0, "count": 0})
    events = sorted(
        (
            e
            for e in payload.get("traceEvents", ())
            if e.get("ph") == "X" and "kernel_details" in e.get("args", {})
        ),
        key=lambda e: e.get("ts", 0),
    )
    edges = [
        e
        for e in events
        if "MstepEdgePhase" in e["name"] or "MstepFullEdgePhase" in e["name"]
    ]
    edge_roles = {}
    if steps and steps % 2 == 0 and len(edges) == 2 * steps:
        phases = [
            re.search(r"Mstep(?:Full)?EdgePhaseILi([01])E", e["name"]) for e in edges
        ]
        if all(p and int(p.group(1)) == i % 2 for i, p in enumerate(phases)):
            edge_roles = {
                id(e): "first" if i % 4 < 2 else "second" for i, e in enumerate(edges)
            }
    for event in events:
        arguments = event.get("args", {})
        if event.get("ph") != "X" or "kernel_details" not in arguments:
            continue
        name = event["name"]
        full = re.search(
            r"MstepFullEdgePhaseILi([01])ELb([01])ELb([01])ELb([01])ELb([01])E", name
        )
        role = edge_roles.get(id(event))
        group = (
            "first-step full edges and core copies"
            if full and full.group(5) == "1"
            else "first-step full edges"
            if full and role == "first"
            else "second-step full edges"
            if full and role == "second"
            else "full edges (stage unassigned)"
            if full
            else "full edges (unparsed template)"
            if "MstepFullEdgePhase" in name
            else f"{role}-step compact edges"
            if role and "MstepEdgePhase" in name
            else "edge phases (including CPML and scratch)"
            if "MstepEdgePhase" in name
            else "two-step column core"
            if "Mstep" in name and "Core" in name
            else "DFT"
            if "Dft" in name
            else "source injection"
            if "SourceGroup" in name
            else "other kernels"
        )
        duration = event["dur"]
        for item in (groups[group], kernels[name]):
            item["duration_us"] += duration
            item["count"] += 1
        kernels[name]["details"] = arguments["kernel_details"]
    total = sum(group["duration_us"] for group in groups.values())
    if not kernels or total <= 0:
        raise ValueError("No GPU kernel events found; check CUPTI profiling support")
    return {
        "interpretation": "Summed profiled GPU kernel durations; excludes CPU gaps and copies; instrumented, not unprofiled wall-time shares",
        "compact_first_second_assignment": "Verified even-step H/E/H/E launch order"
        if edge_roles and any("MstepEdgePhase" in e["name"] for e in edges)
        else "Not assigned",
        "edge_first_second_assignment": "Verified even-step H/E/H/E launch order"
        if edge_roles
        else "Not assigned",
        "total_kernel_s": total / 1e6,
        "groups": [
            dict(group=name, **item, percent=100 * item["duration_us"] / total)
            for name, item in sorted(groups.items(), key=lambda x: -x[1]["duration_us"])
        ],
        "top_kernels": [
            dict(name=name, **item)
            for name, item in sorted(
                kernels.items(), key=lambda x: -x[1]["duration_us"]
            )[:20]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=("mie", "realistic"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=3, default=(256, 512, 512))
    parser.add_argument("--resolution-nm", type=float, default=5)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--backend", choices=("cuda_streamed", "cuda_mstep"), default="cuda_mstep"
    )
    parser.add_argument("--expected-binary-sha", required=True)
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import beamz._cuda as extension
    import jax

    from beamz.simulation.execute import build_scan, initial_program_state

    digest = hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()
    if digest != args.expected_binary_sha:
        raise RuntimeError(
            "Unexpected native binary; refusing ambiguous profile provenance"
        )
    if args.fixture == "mie":
        from scripts.benchmark_cuda_mie import build_simulation, case_spec
        from scripts.benchmark_cuda_mie import parser as mie_parser

        options = [
            "--output",
            "/tmp/profile-unused.json",
            "--resolutions-nm",
            str(args.resolution_nm),
        ]
        if args.steps is not None:
            options += ["--steps", str(args.steps)]
        settings = mie_parser().parse_args(options)
        spec = case_spec(settings, args.resolution_nm)
        sim = build_simulation(settings, spec)
        steps = spec["timesteps"]
        fixture = spec
    else:
        from scripts.benchmark_cuda_realistic import build_simulation

        steps = args.steps or 256
        settings = SimpleNamespace(
            shape=args.shape,
            steps=steps,
            pml=12,
            monitors=2,
            frequencies=3,
            material="binary",
            source="mode",
            monitor_type="mode",
            boundary="all",
        )
        sim = build_simulation(settings)
        fixture = vars(settings)
    program = sim.compile(num_steps=steps, backend=args.backend)
    state = initial_program_state(program, t=0, current_step=0, monitor_steps=steps)
    jax.block_until_ready(state)
    scan = build_scan(program)
    executable = scan.lower(state, program.coefficients).compile()
    for _ in range(2):
        jax.block_until_ready(executable(state, program.coefficients))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with jax.profiler.trace(str(args.output_dir), create_perfetto_trace=True):
        result = executable(state, program.coefficients)
        jax.block_until_ready(result)
    traces = list(args.output_dir.glob("plugins/profile/*/perfetto_trace.json.gz"))
    if len(traces) != 1:
        raise RuntimeError("Expected one exported GPU Perfetto trace")
    report = {
        "fixture": fixture,
        "backend": args.backend,
        "binary_sha": digest,
        "profiler_sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "launch_environment": {
            name: value
            for name, value in os.environ.items()
            if name.startswith("BEAMZ_CUDA_")
        },
        "schedule": getattr(scan, "cuda_schedule_decision", None),
        "profile_and_export_wall_s": time.perf_counter() - start,
        "trace": str(traces[0]),
        **summarize_trace(traces[0], steps=steps),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
