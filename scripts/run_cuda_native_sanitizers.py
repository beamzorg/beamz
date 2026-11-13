#!/usr/bin/env python3
"""Bounded native-only sanitizer runs with nonvacuous branch/parity witnesses.

Preparing commands is CPU-only. Execution requires the coordinator's GPU grant.
Timeouts are incomplete outcomes, never clean passes. Each tool uses the same
fresh ordinary Program reference and point-source/direct-observer packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA = Path("/home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root/opt/cuda")
PYTHON = ROOT / ".cache/cuda-native-autoresearch-2026-11-11-1607/native-only/bin/python"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terminate(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--steps", type=int, default=7)
    p.add_argument("--case-index", type=int, default=1)
    p.add_argument("--timeout", type=float, default=90)
    p.add_argument("--total-budget", type=float, default=240)
    p.add_argument("--branch", choices=("macro", "uniform", "noio"), default="uniform")
    p.add_argument("--tools", nargs="+", choices=("memcheck", "synccheck", "racecheck"),
                   default=["memcheck", "synccheck", "racecheck"])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    worker = args.output / "worker.py"
    source = ROOT / "scripts/validate_cuda_plane_retirement.py"
    worker.write_text(source.read_text().replace(
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))",
        f"sys.path.insert(0, {str(ROOT)!r})"))
    dependencies = {}
    snapshots = args.output / "dependencies"
    snapshots.mkdir(exist_ok=True)
    for name in ["cuda_native_probe_cases.py", "cuda_native_block_cases.py",
                 "cuda_native_const_block_cases.py", "validate_cuda_native_moving.py",
                 "benchmark_cuda_native_runtime.py", "cuda_fixture_store.py"]:
        path = ROOT / "scripts" / name
        dependencies[str(path)] = sha(path)
        (snapshots / name).write_bytes(path.read_bytes())
    planned = []
    for tool in args.tools:
        output = args.output / (tool + ".json")
        cmd = [str(CUDA / "compute-sanitizer/compute-sanitizer"), "--tool", tool,
               "--error-exitcode", "86", "--kernel-name", "kns=Asymmetric", str(PYTHON),
               str(worker.resolve()), "--module", str(args.module.resolve()),
               "--fixture", str(args.fixture.resolve()), "--mode", "probes",
               "--steps", str(args.steps), "--depth", str(args.depth),
               "--case-index", str(args.case_index), "--repeats", "2",
               "--output", str(output.resolve())]
        if args.branch != "macro":
            cmd += ["--require-" + ("noio" if args.branch == "noio" else "uniform") + "-branch"]
        planned.append({"tool": tool, "command": cmd, "helper_output": str(output)})
    report = {"schema": "beamz.bounded-native-sanitizers.v1", "module_sha256": sha(args.module),
              "fixture_sha256": sha(args.fixture), "worker_sha256": sha(worker),
              "scope": "Asymmetric kernels only; fresh ordinary Program oracle; seeded physical fields/CPML; two coincident point sources and directDFT; reset/repeat all30",
              "dependencies": dependencies,
              "required_branch": args.branch, "planned": planned, "rows": []}
    (args.output / "plan.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps({"GPU_used": False, "plan": str(args.output / "plan.json")}))
        return 0
    started = time.monotonic()
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    for row in planned:
        remaining = args.total_budget - (time.monotonic() - started)
        timeout = min(args.timeout, remaining)
        if timeout <= 0:
            report["rows"].append({"tool": row["tool"], "status": "not-run-budget-exhausted"})
            continue
        if any(sha(Path(path)) != value for path, value in dependencies.items()):
            raise RuntimeError("Frozen helper dependency changed during sanitizer campaign")
        if sha(args.module) != report["module_sha256"] or sha(worker) != report["worker_sha256"]:
            raise RuntimeError("Frozen module/worker changed during sanitizer campaign")
        logpath = args.output / (row["tool"] + ".log")
        begin = time.monotonic()
        status = "completed"
        with logpath.open("w") as log:
            process = subprocess.Popen(row["command"], cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                status = "timeout-incomplete"
                terminate(process)
        text = logpath.read_text()
        helper_path = Path(row["helper_output"])
        helper = json.loads(helper_path.read_text()) if helper_path.exists() else {}
        zero = bool(re.search(r"(?:ERROR SUMMARY:\s*0 errors|RACECHECK SUMMARY:\s*0 hazards)", text))
        clean = (status == "completed" and process.returncode == 0 and zero
                 and helper.get("passed", False) and helper.get("positive_macro", False))
        if args.branch != "macro":
            clean = clean and helper.get("positive_" + args.branch + "_branch", False)
        report["rows"].append({"tool": row["tool"], "status": status,
                               "returncode": process.returncode, "wall_s": time.monotonic()-begin,
                               "timeout_s": timeout, "explicit_zero_summary": zero,
                               "positive_macro": helper.get("positive_macro", False),
                               "positive_uniform_branch": helper.get("positive_uniform_branch", False),
                               "positive_noio_branch": helper.get("positive_noio_branch", False),
                               "all30_reset_repeat_passed": helper.get("passed", False),
                               "log": str(logpath), "clean_in_scope": bool(clean)})
        report["all_requested_tools_clean"] = all(r.get("clean_in_scope", False) for r in report["rows"])
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["rows"][-1]), flush=True)
    report["all_requested_tools_clean"] = (len(report["rows"]) == len(planned)
                                            and all(r.get("clean_in_scope", False) for r in report["rows"]))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["all_requested_tools_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
