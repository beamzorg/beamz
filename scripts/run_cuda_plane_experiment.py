#!/usr/bin/env python3
"""Freeze and orchestrate complete native retirement candidates; no implicit GPU use.

`prepare` only writes a source/provenance/command plan. `validate` and `benchmark`
execute it and therefore require the coordinator's GPU grant. Every benchmark
uses the existing temporal v13 control, identical physical data and ownership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AREA = ROOT / ".cache/cuda-plane-retirement-2026-11-12"
NATIVE_PYTHON = ROOT / ".cache/cuda-native-autoresearch-2026-11-11-1607/native-only/bin/python"
BASELINE = ROOT / ".cache/cuda-engine-autoresearch-2026-11-11-2032/moving-fast-uniform-v13/build/_cuda_native.cpython-311-x86_64-linux-gnu.so"
BASELINE_SHA = "c2240384e6e3621e30c0d531333e39b4bd60704143cd36736d3f2de47d67b6c4"
FIXTURES = ROOT / ".cache/cuda-native-autoresearch-2026-11-11-1607/fixtures"
CLOCK = ROOT / ".cache/cuda-native-autoresearch-2026-11-11-1607/nonzero-origin-clockfix/durable-fixture.json"
COMPARE = ROOT / ".cache/main-vs-temporal-2026-11-12"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def freeze_worker(source, target):
    text = source.read_text().replace(
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))",
        f"sys.path.insert(0, {str(ROOT)!r})",
    ).replace("ROOT = Path(__file__).resolve().parents[1]", f"ROOT = Path({str(ROOT)!r})")
    target.write_text(text)
    compile(text, str(target), "exec")
    return {"source": str(source), "source_sha256": digest(source),
            "frozen": str(target), "frozen_sha256": digest(target)}


def prepare(args):
    control_sha = args.expected_control_sha256
    if control_sha is None:
        if args.baseline.resolve() != BASELINE.resolve():
            raise ValueError("Custom baseline needs --expected-control-sha256")
        control_sha = BASELINE_SHA
    if digest(args.baseline.resolve()) != control_sha:
        raise ValueError("Baseline binary differs from the explicit control contract")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    workers = out / "workers"
    workers.mkdir(exist_ok=True)
    scripts = ["validate_cuda_native_moving.py", "validate_cuda_native_clocks.py",
               "validate_cuda_plane_retirement.py"]
    frozen = [freeze_worker(ROOT / "scripts" / name, workers / name) for name in scripts]
    # Use exactly the already-verified native worker from the latest full comparison.
    native_source = COMPARE / "native_worker.py"
    shutil.copy2(native_source, workers / "benchmark_native.py")
    worker = workers / "benchmark_native.py"
    text = worker.read_text()
    anchor = '"module_sha256": file_digest(args.module),'
    if text.count(anchor) != 1:
        raise ValueError("Unexpected module-report identity anchor")
    text = text.replace(anchor, '"module": str(args.module.resolve()),\n            ' + anchor)
    # Bounded macro engines keep the unused Mstep workspace on the host.
    # Preserve exact allocation admission rather than counting those ABI
    # inputs as resident GPU allocations or weakening the equality check.
    allocation = '    payload = inputs + (persistent if args.ownership == "preserve" else 0) + 4'
    if text.count(allocation) != 1:
        raise ValueError("Unexpected frozen exact-allocation admission anchor")
    text = text.replace(allocation, allocation + '''
    deferred_scratch = 0
    if args.method == "moving" and getattr(module, "native_lazy_mstep_scratch", False):
        deferred_scratch = sum(sizes[i] for i in range(80, 86))
        payload -= deferred_scratch''')
    text = text.replace('        "input_payload_bytes": inputs,',
                        '        "input_payload_bytes": inputs,\n        "deferred_scratch_bytes": deferred_scratch,')
    worker.write_text(text)
    if args.depth == 8:
        worker = workers / "benchmark_native.py"
        text = worker.read_text()
        if text.count("choices=[4, 16]") != 1:
            raise ValueError("Unexpected frozen depth-choice anchor")
        worker.write_text(text.replace("choices=[4, 16]", "choices=[4, 8, 16]"))
    frozen.append({"source": str(native_source), "source_sha256": digest(native_source),
                   "frozen": str(workers / "benchmark_native.py"),
                   "frozen_sha256": digest(workers / "benchmark_native.py")})
    frozen.append(freeze_worker(ROOT / "scripts/cuda_research_compare.py",
                                workers / "cuda_research_compare.py"))
    dependencies = [ROOT / "scripts" / n for n in [
        "cuda_native_probe_cases.py", "cuda_native_block_cases.py",
        "cuda_native_const_block_cases.py", "cuda_fixture_store.py",
        "benchmark_cuda_native_runtime.py", "export_fdtdx_comparison.py",
        "export_cuda_large_fixture.py", "export_cuda_native_fixture.py",
        "validate_cuda_native_moving.py"]]
    cmds = []
    for name, depth, steps in [("small-mie-seeded", 4, None),
                               ("small-rect-seeded", 4, None),
                               ("small-mie-seeded", args.depth, 35), ("mie259", args.depth, None)]:
        label = f"{name}-depth{depth}-steps{steps or 'fixture'}"
        cmd = [str(NATIVE_PYTHON), str(workers / scripts[0]), "--module", "{candidate}",
               "--fixture", str(FIXTURES / (name + ".json")), "--depth", str(depth),
               "--oracle-kind", "program", "--output", str(out / (label + ".json"))]
        if steps:
            cmd += ["--steps", str(steps)]
        cmds.append({"label": label, "command": cmd, "output": str(out / (label + ".json"))})
    cmds.append({"label": "clock13", "command": [str(NATIVE_PYTHON), str(workers / scripts[1]),
                 "--mode", "native", "--fixture", str(CLOCK), "--module", "{candidate}",
                 "--method", "moving", "--output", str(out / "clock13.json")],
                 "output": str(out / "clock13.json")})
    for mode in ["materials", "probes"]:
        cmd = [str(NATIVE_PYTHON), str(workers / scripts[2]), "--module", "{candidate}",
               "--fixture", str(FIXTURES / "small-mie-seeded.json"), "--mode", mode,
               "--depth", str(args.depth), "--steps", "35", "--output", str(out / (mode + ".json"))]
        if mode == "materials":
            cmd += ["--require-uniform-branch"]
            if args.require_noio:
                cmd += ["--require-noio-branch"]
        cmds.append({"label": mode, "command": cmd, "output": str(out / (mode + ".json"))})
    plan = {"schema": "beamz.plane-retirement-campaign.v1", "GPU_grant_required_for_execution": True,
            "baseline": str(args.baseline.resolve()), "baseline_sha256": control_sha,
            "expected_control_sha256": control_sha,
            "control_contract": "Explicit parent/strong-control module SHA, independent of candidate naming",
            "comparator": str(workers / "cuda_research_compare.py"),
            "workers": frozen,
            "dependencies": {str(p): digest(p) for p in dependencies},
            "fixture_prefix": str(COMPARE / "fixture-prefix-stock-default.json"),
            "fixture_full": str(COMPARE / "fixture-full-stock-default.json"),
            "validation": cmds,
            "benchmark": {"order": ["baseline", "candidate", "candidate", "baseline"],
                          "worker": str(workers / "benchmark_native.py"), "method": "moving",
                          "depth": args.depth, "baseline_depth": 16, "ownership": "preserve", "chunk": 256,
                          "warmups": 1, "samples": 3},
            "scope": "Same complete physics and raw30leaf reference. Fresh-process repeat units; no CI from within-process samples."}
    plan["fixture_hashes"] = {plan[key]: digest(Path(plan[key])) for key in ["fixture_prefix", "fixture_full"]}
    snapshots = workers / "dependencies"
    snapshots.mkdir(exist_ok=True)
    plan["dependency_snapshots"] = {}
    for source in plan["dependencies"]:
        target = snapshots / Path(source).name
        shutil.copy2(source, target)
        plan["dependency_snapshots"][source] = str(target)
    save(out / "plan.json", plan)
    return plan


def verify(plan):
    control_sha = plan.get("expected_control_sha256")
    if not control_sha or control_sha != plan["baseline_sha256"]:
        raise ValueError("Missing or inconsistent explicit control contract")
    if digest(Path(plan["baseline"])) != control_sha:
        raise ValueError("Baseline binary differs from the explicit control contract")
    expected = dict(plan["dependencies"])
    expected.update(plan["fixture_hashes"])
    expected[plan["baseline"]] = plan["baseline_sha256"]
    for worker in plan["workers"]:
        expected[worker["frozen"]] = worker["frozen_sha256"]
    for path, sha in expected.items():
        if digest(Path(path)) != sha:
            raise ValueError(f"Frozen dependency changed: {path}")


def execute(cmd, label, out, expected_output):
    started = time.monotonic()
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    with (out / (label + ".log")).open("w") as log:
        process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=1800)
    record = {"label": label, "command": cmd, "returncode": process.returncode,
              "wall_s": time.monotonic() - started, "output": str(expected_output)}
    if not process.returncode:
        result = json.loads(Path(expected_output).read_text())
        module_path = Path(cmd[cmd.index("--module") + 1])
        module_matches = result.get("module_sha256") == digest(module_path)
        record["reported_module_matches"] = module_matches
        record["report_sha256"] = digest(Path(expected_output))
        record["passed"] = module_matches and result.get("passed", result.get("status") == "ok")
    else:
        record["passed"] = False
        record["failure_log_tail"] = (out / (label + ".log")).read_text()[-16000:]
    return record


def run(args, plan):
    out = args.output.resolve()
    candidate_sha = digest(args.candidate)
    records = []
    validation = args.stage == "validate"
    if not validation and not plan.get("comparator"):
        raise ValueError("Historical plan needs a frozen guarded comparator before rerun")
    commands = plan["validation"] if validation else []
    if not validation:
        settings = plan["benchmark"]
        order = settings["order"][:2] if args.screen else settings["order"]
        for i, variant in enumerate(order):
            module = plan["baseline"] if variant == "baseline" else str(args.candidate.resolve())
            label = f"prefix-{i}-{variant}"
            output = out / (label + ".json")
            command = [str(NATIVE_PYTHON), settings["worker"], "--fixture", plan["fixture_prefix"],
                       "--backend", "native", "--module", module, "--method", "moving",
                       "--moving-depth", str(settings.get("baseline_depth", 16) if variant == "baseline"
                                              else settings["depth"]), "--ownership", "preserve",
                       "--warmups", str(args.warmups),
                       "--samples", str(args.samples), "--output", str(output)]
            commands.append({"label": label, "command": command, "output": str(output)})
    for row in commands:
        verify(plan)
        if digest(args.candidate) != candidate_sha:
            raise ValueError("Candidate binary changed during campaign")
        cmd = [value.replace("{candidate}", str(args.candidate.resolve())) for value in row["command"]]
        record = execute(cmd, row["label"], out, row["output"])
        record["expected_control_sha256"] = plan["expected_control_sha256"]
        records.append(record)
        save(out / (args.stage + "-processes.json"), {"module_sha256": candidate_sha,
             "expected_control_sha256": plan["expected_control_sha256"],
             "records": records, "completed_all_commands": len(records) == len(commands),
             "passed": len(records) == len(commands) and all(r["passed"] for r in records)})
        print(json.dumps({"case": row["label"], "passed": record["passed"]}), flush=True)
        if not record["passed"]:
            return 1
    if not validation:
        controls = [r["output"] for r in records if r["label"].endswith("-baseline")]
        candidates = [r["output"] for r in records if r["label"].endswith("-candidate")]
        subprocess.run([str(NATIVE_PYTHON), plan["comparator"],
                        "--baseline", *controls, "--candidate", *candidates,
                        "--expected-control-sha256", plan["expected_control_sha256"],
                        "--output", str(out / "prefix-comparison.json")], check=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "validate", "benchmark"), required=True)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--baseline", type=Path, default=BASELINE)
    p.add_argument("--expected-control-sha256")
    p.add_argument("--output", type=Path, default=AREA / "candidate")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--depth", type=int, choices=(4, 8, 16), default=16)
    p.add_argument("--require-noio", action="store_true")
    p.add_argument("--screen", action="store_true", help="One fresh parent/candidate pair; no CI")
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args)
        print(json.dumps({"plan": str(args.output / "plan.json"), "GPU_used": False}))
        return 0
    if not args.candidate or args.samples < 1 or args.warmups < 0:
        p.error("Execution needs --candidate, positive samples and nonnegative warmups")
    plan = json.loads((args.output / "plan.json").read_text())
    return run(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
