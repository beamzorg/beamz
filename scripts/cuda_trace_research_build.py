"""Build a staged immutable trace experiment without replacing installed CUDA.

The moving translation unit must retain non-RDC compilation and ordinary core
and runtime RDC. The default gate requires zero moving stack/local/spills;
explicit exploratory budgets apply only to PersistentTileStep stack/spills.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / ".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(candidate):
    candidate = candidate.resolve()
    source = candidate / "source"
    if not (source / "src/asymmetric_native.cuh").is_file():
        raise ValueError("A separately staged source tree is required")
    if (candidate / "build").exists():
        raise ValueError("Build directory must be new")
    cmd = json.loads((TEMPLATE / "configure-command.json").read_text())
    cmd[cmd.index("-S") + 1] = str(source)
    cmd[cmd.index("-B") + 1] = str(candidate / "build")
    (candidate / "configure-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    (candidate / "build.py").write_bytes((TEMPLATE / "build.py").read_bytes())
    record = {
        "schema": "beamz.trace-experiment-source.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {str(p.relative_to(source)): digest(p)
                          for p in sorted(source.rglob("*")) if p.is_file()},
        "configure_command": cmd,
    }
    (candidate / "staged-source.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def build(candidate, expected, budget=0):
    record = prepare(candidate)
    candidate = candidate.resolve()
    r = subprocess.run([sys.executable, str(candidate / "build.py")], cwd=ROOT)
    if r.returncode:
        return r.returncode
    for name, sha in record["source_hashes"].items():
        if digest(candidate / "source" / name) != sha:
            raise ValueError("Staged source changed during build: " + name)
    return finalize(candidate, expected, budget)


def finalize(candidate, expected, budget=0):
    candidate = candidate.resolve()
    sys.path.insert(0, str(ROOT))
    from scripts.check_cuda_moving_build import is_moving_kernel
    module = candidate / "build/_cuda_native.cpython-311-x86_64-linux-gnu.so"
    subprocess.run([sys.executable, str(ROOT / "scripts/check_cuda_moving_build.py"),
                    "--build", str(candidate / "build"), "--module", str(module),
                    "--cuobjdump", "/home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root/opt/cuda/bin/cuobjdump",
                    "--expected-kernels", str(expected), "--persistent-stack-budget", str(budget),
                    "--output", str(candidate / ("build-contract.json" if not budget else "exploratory-build-contract.json"))], check=True)
    rows, name = [], ""
    for line in (candidate / "build.log").read_text().splitlines():
        if "Compiling entry function" in line:
            name = line.split("'")[1]
        match = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if match and is_moving_kernel(name):
            rows.append(dict(name=name, stack=int(match[1]), spill_stores=int(match[2]), spill_loads=int(match[3])))
    def allowed(row):
        return budget if re.search(r"moving_persistent\d+PersistentTileStep", row["name"]) else 0
    if len(rows) != expected or len({r["name"] for r in rows}) != expected or any(r[k] > allowed(r) for r in rows for k in ("stack", "spill_stores", "spill_loads")):
        raise ValueError("Incomplete or nonzero ptxas moving resource records: " + json.dumps(rows))
    (candidate / "ptxas-resources.json").write_text(json.dumps(rows, indent=2) + "\n")
    strict = not any(r[k] for r in rows for k in ("stack", "spill_stores", "spill_loads"))
    manifest = dict(module=str(module), module_sha256=digest(module),
                    source_hashes={str(p.relative_to(candidate / "source")): digest(p) for p in sorted((candidate / "source").rglob("*")) if p.is_file()},
                    expected_moving_kernels=expected, all_moving_zero_stack_local_spill=strict, gpu_executed=False,
                    persistent_exploratory_stack_and_spill_budget_bytes=budget,
                    scope="strict zero resources" if not budget else "EXPLORATORY: bounded PersistentTileStep spills only; old kernels remain zero",
                    build_harness_sha256=digest(Path(__file__)), resource_checker_sha256=digest(ROOT / "scripts/check_cuda_moving_build.py"))
    (candidate / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    frozen = candidate.parent / "frozen-execution" / candidate.name / digest(module) / module.name
    frozen.parent.mkdir(parents=True)
    with frozen.open("xb") as output:
        output.write(module.read_bytes())
    frozen.chmod(0o444)
    proof = dict(module=str(frozen), sha256=digest(frozen), kernel_count=expected, all_moving_zero_stack_local_spill=strict,
                 persistent_exploratory_stack_and_spill_budget_bytes=budget, gpu_executed=False)
    (candidate / "frozen-execution-manifest.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof), flush=True)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--expected-kernels", type=int, default=62)
    p.add_argument("--persistent-spill-budget", type=int, choices=(0, 8, 16), default=0)
    p.add_argument("--finalize-only", action="store_true")
    args = p.parse_args()
    if args.finalize_only:
        record=json.loads((args.candidate / "staged-source.json").read_text())
        if any(digest(args.candidate / "source" / name) != sha for name, sha in record["source_hashes"].items()):
            raise ValueError("Staged source changed after compilation")
        raise SystemExit(finalize(args.candidate,args.expected_kernels,args.persistent_spill_budget))
    raise SystemExit(build(args.candidate, args.expected_kernels, args.persistent_spill_budget))
