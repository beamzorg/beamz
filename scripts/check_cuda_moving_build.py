"""Fail closed on a research moving-TU build's real compiler/resource contract.

Ninja rules, rather than intended CMake settings, establish compilation mode.
Resource checks cover moving entry points only: unchanged ordinary/DFT kernels
may have their own stack frames. No CUDA context or GPU execution is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path

REQUIRED_DEFINES = (
    "BEAMZ_CUDA_NATIVE_MOVING=1",
    "BEAMZ_CUDA_NATIVE_MOVING_POLICIES=1",
    "BEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES=1",
    "BEAMZ_CUDA_NATIVE_SPARSE_OBSERVATION=1",
)


def ninja_contract(ninja: str, rules: str) -> dict:
    def node(object_path: str) -> dict:
        start = "build CMakeFiles/" + object_path + ":"
        blocks = [b for b in ninja.split("\n\n") if b.startswith(start)]
        if len(blocks) != 1:
            raise ValueError(f"Expected one actual Ninja object: {object_path}")
        block = blocks[0]
        rule = block.splitlines()[0].split(":", 1)[1].strip().split()[0]
        rule_blocks = [
            b for b in rules.split("\n\n") if b.startswith("rule " + rule + "\n")
        ]
        if len(rule_blocks) != 1:
            raise ValueError(f"Expected one actual Ninja compile rule: {rule}")
        return {"node": block, "rule": rule_blocks[0]}

    moving = node("beamz_cuda_moving.dir/src/native_moving_kernel.cu.o")
    core = node("beamz_cuda_core.dir/src/update.cu.o")
    runtime = node("_cuda_native.dir/src/native_runtime.cu.o")
    tokens = shlex.split(moving["node"] + "\n" + moving["rule"])
    for definition in REQUIRED_DEFINES:
        key = definition.split("=", 1)[0]
        found = {token[2:] for token in tokens if token.startswith("-D" + key + "=")}
        if found != {definition}:
            raise ValueError(
                f"Missing or contradictory moving definition: {key}: {found}"
            )
    if "--extended-lambda" not in tokens:
        raise ValueError("Missing actual moving extended-lambda option")
    if any(
        t in tokens
        for t in ("-rdc=true", "--relocatable-device-code=true", "-dc", "--device-c")
    ):
        raise ValueError("Moving TU unexpectedly uses relocatable device code")
    for name, obj in (("ordinary core", core), ("native runtime", runtime)):
        if "-rdc=true" not in shlex.split(obj["node"] + "\n" + obj["rule"]):
            raise ValueError(f"Existing {name} RDC mode changed")
    return {"moving": moving, "core": core, "runtime": runtime, "verified": True}


def is_moving_kernel(name: str) -> bool:
    return bool(re.search(r"moving_native\d+(?:Asymmetric|TemporalSources|TemporalEffects)|moving_persistent\d+(?:PersistentTileStep|ResetPersistent)", name))


def moving_resources(output: str, expected: int, persistent_stack_budget: int = 0) -> list[dict]:
    if persistent_stack_budget not in (0, 8, 16):
        raise ValueError("Explicit persistent exploratory budget must be0,8or16 bytes")
    rows = []
    for name, fields in re.findall(r" Function ([^\n]+):\n([^\n]+)", output):
        if not is_moving_kernel(name):
            continue
        values = dict(re.findall(r"\b(REG|STACK|SHARED|LOCAL):(\d+)", fields))
        if set(values) != {"REG", "STACK", "SHARED", "LOCAL"}:
            raise ValueError(f"Incomplete moving resource record: {name}")
        rows.append(
            {"name": name, **{key: int(value) for key, value in values.items()}}
        )
    if len(rows) != expected or len({r["name"] for r in rows}) != expected:
        raise ValueError(
            f"Moving entry-point count {len(rows)} differs from expected {expected}"
        )
    failures = [row for row in rows if row["LOCAL"] or row["STACK"] > (
        persistent_stack_budget if re.search(r"moving_persistent\d+PersistentTileStep", row["name"]) else 0)]
    if failures:
        raise ValueError(
            "Moving stack/local resource gate failed: " + json.dumps(failures)
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument("--expected-kernels", type=int, required=True)
    parser.add_argument("--persistent-stack-budget", type=int, choices=(0, 8, 16), default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = ninja_contract(
        (args.build / "build.ninja").read_text(),
        (args.build / "CMakeFiles/rules.ninja").read_text(),
    )
    output = subprocess.run(
        [str(args.cuobjdump), "--dump-resource-usage", str(args.module)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = moving_resources(output, args.expected_kernels, args.persistent_stack_budget)
    report = {
        "schema": "beamz.moving-build-contract.v1",
        "module": str(args.module.resolve()),
        "module_sha256": hashlib.sha256(args.module.read_bytes()).hexdigest(),
        "compile_contract": contract,
        "kernel_count": len(rows),
        "kernels": rows,
        "persistent_exploratory_stack_budget_bytes": args.persistent_stack_budget,
        "strict_zero_stack_local": not any(r["STACK"] or r["LOCAL"] for r in rows),
        "gpu_executed": False,
        "passed": True,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ("module_sha256", "kernel_count", "passed")}
        )
    )


if __name__ == "__main__":
    main()
