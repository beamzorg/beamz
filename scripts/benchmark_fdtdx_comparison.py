#!/usr/bin/env python3
"""Gate two FDTD adapters, then compare them in fresh-process ABBA order.

This driver imports neither engine and never installs a component. A plan gives
argv arrays for validation/benchmark workers and an independent numerical gate.
Use --execute only after obtaining exclusive GPU ownership. Without it, this
script only validates the plan and writes the expanded command recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ENGINES = ("beamz", "fdtdx")
ORDER = ("beamz", "fdtdx", "fdtdx", "beamz")
DEFAULT_ENVIRONMENT = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "JAX_ENABLE_COMPILATION_CACHE": "false",
    "JAX_ENABLE_X64": "false",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def command_from(template, values):
    if (
        not isinstance(template, list)
        or not template
        or not all(isinstance(arg, str) for arg in template)
    ):
        raise ValueError(
            "Commands must be nonempty argv arrays; shell strings are forbidden"
        )
    return [arg.format_map(values) for arg in template]


def validate_worker(
    result, *, phase, contract_sha, expected_identity, expected_adapter_sha=None
):
    if result.get("status") != "ok" or result.get("phase") != phase:
        raise ValueError("Adapter did not complete the requested phase")
    if result.get("contract_sha256") != contract_sha:
        raise ValueError("Adapter used an unexpected contract")
    identity = result.get("engine", {})
    for name, expected in expected_identity.items():
        if identity.get(name) != expected:
            raise ValueError(f"Unexpected adapter identity field {name}")
    if (
        expected_adapter_sha is not None
        and result.get("adapter_sha256") != expected_adapter_sha
    ):
        raise ValueError("Adapter source differs from the frozen comparison plan")
    work = result.get("actual_work")
    if not isinstance(work, dict) or not isinstance(work.get("canonical"), dict):
        raise ValueError("Adapter must report actual_work.canonical")
    if not work["canonical"]:
        raise ValueError("Empty canonical work description")
    if phase == "benchmark":
        timing = result.get("timing", {})
        for key in ("warm_solve_samples_s", "warm_public_samples_s"):
            values = timing.get(key)
            if (
                not isinstance(values, list)
                or not values
                or not all(
                    isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value > 0
                    for value in values
                )
            ):
                raise ValueError(f"Missing or invalid synchronized timings: {key}")
        cold = timing.get("cold_scene_to_first_s")
        if not isinstance(cold, (int, float)) or not math.isfinite(cold) or cold <= 0:
            raise ValueError("Missing synchronized cold scene-to-first result timing")
        if timing.get("synchronization") != "all_returned_device_leaves":
            raise ValueError("Worker must synchronize every returned device leaf")
        if timing.get("reset_policy") != "same_immutable_initial_state_each_call":
            raise ValueError(
                "Worker must restart from the same immutable initial state"
            )
        if not isinstance(result.get("memory"), dict) or not result["memory"]:
            raise ValueError("Worker must report memory measurements and their scope")
    return result


def validate_gate(gate, *, contract_sha, bindings):
    if not all(
        gate.get(key) is True
        for key in ("passed", "work_equivalent", "numerical_equivalent")
    ):
        raise ValueError("Cross-engine workload/numerical gate did not pass")
    if gate.get("contract_sha256") != contract_sha:
        raise ValueError("Numerical gate belongs to a different contract")
    if gate.get("inputs") != bindings:
        raise ValueError(
            "Numerical gate is stale or binds different validation artifacts"
        )


def summarize_processes(trials):
    summary = {}
    for metric in ("warm_solve_samples_s", "warm_public_samples_s"):
        medians = {
            name: [
                statistics.median(t["result"]["timing"][metric])
                for t in trials
                if t["engine"] == name
            ]
            for name in ENGINES
        }
        # First pair is A/B; the second is B/A, balancing order within each case.
        ratios = [
            medians["fdtdx"][0] / medians["beamz"][0],
            medians["fdtdx"][1] / medians["beamz"][1],
        ]
        summary[metric] = {
            "process_medians_s": medians,
            "process_median_ranges_s": {
                name: [min(values), max(values)] for name, values in medians.items()
            },
            "fdtdx_over_beamz_paired_ratios": ratios,
            "geometric_paired_ratio": math.sqrt(math.prod(ratios)),
            "paired_ratio_range": [min(ratios), max(ratios)],
        }
    summary["independent_repeat_unit"] = "fresh process"
    summary["processes_per_engine"] = 2
    summary["uncertainty"] = (
        "Observed ranges only; within-process samples are not independent process trials; no CI inferred."
    )
    cold = {
        name: [
            t["result"]["timing"]["cold_scene_to_first_s"]
            for t in trials
            if t["engine"] == name
        ]
        for name in ENGINES
    }
    summary["cold_scene_to_first_s"] = {
        "process_values_s": cold,
        "medians_s": {name: statistics.median(values) for name, values in cold.items()},
        "ranges_s": {name: [min(values), max(values)] for name, values in cold.items()},
    }
    summary["process_memory"] = {
        name: [t["result"].get("memory") for t in trials if t["engine"] == name]
        for name in ENGINES
    }
    return summary


def gpu_status():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,power.limit,power.draw,temperature.gpu,clocks.sm,memory.used",
            "--format=csv",
        ],
        text=True,
    ).strip()


def run_command(command, *, cwd, environment, log, timeout_s):
    started = time.perf_counter()
    with Path(log).open("w") as stream:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    return {
        "command": command,
        "exit_code": result.returncode,
        "process_wall_s": time.perf_counter() - started,
        "log": str(log),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    plan = load_json(args.plan)
    if plan.get("schema") != "beamz.fdtdx-comparison-plan.v1":
        parser.error("Unsupported comparison plan schema")
    if set(plan.get("engines", {})) != set(ENGINES):
        parser.error("Plan must specify exactly beamz and fdtdx adapters")
    for name in ENGINES:
        identity = plan["engines"][name].get("expected_identity", {})
        if identity.get("name") != name or not identity.get("version"):
            parser.error("Each engine requires a pinned name and version")
        if name == "beamz" and len(identity.get("native_sha256", "")) != 64:
            parser.error("BeamZ requires its frozen native component SHA-256")
    cwd = Path(plan.get("cwd", Path.cwd())).resolve()
    contracts = [Path(value).resolve() for value in plan["contracts"]]
    if not contracts or len(set(contracts)) != len(contracts):
        parser.error("Need distinct case contracts")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = args.output_dir.resolve()
    if args.execute:
        device_identity = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        write_json(
            output_dir / "device-identity.json", {"device_identity": device_identity}
        )
    report = {
        "schema": "beamz.fdtdx-comparison-results.v1",
        "plan": plan,
        "plan_sha256": sha256(args.plan),
        "order": list(ORDER),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "cases": [],
        "environment": {**DEFAULT_ENVIRONMENT, **plan.get("environment", {})},
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("BEAMZ_CUDA_", "JAX_COMPILATION_CACHE"))
    }
    environment.update(report["environment"])
    timeout = plan.get("timeout_s", 3600)
    for index, contract in enumerate(contracts):
        case_dir = output_dir / f"case-{index:02d}"
        case_dir.mkdir()
        case = {
            "contract": str(contract),
            "contract_sha256": sha256(contract),
            "validations": {},
            "trials": [],
            "status": "planned",
        }
        report["cases"].append(case)
        values = {"contract": str(contract), "case_dir": str(case_dir)}
        for name in ENGINES:
            values[f"{name}_result"] = str(case_dir / f"{name}-validation.json")
            values[f"{name}_state"] = str(case_dir / f"{name}-validation.npz")
        values["gate_output"] = str(case_dir / "numerical-gate.json")
        case["gate_command"] = command_from(plan["gate_command"], values)
        case["expanded_commands"] = []
        for phase, order in (("validate", ENGINES), ("benchmark", ORDER)):
            for trial_index, name in enumerate(order):
                output = (
                    values[f"{name}_result"]
                    if phase == "validate"
                    else str(case_dir / f"{trial_index:02d}-{name}-benchmark.json")
                )
                command = command_from(
                    plan["engines"][name]["command"],
                    {
                        **values,
                        "phase": phase,
                        "output": output,
                        "state_output": (
                            values[f"{name}_state"]
                            if phase == "validate"
                            else str(Path(output).with_suffix(".npz"))
                        ),
                        "solve_samples": str(plan.get("solve_samples", 3)),
                        "public_samples": str(plan.get("public_samples", 1)),
                        "warmups": str(plan.get("warmups", 2)),
                    },
                )
                case["expanded_commands"].append(
                    {
                        "phase": phase,
                        "engine": name,
                        "output": output,
                        "command": command,
                    }
                )
        write_json(output_dir / "comparison.json", report)
        if not args.execute:
            continue
        try:
            for item in case["expanded_commands"]:
                if item["phase"] != "validate":
                    continue
                if sha256(contract) != case["contract_sha256"]:
                    raise ValueError("Contract changed during the comparison")
                name = item["engine"]
                row = run_command(
                    item["command"],
                    cwd=cwd,
                    environment=environment,
                    log=case_dir / f"{name}-validation.log",
                    timeout_s=timeout,
                )
                if row["exit_code"]:
                    raise ValueError(
                        f"{name} validation worker failed; see {row['log']}"
                    )
                result = validate_worker(
                    load_json(item["output"]),
                    phase="validate",
                    contract_sha=case["contract_sha256"],
                    expected_identity=plan["engines"][name]["expected_identity"],
                    expected_adapter_sha=plan["engines"][name].get(
                        "expected_adapter_sha256"
                    ),
                )
                row["result"] = result
                case["validations"][name] = row
            canonical = [
                case["validations"][name]["result"]["actual_work"]["canonical"]
                for name in ENGINES
            ]
            if canonical[0] != canonical[1]:
                raise ValueError("Adapters report different canonical actual workloads")
            bindings = {
                f"{name}_{kind}_sha256": sha256(values[f"{name}_{kind}"])
                for name in ENGINES
                for kind in ("result", "state")
            }
            gate_run = run_command(
                case["gate_command"],
                cwd=cwd,
                environment=environment,
                log=case_dir / "numerical-gate.log",
                timeout_s=timeout,
            )
            case["gate"] = gate_run
            if gate_run["exit_code"]:
                raise ValueError("Independent numerical gate command failed")
            gate = load_json(values["gate_output"])
            validate_gate(gate, contract_sha=case["contract_sha256"], bindings=bindings)
            gate_run["result"] = gate
            case["status"] = "validated"
            write_json(output_dir / "comparison.json", report)
            if args.validation_only:
                continue
            for item in case["expanded_commands"]:
                if item["phase"] != "benchmark":
                    continue
                if sha256(contract) != case["contract_sha256"]:
                    raise ValueError("Contract changed after numerical validation")
                name = item["engine"]
                row = {"engine": name, "gpu_before": gpu_status()}
                row.update(
                    run_command(
                        item["command"],
                        cwd=cwd,
                        environment=environment,
                        log=Path(item["output"]).with_suffix(".log"),
                        timeout_s=timeout,
                    )
                )
                row["gpu_after"] = gpu_status()
                if row["exit_code"]:
                    raise ValueError(
                        f"{name} benchmark worker failed; see {row['log']}"
                    )
                result = validate_worker(
                    load_json(item["output"]),
                    phase="benchmark",
                    contract_sha=case["contract_sha256"],
                    expected_identity=plan["engines"][name]["expected_identity"],
                    expected_adapter_sha=plan["engines"][name].get(
                        "expected_adapter_sha256"
                    ),
                )
                if (
                    result["actual_work"]
                    != case["validations"][name]["result"]["actual_work"]
                ):
                    raise ValueError(
                        "Timed workload differs from the validated workload"
                    )
                if result["engine"] != case["validations"][name]["result"]["engine"]:
                    raise ValueError(
                        "Engine identity changed after numerical validation"
                    )
                if result.get("adapter_sha256") != case["validations"][name][
                    "result"
                ].get("adapter_sha256"):
                    raise ValueError(
                        "Adapter source changed after numerical validation"
                    )
                timing = result["timing"]
                if len(timing["warm_solve_samples_s"]) != plan.get("solve_samples", 3):
                    raise ValueError(
                        "Solve sample count differs from the shared recipe"
                    )
                if len(timing["warm_public_samples_s"]) != plan.get(
                    "public_samples", 1
                ):
                    raise ValueError(
                        "Public sample count differs from the shared recipe"
                    )
                row["result"] = result
                case["trials"].append(row)
                write_json(output_dir / "comparison.json", report)
            case["summary"] = summarize_processes(case["trials"])
            case["status"] = "passed"
        except Exception as error:
            case.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(output_dir / "comparison.json", report)
            raise
        finally:
            write_json(output_dir / "comparison.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
