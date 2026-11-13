"""Comparison acceptance guards; these tests never start a GPU worker."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.benchmark_fdtdx_comparison import (
    command_from,
    summarize_processes,
    validate_gate,
    validate_worker,
)


def test_gate_requires_both_physical_equivalence_and_current_artifacts():
    bindings = {"beamz_state_sha256": "abc", "fdtdx_state_sha256": "def"}
    gate = {
        "passed": True,
        "work_equivalent": True,
        "numerical_equivalent": True,
        "contract_sha256": "contract",
        "inputs": bindings,
    }
    validate_gate(gate, contract_sha="contract", bindings=bindings)
    with pytest.raises(ValueError, match="numerical gate did not pass"):
        validate_gate(
            {**gate, "numerical_equivalent": False},
            contract_sha="contract",
            bindings=bindings,
        )
    with pytest.raises(ValueError, match="stale"):
        validate_gate(
            gate,
            contract_sha="contract",
            bindings={**bindings, "fdtdx_state_sha256": "new"},
        )


def test_adapter_must_pin_engine_and_synchronize_full_immutable_replay():
    identity = {"name": "fdtdx", "version": "0.6.3"}
    result = {
        "status": "ok",
        "phase": "benchmark",
        "engine": identity,
        "contract_sha256": "contract",
        "actual_work": {"canonical": {"steps": 64}},
        "timing": {
            "warm_solve_samples_s": [1, 1, 1],
            "warm_public_samples_s": [2],
            "cold_scene_to_first_s": 3,
            "synchronization": "all_returned_device_leaves",
            "reset_policy": "same_immutable_initial_state_each_call",
        },
        "memory": {"scope": "device-global", "peak_bytes": 123},
    }
    validate_worker(
        result, phase="benchmark", contract_sha="contract", expected_identity=identity
    )
    with pytest.raises(ValueError, match="frozen comparison plan"):
        validate_worker(
            result,
            phase="benchmark",
            contract_sha="contract",
            expected_identity=identity,
            expected_adapter_sha="frozen",
        )
    with pytest.raises(ValueError, match="identity"):
        validate_worker(
            {**result, "engine": {**identity, "version": "unreleased"}},
            phase="benchmark",
            contract_sha="contract",
            expected_identity=identity,
        )
    with pytest.raises(ValueError, match="synchronize"):
        validate_worker(
            {**result, "timing": {**result["timing"], "synchronization": "one leaf"}},
            phase="benchmark",
            contract_sha="contract",
            expected_identity=identity,
        )


def test_paired_summary_uses_processes_and_correct_ratio_direction():
    trials = []
    for engine, value in zip(
        ("beamz", "fdtdx", "fdtdx", "beamz"), (2, 6, 12, 4), strict=True
    ):
        trials.append(
            {
                "engine": engine,
                "result": {
                    "timing": {
                        "warm_solve_samples_s": [value] * 3,
                        "warm_public_samples_s": [value * 2],
                        "cold_scene_to_first_s": value * 4,
                    },
                    "memory": {"peak_bytes": value * 100},
                },
            }
        )
    summary = summarize_processes(trials)
    assert summary["warm_solve_samples_s"]["geometric_paired_ratio"] == 3
    assert summary["warm_solve_samples_s"]["process_medians_s"]["beamz"] == [2, 4]
    assert summary["processes_per_engine"] == 2
    assert summary["cold_scene_to_first_s"]["medians_s"]["fdtdx"] == 36


def test_command_templates_are_argv_not_shell():
    assert command_from(
        ["python", "adapter.py", "{contract}"], {"contract": "a b.json"}
    ) == ["python", "adapter.py", "a b.json"]
    with pytest.raises(ValueError, match="shell strings"):
        command_from("python adapter.py", {})


def test_failed_numerical_gate_prevents_every_timed_process(tmp_path, monkeypatch):
    # Exercise orchestration without depending on a GPU or NVIDIA utilities.
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/sh\nprintf '%s\\n' 'Mock GPU, mock-uuid, mock-driver'\n")
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    root = Path(__file__).resolve().parents[2]
    worker = tmp_path / "worker.py"
    calls = tmp_path / "calls.txt"
    worker.write_text("""import argparse,hashlib,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--phase');p.add_argument('--engine');p.add_argument('--contract');p.add_argument('--output');p.add_argument('--state-output');p.add_argument('--calls');a=p.parse_args()
with open(a.calls,'a') as f:f.write(a.phase+' '+a.engine+'\\n')
identity={'name':a.engine,'version':'0.30.0' if a.engine=='beamz' else '0.6.3'}
if a.engine=='beamz':identity['native_sha256']='a'*64
pathlib.Path(a.state_output).write_bytes(b'validation')
pathlib.Path(a.output).write_text(json.dumps({'status':'ok','phase':a.phase,'engine':identity,'contract_sha256':hashlib.sha256(pathlib.Path(a.contract).read_bytes()).hexdigest(),'actual_work':{'canonical':{'steps':64}}}))
""")
    gate = tmp_path / "gate.py"
    gate.write_text("""import json,pathlib,sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({'passed':False,'work_equivalent':True,'numerical_equivalent':False}))
""")
    contract = tmp_path / "contract.json"
    contract.write_text('{"steps":64}')
    engines = {}
    for name in ("beamz", "fdtdx"):
        identity = {"name": name, "version": "0.30.0" if name == "beamz" else "0.6.3"}
        if name == "beamz":
            identity["native_sha256"] = "a" * 64
        engines[name] = {
            "expected_identity": identity,
            "command": [
                sys.executable,
                str(worker),
                "--phase",
                "{phase}",
                "--engine",
                name,
                "--contract",
                "{contract}",
                "--output",
                "{output}",
                "--state-output",
                "{state_output}",
                "--calls",
                str(calls),
            ],
        }
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "beamz.fdtdx-comparison-plan.v1",
                "contracts": [str(contract)],
                "engines": engines,
                "gate_command": [sys.executable, str(gate), "{gate_output}"],
            }
        )
    )
    output = tmp_path / "results"
    process = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/benchmark_fdtdx_comparison.py"),
            "--plan",
            str(plan),
            "--output-dir",
            str(output),
            "--execute",
        ],
        text=True,
        capture_output=True,
    )
    assert process.returncode != 0
    assert calls.read_text().splitlines() == ["validate beamz", "validate fdtdx"]
    report = json.loads((output / "comparison.json").read_text())
    assert report["cases"][0]["status"] == "failed"
    assert report["cases"][0]["trials"] == []
