"""Prepare/run an actual-host NoIO-axis fork/join CPU proof and negatives.

Default preparation writes private proof inputs and an exact compiler command.
--execute performs CPU-only C++ compilation/execution; never uses CUDA or GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    consumer = source/"src/asymmetric_native.cuh"
    code = consumer.read_text()
    begin = code.index("template<int Depth,int Layout,bool IsotropicH>\ninline cudaError_t LaunchRanks(")
    end = code.index("template<int Depth,int Layout>\ninline cudaError_t DispatchClass", begin)
    outer = code[begin:end]
    (output/"actual_outer_noio_rank.inc").write_text(outer)
    schedule = (source/"src/concurrent_noio_axis_schedule.inc").read_text()
    (output/"concurrent_noio_axis_schedule.inc").write_text(schedule)
    producer_harness = Path(__file__).with_suffix(".cc").with_name("prove_cuda_concurrent_noio_launch.cc").resolve()
    harness = output/"actual-host-mock.cc"
    harness.write_bytes(producer_harness.read_bytes())
    exe = output/"actual-host-proof"
    command = ["c++", "-std=c++17", "-O2", "-DBEAMZ_CUDA_NATIVE_MOVING_POLICIES=1",
               "-DBEAMZ_CUDA_NATIVE_MOVING_PARALLEL_POLICIES=1", "-I"+str(output),
               "-I"+str(source/"src"), str(harness), str(source/"src/native_moving.cc"), "-o", str(exe)]
    proof = {"source": str(source), "consumer_sha256": sha(consumer),
             "planner_sha256": sha(source/"src/native_moving.cc"),
             "workspace_header_sha256": sha(source/"src/native_moving.h"),
             "schedule_sha256": sha(source/"src/concurrent_noio_axis_schedule.inc"),
             "harness_sha256": sha(harness), "harness_producer_sha256": sha(producer_harness), "compiler_command": command,
             "actual_outer_launch_sha256": sha(output/"actual_outer_noio_rank.inc"),
             "gpu_used": False, "executed": False}
    (output/"proof-plan.json").write_text(json.dumps(proof, indent=2)+"\n")
    return proof


def execute(proof, output):
    output = output.resolve()
    command = proof["compiler_command"]
    build = subprocess.run(command, text=True, capture_output=True)
    (output/"compile.log").write_text(build.stdout+build.stderr)
    if build.returncode:
        raise RuntimeError("CPU proof compilation failed; see compile.log")
    run = subprocess.run([command[-1]], text=True, capture_output=True)
    (output/"run.log").write_text(run.stdout+run.stderr)
    if run.returncode:
        raise RuntimeError("CPU proof rejected schedule; see run.log")
    proof["positive_result"] = json.loads(run.stdout)
    schedule = (output/"concurrent_noio_axis_schedule.inc").read_text()
    negatives = {
        "missing_child_join": ("status=cudaStreamWaitEvent(stream,static_cast<cudaEvent_t>(w.noio_axis_done_events[axis]),0);", "status=cudaSuccess;"),
        "missing_child_fork_wait": ("status=cudaStreamWaitEvent(child,static_cast<cudaEvent_t>(w.noio_axis_fork_event),0);", "status=cudaSuccess;"),
    }
    reports = []
    try:
        for name, (old, new) in negatives.items():
            if schedule.count(old) != 1:
                raise ValueError("Negative mutation anchor changed")
            (output/"concurrent_noio_axis_schedule.inc").write_text(schedule.replace(old, new, 1))
            negative_command = command[:-1]+[str(output/("negative-"+name))]
            compile_run = subprocess.run(negative_command, text=True, capture_output=True)
            if compile_run.returncode:
                raise RuntimeError("Negative must compile before rejection")
            result = subprocess.run([negative_command[-1]], text=True, capture_output=True)
            if result.returncode == 0:
                raise RuntimeError(f"Unsafe schedule escaped:{name}")
            reports.append({"mutation": name, "rejected": True, "stderr": result.stderr.strip()})
    finally:
        (output/"concurrent_noio_axis_schedule.inc").write_text(schedule)
    proof.update({"executed": True, "negative_controls": reports})
    (output/"proof.json").write_text(json.dumps(proof, indent=2)+"\n")
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    proof = prepare(a.source, a.output)
    if a.execute:
        proof = execute(proof, a.output)
    print(json.dumps({"executed": proof["executed"], "output": str(a.output), "positive_result": proof.get("positive_result")}))


if __name__ == "__main__":
    main()
