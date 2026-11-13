"""Non-timing CUPTI launch witness for the explicit persistent native method.

An ordinary Program reference, immutable inputs and all30 outputs are checked.
This instrumentation is separate from every performance comparison.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cuda_fixture_store import digest_array
from scripts.export_fdtdx_comparison import file_digest
from scripts.validate_cuda_native_moving import exact, load_fixture
from scripts.validate_cuda_plane_retirement import module_at, ordinary, state_of


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--expected-module-sha256", required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--collector", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--order", choices=("rank", "locality"), default="rank")
    p.add_argument("--depth", type=int, choices=(4, 16), default=16)
    p.add_argument("--steps", type=int, default=35)
    a = p.parse_args()
    if file_digest(a.module) != a.expected_module_sha256 or a.steps < a.depth:
        p.error("Unchanged explicit candidate and positive macro count required")
    module = module_at(a.module)
    fixture, data, inputs = load_fixture(a.fixture)
    before = [digest_array(v) for v in inputs]
    expected = ordinary(module, fixture, data, inputs, fixture["attributes"], a.steps)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    activity = a.output.with_suffix(".activity.jsonl")
    lib = ctypes.CDLL(str(a.collector.resolve()))
    lib.beamz_profile_start.argtypes = [ctypes.c_char_p]
    lib.beamz_profile_error.restype = ctypes.c_char_p
    session = module.NativeSession(inputs, fixture["attributes"], preserve_inputs=True)
    try:
        if lib.beamz_profile_start(str(activity.resolve()).encode()):
            raise RuntimeError(str(lib.beamz_profile_error()))
        try:
            result = session.run_moving_persistent([a.steps], depth=a.depth, order=a.order, workers=0)
        finally:
            if lib.beamz_profile_stop():
                raise RuntimeError(str(lib.beamz_profile_error()))
        parity = exact(state_of(fixture, data, session, result), expected)
        metadata, clocks = session.metadata(), session.clocks()
    finally:
        session.close()
    rows = [json.loads(line) for line in activity.read_text().splitlines()]
    kernels = [v for v in rows if v.get("kind") == "kernel"]
    persistent = [v for v in kernels if "PersistentTileStep" in v["name"]]
    reset = [v for v in kernels if "ResetPersistent" in v["name"]]
    collectors = [v for v in rows if v.get("kind") == "collector"]
    macros = a.steps // a.depth
    passed = (parity["all30_bitwise"] and before == [digest_array(v) for v in inputs]
              and len(persistent) == macros and len(reset) == macros
              and len(collectors) == 1 and collectors[0]["dropped_records"] == 0
              and all(v["block"] == [1024, 1, 1] and v["grid"] == [metadata["persistent_tile_step_workers"], 1, 1]
                      and v["local_bytes_per_thread"] == 0 for v in persistent)
              and result["algorithm"] == "persistent_tile_step"
              and clocks["current_step"] == int(inputs[127]) + a.steps
              and clocks["elapsed_steps"] == int(inputs[128]) + a.steps)
    report = dict(schema="beamz.persistent-launch-witness.v1", passed=passed,
                  module=str(a.module.resolve()), module_sha256=file_digest(a.module),
                  fixture_sha256=file_digest(a.fixture), worker_sha256=file_digest(Path(__file__)),
                  collector_sha256=file_digest(a.collector), activity=str(activity),
                  activity_sha256=file_digest(activity), scope="instrumented launch/correctness witness; no performance claim",
                  requested_API="run_moving_persistent", order=a.order, depth=a.depth,
                  actual_persistent_launches=persistent, actual_reset_launches=len(reset),
                  all_kernel_launch_count=len(kernels), collector=collectors, parity=parity,
                  input_preservation=before == [digest_array(v) for v in inputs],
                  metadata=metadata, clocks=clocks)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=passed, persistent_launches=len(persistent), macros=macros)), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
