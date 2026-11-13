"""Bounded complete-state persistent case for CUDA sanitizer instrumentation.

Uses seeded fields/CPML, two ordered coincident sources and live direct DFT.
Performance is never inferred from this worker.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cuda_fixture_store import digest_array
from scripts.export_fdtdx_comparison import file_digest
from scripts.validate_cuda_native_moving import exact, load_fixture
from scripts.validate_cuda_native_source_plane import make_packet
from scripts.validate_cuda_plane_retirement import module_at, ordinary, state_of


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--expected-module-sha256", required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--order", choices=("rank", "locality"), default="rank")
    p.add_argument("--depth", type=int, choices=(4, 16), default=16)
    p.add_argument("--steps", type=int, choices=(17, 35), default=17)
    p.add_argument("--repeats", type=int, default=2)
    a = p.parse_args()
    if file_digest(a.module) != a.expected_module_sha256 or a.steps < a.depth or a.repeats < 1:
        p.error("Unchanged explicit candidate and positive macro/repeat count required")
    module = module_at(a.module)
    fixture, data, inputs = load_fixture(a.fixture)
    packet, attrs = make_packet(inputs, fixture["attributes"], "direct-source-DFT", a.steps, True)
    before = [digest_array(v) for v in packet]
    expected = ordinary(module, fixture, data, packet, attrs, a.steps)
    quiet = list(packet)
    for group in (5, 7):
        quiet[87 + 3 * group] = np.zeros_like(packet[87 + 3 * group])
    quiet_state = ordinary(module, fixture, data, quiet, attrs, a.steps)
    sensitive = [name for name in expected if not np.array_equal(expected[name], quiet_state[name])]
    if not any("dft" in name.lower() for name in sensitive):
        raise RuntimeError("Source waveform lacks a live DFT sensitivity witness")
    records = []
    session = module.NativeSession(packet, attrs, preserve_inputs=False)
    try:
        for repeat in range(a.repeats):
            if repeat:
                session.reset()
            result = session.run_moving_persistent([a.steps], depth=a.depth, order=a.order, workers=0)
            metadata, clocks = session.metadata(), session.clocks()
            parity = exact(state_of(fixture, data, session, result), expected)
            positive = (result.get("algorithm") == "persistent_tile_step"
                        and result.get("persistent_order") == a.order
                        and result.get("moving_macro_count") == a.steps // a.depth
                        and metadata.get("persistent_tile_step_ready")
                        and metadata.get("persistent_tile_step_jobs", 0) > 0)
            clock_ok = (clocks["current_step"] == int(packet[127]) + a.steps
                        and clocks["elapsed_steps"] == int(packet[128]) + a.steps
                        and np.float32(clocks["observation_origin"]).view(np.uint32)
                        == np.asarray(packet[126]).view(np.uint32))
            records.append(dict(repeat=repeat, parity=parity, positive_persistent=positive,
                                clocks=clocks, clocks_correct=bool(clock_ok), metadata=metadata))
    finally:
        session.close()
    unchanged = before == [digest_array(v) for v in packet]
    passed = unchanged and all(v["parity"]["all30_bitwise"] and v["positive_persistent"] and v["clocks_correct"] for v in records)
    report = dict(schema="beamz.persistent-sanitizer-case.v1", passed=bool(passed),
                  module=str(a.module.resolve()), module_sha256=file_digest(a.module),
                  worker_sha256=file_digest(Path(__file__)), fixture_sha256=file_digest(a.fixture),
                  input_preservation=unchanged, source_sensitive_leaves=sensitive,
                  order=a.order, depth=a.depth, steps=a.steps, records=records,
                  positive_persistent=all(v["positive_persistent"] for v in records))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=bool(passed), records=len(records), source_live_DFT=True)), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
