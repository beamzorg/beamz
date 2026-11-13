"""Freeze explicit persistent-method workers from the existing checked campaign.

prepare uses CPU only. validate/benchmark execute one worker and require the
coordinator's GPU ownership. Normal moving baseline/comparator bytes are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AREA = ROOT / ".cache/cuda-trace-five-hour-2026-11-13-1059"
NATIVE_PYTHON = ROOT / ".cache/cuda-native-autoresearch-2026-11-11-1607/native-only/bin/python"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Exactly one frozen persistent-worker anchor required:{old!r}")
    return text.replace(old, new, 1)


WITNESS = '''
def persistent_witness(result, metadata, chunks, depth, order, requested_workers):
    macros = sum(int(n) // depth for n in chunks)
    if (result.get("algorithm") != "persistent_tile_step" or not result.get("persistent_tile_step")
            or result.get("persistent_order") != order or result.get("moving_macro_count") != macros):
        raise RuntimeError("Native result does not witness the requested persistent API/macro count")
    if macros:
        if (not metadata.get("persistent_tile_step_ready") or metadata.get("persistent_tile_step_order") != order
                or metadata.get("persistent_tile_step_workers", 0) < 1 or metadata.get("persistent_tile_step_jobs", 0) < 1):
            raise RuntimeError("Positive persistent macros lack the matching prepared native queue")
        if requested_workers and metadata["persistent_tile_step_workers"] != min(requested_workers, metadata["persistent_tile_step_jobs"]):
            raise RuntimeError("Actual persistent worker count differs from the explicit request")
    return {"API": "run_moving_persistent", "algorithm": result["algorithm"],
            "order": order, "requested_workers": requested_workers,
            "actual_workers": metadata.get("persistent_tile_step_workers", 0),
            "actual_jobs": metadata.get("persistent_tile_step_jobs", 0),
            "actual_moving_macros": macros, "persistent_body_positive": macros > 0,
            "ordinary_tail_steps": result.get("ordinary_tail_steps")}
'''


def validation_adapter(text):
    helpers = WITNESS + '''
PERSISTENT_METHOD_WITNESSES = []

def persistent_run(session, chunks, depth, order, workers):
    result = session.run_moving_persistent(chunks, depth=depth, order=order, workers=workers)
    PERSISTENT_METHOD_WITNESSES.append(persistent_witness(result, session.metadata(), chunks, depth, order, workers))
    return result

'''
    text = once(text, "def load_fixture(path):", helpers + "def load_fixture(path):")
    text = once(text, '    p.add_argument("--module", type=Path, required=True)', '    p.add_argument("--module", type=Path, required=True)\n    p.add_argument("--persistent-order", choices=("rank", "locality"), default="rank")\n    p.add_argument("--persistent-workers", type=int, default=0)')
    text = once(text, "    fixture, data, inputs = load_fixture(a.fixture)", "    fixture, data, inputs = load_fixture(a.fixture)\n    if not getattr(module, 'moving_persistent_tile_step', False) or not hasattr(module.NativeSession, 'run_moving_persistent'):\n        raise RuntimeError('Candidate lacks the explicit persistent native API')\n    presence = sum((1 << field) for field in range(6) if inputs[95 + 3 * field].shape[0])\n    if presence not in (0, 20):\n        raise ValueError('Persistent prototype admits source masks0or20 only')")
    count = text.count("session.run_moving(")
    for args in ("chunks, a.depth", "[split], a.depth", "[steps - split], a.depth", "[steps], a.depth", "[a.depth], a.depth"):
        text = text.replace("session.run_moving(" + args + ")", "persistent_run(session, " + args + ", a.persistent_order, a.persistent_workers)")
    if count < 5 or "session.run_moving(" in text:
        raise ValueError("Unexpected unadapted moving calls in the frozen complete-state gate")
    text = once(text, '"schema": "beamz.native-moving-complete-gates.v1",', '"schema": "beamz.native-persistent-moving-complete-gates.v1",\n        "requested_API": "run_moving_persistent",\n        "persistent_order": a.persistent_order,\n        "persistent_workers_requested": a.persistent_workers,\n        "persistent_method_witnesses": PERSISTENT_METHOD_WITNESSES,')
    text = once(text, '        "passed": immutable', '        "passed": any(w["persistent_body_positive"] for w in PERSISTENT_METHOD_WITNESSES)\n        and immutable')
    return text


def benchmark_adapter(text):
    text = once(text, "def exact_admission(fixture, store, args, module=None):", WITNESS + "\ndef exact_admission(fixture, store, args, module=None):")
    text = once(text, '    elif args.method == "moving":', '    elif args.method in ("moving", "moving_persistent"):')
    text = once(text, '        payload += int(moving["extra_gpu_bytes"])', '        payload += int(moving["extra_gpu_bytes"])\n        if args.method == "moving_persistent":\n            queue_bytes = int(moving["tile_count"]) * 16 + int(moving["epochs"]) * int(moving["max_frontier"]) * 4 + 8\n            moving["persistent_queue_expected_bytes"] = queue_bytes\n            payload += queue_bytes')
    dispatch = '''    if args.method == "moving_persistent":
        if not getattr(module, "moving_persistent_tile_step", False):
            raise RuntimeError("Candidate does not expose the persistent native prototype")
        def run(chunks):
            return session.run_moving_persistent(chunks, depth=args.moving_depth,
                                                 order=args.persistent_order, workers=args.persistent_workers)
    elif args.method == "moving":'''
    text = once(text, '    if args.method == "moving":', dispatch)
    text = once(text, "        parity = validate(result)", '''        first_persistent_witness = persistent_witness(result, session.metadata(), chunks,
                args.moving_depth, args.persistent_order, args.persistent_workers)
        if session.metadata()["persistent_tile_step_metadata_bytes"] != admission["moving"]["persistent_queue_expected_bytes"]:
            raise RuntimeError("Actual persistent allocation differs from bound native source formula")
        parity = validate(result)''')
    # Clock relation is checked outside the measured invocation along with readback.
    text = once(text, "        clocks = session.clocks()\n        rows = []", '''        clocks = session.clocks()
        if (clocks["current_step"] != int(inputs[127]) + sum(chunks)
                or clocks["elapsed_steps"] != int(inputs[128]) + sum(chunks)
                or np.float32(clocks["observation_origin"]).view(np.uint32) != np.asarray(inputs[126]).view(np.uint32)):
            raise RuntimeError("Persistent prepared clock/origin contract differs")
        rows = []''')
    text = once(text, '            "native_timing_samples": native_times,', '''            "native_timing_samples": native_times,
            "requested_API": "run_moving_persistent",
            "persistent_order": args.persistent_order,
            "persistent_workers_requested": args.persistent_workers,
            "first_persistent_method_witness": first_persistent_witness,
            "persistent_method_witnesses": [persistent_witness(v, session.metadata(), chunks,
                args.moving_depth, args.persistent_order, args.persistent_workers) for v in native_times],''')
    text = once(text, 'choices=["mstep", "deep4", "rolling4", "moving"]', 'choices=["mstep", "deep4", "rolling4", "moving", "moving_persistent"]')
    text = once(text, '    p.add_argument("--ownership", choices=["preserve", "owned"], default="preserve")', '    p.add_argument("--persistent-order", choices=("rank", "locality"), default="rank")\n    p.add_argument("--persistent-workers", type=int, default=0)\n    p.add_argument("--ownership", choices=["preserve", "owned"], default="preserve")')
    text = once(text, "    a = p.parse_args()", "    a = p.parse_args()\n    if a.backend != 'native' or a.method != 'moving_persistent':\n        p.error('This frozen candidate adapter requires native moving_persistent; use the unchanged baseline worker for normal moving')")
    text = once(text, '"moving_depth": a.moving_depth if a.method == "moving" else None,', '"moving_depth": a.moving_depth if a.method in ("moving", "moving_persistent") else None,')
    text = once(text, 'if a.method in ("deep4", "moving"):', 'if a.method in ("deep4", "moving", "moving_persistent"):')
    return text


def prepare(output, template, session_source):
    output, template, session_source = Path(output).resolve(), Path(template).resolve(), Path(session_source).resolve()
    if output.exists():
        raise ValueError("New frozen persistent worker directory required")
    plan = json.loads((template / "plan.json").read_text())
    expected = {row["frozen"]: row["frozen_sha256"] for row in plan["workers"]}
    expected.update(plan["dependencies"])
    for path, value in expected.items():
        if digest(path) != value:
            raise ValueError("Original checked campaign dependency changed:" + path)
    host = (session_source / "src/native_extension.cc").read_text()
    for token in ('const size_t bytes=count*16+owners*4+8;', 'RunMovingPersistent(', 'persistent?BeamzEnqueueNativePersistent('):
        if token not in host:
            raise ValueError("Persistent native source does not match the explicit adapter admission/API contract")
    output.mkdir(parents=True)
    source_files = {name: template / "workers" / name for name in ("validate_cuda_native_moving.py", "benchmark_native.py", "cuda_research_compare.py")}
    produced = {
        "validate_persistent.py": validation_adapter(source_files["validate_cuda_native_moving.py"].read_text()),
        "benchmark_persistent.py": benchmark_adapter(source_files["benchmark_native.py"].read_text()),
        "benchmark_baseline.py": source_files["benchmark_native.py"].read_text(),
        "cuda_research_compare.py": source_files["cuda_research_compare.py"].read_text(),
    }
    for name, text in produced.items():
        compile(text, str(output / name), "exec")
        (output / name).write_text(text)
    fixture = ROOT / ".cache/cuda-native-autoresearch-2026-11-11-1607/fixtures/small-mie-seeded.json"
    manifest = {"schema": "beamz.frozen-persistent-workers.v1", "GPU_used_by_prepare": False,
                "template_plan": str(template / "plan.json"), "template_plan_sha256": digest(template / "plan.json"),
                "checked_template_sources": {str(p): digest(p) for p in source_files.values()},
                "checked_dependencies": plan["dependencies"],
                "frozen_workers": {str(output / n): digest(output / n) for n in produced},
                "session_source": str(session_source), "session_source_sha256": digest(session_source / "src/native_extension.cc"),
                "persistent_allocation_formula": "tile_count*16+epochs*max_frontier*4+8",
                "small_fixture": str(fixture), "small_fixture_sha256": digest(fixture),
                "prefix_fixture": plan["fixture_prefix"], "prefix_fixture_sha256": digest(plan["fixture_prefix"]),
                "baseline": plan["baseline"], "expected_control_sha256": plan["expected_control_sha256"],
                "normal_baseline_and_strict_comparator_byte_identical": True,
                "candidate_method_identity": "moving_persistent / run_moving_persistent / persistent_tile_step",
                "producer_sha256": digest(__file__)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = dict(manifest["frozen_workers"])
    expected.update(manifest["checked_dependencies"])
    expected.update(manifest["checked_template_sources"])
    expected[manifest["template_plan"]] = manifest["template_plan_sha256"]
    expected[str(Path(manifest["session_source"]) / "src/native_extension.cc")] = manifest["session_source_sha256"]
    expected[manifest["small_fixture"]] = manifest["small_fixture_sha256"]
    expected[manifest["prefix_fixture"]] = manifest["prefix_fixture_sha256"]
    for path, value in expected.items():
        if digest(path) != value:
            raise ValueError("Frozen persistent worker contract changed:" + path)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "validate", "benchmark"), required=True)
    p.add_argument("--workers-directory", type=Path, default=AREA / "persistent-workers")
    p.add_argument("--template", type=Path, default=AREA / "validation-template")
    p.add_argument("--session-source", type=Path, default=AREA / "persistent-session-v2/source")
    p.add_argument("--module", type=Path)
    p.add_argument("--expected-module-sha256")
    p.add_argument("--order", choices=("rank", "locality"), default="rank")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--depth", type=int, choices=(4, 16), default=16)
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--warmups", type=int, default=0)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.stage == "prepare":
        result = prepare(a.workers_directory, a.template, a.session_source)
        print(json.dumps({"workers": result["frozen_workers"], "GPU_used": False}))
        return 0
    manifest = verify(a.workers_directory)
    if not a.module or not a.expected_module_sha256 or not a.output or digest(a.module) != a.expected_module_sha256:
        p.error("Execution requires unchanged explicit module/SHA andoutput")
    if a.workers < 0:
        p.error("Worker count cannot be negative")
    worker = a.workers_directory / ("validate_persistent.py" if a.stage == "validate" else "benchmark_persistent.py")
    command = [str(NATIVE_PYTHON), str(worker), "--module", str(a.module.resolve()),
               "--persistent-order", a.order, "--persistent-workers", str(a.workers), "--output", str(a.output)]
    if a.stage == "validate":
        command += ["--fixture", manifest["small_fixture"], "--depth", str(a.depth), "--steps", str(a.steps), "--oracle-kind", "program"]
    else:
        command += ["--fixture", manifest["prefix_fixture"], "--backend", "native", "--method", "moving_persistent",
                    "--moving-depth", str(a.depth), "--ownership", "preserve", "--samples", str(a.samples), "--warmups", str(a.warmups)]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    run = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), timeout=1800)
    if not run.returncode:
        report = json.loads(a.output.read_text())
        if report.get("module_sha256") != a.expected_module_sha256:
            raise ValueError("Persistent worker report module identity differs")
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
