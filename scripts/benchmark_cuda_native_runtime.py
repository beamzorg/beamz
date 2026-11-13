#!/usr/bin/env python3
"""Compare matched preserving/owned full-physics calls with or without JAX.

Native mode requires an interpreter without JAX and loads the separate extension
by filename, bypassing the BeamZ package. All exports/downloads are outside warm
timing. Both backends generate/reset the same initial state for every sample.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_cuda_native_fixture import (
    PERSISTENT_INPUTS,
    WORKSPACE_INPUTS,
    ZERO_WORKSPACE_INPUTS,
    buffer_digest,
)
from scripts.export_fdtdx_comparison import file_digest


def observed_process_gpu_memory():
    """Sample warmed resident allocations outside solve timing, before teardown."""
    queries = {
        "processes": [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        "device_global": [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
    }
    report = {
        "pid": os.getpid(),
        "scope": "Warmed synchronized process before teardown; includes context, graphs, allocator and resident live arrays; device-global usage includes other processes",
        "units": "MiB",
    }
    for name, command in queries.items():
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if name == "processes":
            rows = [row for row in rows if row.split(",", 1)[0].strip() == str(os.getpid())]
        report[name] = rows
        if result.returncode:
            report[f"{name}_error"] = result.stderr.strip()
    return report


def chunks_for(steps, policy):
    if policy == "whole":
        return [steps]
    full, tail = divmod(steps, 256)
    return [256] * full + ([tail] if tail else [])


def compare_state(candidate, reference, *, allow_roundoff=False):
    rows = []
    for name, expected in reference.items():
        actual = np.asarray(candidate[name])
        expected = np.asarray(expected)
        compatible = actual.shape == expected.shape and actual.dtype == expected.dtype
        exact = compatible and np.array_equal(actual, expected)
        row = {
            "name": name,
            "exact": bool(exact),
            "shape": list(actual.shape),
            "dtype": actual.dtype.str,
            "passed": bool(exact),
        }
        if compatible and expected.dtype.kind == "f":
            finite = bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)))
            delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            maximum = float(delta.max(initial=0))
            scale = float(np.max(np.abs(expected), initial=0))
            row.update(max_abs_error=maximum, reference_peak=scale, finite=finite)
            row["passed"] = finite and (
                exact or (allow_roundoff and maximum <= 1e-6 * scale + 1e-30)
            )
        rows.append(row)
    return {
        "passed": all(row["passed"] for row in rows),
        "exact": all(row["exact"] for row in rows),
        "leaves": rows,
    }


def logical_native_state(fixture, data, outputs, clocks, output_bank):
    state = {}
    for leaf in fixture["state_leaves"]:
        mapping = leaf["mapping"]
        if mapping["kind"] == "field":
            value = outputs[(18 if output_bank else 0) + mapping["component"]]
        elif mapping["kind"] == "output":
            value = outputs[mapping["index"]]
        elif mapping["kind"] == "clock":
            key = "time" if mapping["name"] == "t" else "current_step"
            value = np.asarray(clocks[key], dtype=np.dtype(leaf["dtype"]))
        else:
            value = data[leaf["initial_key"]]
        state[leaf["name"]] = np.asarray(value)
    return state


def native_benchmark(args, fixture, data, inputs, chunks):
    if importlib.util.find_spec("jax") is not None:
        raise RuntimeError(
            "Native proof must use the NumPy-only interpreter with JAX absent"
        )
    spec = importlib.util.spec_from_file_location("_cuda_native", args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = [buffer_digest(value) for value in inputs]
    started = time.perf_counter()
    session = module.NativeSession(
        inputs,
        fixture["attributes"],
        preserve_inputs=args.ownership == "preserve",
        kind=args.kind,
        initialize_scratch=True,
    )
    setup_s = time.perf_counter() - started

    def run(schedule):
        if args.native_method == "deep4":
            if not hasattr(session, "run_deep4"):
                raise RuntimeError(
                    "This module does not expose the experimental deep4 method"
                )
            return session.run_deep4(schedule)
        if hasattr(session, "run_schedule"):
            return session.run_schedule(schedule)
        if len(schedule) == 1:
            return session.run(nsteps=schedule[0])
        raise RuntimeError("Matched chunked calls require NativeSession.run_schedule")

    try:
        session.reset()
        started = time.perf_counter()
        first = run(chunks)
        cold_s = time.perf_counter() - started
        outputs = session.download()
        state = logical_native_state(
            fixture, data, outputs, session.clocks(), first["output_bank"]
        )
        reference = {
            leaf["name"]: data[leaf["reference_key"]]
            for leaf in fixture["state_leaves"]
        }
        parity = compare_state(state, reference, allow_roundoff=args.allow_roundoff)
        if not parity["passed"]:
            return {"status": "failed_parity", "parity": parity}
        del outputs, state
        continuation = None
        if fixture["continuation_steps"]:
            session.reset()
            result = run(fixture["continuation_steps"])
            outputs = session.download()
            state = logical_native_state(
                fixture, data, outputs, session.clocks(), result["output_bank"]
            )
            reference = {
                leaf["name"]: data[f"continuation_{i:02d}"]
                for i, leaf in enumerate(fixture["state_leaves"])
            }
            continuation = compare_state(
                state, reference, allow_roundoff=args.allow_roundoff
            )
            if not continuation["passed"]:
                return {
                    "status": "failed_continuation",
                    "parity": parity,
                    "continuation": continuation,
                }
            del outputs, state
        samples, resets, native_times = [], [], []
        for index in range(args.warmups + args.samples):
            reset_s = 0.0
            if args.ownership == "owned":
                started = time.perf_counter()
                session.reset()
                reset_s = time.perf_counter() - started
            started = time.perf_counter()
            result = run(chunks)
            elapsed = time.perf_counter() - started
            if index >= args.warmups:
                samples.append(elapsed)
                resets.append(reset_s)
                native_times.append(result)
        host_immutable = before == [buffer_digest(value) for value in inputs]
        final_outputs = session.download()
        final_state = logical_native_state(
            fixture, data, final_outputs, session.clocks(), result["output_bank"]
        )
        final_reference = {
            leaf["name"]: data[leaf["reference_key"]]
            for leaf in fixture["state_leaves"]
        }
        repeat_parity = compare_state(
            final_state, final_reference, allow_roundoff=args.allow_roundoff
        )
        return {
            "status": "ok"
            if host_immutable and repeat_parity["passed"]
            else "failed_repeat_or_mutation",
            "setup_s": setup_s,
            "first_call_s": cold_s,
            "samples_wall_s": samples,
            "outside_reset_s": resets,
            "native_timing_samples": native_times,
            "median_wall_s": float(np.median(samples)),
            "parity": parity,
            "continuation": continuation,
            "repeat_parity": repeat_parity,
            "host_inputs_unchanged": host_immutable,
            "session_metadata": session.metadata(),
            "jax_available": False,
            "jax_imported": any(
                name == "jax" or name.startswith("jax.") for name in sys.modules
            ),
            "module_sha256": file_digest(args.module),
            "observed_warm_process_gpu_memory": observed_process_gpu_memory()
            if args.sample_memory
            else None,
        }
    finally:
        session.close()


def jax_benchmark(args, fixture, data, inputs, chunks):
    import jax
    import jax.numpy as jnp

    from beamz.simulation.backend import cuda_backend_status

    status = cuda_backend_status()
    if not status.available:
        raise RuntimeError(status.reason)
    import beamz._cuda as extension

    expected_sha = args.expected_component_sha or fixture["native_component_sha256"]
    if file_digest(extension.__file__) != expected_sha:
        raise RuntimeError("Installed reference component changed since fixture export")
    nonworkspace = [i for i in range(len(inputs)) if i not in WORKSPACE_INPUTS]
    positions = {value: i for i, value in enumerate(nonworkspace)}
    passthrough = [
        leaf
        for leaf in fixture["state_leaves"]
        if leaf["mapping"]["kind"] == "unchanged"
    ]
    base = [jax.device_put(inputs[i]) for i in nonworkspace]
    base += [jax.device_put(data[leaf["initial_key"]]) for leaf in passthrough]
    jax.block_until_ready(base)
    attributes = {
        key: np.dtype(fixture["attribute_dtypes"][key]).type(value)
        for key, value in fixture["attributes"].items()
    }
    result_types = tuple(
        jax.ShapeDtypeStruct(tuple(item["shape"]), np.dtype(item["dtype"]))
        for item in fixture["outputs"]
    )
    aliases = {int(key): int(value) for key, value in fixture["output_aliases"].items()}
    ffi = jax.ffi.ffi_call(
        fixture["target"],
        result_types,
        input_output_aliases=aliases,
        vmap_method="sequential",
    )

    def make_call(schedule):
        def call(*arguments):
            values = {
                index: arguments[position] for index, position in positions.items()
            }
            current = values[127]
            elapsed = values[128]
            origin = values[126]
            for count in schedule:
                values[127], values[128] = current, elapsed
                complete = []
                for item in fixture["inputs"]:
                    index, shape, dtype = (
                        item["index"],
                        tuple(item["shape"]),
                        np.dtype(item["dtype"]),
                    )
                    complete.append(
                        jnp.zeros(shape, dtype)
                        if index in ZERO_WORKSPACE_INPUTS
                        else jnp.empty(shape, dtype)
                        if index in WORKSPACE_INPUTS
                        else values[index]
                    )
                outputs = ffi(*complete, **{**attributes, "nsteps": np.int32(count)})
                bank = 18 if (count // 2) % 2 else 0
                for c in range(6):
                    values[c] = outputs[bank + c]
                for c in range(6):
                    values[31 + c], values[62 + c] = outputs[6 + c], outputs[12 + c]
                for c in range(3):
                    values[120 + c] = outputs[30 + c]
                current = current + np.int32(count)
                elapsed = elapsed + np.int32(count)
            passthrough_values = {
                leaf["name"]: arguments[len(nonworkspace) + i]
                for i, leaf in enumerate(passthrough)
            }
            result = []
            for leaf in fixture["state_leaves"]:
                name, mapping = leaf["name"], leaf["mapping"]
                if mapping["kind"] == "field":
                    value = values[mapping["component"]]
                elif mapping["kind"] == "output":
                    output = mapping["index"]
                    value = (
                        values[31 + output - 6]
                        if output < 12
                        else values[62 + output - 12]
                        if output < 18
                        else values[120 + output - 30]
                    )
                elif name == "current_step":
                    value = current
                elif name == "t":
                    value = origin + np.float32(attributes["dt"]) * elapsed.astype(
                        jnp.float32
                    )
                else:
                    value = passthrough_values[name]
                result.append(value)
            return tuple(result)

        donated = (
            tuple(positions[index] for index in PERSISTENT_INPUTS)
            if args.ownership == "owned"
            else ()
        )
        return jax.jit(call, donate_argnums=donated)

    call = make_call(chunks)
    started = time.perf_counter()
    executable = call.lower(*base).compile()
    compile_s = time.perf_counter() - started

    def fresh():
        if args.ownership == "preserve":
            return list(base)
        prepared = list(base)
        for index in PERSISTENT_INPUTS:
            prepared[positions[index]] = jax.device_put(
                np.array(inputs[index], copy=True)
            )
        return jax.block_until_ready(prepared)

    initial = fresh()
    final = jax.block_until_ready(executable(*initial))
    reference = {
        leaf["name"]: data[leaf["reference_key"]] for leaf in fixture["state_leaves"]
    }
    actual = {
        leaf["name"]: np.asarray(value)
        for leaf, value in zip(fixture["state_leaves"], final, strict=True)
    }
    parity = compare_state(actual, reference, allow_roundoff=args.allow_roundoff)
    if not parity["passed"]:
        return {"status": "failed_parity", "parity": parity}
    samples, resets = [], []
    for index in range(args.warmups + args.samples):
        final = None
        started = time.perf_counter()
        prepared = fresh()
        reset_s = time.perf_counter() - started
        started = time.perf_counter()
        final = jax.block_until_ready(executable(*prepared))
        if index >= args.warmups:
            samples.append(time.perf_counter() - started)
            resets.append(reset_s if args.ownership == "owned" else 0.0)
    actual = {
        leaf["name"]: np.asarray(value)
        for leaf, value in zip(fixture["state_leaves"], final, strict=True)
    }
    repeat = compare_state(actual, reference, allow_roundoff=args.allow_roundoff)
    return {
        "status": "ok" if repeat["passed"] else "failed_repeat",
        "compile_s": compile_s,
        "samples_wall_s": samples,
        "outside_reset_s": resets,
        "median_wall_s": float(np.median(samples)),
        "parity": parity,
        "repeat_parity": repeat,
        "jax_version": jax.__version__,
        "scope": "Ready full FFI invocation; workspaces created inside JIT; preserving copies included or persistent-state donation explicit",
        "observed_warm_process_gpu_memory": observed_process_gpu_memory()
        if args.sample_memory
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--backend", choices=("native", "jax"), required=True)
    parser.add_argument("--module", type=Path)
    parser.add_argument("--kind", choices=("mstep", "program"), default="mstep")
    parser.add_argument("--native-method", choices=("mstep", "deep4"), default="mstep")
    parser.add_argument(
        "--ownership", choices=("preserve", "owned"), default="preserve"
    )
    parser.add_argument("--schedule", choices=("chunk256", "whole"), default="chunk256")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--allow-roundoff", action="store_true")
    parser.add_argument("--expected-component-sha")
    parser.add_argument("--sample-memory", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    fixture = json.loads(args.fixture.read_text())
    if fixture["schema"] != "beamz.cuda-native-fixture.v1":
        raise ValueError("Wrong native fixture schema")
    data_path = args.fixture.parent / fixture["data_file"]
    if file_digest(data_path) != fixture["data_sha256"]:
        raise ValueError("Fixture NPZ hash differs")
    if args.backend == "native" and args.module is None:
        parser.error("Native mode requires --module")
    if args.backend == "jax" and args.native_method != "mstep":
        parser.error(
            "Experimental native methods cannot be silently used as JAX baselines"
        )
    report = {
        "fixture": str(args.fixture),
        "fixture_sha256": file_digest(args.fixture),
        "backend": args.backend,
        "native_method": args.native_method,
        "ownership": args.ownership,
        "schedule": args.schedule,
        "chunks": chunks_for(fixture["attributes"]["nsteps"], args.schedule),
        "worker_sha256": file_digest(__file__),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with np.load(data_path, allow_pickle=False) as data:
            inputs = [np.asarray(data[item["key"]]) for item in fixture["inputs"]]
            if any(
                buffer_digest(value) != item["sha256"]
                for value, item in zip(inputs, fixture["inputs"], strict=True)
            ):
                raise ValueError("Native fixture input hash differs")
            function = native_benchmark if args.backend == "native" else jax_benchmark
            report.update(function(args, fixture, data, inputs, report["chunks"]))
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "median_wall_s": report.get("median_wall_s"),
                "output": str(args.output),
            }
        )
    )
    return int(report["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
