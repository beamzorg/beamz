#!/usr/bin/env python3
r"""Validate nonzero native source/observation clocks against an independent JAX oracle.

Export a reference JSON/NPZ with a selected FFI component loaded process-locally::

    python scripts/launch_cuda_component.py --component FFI.so --sha256 SHA \
        scripts/validate_cuda_native_clocks.py --mode reference \
        --fixture seeded.json --expected-component-sha SHA --output clocks.json

Then use a separate interpreter with JAX absent::

    native-only/bin/python scripts/validate_cuda_native_clocks.py --mode native \
        --fixture clocks.json --module _cuda_native.so --output native-clocks.json

Use --method deep4 to validate the experimental native schedule. Each native mode
checks owned/preserving calls, reset/repeat, odd schedules and cross-call
continuation. The elapsed-nine checkpoint distinguishes split FP32 multiply/add
from a fused clock update on the frozen Mie fixture. No native results form the
reference, and every mismatch is retained in the output report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_cuda_native_runtime import compare_state, logical_native_state
from scripts.export_cuda_native_fixture import (
    WORKSPACE_INPUTS,
    ZERO_WORKSPACE_INPUTS,
    buffer_digest,
)
from scripts.export_fdtdx_comparison import file_digest

SCHEDULES = {"s7": [7], "s34": [3, 4], "s331": [3, 3, 1], "s3": [3], "s33": [3, 3]}
SCHEMA = "beamz.cuda-native-nonzero-origin.v1"


def float32_bits(value):
    return int(np.asarray(value, dtype=np.float32).view(np.uint32).item())


def read_fixture(path, expected_schema):
    fixture = json.loads(path.read_text())
    if fixture["schema"] != expected_schema:
        raise ValueError(f"Expected {expected_schema}")
    data_path = path.parent / fixture["data_file"]
    if file_digest(data_path) != fixture["data_sha256"]:
        raise ValueError("Fixture NPZ hash differs")
    return fixture, data_path


def verify_inputs(fixture, inputs):
    if len(inputs) != 134 or len(fixture["state_leaves"]) != 30:
        raise ValueError("Clock validation requires the complete 134-input/30-leaf ABI")
    if any(
        buffer_digest(value) != item["sha256"]
        for value, item in zip(inputs, fixture["inputs"], strict=True)
    ):
        raise ValueError("Fixture input hash differs")


def reference_export(args):
    # Deliberately imported only in the reference process.
    import jax
    import jax.numpy as jnp

    from beamz.simulation.backend import cuda_backend_status

    status = cuda_backend_status()
    if not status.available:
        raise RuntimeError(status.reason)
    import beamz._cuda as extension

    if file_digest(extension.__file__) != args.expected_component_sha:
        raise ValueError("Loaded FFI component differs from requested SHA256")
    fixture, data_path = read_fixture(args.fixture, "beamz.cuda-native-fixture.v1")
    with np.load(data_path, allow_pickle=False) as data:
        arrays = {key: np.array(data[key], copy=True) for key in data.files}
    inputs = [arrays[item["key"]] for item in fixture["inputs"]]
    verify_inputs(fixture, inputs)
    dt = np.float32(fixture["attributes"]["dt"])
    origin = np.float32(float(dt) * 3.125)
    for index, value in ((126, origin), (127, np.int32(5)), (128, np.int32(3))):
        arrays[fixture["inputs"][index]["key"]] = np.asarray(value)
    for leaf in fixture["state_leaves"]:
        if leaf["name"] == "current_step":
            arrays[leaf["initial_key"]] = np.asarray(5, np.int32)
        elif leaf["name"] == "t":
            arrays[leaf["initial_key"]] = np.asarray(
                np.float32(origin + np.float32(dt * np.float32(3)))
            )
    inputs = [arrays[item["key"]] for item in fixture["inputs"]]
    before = [buffer_digest(value) for value in inputs]
    for group in range(9):
        if inputs[86 + 3 * group].shape[0] and inputs[87 + 3 * group].shape[1] < 12:
            raise ValueError("Source waveform must cover source steps 5 through 11")
    nonworkspace = [i for i in range(len(inputs)) if i not in WORKSPACE_INPUTS]
    positions = {value: index for index, value in enumerate(nonworkspace)}
    passthrough = [
        leaf
        for leaf in fixture["state_leaves"]
        if leaf["mapping"]["kind"] == "unchanged"
    ]
    base = [jax.device_put(inputs[i]) for i in nonworkspace]
    base += [jax.device_put(arrays[leaf["initial_key"]]) for leaf in passthrough]
    jax.block_until_ready(base)
    attributes = {
        key: np.dtype(fixture["attribute_dtypes"][key]).type(value)
        for key, value in fixture["attributes"].items()
    }
    results = tuple(
        jax.ShapeDtypeStruct(tuple(item["shape"]), np.dtype(item["dtype"]))
        for item in fixture["outputs"]
    )
    ffi = jax.ffi.ffi_call(
        fixture["target"],
        results,
        input_output_aliases={
            int(k): int(v) for k, v in fixture["output_aliases"].items()
        },
        vmap_method="sequential",
    )

    def make_call(schedule):
        def call(*arguments):
            values = {
                index: arguments[position] for index, position in positions.items()
            }
            current, elapsed, anchor = values[127], values[128], values[126]
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
                for component in range(6):
                    values[component] = outputs[bank + component]
                    values[31 + component] = outputs[6 + component]
                    values[62 + component] = outputs[12 + component]
                for component in range(3):
                    values[120 + component] = outputs[30 + component]
                current += np.int32(count)
                elapsed += np.int32(count)
            unchanged = {
                leaf["name"]: arguments[len(nonworkspace) + i]
                for i, leaf in enumerate(passthrough)
            }
            result = []
            for leaf in fixture["state_leaves"]:
                name, mapping = leaf["name"], leaf["mapping"]
                if mapping["kind"] == "field":
                    value = values[mapping["component"]]
                elif mapping["kind"] == "output":
                    index = mapping["index"]
                    value = (
                        values[31 + index - 6]
                        if index < 12
                        else values[62 + index - 12]
                        if index < 18
                        else values[120 + index - 30]
                    )
                elif name == "current_step":
                    value = current
                elif name == "t":
                    value = anchor + dt * elapsed.astype(jnp.float32)
                else:
                    value = unchanged[name]
                result.append(value)
            return (*result, elapsed)

        return call

    references = {}
    for label, schedule in SCHEDULES.items():
        compiled = jax.jit(make_call(tuple(schedule))).lower(*base).compile()
        values = jax.block_until_ready(compiled(*base))
        repeat = jax.block_until_ready(compiled(*base))
        if not all(np.array_equal(a, b) for a, b in zip(values, repeat, strict=True)):
            raise ValueError(f"JAX reference repeat differs: {label}")
        keys = []
        for index, value in enumerate(values[:-1]):
            key = f"nonzero_{label}_{index:02d}"
            arrays[key] = np.asarray(value)
            keys.append(key)
        references[label] = {
            "chunks": schedule,
            "keys": keys,
            "elapsed_steps": int(np.asarray(values[-1])),
        }
    if before != [buffer_digest(value) for value in inputs]:
        raise ValueError("Reference export mutated host inputs")
    for item, value in zip(fixture["inputs"], inputs, strict=True):
        item["sha256"] = buffer_digest(value)
    fixture.update(
        schema=SCHEMA,
        references=references,
        native_component_sha256=args.expected_component_sha,
        nonzero_clock={
            "origin": float(origin),
            "origin_bits": float32_bits(origin),
            "current_step": 5,
            "elapsed_steps": 3,
            "origin_dt_scale": 3.125,
        },
        data_file=args.output.with_suffix(".npz").name,
    )
    np.savez(args.output.with_suffix(".npz"), **arrays)
    fixture["data_sha256"] = file_digest(args.output.with_suffix(".npz"))
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    time_index = next(
        i for i, leaf in enumerate(fixture["state_leaves"]) if leaf["name"] == "t"
    )
    checkpoint = arrays[references["s33"]["keys"][time_index]]
    fused = np.float32(float(origin) + float(dt) * 9)
    split = np.float32(origin + np.float32(dt * np.float32(9)))
    report = {
        "status": "ok",
        "mode": "reference",
        "oracle": "Independent JAX minimal FFI and JAX clock expression; no native execution used as oracle",
        "component_sha256": args.expected_component_sha,
        "source_fixture_sha256": file_digest(args.fixture),
        "fixture_sha256": file_digest(args.output),
        "worker_sha256": file_digest(__file__),
        "schedules": SCHEDULES,
        "clock": fixture["nonzero_clock"],
        "repeat_exact": True,
        "host_inputs_unchanged": True,
        "clock_rounding": {
            "jax_checkpoint_bits": float32_bits(checkpoint),
            "fused_bits": float32_bits(fused),
            "split_bits": float32_bits(split),
            "fused_and_split_differ": bool(fused != split),
        },
    }
    if float32_bits(checkpoint) != float32_bits(split):
        raise ValueError(
            "Independent JAX clock checkpoint does not match split multiply/add"
        )
    args.output.with_suffix(".reference.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def native_check(args):
    if importlib.util.find_spec("jax") is not None:
        raise RuntimeError("Native validation requires an interpreter with JAX absent")
    fixture, data_path = read_fixture(args.fixture, SCHEMA)
    if {k: v["chunks"] for k, v in fixture["references"].items()} != SCHEDULES:
        raise ValueError("Reference must retain all five required schedules")
    spec = importlib.util.spec_from_file_location("_cuda_native", args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {
        "status": "ok",
        "mode": "native",
        "method": args.method,
        "moving_depth": 4 if args.method == "moving" else None,
        "module_sha256": file_digest(args.module),
        "fixture_sha256": file_digest(args.fixture),
        "worker_sha256": file_digest(__file__),
        "oracle_component_sha256": fixture["native_component_sha256"],
        "checks": [],
    }
    with np.load(data_path, allow_pickle=False) as data:
        inputs = [data[item["key"]] for item in fixture["inputs"]]
        verify_inputs(fixture, inputs)
        before = [buffer_digest(value) for value in inputs]

        def check(session, result, label, name):
            clocks = session.clocks()
            state = logical_native_state(
                fixture, data, session.download(), clocks, result["output_bank"]
            )
            reference = fixture["references"][label]
            expected = {
                leaf["name"]: data[key]
                for leaf, key in zip(
                    fixture["state_leaves"], reference["keys"], strict=True
                )
            }
            parity = compare_state(state, expected)
            elapsed_exact = clocks["elapsed_steps"] == reference["elapsed_steps"]
            origin_exact = (
                float32_bits(clocks["observation_origin"])
                == fixture["nonzero_clock"]["origin_bits"]
            )
            passed = (
                parity["passed"] and parity["exact"] and elapsed_exact and origin_exact
            )
            report["checks"].append(
                {
                    "case": name,
                    "reference": label,
                    "passed": bool(passed),
                    "all30_exact": parity["exact"],
                    "elapsed_exact": bool(elapsed_exact),
                    "origin_exact": origin_exact,
                    "expected_elapsed_steps": reference["elapsed_steps"],
                    "clocks": clocks,
                    "time_bits": float32_bits(clocks["time"]),
                    "expected_time_bits": float32_bits(expected["t"]),
                    "parity": parity,
                }
            )
            if not passed:
                report["status"] = "failed_parity"

        for preserve in (False, True):
            session = module.NativeSession(
                inputs, fixture["attributes"], preserve_inputs=preserve
            )
            if args.method == "moving":
                run = partial(session.run_moving, depth=4)
            else:
                run = (
                    session.run_deep4
                    if args.method == "deep4"
                    else session.run_schedule
                )
            prefix = "preserve" if preserve else "owned"
            try:
                check(session, run([7]), "s7", f"{prefix}:whole")
                if not preserve:
                    session.reset()
                check(session, run([7]), "s7", f"{prefix}:repeat")
                for schedule, label in (([3, 4], "s34"), ([3, 3, 1], "s331")):
                    session.reset()
                    check(session, run(schedule), label, f"{prefix}:schedule-{label}")
                if not preserve:
                    session.reset()
                    for count, label in ((3, "s3"), (3, "s33"), (1, "s331")):
                        check(
                            session, run([count]), label, f"owned:continuation-{label}"
                        )
                    session.reset()
                    check(session, run([3]), "s3", "owned:before-real-D4")
                    check(session, run([4]), "s34", "owned:continuation3+4")
            except Exception as error:
                report.update(status="error", error=f"{type(error).__name__}: {error}")
            finally:
                session.close()
        report["host_inputs_unchanged"] = before == [
            buffer_digest(value) for value in inputs
        ]
        report["jax_available"] = False
        report["jax_imported"] = any(
            name == "jax" or name.startswith("jax.") for name in sys.modules
        )
        if not report["host_inputs_unchanged"] or report["jax_imported"]:
            report["status"] = "failed_host_mutation_or_jax_import"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode", choices=("reference", "export", "native"), required=True
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file; reference mode also writes sibling NPZ and .reference.json",
    )
    parser.add_argument("--module", type=Path)
    parser.add_argument("--expected-component-sha")
    parser.add_argument(
        "--method", choices=("mstep", "deep4", "moving"), default="mstep"
    )
    args = parser.parse_args()
    if args.mode == "native" and args.module is None:
        parser.error("Native mode requires --module")
    if args.mode != "native" and not args.expected_component_sha:
        parser.error("Reference mode requires --expected-component-sha")
    if args.fixture.resolve() == args.output.resolve():
        parser.error("Output must differ from the input fixture")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = native_check(args) if args.mode == "native" else reference_export(args)
    except Exception as error:
        args.output.with_suffix(".error.json").write_text(
            json.dumps(
                {
                    "status": "error",
                    "mode": args.mode,
                    "error": f"{type(error).__name__}: {error}",
                    "worker_sha256": file_digest(__file__),
                },
                indent=2,
            )
            + "\n"
        )
        raise
    print(
        json.dumps(
            {
                "status": report["status"],
                "mode": args.mode,
                "checks": len(report.get("checks", [])),
                "output": str(args.output),
            }
        )
    )
    return int(report["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
