#!/usr/bin/env python3
"""Measure complete mmap-backed CUDA workloads with bounded host parity checks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_cuda_native_runtime import (
    chunks_for,
    observed_process_gpu_memory,
)
from scripts.cuda_fixture_store import (
    ArrayReader,
    compare_arrays,
    digest_array,
    drop_pages,
)
from scripts.export_cuda_large_fixture import available_host_bytes
from scripts.export_cuda_native_fixture import (
    PERSISTENT_INPUTS,
    WORKSPACE_INPUTS,
    ZERO_WORKSPACE_INPUTS,
)
from scripts.export_fdtdx_comparison import file_digest


def exact_admission(fixture, store, args, module=None):
    import subprocess

    sizes = {
        item["index"]: store.metadata[item["key"]]["nbytes"]
        for item in fixture["inputs"]
    }
    persistent = sum(sizes[i] for i in (*PERSISTENT_INPUTS, 127, 128))
    inputs = sum(sizes.values())
    payload = inputs + (persistent if args.ownership == "preserve" else 0) + 4
    deep = {}
    moving = {}
    if args.method == "deep4":
        shapes = [tuple(x["shape"]) for x in fixture["inputs"][:6]]
        plan = module.deep_observation_plan(
            shapes, store[fixture["inputs"][113]["key"]]
        )
        psi = sum(sizes[i] for i in (*range(31, 37), *range(62, 68)))
        observation = (
            sum(
                plan[k].nbytes
                for k in ["keys", "remapped_indices", "publication_words"]
            )
            + plan["staged_bytes_d4"]
        )
        deep = {
            "CPML_shadow_bytes": psi,
            "observation_bytes": observation,
            "key_count": plan["key_count"],
        }
        payload += psi + observation
        del plan
    elif args.method == "moving":
        prepared = [store[item["key"]] for item in fixture["inputs"]]
        moving = dict(module.moving_resource_plan(prepared, depth=args.moving_depth))
        payload += int(moving["extra_gpu_bytes"])
    free = (
        int(
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        << 20
    )
    host = available_host_bytes()
    # Native constructor keeps private host snapshots; uploads may touch every mmap page.
    host_required = (
        inputs + persistent + (2 << 30)
        if args.backend == "native"
        else (3 << 30) + max(sizes.values())
    )
    record = {
        "input_payload_bytes": inputs,
        "persistent_bytes": persistent,
        "GPU_payload_expected_bytes": payload,
        "GPU_required_with2GiB_reserve": payload + (2 << 30),
        "GPU_free_bytes": free,
        "host_required_conservative_bytes": host_required,
        "host_available_bytes": host,
        "deep": deep,
        "moving": moving,
    }
    if record["GPU_required_with2GiB_reserve"] > free or host_required > host:
        raise MemoryError(f"Pre-init exact packet admission rejected: {record}")
    return record


def verify_state(fixture, store, values):
    rows = []
    for leaf, value in zip(fixture["state_leaves"], values, strict=True):
        expected = store[leaf["reference_key"]]
        row = compare_arrays(value, expected)
        row["name"] = leaf["name"]
        rows.append(row)
        drop_pages(expected)
    return {
        "passed": all(x["passed"] for x in rows),
        "exact": all(x["exact"] for x in rows),
        "bitwise_exact": all(x["bitwise_exact"] for x in rows),
        "leaves": rows,
    }


def native(args, f, store, inputs, chunks):
    if importlib.util.find_spec("jax") is not None:
        raise RuntimeError("Native worker requires JAX absent")
    spec = importlib.util.spec_from_file_location("_cuda_native", args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    admission = exact_admission(f, store, args, module)
    start = time.perf_counter()
    session = module.NativeSession(
        inputs, f["attributes"], preserve_inputs=args.ownership == "preserve"
    )
    setup = time.perf_counter() - start
    for value in inputs:
        drop_pages(value)
    if args.method == "moving":
        def run(chunks):
            return session.run_moving(chunks, depth=args.moving_depth)
    else:
        run = getattr(session, {"deep4": "run_deep4", "rolling4": "run_rolling4",
                                "mstep": "run_schedule"}[args.method])
    scratch = args.output.parent / (args.output.stem + ".readback")
    scratch.mkdir(exist_ok=True)

    def validate(result):
        clocks = session.clocks()
        rows = []
        for index, leaf in enumerate(f["state_leaves"]):
            m = leaf["mapping"]
            raw = None
            if m["kind"] == "field":
                raw = (18 if result["output_bank"] else 0) + m["component"]
            elif m["kind"] == "output":
                raw = m["index"]
            expected = store[leaf["reference_key"]]
            if raw is not None:
                path = scratch / f"{index}.npy"
                value = np.lib.format.open_memmap(
                    path,
                    mode="w+",
                    dtype=np.dtype(leaf["dtype"]),
                    shape=tuple(leaf["shape"]),
                )
                session.download_into(raw, value)
            elif m["kind"] == "clock":
                value = np.asarray(
                    clocks["time" if m["name"] == "t" else "current_step"],
                    dtype=np.dtype(leaf["dtype"]),
                )
            else:
                value = store[leaf["initial_key"]]
            row = compare_arrays(value, expected)
            row["name"] = leaf["name"]
            rows.append(row)
            if raw is not None:
                value.flush()
                drop_pages(value)
                if row["passed"]:
                    del value
                    path.unlink()
                else:
                    row["failed_readback"] = str(path)
            drop_pages(expected)
        return {
            "passed": all(x["passed"] for x in rows),
            "exact": all(x["exact"] for x in rows),
            "bitwise_exact": all(x["bitwise_exact"] for x in rows),
            "leaves": rows,
            "clocks": clocks,
        }

    try:
        session.reset()
        start = time.perf_counter()
        result = run(chunks)
        first = time.perf_counter() - start
        if args.method == "moving":
            # Admission before construction is an upper bound. The bounded
            # moving engine retains unused ordinary-Mstep bytes on the host.
            # Account for those declared buffers before checking actual payload.
            prepared = session.metadata()
            admission["GPU_payload_upper_bound_bytes"] = admission["GPU_payload_expected_bytes"]
            admission["deferred_scratch_bytes"] = prepared.get("deferred_scratch_bytes", 0)
            admission["deferred_plan_bytes"] = prepared.get("deferred_plan_bytes", 0)
            admission["GPU_payload_expected_bytes"] -= (
                admission["deferred_scratch_bytes"] + admission["deferred_plan_bytes"]
            )
        if (
            session.metadata()["allocated_payload_bytes"]
            != admission["GPU_payload_expected_bytes"]
        ):
            raise RuntimeError(
                "Actual native allocation differs from CPU packet admission"
            )
        parity = validate(result)
        if not parity["passed"]:
            return {"status": "failed_parity", "parity": parity, "admission": admission}
        samples = []
        resets = []
        native_times = []
        for i in range(args.warmups + args.samples):
            reset = 0.0
            if args.ownership == "owned":
                start = time.perf_counter()
                session.reset()
                reset = time.perf_counter() - start
            start = time.perf_counter()
            result = run(chunks)
            elapsed = time.perf_counter() - start
            if i >= args.warmups:
                samples.append(elapsed)
                resets.append(reset)
                native_times.append(result)
        repeat = validate(result)
        return {
            "status": "ok" if repeat["passed"] else "failed_repeat",
            "setup_s": setup,
            "first_call_s": first,
            "median_wall_s": float(np.median(samples)),
            "samples_wall_s": samples,
            "outside_reset_s": resets,
            "native_timing_samples": native_times,
            "parity": parity,
            "repeat_parity": repeat,
            "admission": admission,
            "session_metadata": session.metadata(),
            "module_sha256": file_digest(args.module),
            "jax_available": False,
            "jax_imported": any(
                n == "jax" or n.startswith("jax.") for n in sys.modules
            ),
            "observed_warm_process_gpu_memory": observed_process_gpu_memory()
            if args.sample_memory
            else None,
        }
    finally:
        session.close()


def jax_backend(args, f, store, inputs, chunks):
    admission = exact_admission(f, store, args)
    import beamz._cuda as extension
    import jax
    import jax.numpy as jnp

    from beamz.simulation.backend import cuda_backend_status

    status = cuda_backend_status()
    if not status.available:
        raise RuntimeError(status.reason)

    if file_digest(extension.__file__) != args.expected_component_sha:
        raise ValueError("FFI component hash differs")
    indices = [i for i in range(134) if i not in WORKSPACE_INPUTS]
    positions = {v: i for i, v in enumerate(indices)}
    passthrough = [
        leaf for leaf in f["state_leaves"] if leaf["mapping"]["kind"] == "unchanged"
    ]
    host = [inputs[i] for i in indices] + [
        store[leaf["initial_key"]] for leaf in passthrough
    ]
    specs = [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in host]
    attrs = {
        k: np.dtype(f["attribute_dtypes"][k]).type(v)
        for k, v in f["attributes"].items()
    }
    ffi = jax.ffi.ffi_call(
        f["target"],
        tuple(
            jax.ShapeDtypeStruct(tuple(x["shape"]), np.dtype(x["dtype"]))
            for x in f["outputs"]
        ),
        input_output_aliases={int(k): int(v) for k, v in f["output_aliases"].items()},
        vmap_method="sequential",
    )

    def call(*arguments):
        values = {i: arguments[p] for i, p in positions.items()}
        current, elapsed, origin = values[127], values[128], values[126]
        for count in chunks:
            values[127], values[128] = current, elapsed
            complete = []
            for item in f["inputs"]:
                i = item["index"]
                shape = tuple(item["shape"])
                dtype = np.dtype(item["dtype"])
                complete.append(
                    jnp.zeros(shape, dtype)
                    if i in ZERO_WORKSPACE_INPUTS
                    else jnp.empty(shape, dtype)
                    if i in WORKSPACE_INPUTS
                    else values[i]
                )
            out = ffi(*complete, **{**attrs, "nsteps": np.int32(count)})
            bank = 18 if (count // 2) % 2 else 0
            for c in range(6):
                values[c] = out[bank + c]
                values[31 + c] = out[6 + c]
                values[62 + c] = out[12 + c]
            for c in range(3):
                values[120 + c] = out[30 + c]
            current += np.int32(count)
            elapsed += np.int32(count)
        unchanged = {
            leaf["name"]: arguments[len(indices) + i]
            for i, leaf in enumerate(passthrough)
        }
        result = []
        for leaf in f["state_leaves"]:
            name, m = leaf["name"], leaf["mapping"]
            if m["kind"] == "field":
                value = values[m["component"]]
            elif m["kind"] == "output":
                i = m["index"]
                value = (
                    values[31 + i - 6]
                    if i < 12
                    else values[62 + i - 12]
                    if i < 18
                    else values[120 + i - 30]
                )
            elif name == "current_step":
                value = current
            elif name == "t":
                value = origin + np.float32(attrs["dt"]) * elapsed.astype(jnp.float32)
            else:
                value = unchanged[name]
            result.append(value)
        return tuple(result)

    donated = (
        tuple(positions[i] for i in PERSISTENT_INPUTS)
        if args.ownership == "owned"
        else ()
    )
    start = time.perf_counter()
    executable = jax.jit(call, donate_argnums=donated).lower(*specs).compile()
    compile_s = time.perf_counter() - start
    base = []
    for position, value in enumerate(host):
        if (
            args.ownership == "owned"
            and position < len(indices)
            and indices[position] in PERSISTENT_INPUTS
        ):
            base.append(None)
        else:
            base.append(jax.device_put(value).block_until_ready())
            drop_pages(value)

    def fresh():
        prepared = list(base)
        if args.ownership == "owned":
            for i in PERSISTENT_INPUTS:
                prepared[positions[i]] = jax.device_put(inputs[i]).block_until_ready()
                drop_pages(inputs[i])
        return prepared

    prepared = fresh()
    start = time.perf_counter()
    final = jax.block_until_ready(executable(*prepared))
    first = time.perf_counter() - start
    parity = verify_state(f, store, final)
    if not parity["passed"]:
        return {"status": "failed_parity", "parity": parity, "admission": admission}
    samples = []
    resets = []
    for i in range(args.warmups + args.samples):
        final = None
        prepared = None
        start = time.perf_counter()
        prepared = fresh()
        reset = time.perf_counter() - start
        start = time.perf_counter()
        final = jax.block_until_ready(executable(*prepared))
        elapsed = time.perf_counter() - start
        if i >= args.warmups:
            samples.append(elapsed)
            resets.append(reset if args.ownership == "owned" else 0.0)
    repeat = verify_state(f, store, final)
    return {
        "status": "ok" if repeat["passed"] else "failed_repeat",
        "compile_s": compile_s,
        "first_call_s": first,
        "median_wall_s": float(np.median(samples)),
        "samples_wall_s": samples,
        "outside_reset_s": resets,
        "parity": parity,
        "repeat_parity": repeat,
        "admission": admission,
        "jax_version": jax.__version__,
        "component_sha256": file_digest(extension.__file__),
        "scope": "Prepared complete FFI call; preserving copies included or state donation explicit. Owned compile uses abstract values and retains no redundant original device-state bank.",
        "observed_warm_process_gpu_memory": observed_process_gpu_memory()
        if args.sample_memory
        else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--backend", choices=["native", "jax"], required=True)
    p.add_argument("--module", type=Path)
    p.add_argument("--expected-component-sha")
    p.add_argument("--method", choices=["mstep", "deep4", "rolling4", "moving"], default="mstep")
    p.add_argument("--moving-depth", type=int, choices=[4, 16], default=16,
                   help="Native consumer currently implements depths4 and16")
    p.add_argument("--ownership", choices=["preserve", "owned"], default="preserve")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--sample-memory", action="store_true")
    p.add_argument("--admission-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.backend == "native" and a.module is None:
        p.error("Native requires --module")
    if a.backend == "jax" and (a.method != "mstep" or not a.expected_component_sha):
        p.error("JAX supports mstep and requires component SHA")
    if a.samples < 1 or a.warmups < 0:
        p.error("Need>=1sample and>=0warmups")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    f = json.loads(a.fixture.read_text())
    manifest = a.fixture.parent / f["data_file"]
    assert file_digest(manifest) == f["data_sha256"]
    store = ArrayReader(a.fixture.parent / f["data_directory"])
    if a.admission_only:
        module = None
        if a.method in ("deep4", "moving"):
            spec = importlib.util.spec_from_file_location("_cuda_native", a.module)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        print(
            json.dumps(
                {"status": "admitted", "memory": exact_admission(f, store, a, module)}
            )
        )
        return 0
    inputs = []
    for item in f["inputs"]:
        store.verify(item["key"])
        inputs.append(store[item["key"]])
    for leaf in f["state_leaves"]:
        store.verify(leaf["reference_key"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    chunks = chunks_for(f["attributes"]["nsteps"], "chunk256")
    report = {
        "fixture": str(a.fixture),
        "fixture_sha256": file_digest(a.fixture),
        "backend": a.backend,
        "method": a.method,
        "moving_depth": a.moving_depth if a.method == "moving" else None,
        "ownership": a.ownership,
        "chunks": chunks,
        "worker_sha256": file_digest(__file__),
    }
    try:
        report.update(
            (native if a.backend == "native" else jax_backend)(
                a, f, store, inputs, chunks
            )
        )
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        immutable = all(
            digest_array(value) == item["sha256"]
            for value, item in zip(inputs, f["inputs"], strict=True)
        )
        report["host_inputs_unchanged"] = immutable
        if not immutable:
            report["status"] = "failed_host_mutation"
        report["host_peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "median_wall_s": report.get("median_wall_s"),
                "output": str(a.output),
            }
        )
    )
    return int(report["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
