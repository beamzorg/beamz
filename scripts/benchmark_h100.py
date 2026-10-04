#!/usr/bin/env python3
"""Run the canonical BeamZ H100 throughput workloads and emit schema-v3 JSON."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import jax
import jaxlib

import beamz
from beamz.simulation import observe as monitor_runtime
from beamz.simulation import sharding as sharding_runtime
from beamz.simulation.backend import cuda_backend_status
from beamz.simulation.execute import build_scan, initial_program_state
from beamz.simulation.model import ShardingConfig
from tests.performance.benchmark_schema import BenchmarkRecord
from tests.performance.h100_workloads import H100_WORKLOADS


def _block(state) -> None:
    # Include every field, CPML recurrence and monitor accumulator on all ranks.
    jax.block_until_ready(state)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    suffix = "-dirty" if status.stdout.strip() else ""
    return result.stdout.strip() + suffix


def _peak_memory_bytes(devices, fallback: int) -> int:
    total = 0
    found = False
    for device in devices:
        stats = device.memory_stats() or {}
        values = [
            int(stats[key])
            for key in ("peak_bytes_in_use", "bytes_in_use")
            if key in stats
        ]
        if values:
            found = True
            total += max(values)
    return total if found and total > 0 else int(fallback)


def _array_devices(array) -> tuple[object, ...]:
    devices = getattr(array, "devices", None)
    if callable(devices):
        return tuple(devices())
    device = getattr(array, "device", None)
    if callable(device):
        device = device()
    return () if device is None else (device,)


def _time_call(callable_):
    started = time.perf_counter()
    value = callable_()
    _block(value)
    return value, time.perf_counter() - started


def run_benchmark(
    args: argparse.Namespace, *, final_state_callback=None
) -> BenchmarkRecord:
    visible_devices = jax.devices()
    if not visible_devices:
        raise RuntimeError("JAX reported no execution devices")
    if not args.allow_cpu and not all(
        device.platform == "gpu" for device in visible_devices
    ):
        raise RuntimeError(
            "H100 benchmark requires GPU devices; use --allow-cpu for smoke tests"
        )
    sharding = (
        None
        if args.devices == 1
        else ShardingConfig(
            enabled=True,
            axis=args.shard_axis,
            num_devices=args.devices,
        )
    )

    workload = H100_WORKLOADS[args.workload].resized(
        shape_zyx=None if args.shape is None else tuple(args.shape),
        timesteps=args.timesteps,
    )
    setup_device = (
        jax.default_device(jax.devices("cpu")[0])
        if getattr(args, "host_setup", False)
        else nullcontext()
    )
    with setup_device:
        sim = workload.build()
        sim.clear_compiled_cache()
        program = sim.compile(
            num_steps=workload.timesteps,
            sharding=sharding,
            backend=args.backend,
        )
        state = initial_program_state(
            program,
            t=float(sim.time[0]),
            current_step=0,
            monitor_steps=workload.timesteps,
        )
    state = sharding_runtime.prepare_state(
        program,
        state,
        replicated_fields=(*monitor_runtime.MONITOR_FIELDS, "t", "current_step"),
    )
    coefficients = sharding_runtime.place_tree(program, program.coefficients)
    if getattr(args, "host_setup", False) and args.devices == 1:
        state = jax.device_put(state, visible_devices[0])
        coefficients = jax.device_put(coefficients, visible_devices[0])
    execution_devices = _array_devices(state.ex)
    if not execution_devices:
        raise RuntimeError("could not determine devices used by the benchmark state")
    scan = build_scan(program, donate_state=False)

    started = time.perf_counter()
    lowered = scan.lower(state, coefficients)
    trace_lower_s = time.perf_counter() - started
    started = time.perf_counter()
    executable = lowered.compile()
    compile_s = time.perf_counter() - started

    # One unreported launch primes allocator and clocks before the measured samples.
    warm_state = executable(state, coefficients)
    _block(warm_state)
    del warm_state
    kernel_samples = tuple(
        _time_call(
            lambda state=state, coefficients=coefficients, executable=executable: (
                executable(state, coefficients)
            )
        )[1]
        for _ in range(args.samples)
    )
    cpml_psi_precision = (
        str(state.cpml_psi_h_terms[0].dtype) if state.cpml_psi_h_terms else "float32"
    )
    # Public runs own their input placement. Do not keep a second simulation's
    # standalone benchmark inputs alive while measuring their allocator peak.
    del state, coefficients, executable, lowered, scan

    # Public-path latency includes input placement, state allocation and result decode.
    warm_run = sim.advance(
        num_steps=workload.timesteps,
        sharding=sharding,
        backend=args.backend,
    )
    _block(warm_run.state)
    del warm_run
    end_to_end_samples = []
    for sample in range(args.samples):
        result, elapsed = _time_call(
            lambda: (
                sim.advance(
                    num_steps=workload.timesteps,
                    sharding=sharding,
                    backend=args.backend,
                ).state
            )
        )
        end_to_end_samples.append(elapsed)
        if sample == args.samples - 1 and final_state_callback is not None:
            final_state_callback(result)
        del result
    peak_memory = _peak_memory_bytes(execution_devices, 0)
    if not peak_memory:
        # H100 allocator statistics are authoritative. Building a second default
        # backend plan just for an unused fallback can allocate the global grid
        # on GPU 0 and OOM after every timed sample already succeeded.
        with setup_device:
            peak_memory = sim.memory_estimate(
                num_steps=workload.timesteps,
                sharding=sharding,
            )["total_bytes"]
    features = workload.feature_labels
    cuda_component_version = None
    cuda_abi_version = None
    if program.config.backend.startswith("cuda"):
        status = cuda_backend_status(register=False)
        if not status.available:
            raise RuntimeError(status.reason or "CUDA component unavailable")
        cuda_component_version = status.extension_version
        cuda_abi_version = status.abi_version
    return BenchmarkRecord(
        beamz_commit=_git_commit(),
        beamz_version=beamz.__version__,
        python_version=platform.python_version(),
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        workload=workload.name,
        backend=program.config.backend,
        device="; ".join(sorted({device.device_kind for device in execution_devices})),
        device_count=len(execution_devices),
        precision="float32",
        grid_dimensions=workload.shape_zyx,
        timesteps=workload.timesteps,
        boundaries=features["boundaries"],
        sources=features["sources"],
        monitors=features["monitors"],
        trace_lower_s=trace_lower_s,
        compile_s=compile_s,
        warm_runtime_samples_s=kernel_samples,
        warm_end_to_end_samples_s=tuple(end_to_end_samples),
        peak_memory_bytes=peak_memory,
        cpml_psi_precision=cpml_psi_precision,
        cuda_component_version=cuda_component_version,
        cuda_abi_version=cuda_abi_version,
        cuda_flags=int(program.config.cuda_flags),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=H100_WORKLOADS, default="realistic_3d")
    parser.add_argument("--shape", nargs=3, type=int, metavar=("NZ", "NY", "NX"))
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--backend",
        choices=("auto", "jax", "cuda", "cuda_streamed"),
        default="auto",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="number of JAX devices; values greater than one enable sharding",
    )
    parser.add_argument(
        "--shard-axis",
        choices=("auto", "z", "y", "x"),
        default="auto",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host-setup", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow a non-comparable smoke run when no accelerator is available",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.samples < 3:
        raise SystemExit("--samples must be at least three")
    if args.devices < 1:
        raise SystemExit("--devices must be positive")
    if args.timesteps is None:
        args.timesteps = H100_WORKLOADS[args.workload].timesteps
    record = run_benchmark(args)
    payload = json.dumps(record.as_dict(), indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
