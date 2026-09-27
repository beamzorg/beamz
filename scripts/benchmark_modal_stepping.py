#!/usr/bin/env python3
"""Measure realistic modal stepping without repeating public API construction.

Five synchronized samples follow one warmup. Full optical completion and public
API wall times belong to benchmark_device_completion.py and benchmark_h100.py.
"""

import argparse
import cProfile
import hashlib
import json
import os
import pstats
import resource
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_h100_backends import ModalWorkload

from beamz.simulation import observe, sharding
from beamz.simulation.execute import build_scan, initial_program_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", type=int, nargs=3, required=True)
    p.add_argument("--devices", type=int, required=True)
    p.add_argument("--backend", choices=["cuda_streamed", "jax"], required=True)
    p.add_argument("--frequencies", type=int, default=101)
    p.add_argument("--resolution-nm", type=float, default=80.0)
    p.add_argument("--timesteps", type=int, default=256)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--host-setup", action="store_true")
    p.add_argument("--donate-state", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--workload", choices=["modal_cpml12"], default="modal_cpml12")
    p.add_argument("--shard-axis", choices=["x", "y", "z"], default="x")
    p.add_argument("--compile-diagnostics", type=Path)
    p.add_argument("--trace", type=Path)
    p.add_argument("--profile-public", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.donate_state and (a.trace or a.profile_public):
        p.error("donating capacity runs do not support extra trace/public replays")
    devices = jax.devices()
    if (
        len(devices) != a.devices
        or any("H100" not in d.device_kind for d in devices)
        or a.samples < 5
        or a.timesteps < 32
        or min(a.shape) <= 24
        or a.frequencies < 1
        or not np.isfinite(a.resolution_nm)
        or a.resolution_nm <= 0
    ):
        p.error(
            "requires requested H100 count, >=5 samples, >=32 steps, dimensions >24"
        )
    started = time.perf_counter()
    cfg = (
        None
        if a.devices == 1
        else dict(axis=a.shard_axis, num_devices=a.devices, backend="gpu")
    )
    context = (
        jax.default_device(jax.devices("cpu")[0]) if a.host_setup else nullcontext()
    )
    with context:
        sim = ModalWorkload(
            shape_zyx=tuple(a.shape),
            timesteps=a.timesteps,
            frequencies=a.frequencies,
            resolution_nm=a.resolution_nm,
        ).build()
        sim.clear_compiled_cache()
        program = sim.compile(num_steps=a.timesteps, backend=a.backend, sharding=cfg)
        state = initial_program_state(
            program, t=float(sim.time[0]), current_step=0, monitor_steps=a.timesteps
        )
    if program.config.backend != a.backend:
        raise RuntimeError("Resolved backend differs from request")
    state = sharding.prepare_state(
        program, state, replicated_fields=(*observe.MONITOR_FIELDS, "t", "current_step")
    )
    coeffs = sharding.place_tree(program, program.coefficients)
    jax.block_until_ready((state, coeffs))
    prepared_memory = [dict(id=d.id, stats=d.memory_stats()) for d in devices]
    cuda_schedule = None
    if a.backend == "cuda_streamed" and hasattr(program.config, "cuda_memory_policy"):
        from beamz.simulation.memory import (
            cuda_capacity_schedule,
            cuda_workspace_estimate,
        )

        cuda_schedule = dict(
            policy=program.config.cuda_memory_policy,
            capacity=cuda_capacity_schedule(program, donate_state=a.donate_state),
            estimated_fast_workspace_bytes=cuda_workspace_estimate(
                program, donate_state=a.donate_state
            ),
        )
    setup_s = time.perf_counter() - started
    print(json.dumps(dict(stage="prepared", setup_s=setup_s)), flush=True)
    tick = time.perf_counter()
    if a.compile_diagnostics:
        a.compile_diagnostics.mkdir(parents=True, exist_ok=True)
        (a.compile_diagnostics / "prepared.json").write_text(
            json.dumps(
                dict(shape=a.shape, setup_s=setup_s, prepared_memory=prepared_memory),
                indent=2,
            )
            + "\n"
        )
    try:
        lowered = build_scan(program, donate_state=a.donate_state).lower(state, coeffs)
        if a.compile_diagnostics:
            # Elide literal payloads: a field-sized constant must not produce a
            # multi-gigabyte text dump while diagnosing near-capacity failures.
            module = lowered.compiler_ir()
            (a.compile_diagnostics / "lowered.mlir").write_text(
                module.operation.get_asm(
                    large_elements_limit=16, large_resource_limit=16
                )
            )
        exe = lowered.compile()
    except Exception as error:
        if a.compile_diagnostics:
            (a.compile_diagnostics / "failure.json").write_text(
                json.dumps(
                    dict(
                        error=repr(error),
                        elapsed_s=time.perf_counter() - tick,
                        device_memory=[
                            dict(id=d.id, stats=d.memory_stats()) for d in devices
                        ],
                    ),
                    indent=2,
                )
                + "\n"
            )
        raise
    compile_s = time.perf_counter() - tick
    analysis = exe.memory_analysis()
    compiled_memory = {
        key: getattr(analysis, key, None)
        for key in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
            "temp_size_in_bytes",
        )
    }
    print(json.dumps(dict(stage="compiled", compile_s=compile_s)), flush=True)
    warm = exe(state, coeffs)
    jax.block_until_ready(warm)
    if a.donate_state:
        state = warm
    del warm
    warm_memory = [dict(id=d.id, stats=d.memory_stats()) for d in devices]
    samples = []
    continuation_memory = []
    for sample in range(a.samples):
        tick = time.perf_counter()
        result = exe(state, coeffs)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - tick)
        if a.donate_state:
            state = result
            continuation_memory.append(
                [dict(id=d.id, stats=d.memory_stats()) for d in devices]
            )
        if sample < a.samples - 1:
            del result
    stepped_memory = [dict(id=d.id, stats=d.memory_stats()) for d in devices]
    # Fuse the reduction rather than materializing a field-sized boolean array
    # merely to validate a near-capacity simulation after timing.
    all_finite = jax.jit(lambda value: jnp.all(jnp.isfinite(value)))
    finite = all(bool(jax.device_get(all_finite(x))) for x in jax.tree.leaves(result))
    timed_memory = [dict(id=d.id, stats=d.memory_stats()) for d in devices]
    weights = np.asarray(jax.device_get(result.dft_weight_sum))
    final_current_step = int(jax.device_get(result.current_step))
    if (
        len(program.monitors) != 2
        or weights.size != 2 * a.frequencies
        or not np.all(weights > 0)
    ):
        raise RuntimeError("Both frequency monitors must accumulate nonzero weights")
    del result
    if not finite:
        raise RuntimeError("Non-finite stepping output")
    if a.trace:
        a.trace.mkdir(parents=True, exist_ok=True)
        (a.trace / "optimized-hlo.txt").write_text(exe.as_text())
        with jax.profiler.trace(str(a.trace), create_perfetto_trace=True):
            jax.block_until_ready(exe(state, coeffs))
    public_profile_s = None
    if a.profile_public:
        del state, coeffs, exe
        a.profile_public.parent.mkdir(parents=True, exist_ok=True)
        profiler = cProfile.Profile()
        tick = time.perf_counter()
        profiler.enable()
        public = sim.advance(
            num_steps=a.timesteps, backend=a.backend, sharding=cfg, performance=False
        )
        jax.block_until_ready(public.state)
        profiler.disable()
        public_profile_s = time.perf_counter() - tick
        profiler.dump_stats(str(a.profile_public))
        with a.profile_public.with_suffix(".txt").open("w") as stream:
            pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(80)
        del public
    memory = [dict(id=d.id, stats=d.memory_stats()) for d in devices]
    data = dict(
        protocol=(
            "five_donating_continuation_samples"
            if a.donate_state
            else "five_synchronized_warm_stepping_samples"
        ),
        donate_state=a.donate_state,
        cuda_schedule=cuda_schedule,
        compiled_memory=compiled_memory,
        final_current_step=final_current_step,
        warm_memory=warm_memory,
        continuation_memory=continuation_memory,
        backend=a.backend,
        devices=a.devices,
        shard_axis=a.shard_axis,
        shape=a.shape,
        steps=a.timesteps,
        frequencies=a.frequencies,
        cpml_cells=12,
        resolution_nm=a.resolution_nm,
        physical_size_zyx_um=(np.asarray(a.shape) * a.resolution_nm / 1000).tolist(),
        prepared_memory=prepared_memory,
        timed_memory=timed_memory,
        stepped_memory=stepped_memory,
        runtime_cv=statistics.stdev(samples) / statistics.mean(samples),
        monitor_weight_min=float(weights.min()),
        monitor_weight_max=float(weights.max()),
        monitor_count=len(program.monitors),
        setup_s=setup_s,
        compile_s=compile_s,
        warm_runtime_samples_s=samples,
        kernel_gcups=int(np.prod(a.shape))
        * a.timesteps
        / statistics.median(samples)
        / 1e9,
        final_state_finite=finite,
        public_api_measured=False,
        diagnostic_profiled_public_call_s=public_profile_s,
        peak_memory_bytes=sum(
            (d["stats"] or {}).get("peak_bytes_in_use", 0) for d in memory
        ),
        device_memory=memory,
        worker_measurement_wall_s=time.perf_counter() - started,
        host_max_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if sys.platform == "darwin" else 1024),
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        environment={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("XLA_", "BEAMZ_CUDA_", "NCCL_", "NUMPY_MADVISE"))
        },
    )
    if a.backend == "cuda_streamed":
        import beamz._cuda as native

        data["native_sha256"] = hashlib.sha256(
            Path(native.__file__).read_bytes()
        ).hexdigest()
        data["cuda_flags"] = int(program.config.cuda_flags)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in data.items() if k not in {"device_memory", "environment"}}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
