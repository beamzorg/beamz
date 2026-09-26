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
import statistics
import subprocess
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
    p.add_argument("--timesteps", type=int, default=256)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--host-setup", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--workload", choices=["modal_cpml12"], default="modal_cpml12")
    p.add_argument("--shard-axis", choices=["x"], default="x")
    p.add_argument("--trace", type=Path)
    p.add_argument("--profile-public", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    devices = jax.devices()
    if (
        len(devices) != a.devices
        or any("H100" not in d.device_kind for d in devices)
        or a.samples < 5
        or a.timesteps < 32
        or min(a.shape) <= 24
        or a.frequencies < 1
    ):
        p.error(
            "requires requested H100 count, >=5 samples, >=32 steps, dimensions >24"
        )
    started = time.perf_counter()
    cfg = (
        None if a.devices == 1 else dict(axis="x", num_devices=a.devices, backend="gpu")
    )
    context = (
        jax.default_device(jax.devices("cpu")[0]) if a.host_setup else nullcontext()
    )
    with context:
        sim = ModalWorkload(
            shape_zyx=tuple(a.shape), timesteps=a.timesteps, frequencies=a.frequencies
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
    setup_s = time.perf_counter() - started
    print(json.dumps(dict(stage="prepared", setup_s=setup_s)), flush=True)
    tick = time.perf_counter()
    exe = build_scan(program, donate_state=False).lower(state, coeffs).compile()
    compile_s = time.perf_counter() - tick
    print(json.dumps(dict(stage="compiled", compile_s=compile_s)), flush=True)
    warm = exe(state, coeffs)
    jax.block_until_ready(warm)
    del warm
    samples = []
    for sample in range(a.samples):
        tick = time.perf_counter()
        result = exe(state, coeffs)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - tick)
        if sample < a.samples - 1:
            del result
    finite = all(
        bool(jax.device_get(jnp.all(jnp.isfinite(x)))) for x in jax.tree.leaves(result)
    )
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
        protocol="five_synchronized_warm_stepping_samples",
        backend=a.backend,
        devices=a.devices,
        shape=a.shape,
        steps=a.timesteps,
        frequencies=a.frequencies,
        cpml_cells=12,
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
