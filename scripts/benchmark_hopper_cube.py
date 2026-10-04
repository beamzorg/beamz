"""Fresh-process H100 cube capacity/throughput trial with fixed CPML12 physics."""

import argparse
import faulthandler
import gc
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace


def main():
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side", type=int, required=True)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-device", default="H100")
    p.add_argument(
        "--state-output", type=Path, help="Save final leaves for small parity cases"
    )
    a = p.parse_args()
    if a.side <= 24 or a.samples < 3 or a.steps < 1:
        p.error("Need nonempty interior, >=3 samples and positive steps")
    if a.output.exists():
        p.error("Use a fresh output path")
    os.environ.update(
        BEAMZ_CUDA_AUTOTUNE="off",
        BEAMZ_CUDA_CPML_PSI_PRECISION="fp32",
        BEAMZ_CUDA_TEMPORAL_STEPS="1",
        BEAMZ_CUDA_CPML_TEMPORAL="0",
        BEAMZ_CUDA_CPML_SPATIAL="0",
        BEAMZ_CUDA_FIELD_PADDING="none",
        BEAMZ_CUDA_CPML_SHELL_TILE="32x4",
        BEAMZ_CUDA_STORAGE_AXES="012",
        BEAMZ_CUDA_CPML_CORE_FUSION="0",
    )
    import beamz._cuda as extension
    import jax
    import numpy as np

    from beamz.simulation.cuda import runtime
    from beamz.simulation.execute import build_scan, initial_program_state
    from scripts.benchmark_cuda_realistic import build_simulation

    device = jax.devices()[0]
    assert len(jax.devices()) == 1 and a.expected_device in device.device_kind
    data = dict(
        side=a.side,
        shape=[a.side] * 3,
        cells=a.side**3,
        steps=a.steps,
        pml=12,
        source="mode",
        monitors=1,
        monitor_type="mode",
        frequencies=3,
        field_precision="fp32",
        cpml_precision="fp32",
        shell_tile="32x4",
        storage_axes="012",
        fusion=False,
        donate_state=True,
        fresh_initial_state_per_sample=True,
        warmups=3,
        native_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        jax_version=jax.__version__,
        device=device.device_kind,
        allocator_env={
            k: v for k, v in os.environ.items() if k.startswith("XLA_PYTHON_CLIENT")
        },
        samples_s=[],
        sample_windows_unix=[],
        stages=[],
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    start_all = time.perf_counter()

    def save(stage):
        data["stage"] = stage
        data["elapsed_s"] = time.perf_counter() - start_all
        data["host_peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        snapshot = dict(
            stage=stage,
            unix_time=time.time(),
            elapsed_s=data["elapsed_s"],
            memory_stats=device.memory_stats() or {},
        )
        snapshot["gpu"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        data["stages"].append(snapshot)
        a.output.write_text(json.dumps(data, indent=2) + "\n")
        print(stage, round(data["elapsed_s"], 2), snapshot["gpu"], flush=True)

    def report_failure(kind, value, traceback):
        data["status"] = "failed"
        data["error"] = f"{kind.__name__}: {value}"
        with suppress(Exception):
            save("failed")
        sys.__excepthook__(kind, value, traceback)

    sys.excepthook = report_failure
    save("building")
    sim = build_simulation(
        SimpleNamespace(
            shape=[a.side] * 3,
            steps=a.steps,
            presteps=0,
            pml=12,
            monitors=1,
            frequencies=3,
            material="binary",
            source="mode",
            monitor_type="mode",
        )
    )
    save("simulation_built")
    program = sim.compile(num_steps=a.steps, backend="cuda_streamed")

    def fresh():
        state = initial_program_state(
            program, t=0, current_step=0, monitor_steps=a.steps
        )
        return jax.block_until_ready(state)

    state = fresh()
    data["setup_s"] = time.perf_counter() - start_all
    save("program_built")
    plans = []
    choose = runtime._native_schedule_plan

    def record(*args, **kwargs):
        plan = choose(*args, **kwargs)
        plans.append(asdict(plan))
        return plan

    runtime._native_schedule_plan = record
    start = time.perf_counter()
    try:
        executable = (
            build_scan(program, donate_state=True)
            .lower(state, program.coefficients)
            .compile()
        )
    finally:
        runtime._native_schedule_plan = choose
    data["compile_s"] = time.perf_counter() - start
    data["native_plans"] = plans
    analysis = executable.memory_analysis()
    data["executable_memory"] = {
        k: int(getattr(analysis, k))
        for k in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
            "temp_size_in_bytes",
            "generated_code_size_in_bytes",
        )
    }
    save("compiled")
    for i in range(3 + a.samples):
        if i:
            state = fresh()
        begin_unix = time.time()
        begin = time.perf_counter()
        result = executable(state, program.coefficients)
        jax.block_until_ready(result)
        seconds = time.perf_counter() - begin
        if i >= 3:
            data["samples_s"].append(seconds)
            data["sample_windows_unix"].append([begin_unix, time.time()])
            data["gcups"] = (
                a.side**3 * a.steps / statistics.median(data["samples_s"]) / 1e9
            )
        save(f"iteration_{i}")
        if i == 2 + a.samples:
            finite = True
            max_abs_fields = {}
            saved_leaves = {}

            def host_chunks(leaf):
                # np.asarray(leaf) caches a host copy on the JAX array itself.
                # Slice large leaves so validation cannot retain a second full
                # state in host RAM even after the local NumPy reference dies.
                if not leaf.ndim or a.state_output:
                    yield np.asarray(leaf)
                    return
                plane_bytes = max(1, int(np.prod(leaf.shape[1:])) * leaf.dtype.itemsize)
                stride = max(1, (4 << 20) // plane_bytes)
                for start in range(0, leaf.shape[0], stride):
                    yield np.asarray(leaf[start : start + stride])

            for leaf_index, leaf in enumerate(jax.tree.leaves(result)):
                for host in host_chunks(leaf):
                    finite = finite and bool(np.isfinite(host).all())
                    if a.state_output:
                        saved_leaves[f"leaf_{leaf_index}"] = host
                    del host
            if a.state_output:
                np.savez_compressed(a.state_output, **saved_leaves)
            for name in ("ex", "ey", "ez", "hx", "hy", "hz"):
                max_abs_fields[name] = max(
                    float(np.max(np.abs(host)))
                    for host in host_chunks(getattr(result, name))
                )
            data["finite_complete_state"] = finite
            data["max_abs_fields"] = max_abs_fields
            data["final_step"] = int(result.current_step)
            assert finite and data["final_step"] == a.steps
        del result, state
        gc.collect()
    data["status"] = "complete"
    save("complete")
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
