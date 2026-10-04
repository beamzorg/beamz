#!/usr/bin/env python3
"""Isolated, fixed-geometry cosine-crossing resolution/capacity benchmark.

Run from the repository root. CUDA, native rasterizer and GPU JAX are required.
Each trial is a new process. Results and failure logs are checkpointed immediately.
The geometry and monitor definitions match cosine_waveguide_crossing.ipynb;
CPML thickness is held at its notebook reference value throughout refinement.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path


def crossing(resolution_nm: float):
    import numpy as np
    from scipy.optimize import fsolve
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    import beamz as bz

    um = bz.um
    h, w_in, w_out, w_m, l_t = np.array([0.161, 0.35, 1.1, 0.75, 5.3]) * um
    wavelength = 1.31 * um
    freq0 = bz.LIGHT_SPEED / wavelength
    dx = resolution_nm * 1e-9
    lengths = [13.7 * um, 13.7 * um, 1.5 * wavelength]
    domain = tuple(np.nextafter(np.ceil(x / dx) * dx, np.inf) for x in lengths)
    lx, ly, _ = domain

    def equations(ab):
        a, b = ab
        return (
            w_m * np.cos(a * (-w_out / 2) + b) - w_out / 2,
            w_m * np.cos(a * (-w_out / 2 - l_t) + b) - w_in / 2,
        )

    a, b = fsolve(equations, (0.5 / um, 2.0))
    x = np.linspace(-w_out / 2 - l_t, -w_out / 2, 30)
    w = w_m * np.cos(a * x + b)
    left = list(zip(np.r_[x, x[::-1]], np.r_[w, -w[::-1]], strict=False))
    pieces = [
        Polygon(v)
        for v in (
            left,
            [(-x, y) for x, y in left],
            [(y, -x) for x, y in left],
            [(y, x) for x, y in left],
        )
    ]
    pieces += [
        box(-w_out / 2, -w_out / 2, w_out / 2, w_out / 2),
        box(-lx / 2, -w_in / 2, lx / 2, w_in / 2),
        box(-w_in / 2, -ly / 2, w_in / 2, ly / 2),
    ]
    geometry = unary_union(pieces)
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    polygons = [geometry] if geometry.geom_type == "Polygon" else geometry.geoms
    design = bz.Design(background=bz.Material(permittivity=1.45**2))
    for polygon in polygons:

        def vertices(ring):
            return [(float(x), float(y), -h / 2) for x, y in list(ring.coords)[:-1]]

        design += bz.Polygon(
            vertices=vertices(polygon.exterior),
            interiors=[vertices(r) for r in polygon.interiors],
            material=bz.Material(permittivity=3.67**2),
            depth=h,
            z=-h / 2,
        )
    frequencies = bz.LIGHT_SPEED / (np.linspace(1.26, 1.36, 101) * um)
    source = bz.ModeSource(
        center=(-lx / 2 + 0.5 * um, 0, 0),
        size=(0, 4 * w_in, 4 * h),
        direction="+",
        source_time=bz.GaussianPulse(freq0=freq0, fwidth=freq0 / 10, offset=4.0),
        mode_spec=bz.ModeSpec(num_modes=1, target_neff=3.455, polarization="te"),
        power=1.0,
    )
    monitors = [
        bz.FieldMonitor(
            center=(0, 0, 0),
            size=(lx, ly, 0),
            freqs=[freq0],
            fields=("Ex", "Ey", "Ez"),
            name="field",
        ),
        bz.FluxMonitor(
            center=(lx / 2 - 0.5 * um, 0, 0),
            size=(0, 4 * w_in, 4 * h),
            freqs=frequencies,
            name="flux_through",
        ),
        bz.FluxMonitor(
            center=(0, ly / 2 - 0.5 * um, 0),
            size=(4 * w_in, 0, 4 * h),
            freqs=frequencies,
            name="flux_cross",
        ),
    ]
    return bz.Simulation(
        domain=domain,
        grid_spec=bz.GridSpec.uniform(dx, max_total_cells=None),
        design=design,
        sources=[source],
        monitors=monitors,
        run_time=1e-12,
        boundaries=[
            bz.PML(
                thickness=12 * wavelength / (10 * 3.67),
                formulation="cpml",
                m=3,
                kappa_max=3.0,
                alpha_max=0.0,
            )
        ],
    )


def stage(name):
    print(json.dumps({"stage": name, "elapsed_epoch": time.time()}), flush=True)


def child(args):
    import faulthandler

    faulthandler.dump_traceback_later(120, repeat=True)
    import jax
    import jaxlib
    import numpy as np

    import beamz
    from beamz.simulation import observe
    from beamz.simulation import sharding as placement
    from beamz.simulation.backend import cuda_backend_status
    from beamz.simulation.execute import build_scan, initial_program_state

    devices = jax.devices("gpu")
    assert len(devices) == args.devices and all(
        "H100" in d.device_kind for d in devices
    )
    device = devices[0]
    sharding = (
        None
        if args.devices == 1
        else dict(axis=args.shard_axis, num_devices=args.devices, backend="gpu")
    )
    stage("geometry")
    start = time.perf_counter()
    sim = crossing(args.resolution)
    geometry_s = time.perf_counter() - start
    steps = sim.num_steps if args.mode == "full" else args.steps
    stage("rasterization")
    start = time.perf_counter()
    sim.to_request(num_steps=steps, backend=args.backend, sharding=sharding)
    raster_s = time.perf_counter() - start
    stage("mode_and_program_setup")
    start = time.perf_counter()
    program = sim.compile(num_steps=steps, backend=args.backend, sharding=sharding)
    assert program.sharding.layout.num_devices == args.devices
    setup_s = time.perf_counter() - start
    shape = tuple(int(n) for n in program.grid.permittivity.shape)
    stage("state_allocation")
    coeffs = placement.place_tree(program, program.coefficients)

    def new_state():
        state = initial_program_state(
            program, t=float(sim.time[0]), current_step=0, monitor_steps=steps
        )
        if args.devices > 1:
            state = placement.prepare_state(
                program,
                state,
                replicated_fields=(*observe.MONITOR_FIELDS, "t", "current_step"),
            )
        return jax.block_until_ready(state)

    state = new_state()
    scan = build_scan(program, donate_state=True)
    stage("trace_lower")
    start = time.perf_counter()
    lowered = scan.lower(state, coeffs)
    lower_s = time.perf_counter() - start
    stage("xla_compile")
    start = time.perf_counter()
    executable = lowered.compile()
    compile_s = time.perf_counter() - start
    stage("execution")
    samples = []
    count = 1 if args.mode == "full" else args.samples + 1
    result = None
    for index in range(count):
        if index:
            del result
            gc.collect()
            state = new_state()
        start = time.perf_counter()
        result = executable(state, coeffs)
        jax.block_until_ready(result)
        duration = time.perf_counter() - start
        if index or args.mode == "full":
            samples.append(duration)
        print(json.dumps({"sample": index, "seconds": duration}), flush=True)
    metrics = {
        "mode": args.mode,
        "resolution_nm": args.resolution,
        "backend": args.backend,
        "grid_zyx": shape,
        "cells": math.prod(shape),
        "steps": steps,
        "physical_run_time_s": float(sim.run_time),
        "full_steps": sim.num_steps,
        "geometry_s": geometry_s,
        "raster_mode_program_setup_s": raster_s + setup_s,
        "rasterization_s": raster_s,
        "mode_and_program_setup_s": setup_s,
        "trace_lower_s": lower_s,
        "xla_compile_s": compile_s,
        "runtime_samples_s": samples,
        "median_runtime_s": statistics.median(samples),
        "gcups": math.prod(shape) * steps / statistics.median(samples) / 1e9,
        "device": device.device_kind,
        "device_count": len(devices),
        "shard_axis": args.shard_axis if args.devices > 1 else None,
        "nccl_env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "beamz_version": beamz.__version__,
        "field_dtype": str(result.ex.dtype),
        "cpml_psi_precision": (
            str(result.cpml_psi_h_terms[0].dtype)
            if result.cpml_psi_h_terms
            else "float32"
        ),
        "cuda_flags": int(program.config.cuda_flags),
        "numpy_version": np.__version__,
        "numpy_madvise_hugepage": os.environ.get("NUMPY_MADVISE_HUGEPAGE", "default"),
        "donate_state": True,
        "memory_stats": device.memory_stats(),
        "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "finite_fields": bool(np.isfinite(np.asarray(result.ez)).all()),
    }
    if args.backend.startswith("cuda"):
        import beamz._cuda as extension

        status = cuda_backend_status(register=False)
        metrics.update(
            cuda_component_version=status.extension_version,
            cuda_abi_version=status.abi_version,
            extension_sha256=hashlib.sha256(
                Path(extension.__file__).read_bytes()
            ).hexdigest(),
        )
    if args.mode == "full":
        stage("result_extraction")
        from beamz.simulation.execute import (
            _compiled_source_launch_powers,
            _decode_monitor_results,
        )
        from beamz.simulation.results import SimulationResults

        start = time.perf_counter()
        results = SimulationResults.from_run(
            sim,
            runtime_fields=program.grid,
            monitor_results=_decode_monitor_results(sim, program, result),
            store_full_materials=False,
            source_launch_powers=_compiled_source_launch_powers(program, 1),
        )
        through = np.asarray(results["flux_through"].flux, dtype=float)
        cross = np.asarray(results["flux_cross"].flux, dtype=float)
        metrics["flux_through"] = through.tolist()
        metrics["flux_cross"] = cross.tolist()
        source_power = results.launched_power(0)
        metrics["source_launch_power"] = source_power
        metrics["transmission_db"] = (
            10 * np.log10(np.maximum(through / float(source_power), 1e-30))
        ).tolist()
        cross_ratio = np.divide(
            cross, through, out=np.zeros_like(cross), where=np.abs(through) >= 1e-30
        )
        metrics["crosstalk_db"] = (
            10 * np.log10(np.maximum(cross_ratio, 1e-30))
        ).tolist()
        metrics["result_extraction_s"] = time.perf_counter() - start
        metrics["finite_fields"] = all(
            np.isfinite(np.asarray(getattr(result, name))).all()
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        )
        metrics["negative_flux_bins"] = {
            "through": int((through < 0).sum()),
            "cross": int((cross < 0).sum()),
        }
    metrics["host_peak_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    metrics["memory_stats"] = device.memory_stats()
    metrics["finite_fields_checked"] = (
        ["ex", "ey", "ez", "hx", "hy", "hz"] if args.mode == "full" else ["ez"]
    )
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    faulthandler.cancel_dump_traceback_later()
    stage("complete")


def trial(args, resolution, mode="throughput", backend=None):
    backend = backend or args.backend
    stem = f"{mode}-{backend}-{resolution:g}nm"
    if args.devices > 1:
        stem += f"-{args.devices}gpu-{args.shard_axis}"
    target = args.output_dir / (stem + ".json")
    if target.exists():
        raise FileExistsError(
            f"Refusing to reuse {target}; choose a fresh output directory for this run."
        )
    log = args.output_dir / (stem + ".log")
    env = dict(
        os.environ,
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION=".95",
        BEAMZ_DISABLE_JAX_PERSISTENT_CACHE="1",
        BEAMZ_RASTER_CACHE="0",
        BEAMZ_CUDA_CPML_PSI_PRECISION="fp32",
        MPLBACKEND="Agg",
        NUMPY_MADVISE_HUGEPAGE="0",
        OPENBLAS_NUM_THREADS="4",
        OMP_NUM_THREADS="8",
    )
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--resolution",
        str(resolution),
        "--mode",
        mode,
        "--backend",
        backend,
        "--devices",
        str(args.devices),
        "--shard-axis",
        args.shard_axis,
        "--steps",
        str(args.steps),
        "--samples",
        str(args.samples),
        "--output",
        str(target),
    ]
    print(f"Starting {stem}", flush=True)
    start = time.monotonic()
    gpu_peak = 0
    host_peak = 0
    cgroup = Path("/sys/fs/cgroup")

    def oom_kills():
        path = cgroup / "memory.events"
        if not path.exists():
            return 0
        values = dict(line.split() for line in path.read_text().splitlines())
        return int(values.get("oom_kill", 0))

    prior_oom_kills = oom_kills()
    timed_out = False
    with log.open("w") as stream:
        proc = subprocess.Popen(command, env=env, stdout=stream, stderr=stream)
        while proc.poll() is None:
            current = cgroup / "memory.current"
            if current.exists():
                host_peak = max(host_peak, int(current.read_text()))
            if time.monotonic() - start > args.timeout:
                proc.kill()
                timed_out = True
                break
            smi = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if smi.returncode == 0:
                gpu_peak = max(gpu_peak, int(float(smi.stdout.strip())) * 2**20)
            time.sleep(0.25)
        code = proc.wait()
    if target.exists():
        record = json.loads(target.read_text())
        record.update(
            status="ok",
            nvidia_smi_peak_bytes=gpu_peak,
            cgroup_peak_sampled_bytes=host_peak,
            child_wall_s=time.monotonic() - start,
        )
    else:
        message = log.read_text()
        gpu_oom = any(
            x in message.lower()
            for x in [
                "cuda_error_out_of_memory",
                "resource_exhausted",
                "gpu memory",
                "cublas_status_alloc_failed",
            ]
        )
        record = dict(
            status=(
                "host_oom"
                if oom_kills() > prior_oom_kills
                else "gpu_oom"
                if gpu_oom
                else "timeout"
                if timed_out
                else "error"
            ),
            returncode=code,
            resolution_nm=resolution,
            mode=mode,
            backend=backend,
            nvidia_smi_peak_bytes=gpu_peak,
            cgroup_peak_sampled_bytes=host_peak,
            detail=message[-5000:],
        )
    target.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{stem}: {record['status']}, peak {gpu_peak / 2**30:.2f} GiB", flush=True)
    return record


def sweep(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    last_ok = None
    first_oom = None
    for resolution in args.resolutions:
        record = trial(args, resolution)
        records.append(record)
        if record["status"] == "ok":
            last_ok = resolution
        elif record["status"] == "gpu_oom":
            first_oom = resolution
            break
        else:
            break
    if first_oom is not None and last_ok is not None:
        for _ in range(args.refinements):
            middle = round((first_oom + last_ok) / 2, 3)
            record = trial(args, middle)
            records.append(record)
            if record["status"] == "ok":
                last_ok = middle
            elif record["status"] == "gpu_oom":
                first_oom = middle
            else:
                break
    for resolution in args.full_resolutions:
        records.append(trial(args, resolution, mode="full"))
    summary = dict(
        records=records,
        last_success_nm=last_ok,
        first_gpu_oom_nm=first_oom,
        protocol=dict(
            steps=args.steps,
            warmups=1,
            samples=args.samples,
            pml_thickness_m=12 * 1.31e-6 / (10 * 3.67),
            flux_frequencies=101,
            field_frequencies=1,
            nominal_domain_m=[13.7e-6, 13.7e-6, 1.965e-6],
        ),
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    good = [r for r in records if r["status"] == "ok"]
    columns = [
        "mode",
        "backend",
        "resolution_nm",
        "cells",
        "steps",
        "gcups",
        "median_runtime_s",
        "raster_mode_program_setup_s",
        "xla_compile_s",
        "nvidia_smi_peak_bytes",
        "host_peak_rss_bytes",
    ]
    with (args.output_dir / "measurements.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(good)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--resolution", type=float, default=36.0)
    parser.add_argument(
        "--resolutions",
        type=float,
        nargs="+",
        default=[36, 30, 25, 21, 18, 15, 12.5, 10.5, 9, 7.5, 6],
    )
    parser.add_argument(
        "--full-resolutions", type=float, nargs="*", default=[36, 25, 18]
    )
    parser.add_argument("--refinements", type=int, default=3)
    parser.add_argument("--mode", choices=["throughput", "full"], default="throughput")
    parser.add_argument(
        "--backend",
        choices=["cuda_streamed", "jax"],
        default="cuda_streamed",
    )
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--devices", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--shard-axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--output", type=Path, default=Path("trial.json"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/results/cosine-h100")
    )
    args = parser.parse_args()
    child(args) if args.child else sweep(args)


if __name__ == "__main__":
    main()
