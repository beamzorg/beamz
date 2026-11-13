#!/usr/bin/env python3
"""CUDA nanoparticle throughput benchmark inspired by the FDTDX Mie post.

This is NOT an exact reproduction: the post does not specify GPU, particle,
illumination, duration, boundary thickness, monitors, or software revision.
Defaults here are explicit assumptions: a 680 nm cube INCLUDING 12 CPML cells,
100 nm radius, nondispersive n=2 sphere in vacuum, a finite-aperture +x plane
wave at 500 nm (30% Gaussian bandwidth, peak at 1/bandwidth), six field-DFT
box faces, and 30 fs. CPML is always 12 cells per face at every resolution;
its physical thickness therefore changes with grid spacing.
The binary cell-center sphere has no subpixel smoothing. This runner measures
a Mie-like workload, not an analytically validated scattering cross section.

Each resolution/schedule starts in a fresh process. Only CUDA schedules are
allowed. BeamZ's existing JAX FFI host launches the native CUDA engine; there
is no JAX numerical baseline. Setup, lowering/compilation, synchronized warm
executable timing, synchronized public calls, and first-call time are separate.
The first-call metric starts at scene construction (excludes Python imports).

Examples (run from checkout with a CUDA-enabled BeamZ Python environment)::

    python scripts/benchmark_cuda_mie.py --dry-run --output /tmp/mie-plan.json
    python scripts/benchmark_cuda_mie.py --cube-nm 800 --resolutions-nm 10 --steps 32 \
        --samples 3 --warmups 1 --validate-schedules --output /tmp/mie-smoke.json
    python scripts/benchmark_cuda_mie.py --output /tmp/mie-sweep.json

Full 1 nm cases may exhaust memory; structured OOM records remain in output.
Use --steps only for kernel experiments: it changes the simulated duration,
so those runs must not be compared with fixed-duration resolution curves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

LIGHT_SPEED = 299792458.0
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIELDS = ("ex", "ey", "ez", "hx", "hy", "hz")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resolutions-nm", nargs="+", type=float, default=[5, 4, 2.5, 2, 1])
    p.add_argument(
        "--schedules",
        nargs="+",
        choices=["streamed", "mstep"],
        default=["streamed", "mstep"],
    )
    p.add_argument("--cube-nm", type=float, default=680)
    p.add_argument("--radius-nm", type=float, default=100)
    p.add_argument("--particle-index", type=float, default=2)
    p.add_argument(
        "--pml-cells",
        type=int,
        choices=[12],
        default=12,
        help="Benchmark contract: exactly 12 CPML cells per face at every resolution",
    )
    p.add_argument("--wavelength-nm", type=float, default=500)
    p.add_argument("--bandwidth-fraction", type=float, default=0.3)
    p.add_argument(
        "--pulse-offset",
        type=float,
        default=1,
        help="Gaussian pulse peak measured in inverse-bandwidth units",
    )
    p.add_argument("--duration-fs", type=float, default=30)
    p.add_argument(
        "--courant",
        type=float,
        default=0.95,
        help="Fraction of the 3D vacuum CFL limit",
    )
    p.add_argument("--steps", type=int)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--public-samples", type=int, default=3)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--monitor-faces", type=int, choices=[0, 6], default=6)
    p.add_argument(
        "--validate-schedules",
        action="store_true",
        help="Outside timing compare every state leaf against streamed CUDA",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--prepare-backend", choices=("cuda_streamed", "cuda_mstep"))
    return p


def case_spec(args, resolution_nm):
    """Compute physical geometry and work without allocating any field arrays."""
    positives = [
        resolution_nm,
        args.cube_nm,
        args.radius_nm,
        args.particle_index,
        args.wavelength_nm,
        args.bandwidth_fraction,
        args.duration_fs,
    ]
    if not all(math.isfinite(x) and x > 0 for x in positives):
        raise ValueError(
            "Physical dimensions and optical parameters must be positive and finite"
        )
    if not 0 < args.courant <= 1:
        raise ValueError("courant must be in (0, 1]")
    cells = round(args.cube_nm / resolution_nm)
    if cells < 8 or not math.isclose(cells * resolution_nm, args.cube_nm, abs_tol=1e-8):
        raise ValueError("cube-nm must be an integer multiple of each resolution")
    if args.pml_cells != 12:
        raise ValueError("Benchmark requires exactly 12 CPML cells per face")
    if not math.isfinite(args.pulse_offset) or args.pulse_offset <= 0:
        raise ValueError("pulse-offset must be positive and finite")
    pml_cells = 12
    pml_nm = pml_cells * resolution_nm
    # Surface encloses the sphere, source is upstream of that surface, both
    # remain outside CPML. Fixed physical positions across the resolution sweep.
    half_box_nm = args.radius_nm + 60
    source_x_nm = args.cube_nm / 2 - half_box_nm - 40
    if source_x_nm <= pml_nm + 2 * resolution_nm:
        raise ValueError("Source, monitor box and sphere need more space outside CPML")
    dt = args.courant * resolution_nm * 1e-9 / (LIGHT_SPEED * math.sqrt(3))
    steps = (
        args.steps
        if args.steps is not None
        else math.ceil(args.duration_fs * 1e-15 / dt)
    )
    if steps < 2:
        raise ValueError("Need at least two timesteps")
    return {
        "resolution_nm": resolution_nm,
        "shape_zyx": [cells] * 3,
        "material_cells": cells**3,
        "timesteps": steps,
        "dt_s": dt,
        "actual_duration_fs": steps * dt * 1e15,
        "duration_policy": "fixed_steps"
        if args.steps is not None
        else "fixed_physical_time",
        "cube_nm_including_pml": args.cube_nm,
        "pml_cells": pml_cells,
        "realized_pml_nm": pml_nm,
        "radius_nm": args.radius_nm,
        "pml_policy": "fixed_cells",
        "particle_index": args.particle_index,
        "background_index": 1,
        "wavelength_nm": args.wavelength_nm,
        "bandwidth_fraction": args.bandwidth_fraction,
        "pulse_offset": args.pulse_offset,
        "source_x_nm": source_x_nm,
        "monitor_half_box_nm": half_box_nm,
        "monitor_faces": args.monitor_faces,
        "frequencies": 1,
        "field_precision": "float32",
        "rasterization": "binary_cell_centers",
        # Yee components differ slightly in extent; this is deliberately a
        # lower bound, NOT a memory-admission estimate or peak prediction.
        "six_cell_center_fields_lower_bound_bytes": cells**3 * 6 * 4,
    }


def build_simulation(args, spec):
    import numpy as np

    import beamz as bz
    from beamz.design import MaterialGrid

    dx = spec["resolution_nm"] * 1e-9
    n = spec["shape_zyx"][0]
    cube = args.cube_nm * 1e-9
    distance = ((np.arange(n, dtype=np.float64) + 0.5) * dx - cube / 2) ** 2
    eps = np.ones((n, n, n), dtype=np.float32)
    # Rasterize one plane at a time to avoid a full float64 distance volume.
    transverse_distance = distance[:, None] + distance[None, :]
    for z in range(n):
        eps[z][transverse_distance + distance[z] <= (args.radius_nm * 1e-9) ** 2] = (
            args.particle_index**2
        )
    grid = MaterialGrid(
        permittivity=eps,
        conductivity=np.float32(0),
        permeability=np.float32(1),
        resolution=dx,
        shape=eps.shape,
    )
    frequency = bz.LIGHT_SPEED / (args.wavelength_nm * 1e-9)
    pml = spec["realized_pml_nm"] * 1e-9
    center = cube / 2
    aperture = cube - 2 * pml
    source = bz.PlaneWaveSource(
        center=(spec["source_x_nm"] * 1e-9, center, center),
        size=(0, aperture, aperture),
        direction="+x",
        source_time=bz.GaussianPulse(
            frequency, frequency * args.bandwidth_fraction, offset=args.pulse_offset
        ),
        background_index=1,
    )
    monitors = []
    half_box = spec["monitor_half_box_nm"] * 1e-9
    if args.monitor_faces:
        for axis in range(3):
            for sign in (-1, 1):
                position = [center] * 3
                position[axis] += sign * half_box
                extent = [2 * half_box] * 3
                extent[axis] = 0
                monitors.append(
                    bz.FieldMonitor(
                        center=tuple(position),
                        size=tuple(extent),
                        freqs=[frequency],
                        fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                        name=f"box_{'xyz'[axis]}_{'minus' if sign < 0 else 'plus'}",
                    )
                )
    return bz.Simulation(
        material_grid=grid,
        size=(cube,) * 3,
        sources=[source],
        monitors=monitors,
        boundaries=[bz.PML(edges="all", thickness=pml, formulation="cpml")],
        time=np.arange(spec["timesteps"]) * spec["dt_s"],
    )


def gpu_status():
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,memory.used,"
                "utilization.gpu,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu",
                "--format=csv",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source_provenance():
    paths = sorted([*ROOT.glob("beamz/**/*.py"), *ROOT.glob("cuda/src/*")])
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return {
        "source_sha256": digest.hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "python": platform.python_version(),
    }


def compare_states(reference, candidate):
    """Full-state numerical gate, including CPML, DFT accumulators and clocks."""
    import jax
    import numpy as np

    if jax.tree_util.tree_structure(reference) != jax.tree_util.tree_structure(
        candidate
    ):
        raise ValueError("CUDA state structures differ")
    worst = 0.0
    for expected, actual in zip(
        jax.tree.leaves(reference), jax.tree.leaves(candidate), strict=True
    ):
        a, b = np.asarray(expected), np.asarray(actual)
        if a.shape != b.shape:
            raise ValueError("CUDA state shapes differ")
        if np.issubdtype(a.dtype, np.inexact):
            error = float(np.max(np.abs(a - b), initial=0))
            scale = float(np.max(np.abs(a), initial=0))
            if (
                not np.isfinite(a).all()
                or not np.isfinite(b).all()
                or error > 1e-5 * scale + 1e-30
            ):
                raise ValueError(
                    f"CUDA state parity failed: error={error}, scale={scale}"
                )
            worst = max(worst, error / max(scale, 1e-30))
        elif not np.array_equal(a, b):
            raise ValueError("CUDA state integer/clock parity failed")
    return {
        "passed": True,
        "max_leaf_relative_error": worst,
        "relative_tolerance": 1e-5,
        "absolute_tolerance": 1e-30,
    }


def run_worker(args, spec, record):
    import jax
    import numpy as np

    from beamz.simulation.backend import cuda_backend_status
    from beamz.simulation.execute import build_scan, initial_program_state
    from scripts.benchmark_cuda_revision import DeviceMemorySampler

    status = cuda_backend_status()
    if not status.available:
        raise RuntimeError(f"CUDA unavailable: {status.reason}")
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("CUDA benchmark requires a GPU; CPU fallback is forbidden")
    import beamz._cuda as extension

    record.update(source_provenance())
    record.update(
        {
            "device": str(jax.devices()[0]),
            "gpu_before": gpu_status(),
            "cuda_component": extension.__version__,
            "cuda_abi": extension.__abi_version__,
            "native_binary_sha256": hashlib.sha256(
                Path(extension.__file__).read_bytes()
            ).hexdigest(),
            "environment": {
                k: v
                for k, v in os.environ.items()
                if k.startswith(("BEAMZ_CUDA_", "XLA_", "JAX_"))
            },
        }
    )
    memory_sampler = DeviceMemorySampler(1)
    start = time.perf_counter()
    sim = build_simulation(args, spec)
    record["scene_build_s"] = time.perf_counter() - start
    kwargs = {
        "num_steps": spec["timesteps"],
        "backend": "cuda",
        "schedule": args.schedules[0],
    }
    prepared = None
    if args.prepare_backend:
        preparation_start = time.perf_counter()
        prepared = sim.compile(
            num_steps=spec["timesteps"], backend=args.prepare_backend
        )
        jax.block_until_ready(prepared.coefficients)
        record["preparation_s"] = time.perf_counter() - preparation_start
    record["prepare_backend"] = args.prepare_backend
    record["preparation_included_in_first_result"] = True
    # This is the real first public call, before executable warmup/compilation.
    public_start = time.perf_counter()
    result = sim.advance(**kwargs, performance=False)
    jax.block_until_ready(result.state)
    record["scene_to_first_public_result_s"] = time.perf_counter() - start
    record["cold_public_advance_s"] = time.perf_counter() - public_start
    del result
    program = sim.compile(**kwargs)
    record["shared_prepared_coefficients"] = (
        None if prepared is None else prepared.coefficients is program.coefficients
    )
    record["coefficient_storage"] = {
        name: {
            "shape": list(getattr(program.coefficients, name).shape),
            "dtype": str(getattr(program.coefficients, name).dtype),
        }
        for name in (f"e_{kind}_{c}" for c in "xyz" for kind in ("decay", "source"))
    }
    start = time.perf_counter()
    state = initial_program_state(
        program, t=0, current_step=0, monitor_steps=spec["timesteps"]
    )
    jax.block_until_ready(state)
    record["initial_state_s"] = time.perf_counter() - start
    start = time.perf_counter()
    scan = build_scan(program)
    lowered = scan.lower(state, program.coefficients)
    record["lower_s"] = time.perf_counter() - start
    start = time.perf_counter()
    executable = lowered.compile()
    record["explicit_compile_s"] = time.perf_counter() - start
    record["explicit_compile_note"] = (
        "May reuse compilation cached by the separately timed first public call"
    )
    record["schedule_decision"] = scan.cuda_schedule_decision
    for _ in range(args.warmups):
        jax.block_until_ready(executable(state, program.coefficients))
    samples = []
    for _ in range(args.samples):
        # Retaining the previous output during a new invocation artificially
        # raises the capacity requirement by another complete field bank.
        final = None
        start = time.perf_counter()
        final = executable(state, program.coefficients)
        jax.block_until_ready(final)
        samples.append(time.perf_counter() - start)
    record["execution_nvml_memory"] = memory_sampler.finish()
    # Host copies/checks happen after timing and are deliberately exhaustive.
    finite, peak = True, 0.0
    for name in FIELDS:
        field = np.asarray(getattr(final, name))
        finite = finite and bool(np.isfinite(field).all())
        peak = max(peak, float(np.max(np.abs(field), initial=0)))
        del field
    dft_peak = max(
        float(np.max(np.abs(np.asarray(getattr(final, name))), initial=0))
        for name in ("dft_vec_re", "dft_vec_im")
    )
    record["signal_check"] = {
        "fields_finite": finite,
        "field_peak": peak,
        "dft_peak": dft_peak,
        "analytic_mie_validated": False,
    }
    if not finite or peak <= 0 or (args.monitor_faces and dft_peak <= 0):
        raise ValueError("Nonfinite or zero physical signal")
    if int(np.asarray(final.current_step)) != spec["timesteps"]:
        raise ValueError("Wrong final simulation clock")
    if args.validate_schedules:
        if args.schedules[0] == "streamed":
            record["schedule_parity"] = {"passed": True, "reference": "self (streamed)"}
        else:
            reference_program = sim.compile(
                num_steps=spec["timesteps"], backend="cuda", schedule="streamed"
            )
            reference_state = initial_program_state(
                reference_program, t=0, current_step=0, monitor_steps=spec["timesteps"]
            )
            reference = jax.block_until_ready(
                build_scan(reference_program)(
                    reference_state, reference_program.coefficients
                )
            )
            record["schedule_parity"] = compare_states(reference, final)
            del reference, reference_state, reference_program
    del final
    public = []
    for _ in range(args.public_samples):
        start = time.perf_counter()
        result = sim.advance(**kwargs, performance=False)
        jax.block_until_ready(result.state)
        public.append(time.perf_counter() - start)
        del result
    memory = executable.memory_analysis()
    record.update(
        {
            "status": "ok",
            "warm_executable_samples_s": samples,
            "warm_public_samples_s": public,
            "warmups": args.warmups,
            "median_warm_executable_s": float(np.median(samples)),
            "median_warm_public_s": float(np.median(public)) if public else None,
            "median_gcups": spec["material_cells"]
            * spec["timesteps"]
            / float(np.median(samples))
            / 1e9,
            "device_memory_stats": jax.devices()[0].memory_stats(),
            "device_memory_note": "Allocator peak over cold/warm calls and optional full-state validation, not isolated kernel peak",
            "executable_memory": {
                key: getattr(memory, key)
                for key in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "alias_size_in_bytes",
                    "temp_size_in_bytes",
                )
            },
            "gpu_after": gpu_status(),
        }
    )


def failure_status(error):
    message = str(error).lower()
    if any(
        token in message
        for token in ("out of memory", "resource_exhausted", "cuda_error_out_of_memory")
    ):
        return "oom"
    return "error"


def main():
    p = parser()
    args = p.parse_args()
    try:
        specs = [case_spec(args, resolution) for resolution in args.resolutions_nm]
        if args.samples < 3 or args.public_samples < 0 or args.warmups < 1:
            raise ValueError("Need >=3 samples, >=1 warmup and >=0 public samples")
        if args.worker and (len(specs) != 1 or len(args.schedules) != 1):
            raise ValueError("Worker requires exactly one resolution and schedule")
    except ValueError as error:
        p.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        record = {"case": specs[0], "schedule": args.schedules[0]}
        try:
            run_worker(args, specs[0], record)
        except Exception as error:
            record.update(
                status=failure_status(error),
                error=str(error),
                traceback=traceback.format_exc(),
            )
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        return 0 if record["status"] == "ok" else 1
    payload = {
        "schema_version": 1,
        "workload": "cuda_mie_like_nanoparticle",
        "exact_fdtdx_reproduction": False,
        "assumptions": __doc__,
        "cases": specs,
        "results": [],
    }
    if args.dry_run:
        payload["status"] = "planned_only"
    else:
        with tempfile.TemporaryDirectory(prefix="beamz-mie-") as temporary:
            for spec in specs:
                for schedule in args.schedules:
                    child_output = Path(temporary) / "worker.json"
                    # Re-emit every physical option. Never forward --dry-run.
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--output",
                        str(child_output),
                        "--resolutions-nm",
                        str(spec["resolution_nm"]),
                        "--schedules",
                        schedule,
                    ]
                    for name in (
                        "cube_nm",
                        "radius_nm",
                        "particle_index",
                        "pml_cells",
                        "wavelength_nm",
                        "bandwidth_fraction",
                        "pulse_offset",
                        "duration_fs",
                        "courant",
                        "samples",
                        "public_samples",
                        "warmups",
                        "monitor_faces",
                        "steps",
                        "prepare_backend",
                    ):
                        value = getattr(args, name)
                        if value is not None:
                            command.extend(["--" + name.replace("_", "-"), str(value)])
                    if args.validate_schedules:
                        command.append("--validate-schedules")
                    env = dict(os.environ)
                    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
                    env["PYTHONPATH"] = (
                        str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                    )
                    child_output.unlink(missing_ok=True)
                    start = time.perf_counter()
                    completed = subprocess.run(
                        command, cwd=ROOT, env=env, text=True, capture_output=True
                    )
                    record = (
                        json.loads(child_output.read_text())
                        if child_output.exists()
                        else {
                            "case": spec,
                            "schedule": schedule,
                            "status": failure_status(completed.stderr),
                            "error": "Worker exited without a result (including possible host OOM kill)",
                        }
                    )
                    record.update(
                        process_wall_s=time.perf_counter() - start,
                        exit_code=completed.returncode,
                        stdout=completed.stdout[-16000:],
                        stderr=completed.stderr[-16000:],
                    )
                    payload["results"].append(record)
                    # Persist completed cases even if the next worker is killed.
                    args.output.write_text(json.dumps(payload, indent=2) + "\n")
                    print(
                        f"{spec['resolution_nm']:g} nm {schedule}: {record['status']}",
                        flush=True,
                    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return int(any(record["status"] != "ok" for record in payload["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
