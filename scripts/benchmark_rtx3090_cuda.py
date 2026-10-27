#!/usr/bin/env python3
"""Compare PR CUDA FDTD with origin/main's JAX/XLA FDTD on one RTX 3090.

The parent process creates a detached ``origin/main`` worktree and runs it and the
checked-out PR in separate Python processes.  Both are therefore timed with the
same JAX/CUDA environment while importing their own revision of BeamZ.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:  # ``python scripts/...`` exposes the sibling module directly.
    from rtx3090_benchmark import (
        BackendMeasurement,
        RTX3090Comparison,
        source_environment,
        source_revision,
        write_report_artifacts,
    )
except ModuleNotFoundError:  # ``import scripts.benchmark...`` is useful in tests.
    from scripts.rtx3090_benchmark import (
        BackendMeasurement,
        RTX3090Comparison,
        source_environment,
        source_revision,
        write_report_artifacts,
    )


def _nvidia_smi() -> dict[str, str | None]:
    """Collect stable driver/CUDA metadata without making it part of timing."""
    try:
        output = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        name, driver = (part.strip() for part in output.splitlines()[0].split(",", 1))
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        name, driver = "unknown", None
    try:
        output = subprocess.run(
            ("nvidia-smi",), check=True, capture_output=True, text=True
        ).stdout
        cuda = next(
            (
                line.split(marker, 1)[1].split()[0]
                for line in output.splitlines()
                for marker in ("CUDA Version:", "CUDA UMD Version:")
                if marker in line
            ),
            None,
        )
    except (OSError, subprocess.CalledProcessError):
        cuda = None
    return {"device": name, "driver_version": driver, "cuda_version": cuda}


def _child_environment(source_root: Path, runner_root: Path) -> dict[str, str]:
    return source_environment(source_root, runner_root)


def _run_child(
    *,
    script: Path,
    source_root: Path,
    role: str,
    backend: str,
    shape: tuple[int, int, int],
    timesteps: int,
    samples: int,
    warmups: int,
    profile: str,
) -> dict[str, object]:
    command = (
        sys.executable,
        str(script),
        "--child",
        "--role",
        role,
        "--backend",
        backend,
        "--shape",
        *(str(value) for value in shape),
        "--timesteps",
        str(timesteps),
        "--samples",
        str(samples),
        "--warmups",
        str(warmups),
        "--profile",
        profile,
    )
    completed = subprocess.run(
        command,
        cwd=source_root,
        env=_child_environment(source_root, script.parents[1]),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark {role} child failed with exit code "
            f"{completed.returncode}:\n{completed.stdout}\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"benchmark child emitted invalid JSON:\n{completed.stdout}\n{completed.stderr}"
        ) from error


def _measurement(payload: dict[str, object], label: str) -> BackendMeasurement:
    return BackendMeasurement(
        label=label,
        revision=str(payload["revision"]),
        backend=str(payload["backend"]),
        device=str(payload["device"]),
        grid_zyx=tuple(int(value) for value in payload["grid_zyx"]),  # type: ignore[arg-type]
        timesteps=int(payload["timesteps"]),
        trace_lower_s=float(payload["trace_lower_s"]),
        compile_s=float(payload["compile_s"]),
        warm_runtime_samples_s=tuple(
            float(value)
            for value in payload["warm_runtime_samples_s"]  # type: ignore[arg-type]
        ),
        profile=str(payload["profile"]),
        field_precision=str(payload["field_precision"]),
        cpml_psi_precision=str(payload["cpml_psi_precision"]),
        python_version=str(payload["python_version"]),
        jax_version=str(payload["jax_version"]),
        jaxlib_version=str(payload["jaxlib_version"]),
        beamz_version=str(payload["beamz_version"]),
        cuda_component_version=(
            None
            if payload.get("cuda_component_version") is None
            else str(payload["cuda_component_version"])
        ),
        cuda_abi_version=(
            None
            if payload.get("cuda_abi_version") is None
            else int(payload["cuda_abi_version"])
        ),
        cuda_flags=int(payload["cuda_flags"]),
        driver_version=(
            None
            if payload.get("driver_version") is None
            else str(payload["driver_version"])
        ),
        cuda_version=(
            None
            if payload.get("cuda_version") is None
            else str(payload["cuda_version"])
        ),
    )


def run_comparison(args: argparse.Namespace) -> RTX3090Comparison:
    root = Path(__file__).resolve().parents[1]
    script = Path(__file__).resolve()
    metadata = _nvidia_smi()
    device = str(metadata["device"])
    if not args.allow_other_gpu and "RTX 3090" not in device.upper():
        raise RuntimeError(
            f"this protocol is calibrated for an RTX 3090; detected {device!r}. "
            "Pass --allow-other-gpu to collect a non-canonical record."
        )
    temporary_parent = Path(tempfile.mkdtemp(prefix="beamz-origin-main-"))
    baseline_root = temporary_parent / "origin-main"
    try:
        subprocess.run(
            ("git", "worktree", "add", "--detach", str(baseline_root), "origin/main"),
            cwd=root,
            check=True,
        )
        baseline_payload = _run_child(
            script=script,
            source_root=baseline_root,
            role="baseline",
            backend="jax",
            shape=tuple(args.shape),
            timesteps=args.timesteps,
            samples=args.samples,
            warmups=args.warmups,
            profile=args.profile,
        )
        cuda_payload = _run_child(
            script=script,
            source_root=root,
            role="cuda",
            backend=args.backend,
            shape=tuple(args.shape),
            timesteps=args.timesteps,
            samples=args.samples,
            warmups=args.warmups,
            profile=args.profile,
        )
    finally:
        if baseline_root.exists():
            subprocess.run(
                ("git", "worktree", "remove", "--force", str(baseline_root)),
                cwd=root,
                check=False,
            )
        shutil.rmtree(temporary_parent, ignore_errors=True)
    comparison = RTX3090Comparison(
        _measurement(baseline_payload, "origin/main JAX/XLA"),
        _measurement(cuda_payload, "PR CUDA streamed"),
    )
    return comparison


def _block(value) -> None:
    value.ez.block_until_ready()


def _run_child_benchmark(args: argparse.Namespace) -> None:
    """Run one source revision.  This path intentionally imports no PR-only helpers."""
    import platform
    import time
    from dataclasses import replace

    import jax
    import jaxlib
    import numpy as np

    import beamz as bz
    from beamz.design import MaterialGrid, RectilinearGrid
    from beamz.simulation import sharding as sharding_runtime
    from beamz.simulation.execute import build_scan, initial_program_state

    devices = tuple(jax.devices("gpu"))
    if len(devices) != 1:
        raise RuntimeError(f"expected exactly one visible GPU, found {devices!r}")
    device = devices[0]
    shape = tuple(args.shape)
    resolution = 50e-9
    metric_grid = None
    if args.profile == "axis_uniform_pec":
        metric_grid = RectilinearGrid.from_spacing(
            (shape[2], shape[1], shape[0]),
            (resolution, 1.25 * resolution, 1.5 * resolution),
        )
    elif args.profile in {"rectilinear_pec", "rectilinear_cpml"}:

        def graded_edges(count, scale, growth):
            widths = scale * np.linspace(1.0, growth, count, dtype=np.float64)
            return np.concatenate(([0.0], np.cumsum(widths)))

        metric_grid = RectilinearGrid(
            graded_edges(shape[2], 0.85 * resolution, 1.30),
            graded_edges(shape[1], 1.05 * resolution, 1.24),
            graded_edges(shape[0], 1.20 * resolution, 1.18),
        )
    minimum_spacing = resolution if metric_grid is None else metric_grid.minimum_spacing
    dt = 0.95 * minimum_spacing / (bz.LIGHT_SPEED * np.sqrt(3.0))
    size_xyz = (
        (shape[2] * resolution, shape[1] * resolution, shape[0] * resolution)
        if metric_grid is None
        else metric_grid.extent
    )
    permittivity = np.ones(shape, dtype=np.float32)
    if metric_grid is None and args.profile != "uniform_pec":
        nz, ny, _nx = shape
        permittivity[nz * 3 // 8 : nz * 5 // 8, ny * 3 // 8 : ny * 5 // 8, :] = (
            np.float32(3.45**2)
        )
    conductivity = np.float32(0.0)
    if args.profile == "conductive_pec":
        conductivity_grid = np.zeros(shape, dtype=np.float32)
        nz, ny, _nx = shape
        conductivity_grid[nz // 4 : 3 * nz // 4, ny // 4 : 3 * ny // 4, :] = np.float32(
            2.5e3
        )
        conductivity = conductivity_grid
    material_grid = MaterialGrid(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=np.float32(1.0),
        resolution=resolution,
        shape=shape,
        grid=metric_grid,
    )
    pml_cells = min(8, (min(shape) - 1) // 2)
    uses_cpml = args.profile in {
        "heterogeneous_cpml",
        "cpml_source",
        "cpml_source_monitor",
        "cpml_multiple_monitors",
        "rectilinear_cpml",
    }
    boundaries = [bz.PEC(edges="all")]
    if uses_cpml:
        boundaries = [
            bz.PML(
                edges="all",
                thickness=pml_cells * resolution,
                formulation="cpml",
            )
        ]
    elif args.profile == "sponge":
        boundaries = [
            bz.PML(
                edges="all",
                thickness=pml_cells * resolution,
                formulation="sponge",
            )
        ]
    elif args.profile == "mixed_boundaries":
        boundaries = [
            bz.PML(
                edges=("left", "right"),
                thickness=pml_cells * resolution,
                formulation="sponge",
            ),
            bz.PEC(edges=("front", "back", "bottom", "top")),
        ]
    sources = []
    if args.profile in {
        "pec_source",
        "pec_source_monitor",
        "cpml_source",
        "cpml_source_monitor",
        "cpml_multiple_monitors",
        "multiple_sources",
        "multiple_monitors",
        "ragged_monitors",
        "scheduled_windowed_monitor",
    }:
        omega = 2.0 * np.pi * 193.414e12
        steps = np.arange(args.timesteps)
        envelope = np.exp(-(((steps - 24.0) / 8.0) ** 2))
        source_positions = (
            (0.25, 0.40)
            if args.profile
            in {
                "multiple_sources",
                "multiple_monitors",
                "cpml_multiple_monitors",
                "scheduled_windowed_monitor",
            }
            else (0.25,)
        )
        sources = [
            bz.GaussianSource(
                position=(fraction * size_xyz[0], 0.5 * size_xyz[1], 0.5 * size_xyz[2]),
                width=max(2.0 * resolution, 0.08 * min(size_xyz[1:])),
                signal=(
                    (1.0 if index == 0 else 0.6) * envelope * np.sin(omega * steps * dt)
                ).astype(np.float32),
            )
            for index, fraction in enumerate(source_positions)
        ]
    elif args.profile == "h_source":
        target_shape = (shape[0] + 1, shape[1], shape[2])
        source_index = (
            slice(target_shape[0] // 2, target_shape[0] // 2 + 1),
            slice(shape[1] // 4, 3 * shape[1] // 4),
            slice(shape[2] // 4, 3 * shape[2] // 4),
        )
        coefficient_shape = tuple(key.stop - key.start for key in source_index)
        sources = [
            bz.CustomSource(
                component="Hz",
                timing="h",
                index=source_index,
                coeff=np.full(coefficient_shape, 1e-4, dtype=np.float32),
                waveform=np.sin(np.linspace(0.0, 6.0 * np.pi, args.timesteps)).astype(
                    np.float32
                ),
                target_shape=target_shape,
            )
        ]
    monitors = []
    if args.profile in {
        "pec_source_monitor",
        "cpml_source_monitor",
        "multiple_monitors",
        "cpml_multiple_monitors",
        "ragged_monitors",
        "scheduled_windowed_monitor",
    }:
        clear_y = max(resolution, size_xyz[1] - 2 * pml_cells * resolution)
        clear_z = max(resolution, size_xyz[2] - 2 * pml_cells * resolution)
        if args.profile == "ragged_monitors":
            monitor_plans = (
                (0.68, clear_y, clear_z, np.asarray((193.414e12,))),
                (
                    0.78,
                    max(resolution, clear_y / 4.0),
                    max(resolution, clear_z / 4.0),
                    np.linspace(190e12, 197e12, 12),
                ),
            )
        else:
            monitor_positions = (
                (0.68, 0.78)
                if args.profile in {"multiple_monitors", "cpml_multiple_monitors"}
                else (0.75,)
            )
            monitor_plans = tuple(
                (
                    fraction,
                    clear_y,
                    clear_z,
                    np.asarray((190e12, 193.414e12, 196e12)),
                )
                for fraction in monitor_positions
            )
        monitors = [
            bz.FieldMonitor(
                center=(fraction * size_xyz[0], 0.5 * size_xyz[1], 0.5 * size_xyz[2]),
                size=(0.0, monitor_y, monitor_z),
                freqs=frequencies,
                fields=("Ey", "Ez", "Hy", "Hz"),
                interval=3 if args.profile == "scheduled_windowed_monitor" else 1,
                name=f"transmission_{index}",
            )
            for index, (fraction, monitor_y, monitor_z, frequencies) in enumerate(
                monitor_plans
            )
        ]
    simulation = bz.Simulation(
        material_grid=material_grid,
        boundaries=boundaries,
        sources=sources,
        monitors=monitors,
        size=size_xyz,
        time=np.arange(args.timesteps, dtype=np.float64) * dt,
    )
    simulation.clear_compiled_cache()
    if args.role == "baseline":
        program = simulation.compile(num_steps=args.timesteps)
    else:
        program = simulation.compile(num_steps=args.timesteps, backend=args.backend)
    if args.profile == "scheduled_windowed_monitor":
        monitor = replace(
            program.monitors[0],
            dft_t_start=float(simulation.time[3]),
            dft_t_end=float(simulation.time[-4]),
            dft_window_code=1,
        )
        program = replace(program, monitors=(monitor,))
    state = initial_program_state(
        program,
        t=float(simulation.time[0]),
        current_step=0,
        monitor_steps=args.timesteps,
    )
    coefficients = sharding_runtime.place_tree(program, program.coefficients)
    scan = build_scan(program, donate_state=False)
    trace_started = time.perf_counter()
    lowered = scan.lower(state, coefficients)
    trace_lower_s = time.perf_counter() - trace_started
    compile_started = time.perf_counter()
    executable = lowered.compile()
    compile_s = time.perf_counter() - compile_started
    for _ in range(args.warmups):
        warmed = executable(state, coefficients)
        _block(warmed)
    samples = []
    for _ in range(args.samples):
        started = time.perf_counter()
        result = executable(state, coefficients)
        _block(result)
        samples.append(time.perf_counter() - started)
    smi = _nvidia_smi()
    resolved_backend = getattr(program.config, "backend", "jax")
    cuda_component_version = None
    cuda_abi_version = None
    if str(resolved_backend).startswith("cuda"):
        from beamz.simulation.backend import cuda_backend_status

        status = cuda_backend_status(register=False)
        if not status.available:
            raise RuntimeError(status.reason or "CUDA component unavailable")
        cuda_component_version = status.extension_version
        cuda_abi_version = status.abi_version
    cpml_psi_precision = (
        str(state.cpml_psi_h_terms[0].dtype) if state.cpml_psi_h_terms else "float32"
    )
    payload = {
        "revision": source_revision(Path.cwd()),
        "backend": resolved_backend,
        "device": str(getattr(device, "device_kind", device)),
        "grid_zyx": shape,
        "timesteps": args.timesteps,
        "profile": args.profile,
        "trace_lower_s": trace_lower_s,
        "compile_s": compile_s,
        "warm_runtime_samples_s": samples,
        "driver_version": smi["driver_version"],
        "cuda_version": smi["cuda_version"],
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "beamz_version": bz.__version__,
        "field_precision": str(state.ex.dtype),
        "cpml_psi_precision": cpml_psi_precision,
        "cuda_component_version": cuda_component_version,
        "cuda_abi_version": cuda_abi_version,
        "cuda_flags": int(getattr(program.config, "cuda_flags", 0)),
    }
    print(json.dumps(payload, allow_nan=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape", nargs=3, type=int, default=(96, 128, 192), metavar=("NZ", "NY", "NX")
    )
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--profile",
        choices=(
            "uniform_pec",
            "heterogeneous_pec",
            "heterogeneous_cpml",
            "pec_source",
            "pec_source_monitor",
            "cpml_source",
            "cpml_source_monitor",
            "cpml_multiple_monitors",
            "axis_uniform_pec",
            "rectilinear_pec",
            "rectilinear_cpml",
            "conductive_pec",
            "sponge",
            "mixed_boundaries",
            "multiple_sources",
            "h_source",
            "multiple_monitors",
            "ragged_monitors",
            "scheduled_windowed_monitor",
        ),
        default="uniform_pec",
    )
    parser.add_argument(
        "--backend", choices=("jax", "cuda", "cuda_streamed"), default="cuda_streamed"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/results/rtx3090")
    )
    parser.add_argument("--allow-other-gpu", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--role", choices=("baseline", "cuda"), help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.samples < 3:
        raise SystemExit("--samples must be at least three")
    if args.warmups < 1:
        raise SystemExit("--warmups must be positive")
    if any(value <= 2 for value in args.shape):
        raise SystemExit("each --shape dimension must be greater than two")
    if args.timesteps < 1:
        raise SystemExit("--timesteps must be positive")
    if not args.child and args.backend == "jax":
        raise SystemExit("--backend jax is reserved for the origin/main child run")
    if args.child:
        if args.role is None:
            raise SystemExit("--child requires --role")
        _run_child_benchmark(args)
        return
    comparison = run_comparison(args)
    paths = write_report_artifacts(comparison, args.output_dir)
    print(
        f"CUDA {'wins' if comparison.cuda_is_faster else 'regresses'}: "
        f"{comparison.runtime_speedup:.3f}x runtime speedup"
    )
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    if not comparison.cuda_is_faster:
        raise SystemExit("custom CUDA is slower than origin/main JAX/XLA")


if __name__ == "__main__":
    main()
