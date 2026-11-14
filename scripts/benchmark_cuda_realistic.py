#!/usr/bin/env python3
"""Measure a complete mode-source/CPML/DFT workload on the selected checkout.

Run each case in a fresh process with PYTHONPATH pointing at the source checkout.
Setup and compilation are excluded from warm GCUPS and reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import beamz as bz
from beamz.design import MaterialGrid
from beamz.simulation.execute import build_scan, initial_program_state


def build_simulation(args):
    nz, ny, nx = args.shape
    dx = float(getattr(args, "resolution_nm", 80.0)) * 1e-9
    size = np.array((nx, ny, nz)) * dx
    dt = 0.95 * dx / (bz.LIGHT_SPEED * np.sqrt(3))
    total_steps = args.steps + getattr(args, "presteps", 0)
    times = np.arange(total_steps) * dt
    eps = np.full((nz, ny, nx), 1.44**2, dtype=np.float32)
    transverse_offset = int(getattr(args, "transverse_offset_cells", 0))
    center_z = nz // 2 + transverse_offset
    center_y = ny // 2 + transverse_offset
    physical_y = size[1] / 2 + transverse_offset * dx
    physical_z = size[2] / 2 + transverse_offset * dx
    if getattr(args, "transverse_anchor", "center") == "low":
        # Keep the entire mode aperture inside CPML while finding nonuniform
        # material near the start of region-wise coefficient preparation.
        center_z, center_y = (min(n // 2 + 1, args.pml + 12) for n in (nz, ny))
        physical_y, physical_z = center_y * dx, center_z * dx
    aperture_y, aperture_z = size[1] - 2 * args.pml * dx, size[2] - 2 * args.pml * dx
    if getattr(args, "transverse_anchor", "center") == "low":
        aperture_y = 2 * (min(physical_y, size[1] - physical_y) - args.pml * dx)
        aperture_z = 2 * (min(physical_z, size[2] - physical_z) - args.pml * dx)
    half_z = max(1, round(160e-9 / dx))
    half_y = max(1, round(240e-9 / dx))
    eps[
        center_z - half_z : center_z + half_z,
        center_y - half_y : center_y + half_y,
        :,
    ] = 3.48**2
    if args.material == "smooth":
        eps += np.linspace(0, 0.01, nx, dtype=np.float32)[None, None, :]
    if args.material == "scalar":
        eps.fill(1.44**2)
    sigma = np.float32(0)
    if args.material == "lossy":
        sigma = np.where(eps > 3, np.float32(100), np.float32(0))
        # Loss starts downstream of the launch cross-section.
        sigma[:, :, : nx // 2] = 0
    if args.material == "dense_conductive":
        sigma = np.broadcast_to(
            np.linspace(0, 100, nx, dtype=np.float32), eps.shape
        ).copy()
    grid = MaterialGrid(
        permittivity=eps,
        conductivity=sigma,
        permeability=np.float32(1),
        resolution=dx,
        shape=eps.shape,
    )
    freq = bz.LIGHT_SPEED / 1.55e-6
    steps = np.arange(total_steps)
    refinement = 80e-9 / dx
    signal = np.exp(-(((steps - 24 * refinement) / (8 * refinement)) ** 2)) * np.sin(
        2 * np.pi * freq * times
    )
    sources = [
        bz.ModeSource(
            center=(0.3 * size[0], physical_y, physical_z),
            size=(
                0,
                min(1.6e-6, aperture_y),
                min(0.8e-6, aperture_z),
            ),
            source_time=bz.SampledSignal(signal.astype(np.float32), dt=dt, freq0=freq),
            direction="+",
            mode_spec=bz.ModeSpec(polarization="te"),
        )
    ]
    if args.source == "gaussian":
        sources = [
            bz.GaussianSource(
                position=(0.3 * size[0], physical_y, physical_z),
                width=3 * dx,
                signal=signal.astype(np.float32),
            )
        ]
    monitors = [
        bz.FieldMonitor(
            center=((0.65 + i * 0.1) * size[0], physical_y, physical_z),
            size=(0, size[1] - 2 * args.pml * dx, size[2] - 2 * args.pml * dx),
            freqs=np.linspace(freq * 0.97, freq * 1.03, args.frequencies),
            fields=("Ey", "Ez", "Hy", "Hz"),
            name=f"plane_{i}",
        )
        for i in range(args.monitors)
    ]
    if getattr(args, "monitor_type", "field") == "mode":
        offset = getattr(args, "monitor_offset_cells", None)
        monitors = [
            bz.ModeMonitor(
                center=(
                    (0.65 + i * 0.1) * size[0]
                    if offset is None
                    else 0.3 * size[0] + (i + 1) * offset * dx,
                    physical_y,
                    physical_z,
                ),
                size=(
                    0,
                    min(1.6e-6, aperture_y),
                    min(0.8e-6, aperture_z),
                ),
                freqs=np.linspace(freq * 0.97, freq * 1.03, args.frequencies),
                mode_spec=bz.ModeSpec(polarization="te"),
                name=f"mode_{i}",
            )
            for i in range(args.monitors)
        ]
    boundary = getattr(args, "boundary", "all")
    if boundary in {"all", "sponge"}:
        boundaries = [
            bz.PML(
                edges="all",
                thickness=args.pml * dx,
                formulation="sponge" if boundary == "sponge" else "cpml",
            )
        ]
    elif boundary == "right":
        boundaries = [
            bz.PML(edges=["right"], thickness=args.pml * dx, formulation="cpml"),
            bz.PEC(edges=["left", "top", "bottom", "front", "back"]),
        ]
    else:
        boundaries = [bz.PEC(edges="all")]
    return bz.Simulation(
        material_grid=grid,
        size=tuple(size),
        sources=sources,
        monitors=monitors,
        boundaries=boundaries,
        time=times,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=(128, 256, 512))
    parser.add_argument(
        "--backend",
        choices=("cuda", "cuda_streamed", "cuda_mstep"),
        default="cuda_streamed",
    )
    parser.add_argument("--schedule", choices=("auto", "streamed", "mstep"))
    parser.add_argument("--expected-runs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--pml", type=int, default=12)
    parser.add_argument(
        "--boundary", choices=("all", "right", "pec", "sponge"), default="all"
    )
    parser.add_argument("--monitors", type=int, default=2)
    parser.add_argument("--frequencies", type=int, default=3)
    parser.add_argument(
        "--material",
        choices=("binary", "smooth", "scalar", "lossy", "dense_conductive"),
        default="binary",
    )
    parser.add_argument("--source", choices=("mode", "gaussian"), default="mode")
    parser.add_argument("--monitor-type", choices=("field", "mode"), default="field")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--state-digests",
        action="store_true",
        help="Hash every final state leaf outside timing for exact build comparisons",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Experiment: use the existing single-bank CPML schedule",
    )
    parser.add_argument("--public-samples", type=int, default=0)
    parser.add_argument(
        "--donate",
        action="store_true",
        help="Use private copied state for each donated trial; copies are outside execution timing",
    )
    parser.add_argument(
        "--presteps",
        type=int,
        default=0,
        help="Advance the pulse before timing to measure developed fields",
    )
    args = parser.parse_args()
    if args.material in {"lossy", "dense_conductive"} and args.source == "mode":
        parser.error(
            "ModeSource currently rejects conductive simulations; use --source gaussian for the loss test"
        )
    if args.in_place:
        from beamz.simulation.cuda import runtime

        runtime._temporal_cpml_fields_supported = lambda *_args: False
    if (
        min(args.shape) <= 2 * args.pml
        or args.pml < 1
        or args.steps < 2
        or args.samples < 3
        or args.warmups < 1
        or args.presteps < 0
        or not 0 <= args.monitors <= 3
        or args.frequencies < 1
    ):
        parser.error(
            "need a nonempty CPML interior, positive PML/frequencies/warmups, >=2 steps, >=3 samples, 0–3 monitors and nonnegative presteps"
        )
    start = time.perf_counter()
    sim = build_simulation(args)
    program = sim.compile(
        num_steps=args.steps,
        backend=args.backend,
        schedule=args.schedule,
        expected_runs=args.expected_runs,
    )
    state = initial_program_state(
        program, t=0, current_step=0, monitor_steps=args.steps
    )
    jax.block_until_ready(state)
    setup_s = time.perf_counter() - start
    start = time.perf_counter()
    from beamz.simulation.cuda import mstep

    mstep_plans = []
    choose_mstep_plan = mstep.build_mstep_plan

    def record_mstep_plan(*positional, **keywords):
        plan_start = time.perf_counter()
        plan = choose_mstep_plan(*positional, **keywords)
        from beamz.simulation.cuda.native.mstep import variant_summary

        mstep_plans.append(
            {
                **plan.summary(),
                **variant_summary(program.config.cuda_flags),
                "build_s": time.perf_counter() - plan_start,
            }
        )
        return plan

    mstep.build_mstep_plan = record_mstep_plan
    try:
        scan = build_scan(program, donate_state=args.donate)
    finally:
        mstep.build_mstep_plan = choose_mstep_plan
    build_scan_s = time.perf_counter() - start
    from beamz.simulation.cuda import runtime

    plans = []
    choose_plan = runtime._native_schedule_plan

    def record_plan(*positional, **keywords):
        plan = choose_plan(*positional, **keywords)
        plans.append(asdict(plan))
        return plan

    runtime._native_schedule_plan = record_plan
    try:
        lowering_start = time.perf_counter()
        lowered = scan.lower(state, program.coefficients)
        lower_s = time.perf_counter() - lowering_start
        compilation_start = time.perf_counter()
        executable = lowered.compile()
        xla_compile_s = time.perf_counter() - compilation_start
    finally:
        runtime._native_schedule_plan = choose_plan
    compile_s = time.perf_counter() - start

    def trial_state():
        if not args.donate:
            return state
        copied = jax.tree.map(
            lambda a: jnp.array(a, copy=True),
            state,
        )
        return jax.block_until_ready(copied)

    conditioning_s = 0.0
    if args.presteps:
        start = time.perf_counter()
        state = sim.advance(
            state=state,
            num_steps=args.presteps,
            backend=args.backend,
            schedule=args.schedule,
            expected_runs=args.expected_runs,
        ).state
        jax.block_until_ready(state)
        conditioning_s = time.perf_counter() - start
    current = trial_state()
    first_start = time.perf_counter()
    jax.block_until_ready(executable(current, program.coefficients))
    first_run_s = time.perf_counter() - first_start
    for _ in range(args.warmups):
        jax.block_until_ready(executable(trial_state(), program.coefficients))
    samples = []
    for _ in range(args.samples):
        current = trial_state()
        start = time.perf_counter()
        result = executable(current, program.coefficients)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    public_samples = []
    if args.public_samples:
        jax.block_until_ready(
            sim.advance(
                num_steps=args.steps,
                backend=args.backend,
                schedule=args.schedule,
                expected_runs=args.expected_runs,
            ).state
        )
        for _ in range(args.public_samples):
            start = time.perf_counter()
            jax.block_until_ready(
                sim.advance(
                    num_steps=args.steps,
                    backend=args.backend,
                    schedule=args.schedule,
                    expected_runs=args.expected_runs,
                ).state
            )
            public_samples.append(time.perf_counter() - start)
    import beamz._cuda as extension

    memory = executable.memory_analysis()

    root = Path(bz.__file__).resolve().parents[1]
    sources = sorted([*root.glob("beamz/**/*.py"), *root.glob("cuda/src/*")])
    source_hash = hashlib.sha256()
    for path in sources:
        if path.is_file():
            source_hash.update(str(path.relative_to(root)).encode())
            source_hash.update(path.read_bytes())
    data = {
        "source_sha256": source_hash.hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "warmups": args.warmups,
        "backend": args.backend,
        "schedule": args.schedule,
        "expected_runs": args.expected_runs,
        "schedule_decision": scan.cuda_schedule_decision,
        "allocator": {
            "preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
            "memory_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ""),
            "implementation": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", ""),
        },
        "cpml_core_fusion_env": os.environ.get("BEAMZ_CUDA_CPML_CORE_FUSION", "auto"),
        "field_padding_env": os.environ.get("BEAMZ_CUDA_FIELD_PADDING", "none"),
        "storage_axes": program.config.cuda_storage_axes,
        "cpml_tile_env": os.environ.get("BEAMZ_CUDA_CPML_TILE", "auto"),
        "cpml_pair_tile_env": os.environ.get("BEAMZ_CUDA_CPML_PAIR_TILE", "16x8x16"),
        "cpml_temporal_env": os.environ.get("BEAMZ_CUDA_CPML_TEMPORAL", "0"),
        "cpml_shell_tile_env": os.environ.get("BEAMZ_CUDA_CPML_SHELL_TILE", "64x4"),
        "temporal_steps_env": os.environ.get("BEAMZ_CUDA_TEMPORAL_STEPS", "1"),
        "source_root": str(root),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "extension_sha256": hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest(),
        "device": str(jax.devices()[0]),
        "jax": jax.__version__,
        "shape": args.shape,
        "steps": args.steps,
        "material": args.material,
        "pml": args.pml,
        "boundary": args.boundary,
        "monitors": args.monitors,
        "frequencies": args.frequencies,
        "in_place_experiment": args.in_place,
        "source_specs": len(program.sources),
        "setup_s": setup_s,
        "compile_s": compile_s,
        "build_scan_s": build_scan_s,
        "lower_s": lower_s,
        "xla_compile_s": xla_compile_s,
        "donate_state": args.donate,
        "cuda_flags": program.config.cuda_flags,
        "first_run_s": first_run_s,
        "executable_memory": {
            name: getattr(memory, name)
            for name in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "alias_size_in_bytes",
                "temp_size_in_bytes",
                "generated_code_size_in_bytes",
            )
        },
        "device_memory": jax.devices()[0].memory_stats(),
        "source": args.source,
        "monitor_type": args.monitor_type,
        "native_plans": plans,
        "mstep_plans": mstep_plans,
        "presteps": args.presteps,
        "conditioning_s": conditioning_s,
        "public_samples_s": public_samples,
        "samples_s": samples,
        "median_gcups": float(
            np.prod(args.shape) * args.steps / np.median(samples) / 1e9
        ),
        "field_dtype": str(state.ex.dtype),
        "field_shapes": {
            name: list(getattr(state, name).shape)
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        },
        "monitor_regions": [
            {"center": list(m.center), "size": list(m.size)} for m in sim.monitors
        ],
        "cpml_dtypes": sorted(
            {
                str(value.dtype)
                for value in (*state.cpml_psi_h_terms, *state.cpml_psi_e_terms)
            }
        ),
        "h_coefficient_shapes": {
            name: list(getattr(program.coefficients, name).shape)
            for name in (
                "h_decay_x",
                "h_decay_y",
                "h_decay_z",
                "h_source_x",
                "h_source_y",
                "h_source_z",
            )
        },
        "e_coefficient_shapes": [
            list(v.shape)
            for v in (
                program.coefficients.e_source_x,
                program.coefficients.e_source_y,
                program.coefficients.e_source_z,
            )
        ],
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used",
                "--format=csv",
            ],
            text=True,
        ).strip(),
    }
    if args.state_digests:
        data["state_sha256"] = [
            hashlib.sha256(np.asarray(leaf).tobytes()).hexdigest()
            for leaf in jax.tree_util.tree_leaves(result)
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
