"""Measure matched FP32 CUDA forward and checkpointed material gradients.

Run from the checkout with the CUDA extension and JAX GPU installed::

    python scripts/benchmark_cuda_checkpoint_adjoint.py --grid graded --trace /tmp/trace

Run each grid in its own process for comparable peak VRAM. Timings exclude
compilation, warm-up, and result decoding, and synchronize each measured call.
"""

import argparse
import gzip
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

import beamz as bz
from beamz.design import MaterialGrid
from beamz.design.grid import RectilinearGrid
from beamz.lattice import sample_voxel_grid_at_component_3d
from beamz.simulation.execute import build_scan


def make_fixture(shape, steps, grid_kind, boundary, cpml_cells=6):
    dx = 80e-9
    dt = 0.45 * dx / bz.LIGHT_SPEED
    rng = np.random.default_rng(53)
    epsilon = rng.uniform(1.5, 2.5, shape).astype(np.float32)
    grid, yee = None, {}
    if grid_kind != "uniform":
        widths = [
            dx
            * (1 + 0.1 * axis)
            * (np.linspace(1, 1.8, size) if grid_kind == "graded" else np.ones(size))
            for axis, size in enumerate(shape[::-1])
        ]
        grid = RectilinearGrid(*(np.r_[0, np.cumsum(w)] for w in widths))
        yee = {
            f"eps_{a}": np.asarray(sample_voxel_grid_at_component_3d(epsilon, f"E{a}"))
            for a in "xyz"
        }
    material = MaterialGrid(
        epsilon, np.float32(0), np.float32(1), dx, shape, grid=grid, yee_materials=yee
    )
    cpml = bz.PML(thickness=cpml_cells * dx, formulation="cpml")
    if boundary != "cpml":
        cpml = bz.PML(
            edges=("left", "right", "top", "back", "front"),
            thickness=cpml_cells * dx,
            formulation="cpml",
        )
    boundaries = [cpml]
    if boundary == "pec":
        boundaries.append(bz.PEC(edges="bottom"))
    z, y, x = (n // 2 for n in shape)
    coordinates = (
        np.arange(shape[2] + 1) * dx,
        np.arange(shape[1] + 1) * dx,
        np.arange(shape[0] + 1) * dx,
    )
    if grid is not None:
        coordinates = (grid.x_edges, grid.y_edges, grid.z_edges)
    xx, yy, zz = coordinates
    sim = bz.Simulation(
        material_grid=material,
        time=np.arange(steps + 1, dtype=np.float64) * dt,
        boundaries=boundaries,
        sources=[
            bz.CustomSource(
                component="Ez",
                timing="e",
                index=(slice(z, z + 2), slice(y, y + 2), slice(x - 4, x - 2)),
                coeff=np.full((2, 2, 2), 0.0625, np.float32),
                waveform=np.sin(2 * np.pi * 2e14 * dt * np.arange(steps + 1)).astype(
                    np.float32
                ),
                target_shape=(shape[0], shape[1] + 1, shape[2] + 1),
            )
        ],
        monitors=[
            bz.FieldMonitor(
                center=(xx[x], yy[y], zz[z]),
                size=(0, yy[y + 4] - yy[y], zz[z + 4] - zz[z]),
                fields=("Ez", "Hy"),
                freqs=(1.8e14, 2.1e14, 2.8e14),
                name="output",
            )
        ],
    )
    return sim


def sample_vram(stop, samples):
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    while not stop.is_set():
        rows = subprocess.check_output(cmd, text=True).splitlines()
        samples.extend(
            int(memory)
            for row in rows
            if "," in row
            for pid, memory in [row.split(",", 1)]
            if int(pid) == os.getpid()
        )
        stop.wait(0.03)


def reverse_profile(path):
    """Summed CUPTI device durations, excluding graph/allocator host time."""
    with gzip.open(path, "rt") as trace:
        events = json.load(trace)["traceEvents"]
    region = next(
        e for e in events if e.get("name") == "reverse" and e.get("ph") == "X"
    )
    device_pids = {
        e["pid"]
        for e in events
        if e.get("name") == "process_name"
        and e.get("args", {}).get("name", "").startswith("/device:GPU:")
    }
    if not device_pids:
        raise RuntimeError("The trace has no CUDA device timeline")
    groups = {
        "replay_compute": (
            "UpdateFusedComponents",
            "UpdateCombinedCpml",
            "UpdateCpml",
            "OrdinaryPhase",
            "RecordElectricCurl",
            "ApplySource",
            "ApplySingleSource",
        ),
        "yee_transpose": ("PhaseAdjoint",),
        "dft_transpose": ("PrepareDftPhases", "ScatterDftTranspose"),
        "reverse_device_copies": ("MemcpyD2D", "memcpy32_post"),
    }
    result = {kind: {"seconds": 0.0, "kernels": 0} for kind in groups}
    for event in events:
        if (
            event.get("ph") != "X"
            or event.get("pid") not in device_pids
            or not region["ts"] <= event["ts"] < region["ts"] + region["dur"]
        ):
            continue
        for kind, needles in groups.items():
            if any(name in event["name"] for name in needles):
                result[kind]["seconds"] += event["dur"] * 1e-6
                result[kind]["kernels"] += 1
                break
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", choices=("uniform", "axis", "graded"), default="uniform"
    )
    parser.add_argument("--boundary", choices=("cpml", "pec"), default="cpml")
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--cpml-cells", type=int, default=6)
    parser.add_argument(
        "--conductivity",
        type=float,
        default=0,
        help="Also differentiate scalar electric conductivity in S/m",
    )
    parser.add_argument("--checkpoint", type=int, default=32)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        parser.error("JAX GPU is required")
    if args.size < 12 or min(args.steps, args.checkpoint, args.samples) < 1:
        parser.error("Use size >= 12 and positive steps, checkpoint and sample counts")
    if not 1 <= args.cpml_cells <= args.size // 2:
        parser.error("Use a positive CPML thickness at most half the grid size")
    if not np.isfinite(args.conductivity) or args.conductivity < 0:
        parser.error("Conductivity must be finite and nonnegative")
    start = time.perf_counter()
    sim = make_fixture(
        (args.size, args.size, args.size),
        args.steps,
        args.grid,
        args.boundary,
        args.cpml_cells,
    )
    conductivity = jnp.float32(args.conductivity) if args.conductivity else None
    if conductivity is not None:
        sim = sim.updated_copy(
            material_grid=replace(
                sim.material_grid, conductivity=np.float32(args.conductivity)
            )
        )
    solver = sim.differentiable(
        backend="cuda", num_steps=args.steps, checkpoint_interval=args.checkpoint
    )
    program = sim.compile(backend="cuda", num_steps=args.steps)
    eps = tuple(jnp.asarray(getattr(program.grid, f"eps_{axis}")) for axis in "xyz")
    if conductivity is not None:
        eps = (*eps, jnp.float32(1))
    setup_s = time.perf_counter() - start

    def objective(solver, parameters):
        if conductivity is not None:
            out = solver(parameters[:3], conductivity=parameters[3] * conductivity)
        else:
            out = solver(parameters)
        return 1e6 * jnp.sum(jnp.abs(solver.monitor_fields(out, "output")) ** 2)

    f = jax.jit(lambda eps: objective(solver, eps))
    vg = jax.jit(jax.value_and_grad(f))

    def prepare_reverse(parameters):
        # The pullback is a PyTree. Return its residuals to time the true
        # recorded forward and the corresponding reverse independently.
        value, pullback = jax.vjp(f, parameters)
        return value, pullback

    recorded = jax.jit(prepare_reverse)
    reverse = jax.jit(lambda pullback: pullback(jnp.float32(1))[0])
    ordinary_scan = build_scan(program)
    ordinary = jax.jit(lambda state, coefficients: ordinary_scan(state, coefficients))

    def timing(label, fn, *values):
        start = time.perf_counter()
        executable = fn.lower(*values).compile()
        compile_s = time.perf_counter() - start
        for _ in range(3):
            output = jax.block_until_ready(executable(*values))
            del output
        samples = []
        for _ in range(args.samples):
            start = time.perf_counter()
            output = jax.block_until_ready(executable(*values))
            samples.append(time.perf_counter() - start)
            del output
        print(label, np.median(samples), flush=True)
        return executable, {
            "compile_s": compile_s,
            "samples_s": samples,
            "median_s": float(np.median(samples)),
            "xla_temporary_bytes": executable.memory_analysis().temp_size_in_bytes,
        }

    _, forward = timing("forward", f, eps)
    full, total = timing("total", vg, eps)
    samples = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_vram, args=(stop, samples))
    sampler.start()
    try:
        # Sample the warmed gradient process over sustained execution, separately
        # from wall timing. This includes CUDA context and the XLA memory pool.
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            jax.block_until_ready(full(eps))
    finally:
        stop.set()
        sampler.join()
    ordinary_exec, ordinary_stats = timing(
        "ordinary", ordinary, solver.initial_state, program.coefficients
    )
    output = ordinary_exec(solver.initial_state, program.coefficients)
    ordinary_value = 1e6 * jnp.sum(
        jnp.abs(solver.monitor_fields(output, "output")) ** 2
    )
    np.testing.assert_allclose(ordinary_value, f(eps), rtol=5e-5, atol=1e-7)
    recorded_exec, recorded_stats = timing("recorded forward", recorded, eps)
    value, pullback = recorded_exec(eps)
    reverse_exec, reverse_stats = timing("reverse including replay", reverse, pullback)
    dispatch = getattr(solver, "execution_plan", None)
    report = dict(
        grid=args.grid,
        cpml_cells=args.cpml_cells,
        conductivity_s_per_m=args.conductivity,
        boundary=args.boundary,
        metric=program.config.metric_kind,
        shape=program.grid.permittivity.shape,
        steps=program.config.num_steps,
        checkpoint=min(args.steps, args.checkpoint),
        ordinary_schedule=ordinary_scan.cuda_schedule_decision["selected"],
        differentiable_plan=dict(dispatch) if dispatch is not None else None,
        gpu=jax.devices()[0].device_kind,
        jax=jax.__version__,
        setup_s=setup_s,
        process_peak_mib=max(samples, default=None),
        value=float(value),
        ordinary_value=float(ordinary_value),
        forward=forward,
        recorded_forward=recorded_stats,
        reverse=reverse_stats,
        total=total,
        ordinary=ordinary_stats,
    )
    if args.validate:
        exact = sim.differentiable(
            backend="jax", num_steps=args.steps, checkpoint_interval=args.checkpoint
        )
        actual_value, gradient = full(eps)
        expected_value, expected = jax.jit(
            jax.value_and_grad(lambda p: objective(exact, p))
        )(eps)
        np.testing.assert_allclose(actual_value, expected_value, rtol=5e-4, atol=1e-7)

        def norm(xs):
            return float(jnp.sqrt(sum(jnp.vdot(x, x) for x in jax.tree.leaves(xs))))

        error = norm(jax.tree.map(lambda a, b: a - b, gradient, expected)) / norm(
            expected
        )
        assert error < 5e-4, error
        # Gradient-aligned direction gives a well-conditioned FD signal.
        direction = jax.tree.map(lambda g: g / norm(expected), expected)
        delta = 0.015
        high = f(jax.tree.map(lambda p, d: p + delta * d, eps, direction))
        low = f(jax.tree.map(lambda p, d: p - delta * d, eps, direction))
        fd = float((high - low) / (2 * delta))
        analytic = float(
            sum(
                jnp.vdot(g, d)
                for g, d in zip(
                    jax.tree.leaves(gradient), jax.tree.leaves(direction), strict=True
                )
            )
        )
        np.testing.assert_allclose(analytic, fd, rtol=0.01, atol=1e-8)
        report.update(gradient_relative_error=error, fd=fd, analytic=analytic)
    if args.trace:
        trace_paths = set(args.trace.glob("plugins/profile/*/perfetto_trace.json.gz"))
        with jax.profiler.trace(str(args.trace), create_perfetto_trace=True):
            for name, executable, inputs in (
                ("gradient", full, (eps,)),
                ("forward", recorded_exec, (eps,)),
                ("reverse", reverse_exec, (pullback,)),
            ):
                with jax.profiler.TraceAnnotation(name):
                    jax.block_until_ready(executable(*inputs))
        (trace_path,) = (
            set(args.trace.glob("plugins/profile/*/perfetto_trace.json.gz"))
            - trace_paths
        )
        report["reverse_profile"] = reverse_profile(trace_path)
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
