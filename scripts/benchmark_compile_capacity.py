#!/usr/bin/env python3
"""GPU compilation/capacity probe; run each case in a fresh process.

``modal`` measures public material/source/monitor preparation. ``prepared`` loads
an already discretized, dense-coefficient fixture directly into GPU buffers to
isolate compiler and execution capacity from the host material compiler. It has
six FP32 Yee fields, three dense FP32 electric coefficients, 12-cell FP32 CPML,
and a localized initial electric pulse; it is not an optical completion test.

An optional one-rank mesh isolates sharded lowering. --devices > 1 uses real
GPU meshes; --shape selects rectangular domains. Prepared fixtures are not
substitutes for full public modal preparation.
For near-capacity loads use XLA_PYTHON_CLIENT_PREALLOCATE=true and
XLA_PYTHON_CLIENT_MEM_FRACTION=.90, leaving headroom for NCCL and CUDA contexts.
Use PYTHONPATH to select the checkout under test; this script works on PR #288.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0
from beamz.lattice import component_shapes
from beamz.simulation import observe, sharding
from beamz.simulation.execute import build_scan, initial_program_state


def prepare(args, device):
    from scripts.benchmark_h100_backends import ModalWorkload
    from tests.performance.h100_workloads import H100Workload

    count = getattr(args, "devices", 1)
    shape_zyx = tuple(getattr(args, "shape", None) or (args.side,) * 3)
    cfg = (
        None
        if args.axis == "none"
        else dict(axis=args.axis, num_devices=count, backend=device.platform)
    )
    # Production intentionally collapses one-device meshes. Override only device
    # discovery to exercise exactly the multi-device local kernel on one 3090.
    with (
        jax.default_device(jax.devices("cpu")[0]),
        (
            patch.object(sharding, "_jax_devices_for_config", lambda _: (device,))
            if count == 1
            else nullcontext()
        ),
    ):
        if args.workload == "modal":
            sim = ModalWorkload(
                shape_zyx=shape_zyx,
                timesteps=args.steps,
                frequencies=args.frequencies,
            ).build()
        else:
            sim = H100Workload(
                name="capacity",
                shape_zyx=(36,) * 3,
                timesteps=args.steps,
                resolution=80e-9,
                pml_cells=12,
                heterogeneous=False,
                cpml=True,
                source=False,
                monitor=False,
            ).build()
        program = sim.compile(num_steps=args.steps, backend=args.backend, sharding=cfg)
        state = initial_program_state(
            program, t=0, current_step=0, monitor_steps=args.steps
        )
    if args.workload == "modal":
        state = sharding.prepare_state(
            program,
            state,
            replicated_fields=(*observe.MONITOR_FIELDS, "t", "current_step"),
        )
        # A CPU setup context is useful also without a mesh; explicitly place it.
        if not program.sharding.layout.enabled:
            state = jax.device_put(state, device)
        coeffs = sharding.place_tree(program, program.coefficients)
        return program, state, coeffs

    logical = dict(component_shapes(shape_zyx))
    padded = dict(logical)
    layout = program.sharding.layout
    if layout.enabled:
        extent = (
            (max(shape[layout.axis] for shape in logical.values()) + count - 1) // count
        ) * count
        padded = {
            name: tuple(extent if i == layout.axis else n for i, n in enumerate(shape))
            for name, shape in logical.items()
        }
    layout = replace(layout, logical_shapes=logical, padded_shapes=padded)
    plan = replace(program.sharding, layout=layout)
    fields = {
        name: jax.ShapeDtypeStruct(shape, np.float32) for name, shape in logical.items()
    }
    grid = replace(program.grid, component_shapes=logical, **fields)

    def resize_term(term):
        shape = list(padded[term.component])
        shape[term.axis] = term.slab.low + term.slab.high
        return replace(
            term,
            slab=term.slab._replace(
                shape=tuple(shape), logical_stop=logical[term.component][term.axis]
            ),
        )

    cpml = replace(
        program.boundary.cpml,
        h_terms=tuple(map(resize_term, program.boundary.cpml.h_terms)),
        e_terms=tuple(map(resize_term, program.boundary.cpml.e_terms)),
    )
    # This fixture has CPML on every face and no PEC/PMC masks or sources.
    metallic = replace(
        program.boundary.metallic,
        **{name: None for name in program.boundary.metallic.__dataclass_fields__},
    )
    program = replace(
        program,
        grid=grid,
        sharding=plan,
        boundary=replace(
            program.boundary,
            cpml=cpml,
            metallic=metallic,
            logical_component_shapes=logical,
        ),
    )

    # Construct directly on destination GPUs without a global host/device-0 bank.
    def allocate(shape, value=0, *, replicated=False):
        if layout.enabled:
            spec = [None] * len(shape)
            if not replicated:
                spec[layout.axis] = "fdtd"
            target = jax.sharding.NamedSharding(
                plan.mesh, jax.sharding.PartitionSpec(*spec)
            )
        else:
            target = jax.sharding.SingleDeviceSharding(device)
        return jax.jit(
            lambda: jnp.full(shape, np.float32(value), dtype=jnp.float32),
            out_shardings=target,
            compiler_options={"xla_gpu_autotune_level": 0},
        )().block_until_ready()

    # Independent allocations, never a shared zero buffer aliased across fields.
    updates = {}
    for name, shape in padded.items():
        updates[name.lower()] = allocate(shape)
    for phase in ("h", "e"):
        updates[f"cpml_psi_{phase}_terms"] = tuple(
            allocate(
                term.slab.shape, replicated=layout.enabled and term.axis == layout.axis
            )
            for term in getattr(cpml, f"{phase}_terms")
        )
    state = state._replace(**updates)
    state = sharding.prepare_state(
        program, state, replicated_fields=(*observe.MONITOR_FIELDS, "t", "current_step")
    )

    def seeded(value):
        value = value.at[14:18, 14:18, 14:18].set(1e-3)
        # Exercise every real interface, not only all-zero halo traffic.
        if layout.enabled and count > 1:
            extent = padded["Ex"][layout.axis] // count
            for rank in range(1, count):
                patch_slice = [slice(14, 18)] * 3
                patch_slice[layout.axis] = slice(rank * extent - 2, rank * extent + 2)
                value = value.at[tuple(patch_slice)].set(1e-3)
        return value

    seed = jax.jit(
        seeded,
        donate_argnums=(0,),
        compiler_options={"xla_gpu_autotune_level": 0},
    )
    state = state._replace(ex=seed(state.ex))
    coeffs = sharding.place_tree(program, program.coefficients)
    # Load dense discretized coefficients without a second full host volume.
    # Exact homogeneous dielectric, deliberately kept dense like sharded inputs.
    values = {}
    for axis, component in zip("xyz", ("Ex", "Ey", "Ez"), strict=True):
        if args.backend == "cuda_streamed":
            values[f"e_decay_{axis}"] = jnp.asarray(1, dtype=jnp.float32)
        name = "e_source" if args.backend == "cuda_streamed" else "e_permittivity"
        value = (
            program.config.dt / (EPS_0 * 2.25)
            if args.backend == "cuda_streamed"
            else 2.25
        )
        values[f"{name}_{axis}"] = allocate(padded[component], value)
    coeffs = coeffs._replace(**values)
    # Scheduling must see the resized dense buffers, without keeping another
    # owning reference that would prevent releasing coefficients for validation.
    program = replace(
        program,
        coefficients=jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), coeffs
        ),
    )
    return program, state, coeffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    geometry = parser.add_mutually_exclusive_group(required=True)
    geometry.add_argument("--side", type=int)
    geometry.add_argument("--shape", type=int, nargs=3)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--backend", choices=("jax", "cuda_streamed"), default="cuda_streamed"
    )
    parser.add_argument("--workload", choices=("prepared", "modal"), default="prepared")
    parser.add_argument("--axis", choices=("none", "x", "z"), default="none")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--frequencies", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--digests", action="store_true", help="Only for small parity runs"
    )
    args = parser.parse_args()
    shape = tuple(args.shape or (args.side,) * 3)
    if min(shape) <= 24 or args.steps < 1 or args.samples < 1 or args.devices < 1:
        parser.error("dimensions > 24, positive steps, samples and devices required")
    devices = jax.devices("gpu")
    if len(devices) != args.devices or (args.devices > 1 and args.axis == "none"):
        parser.error("visible GPU count must match --devices; multi-GPU needs x/z axis")
    device = devices[0]
    extension = None
    if args.backend == "cuda_streamed":
        import beamz._cuda as extension

    import beamz

    root = Path(beamz.__file__).resolve().parents[1]
    source_hash = hashlib.sha256()
    for path in sorted((root / "beamz").rglob("*.py")):
        source_hash.update(str(path.relative_to(root)).encode())
        source_hash.update(path.read_bytes())
    data = dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        device=device.device_kind,
        device_count=len(devices),
        shape=shape,
        solver_source_sha256=source_hash.hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        jax=jax.__version__,
        root=str(root),
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        extension_sha256=(
            hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()
            if extension
            else None
        ),
        environment={
            k: v for k, v in os.environ.items() if k.startswith(("XLA_", "BEAMZ_"))
        },
        stages=[],
        status="running",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def snapshot(stage, started):
        row = dict(
            stage=stage,
            seconds=time.perf_counter() - started,
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            gpu=device.memory_stats(),
            gpus=[dict(id=d.id, stats=d.memory_stats()) for d in devices],
        )
        data["stages"].append(row)
        args.output.write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    stage = "prepare"
    tick = time.perf_counter()
    try:
        program, state, coeffs = prepare(args, device)
        jax.block_until_ready((state, coeffs))
        gc.collect()
        data["input_bytes"] = sum(
            x.size * x.dtype.itemsize for x in jax.tree.leaves((state, coeffs))
        )
        data["cells"] = math.prod(shape)
        snapshot(stage, tick)
        if args.backend == "cuda_streamed" and hasattr(
            program.config, "cuda_memory_policy"
        ):
            from beamz.simulation.memory import (
                cuda_capacity_schedule,
                cuda_workspace_estimate,
            )

            data["cuda_schedule"] = dict(
                policy=program.config.cuda_memory_policy,
                capacity=cuda_capacity_schedule(program, donate_state=True),
                estimated_fast_workspace_bytes=cuda_workspace_estimate(
                    program, donate_state=True
                ),
            )
        stage = "lower"
        tick = time.perf_counter()
        lowered = build_scan(program, donate_state=True).lower(state, coeffs)
        snapshot(stage, tick)
        args.output.with_suffix(".mlir").write_text(
            lowered.compiler_ir().operation.get_asm(
                large_elements_limit=8, large_resource_limit=8
            )
        )
        stage = "compile"
        tick = time.perf_counter()
        exe = lowered.compile()
        analysis = exe.memory_analysis()
        data["executable_bytes"] = {
            name: getattr(analysis, name)
            for name in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "alias_size_in_bytes",
                "temp_size_in_bytes",
            )
        }
        snapshot(stage, tick)
        for sample in range(args.samples):
            stage = f"run_{sample}"
            tick = time.perf_counter()
            state = exe(state, coeffs)
            jax.block_until_ready(state)
            snapshot(stage, tick)
        # All timed continuations retained the coefficients. Release them only
        # after stepping, so validation has scratch even at the allocator limit.
        del coeffs
        gc.collect()
        stage = "validate"
        tick = time.perf_counter()
        # Bound validation to z slabs: a whole-field reduction can itself require
        # compile-time tuning or enormous scratch, contaminating capacity evidence.
        check = jax.jit(
            lambda x: (jnp.all(jnp.isfinite(x)), jnp.max(jnp.abs(x))),
            compiler_options={"xla_gpu_autotune_level": 0},
        )
        maxima = []
        digests = []
        for value in jax.tree.leaves(state):
            if value.size == 0:
                continue
            maximum = 0.0
            # Validate local buffers directly, without global slice resharding.
            slabs = (
                (
                    shard.data[start : start + 4]
                    for shard in value.addressable_shards
                    for start in (
                        range(0, shard.data.shape[0], 4) if value.ndim >= 3 else (0,)
                    )
                )
                if value.ndim >= 3
                else (value.addressable_shards[0].data,)
            )
            digest = hashlib.sha256()
            for slab in slabs:
                finite, peak = jax.device_get(check(slab))
                if not finite:
                    raise RuntimeError("Non-finite state")
                maximum = max(maximum, float(peak))
                if args.digests:
                    digest.update(np.asarray(slab).tobytes())
            maxima.append(maximum)
            if args.digests:
                digests.append(digest.hexdigest())
        data.update(final_step=int(state.current_step), maxima=maxima, digests=digests)
        if int(state.current_step) != args.steps * args.samples or not any(maxima[:6]):
            raise RuntimeError("Invalid continuation or zero fields")
        data["status"] = "ok"
        snapshot(stage, tick)
    except Exception as error:
        data.update(status="failed", failure_stage=stage, error=repr(error))
        snapshot(stage + "_failed", tick)
        raise


if __name__ == "__main__":
    main()
