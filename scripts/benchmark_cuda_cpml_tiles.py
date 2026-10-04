#!/usr/bin/env python3
"""Paired CPML tile comparisons: one physical problem, interleaved variants.

Run the parent with the CUDA-capable Python. Each case gets a fresh child.
All cases have 12-cell CPML, lossless materials and mode sources. Fields stay
FP32; auxiliary storage can be FP32 or BF16.
The default study varies CPML tiles. Other studies compare temporal schedules,
rolling tiles, or padding. Each variant sees the same state and coefficients.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def cases():
    result = []

    def add(name, shape, **options):
        result.append(dict(name=name, shape=shape, **options))

    for shape in itertools.permutations((64, 256, 1024)):
        add("aspect_" + "_".join(map(str, shape)), shape)
    for shape in [(128, 256, 256), (128, 256, 512), (128, 256, 768), (256, 256, 256)]:
        add("volume_" + "_".join(map(str, shape)), shape)
    for shape in [(97, 289, 593), (71, 479, 503), (1024, 256, 63), (1024, 256, 65)]:
        add("irregular_" + "_".join(map(str, shape)), shape)
    for shape in [(128, 256, 512), (64, 256, 1024), (1024, 256, 64)]:
        add("field_" + "_".join(map(str, shape)), shape, monitor_type="field")
    add("long_short_x", (1024, 256, 64), steps=1024)
    add("developed_short_x", (1024, 256, 64), presteps=1024)
    add("smooth", (128, 256, 512), material="smooth")
    add("eleven_frequencies", (128, 256, 512), frequencies=11)
    add(
        "field_eleven_frequencies",
        (1024, 256, 64),
        monitor_type="field",
        frequencies=11,
    )
    return result


def variant_flags(study):
    from beamz.simulation import _cuda_abi as abi

    tile_flags = {"64x4": 0, "32x8": abi.CUDA_SHELL32X8, "32x4": abi.CUDA_SHELL32X4}
    mask = abi.CUDA_SHELL32X8 | abi.CUDA_SHELL32X4
    if study != "tiles":
        tile_flags = {
            "one_step": 0,
            "pair_64x4": abi.CUDA_TEMPORAL_PAIR,
            "pair_32x8": abi.CUDA_TEMPORAL_PAIR | abi.CUDA_SHELL32X8,
        }
        mask |= abi.CUDA_TEMPORAL_PAIR | abi.CUDA_CPML_PAIR
        if study == "temporal_schedule":
            tile_flags = {
                "one_step": 0,
                "16x8x16": abi.CUDA_TEMPORAL_PAIR,
                "single": abi.CUDA_TEMPORAL_PAIR,
            }
        if study == "cpml_temporal":
            tile_flags = {
                "one_step": 0,
                "single": abi.CUDA_TEMPORAL_PAIR,
                "cpml_pair": abi.CUDA_TEMPORAL_PAIR | abi.CUDA_CPML_PAIR,
            }
        if study == "cpml_spatial":
            tile_flags = {
                "one_step": 0,
                "single": abi.CUDA_TEMPORAL_PAIR,
                "spatial": abi.CUDA_TEMPORAL_PAIR,
            }
        if study == "cpml_pair_tiles":
            tile_flags = {
                t: abi.CUDA_TEMPORAL_PAIR | abi.CUDA_CPML_PAIR
                for t in ("16x8x16", "16x8x4", "oriented")
            }
        if study == "cpml_pair_padding":
            tile_flags = {
                "none": abi.CUDA_TEMPORAL_PAIR | abi.CUDA_CPML_PAIR,
                "pad32": abi.CUDA_TEMPORAL_PAIR
                | abi.CUDA_CPML_PAIR
                | abi.CUDA_FIELD_PAD32,
                "pad64": abi.CUDA_TEMPORAL_PAIR
                | abi.CUDA_CPML_PAIR
                | abi.CUDA_FIELD_PAD64,
            }
            mask |= abi.CUDA_FIELD_PAD32 | abi.CUDA_FIELD_PAD64 | abi.CUDA_FIELD_PAD_Y8
        if study == "pair_tiles":
            tile_flags = {
                t: abi.CUDA_TEMPORAL_PAIR for t in ("16x8x16", "32x4x16", "32x8x16")
            }
        if study == "pair_padding":
            tile_flags = {
                "one_step": 0,
                "pair_none": abi.CUDA_TEMPORAL_PAIR,
                "pair_pad32": abi.CUDA_TEMPORAL_PAIR | abi.CUDA_FIELD_PAD32,
            }
            mask |= abi.CUDA_FIELD_PAD32 | abi.CUDA_FIELD_PAD64 | abi.CUDA_FIELD_PAD_Y8
    return tile_flags, mask


def worker(args, case):
    from dataclasses import asdict, replace
    from types import SimpleNamespace

    import beamz._cuda as extension
    import jax
    import numpy as np

    import beamz as bz
    from beamz.simulation import _cuda_abi as abi
    from beamz.simulation.cuda import runtime
    from beamz.simulation.execute import build_scan, initial_program_state
    from scripts.benchmark_cuda_realistic import build_simulation

    config = dict(
        steps=256,
        presteps=0,
        psi_precision=args.psi_precision,
        pml=12,
        material="binary",
        source="mode",
        monitors=2,
        frequencies=3,
        monitor_type="mode",
    )
    config.update(case)
    options = SimpleNamespace(**config)
    start = time.perf_counter()
    sim = build_simulation(options)
    base = sim.compile(num_steps=options.steps, backend="cuda_streamed")
    state = initial_program_state(
        base, t=0, current_step=0, monitor_steps=options.steps
    )
    if options.presteps:
        state = sim.advance(
            state=state, num_steps=options.presteps, backend="cuda_streamed"
        ).state
    jax.block_until_ready(state)
    setup_s = time.perf_counter() - start
    executables, variants = {}, {}
    tile_flags, mask = variant_flags(args.study)
    baseline_name = next(iter(tile_flags))
    for tile, flag in tile_flags.items():
        flags = (base.config.cuda_flags & ~mask) | flag
        program = replace(base, config=replace(base.config, cuda_flags=flags))
        plans = []
        choose = runtime._native_schedule_plan

        def record(*positional, _choose=choose, _plans=plans, **keywords):
            plan = _choose(*positional, **keywords)
            _plans.append(asdict(plan))
            return plan

        runtime._native_schedule_plan = record
        start = time.perf_counter()
        try:
            executables[tile] = (
                build_scan(program, donate_state=False)
                .lower(state, base.coefficients)
                .compile()
            )
        finally:
            runtime._native_schedule_plan = choose
        expected_depth = 2 if flag & abi.CUDA_TEMPORAL_PAIR else 1
        assert plans and all(p["temporal_steps"] == expected_depth for p in plans)
        assert all(p["flags"] & abi.NATIVE_SCHEDULE_COMBINED_CPML_CORE for p in plans)
        variants[tile] = dict(
            cuda_flags=flags,
            native_plans=plans,
            compile_s=time.perf_counter() - start,
            samples_s=[],
        )

    def run(tile):
        if args.study == "cpml_spatial":
            os.environ["BEAMZ_CUDA_CPML_SPATIAL"] = "1" if tile == "spatial" else "0"
            os.environ["BEAMZ_CUDA_PAIR_TILE"] = "single"
        if args.study == "cpml_pair_tiles":
            os.environ["BEAMZ_CUDA_CPML_PAIR_TILE"] = tile
        if args.study in {"pair_tiles", "temporal_schedule", "cpml_temporal"}:
            os.environ["BEAMZ_CUDA_PAIR_TILE"] = (
                "16x8x16" if tile in {"one_step", "cpml_pair"} else tile
            )
        return executables[tile](state, base.coefficients)

    # Full-state comparison on the actual large problem, outside timing. The
    # independent JAX oracle is exercised separately by the hardware tests.
    old_fusion = os.environ.get("BEAMZ_CUDA_CPML_CORE_FUSION")
    if args.study in {"temporal", "pair_padding", "pair_tiles"}:
        # The rolling core uses the fused one-step arithmetic. The separate
        # H/E queue has different FP32 rounding, accumulating several ppm over
        # long runs. Validate pairs against the same arithmetic, while timing
        # the established auto-dispatched baseline below.
        os.environ["BEAMZ_CUDA_CPML_CORE_FUSION"] = "1"
    if args.study == "pair_tiles":
        reference = (
            build_scan(base, donate_state=False)
            .lower(state, base.coefficients)
            .compile()
        )
        baseline = jax.device_get(reference(state, base.coefficients))
    else:
        baseline = jax.device_get(run(baseline_name))
    if old_fusion is None:
        os.environ.pop("BEAMZ_CUDA_CPML_CORE_FUSION", None)
    else:
        os.environ["BEAMZ_CUDA_CPML_CORE_FUSION"] = old_fusion
    for tile in tile_flags:
        comparison = baseline
        reference_name = (
            "pair_16x8x16"
            if args.study == "cpml_pair_tiles"
            else "pair_unpadded"
            if args.study == "cpml_pair_padding"
            else "auto_one_step"
            if args.study in {"temporal_schedule", "cpml_temporal", "cpml_spatial"}
            else "fused_one_step"
        )
        if (args.study == "temporal_schedule" and tile == "16x8x16") or (
            args.study in {"cpml_temporal", "cpml_spatial"}
            and tile in {"cpml_pair", "spatial"}
        ):
            os.environ["BEAMZ_CUDA_CPML_CORE_FUSION"] = "1"
            comparison = jax.device_get(run(baseline_name))
            if old_fusion is None:
                os.environ.pop("BEAMZ_CUDA_CPML_CORE_FUSION", None)
            else:
                os.environ["BEAMZ_CUDA_CPML_CORE_FUSION"] = old_fusion
            reference_name = "fused_one_step"
        variants[tile]["correctness_reference"] = reference_name
        result = jax.device_get(run(tile))
        differences = []
        for ref, actual in zip(
            jax.tree_util.tree_leaves(comparison),
            jax.tree_util.tree_leaves(result),
            strict=True,
        ):
            ref, actual = np.asarray(ref), np.asarray(actual)
            if ref.dtype.name == "bfloat16":
                ref, actual = ref.astype(np.float32), actual.astype(np.float32)
            if np.issubdtype(ref.dtype, np.inexact):
                scale = float(np.max(np.abs(ref), initial=0))
                if args.psi_precision == "bf16":
                    # Existing BF16 application-state envelope. Long-run physical
                    # accuracy against FP32 is assessed separately, not inferred
                    # from this schedule-comparison gate.
                    try:
                        np.testing.assert_allclose(
                            actual,
                            ref,
                            rtol=3e-5,
                            atol=max(3e-6, 2e-2 * scale),
                            err_msg=f"{tile} versus {reference_name}, leaf {len(differences)}",
                        )
                    except AssertionError as exc:
                        if not args.diagnostic:
                            raise
                        variants[tile].setdefault("parity_failures", []).append(
                            {"leaf_index": len(differences), "error": str(exc)}
                        )
                    variants[tile]["parity_gate"] = (
                        "BF16: rtol=3e-5, atol=max(3e-6,0.02*leaf_peak)"
                    )
                elif args.study != "tiles":
                    if tile != baseline_name or args.study == "pair_tiles":
                        dense_rolling = tile != "single" and not (
                            variants[tile]["native_plans"][0]["flags"]
                            & abi.NATIVE_SCHEDULE_PACKED_MATERIAL
                        )
                        if (
                            dense_rolling
                            or tile in {"cpml_pair", "spatial"}
                            or args.study.startswith("cpml_pair_")
                        ):
                            # A separate diagnostic fixing the multiply/FMA
                            # order eliminates this dense-core rounding delta.
                            # Retain the existing rotated hardware parity gate.
                            np.testing.assert_allclose(
                                actual, ref, rtol=3e-5, atol=3e-6 + 3e-6 * scale
                            )
                            variants[tile]["parity_gate"] = (
                                "rtol=3e-5, atol=3e-6+3e-6*leaf_peak"
                            )
                        else:
                            np.testing.assert_array_equal(actual, ref)
                            variants[tile]["parity_gate"] = "bitwise"
                else:
                    np.testing.assert_allclose(
                        actual, ref, rtol=3e-5, atol=max(3e-6, 1e-6 * scale)
                    )
                differences.append(float(np.max(np.abs(actual - ref), initial=0)))
            else:
                np.testing.assert_array_equal(actual, ref)
        variants[tile]["max_absolute_difference_by_leaf"] = differences
        variants[tile]["parity_passed"] = not variants[tile].get("parity_failures")
        del result, comparison
    del baseline

    for _ in range(4):
        for tile in tile_flags:
            jax.block_until_ready(run(tile))
    # Every variant occupies every position equally over each six rounds.
    orders = list(itertools.permutations(tile_flags)) * (args.rounds // 6)
    random.Random(args.seed).shuffle(orders)
    for order in orders:
        for tile in order:
            start = time.perf_counter()
            jax.block_until_ready(run(tile))
            variants[tile]["samples_s"].append(time.perf_counter() - start)
    cells = int(np.prod(options.shape))
    for value in variants.values():
        value["median_gcups"] = (
            cells * options.steps / np.median(value["samples_s"]) / 1e9
        )
        ratios = np.array(variants[baseline_name]["samples_s"]) / value["samples_s"]
        value["paired_speedups"] = ratios.tolist()
        value["median_paired_speedup"] = float(np.median(ratios))
    if args.profile:
        for tile in tile_flags:
            trace_dir = args.output.with_suffix("") / tile
            with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
                for _ in range(2):
                    jax.block_until_ready(run(tile))
    root = Path(bz.__file__).resolve().parents[1]
    source_hash = hashlib.sha256()
    for path in sorted([*root.glob("beamz/**/*.py"), *root.glob("cuda/src/*")]):
        if path.is_file():
            source_hash.update(str(path.relative_to(root)).encode())
            source_hash.update(path.read_bytes())
    data = dict(
        case=config,
        correctness_reference=(
            "64x4"
            if args.study == "tiles"
            else "auto_one_step"
            if args.study == "temporal_schedule"
            else "fused_one_step"
        ),
        variants=variants,
        orders=orders,
        seed=args.seed,
        setup_s=setup_s,
        warmups=4,
        jax=jax.__version__,
        device=str(jax.devices()[0]),
        source_sha256=source_hash.hexdigest(),
        extension_sha256=hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        builder_sha256=hashlib.sha256(
            (root / "scripts/benchmark_cuda_realistic.py").read_bytes()
        ).hexdigest(),
        field_shapes={
            name: list(getattr(state, name).shape)
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        },
        field_dtype=str(state.ex.dtype),
        cpml_dtypes=sorted(
            {str(v.dtype) for v in (*state.cpml_psi_h_terms, *state.cpml_psi_e_terms)}
        ),
        monitor_regions=[
            dict(center=list(m.center), size=list(m.size)) for m in sim.monitors
        ],
        source_specs=len(base.sources),
    )
    args.output.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument(
        "--study",
        choices=(
            "tiles",
            "temporal",
            "pair_padding",
            "pair_tiles",
            "temporal_schedule",
            "cpml_temporal",
            "cpml_spatial",
            "cpml_pair_tiles",
            "cpml_pair_padding",
        ),
        default="tiles",
    )
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Record BF16 parity failures and still time; does not count as validation",
    )
    parser.add_argument("--psi-precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rounds < 6 or args.rounds % 6:
        parser.error("rounds must be a positive multiple of six")
    if args.worker:
        worker(args, json.loads(args.worker))
        return
    from scripts.benchmark_cuda_shapes import telemetry

    selected = cases()
    if args.cases:
        unknown = set(args.cases) - {c["name"] for c in selected}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        selected = [c for c in selected if c["name"] in args.cases]
    random.Random(args.seed).shuffle(selected)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(selected, indent=2) + "\n")
    (args.output / "settings.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n"
    )
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        PYTHONPATH=str(root),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION=".45",
        BEAMZ_CUDA_FIELD_PADDING="none",
        BEAMZ_CUDA_TEMPORAL_STEPS="1",
        BEAMZ_CUDA_CPML_TEMPORAL="0",
        BEAMZ_CUDA_CPML_SPATIAL="0",
        BEAMZ_CUDA_CPML_TILE="auto",
        BEAMZ_CUDA_CPML_SHELL_TILE="64x4",
        BEAMZ_CUDA_CPML_PSI_PRECISION=args.psi_precision,
    )
    env.pop("BEAMZ_CUDA_CPML_CORE_FUSION", None)
    for i, case in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {case['name']}", flush=True)
        output = args.output / (case["name"] + ".json")
        readings = []
        start = time.monotonic()
        with output.with_suffix(".log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    json.dumps(case),
                    "--study",
                    args.study,
                    "--output",
                    str(output.resolve()),
                    "--rounds",
                    str(args.rounds),
                    "--psi-precision",
                    args.psi_precision,
                    "--seed",
                    str(args.seed),
                    *(["--profile"] if args.profile else []),
                    *(["--diagnostic"] if args.diagnostic else []),
                ],
                cwd=root,
                env=env,
                stdout=log,
                stderr=log,
            )
            try:
                while process.poll() is None:
                    reading = telemetry()
                    reading["elapsed_s"] = time.monotonic() - start
                    readings.append(reading)
                    if reading["free_mib"] < 2048 or reading["temperature_c"] >= 85:
                        raise RuntimeError("GPU memory/temperature headroom reached")
                    if reading["elapsed_s"] > 600:
                        raise RuntimeError("Case exceeded ten-minute timeout")
                    time.sleep(1)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                output.with_suffix(".telemetry.json").write_text(
                    json.dumps(readings, indent=2) + "\n"
                )
            if process.returncode:
                raise RuntimeError(f"Case failed; see {output.with_suffix('.log')}")
        data = json.loads(output.read_text())
        print(
            "  "
            + ", ".join(
                f"{t}: {v['median_gcups']:.3f}" for t, v in data["variants"].items()
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
