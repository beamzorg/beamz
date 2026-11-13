"""Compile and run isolated, memory-admitted moving-box CUDA mechanism probes.

No CPML, sources, DFTs or BeamZ frontend are included. The independent Plain
reference and optional spatial shared-memory control use identical Yee physics.
GPU execution and live memory admission happen only with explicit run options.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_optional_helper(enabled, source, destination):
    if enabled:
        shutil.copy2(source, destination)
        return destination
    return None


def validate_warp_neighbors(parser, args):
    if args.register_warp_neighbors and not all(
        (
            args.register_cache,
            args.register_phase_metadata,
            args.tile == [15, 15, 31],
            args.register_threads in (0, 512),
        )
    ):
        parser.error(
            "Warp neighbors require --register-cache --register-phase-metadata --tile15 15 31 and512threads"
        )


def validate_asymmetric_cache(parser, args):
    if (args.asymmetric_pipeline and not args.asymmetric_cache) or (
        args.asymmetric_async and not args.asymmetric_pipeline
    ):
        parser.error(
            "Pipeline requires --asymmetric-cache; async requires --asymmetric-pipeline"
        )
    if args.asymmetric_high_x and not args.asymmetric_low_x:
        parser.error("High-X packet reuse requires --asymmetric-low-x")
    if args.asymmetric_low_x and (
        not args.asymmetric_cache or args.asymmetric_pipeline
    ):
        parser.error(
            "Low-X packets require --asymmetric-cache without --asymmetric-pipeline"
        )
    if args.asymmetric_cache and (args.tile != [5, 15, 31] or args.split_components):
        parser.error(
            "Asymmetric cache requires --tile5 15 31 without component splitting"
        )


def validate_streamed(parser, args):
    if args.streamed_interior and not args.streamed_uz:
        parser.error("Streamed Interior requires --streamed-uz")
    if args.streamed_uz and (
        args.tile != [64, 13, 29]
        or args.phase_profile
        or any(t != 4 for _, t in args.run + args.admission_only)
        or any(t != 4 for t in args.largest_run)
    ):
        parser.error(
            "Current streamed helper requires --tile 64 13 29, depth4, and no phase profiling"
        )


def maximum_index_size():
    lo, hi = 3, 1000
    while lo < hi:
        middle = (lo + hi + 1) // 2
        if 6 * (middle + 1) ** 3 <= 2**31 - 1:
            lo = middle
        else:
            hi = middle - 1
    return lo


def predict_memory(
    size, steps, epochs=4, field_banks=4, tile=(4, 8, 32), phase_profile=False
):
    """Exact requested CUDA payload and conservative bounded host workspace."""
    if size < 3 or steps not in (4, 8, 16, 32) or field_banks not in (3, 4):
        raise ValueError("Expected size>=3 and steps4/8/16/32")
    if len(tile) != 3 or min(tile) < 1 or tile[0] * tile[1] * tile[2] > (1 << 20):
        raise ValueError("Expected positive owned tile with at most1048576 cells")
    owned = tile[0] * tile[1] * tile[2]
    surface = owned - (tile[0] - 1) * (tile[1] - 1) * (tile[2] - 1)
    fields = 6 * (size + 1) ** 3
    nz, ny, nx = ((size + steps + 1 + b - 1) // b for b in tile)
    ranks = nx + ny + nz - 2
    frontier = [0] * ranks
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                frontier[z + y + x] += 1
    boxes, peak = nx * ny * nz, max(frontier)
    owners = epochs * peak if epochs else boxes
    trail = owners * steps * 6 * surface * 4
    device = field_banks * fields * 4 + trail + 10 * 16 * 4
    device += boxes * (1 + bool(epochs)) * 4 + (owners * 4 if epochs else 0) + 4
    solver = device - (field_banks - 2) * fields * 4 - 4 * 16 * 4
    profile_bytes = boxes * 19 * 8 if phase_profile else 0
    device += profile_bytes + (2 * 16 * 4 if phase_profile else 0)
    host = 2 * (1 << 20) * 4 + 2 * boxes * 4 + (ranks + 1) * 4 + 65536
    return {
        "size": size,
        "steps": steps,
        "int32_admitted": fields <= 2**31 - 1,
        "tile_zyx": list(tile),
        "owned_cells_per_box": owned,
        "surface_cells_per_box": surface,
        "boxes": boxes,
        "max_frontier": peak,
        "field_bytes": fields * 4,
        "trail_bytes": trail,
        "device_payload_bytes": device,
        "solver_payload_bytes": solver,
        "phase_profile_bytes": profile_bytes,
        "harness_field_banks": field_banks,
        "device_reserve_bytes": 256 << 20,
        "host_workspace_bound_bytes": host,
        "host_reserve_bytes": 512 << 20,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--nvcc")
    p.add_argument("--arch", default="86")
    p.add_argument(
        "--tile", nargs=3, type=int, default=[4, 8, 32], metavar=("Z", "Y", "X")
    )
    p.add_argument("--relative-addressing", action="store_true", default=True)
    p.add_argument(
        "--generic-addressing", action="store_false", dest="relative_addressing"
    )
    p.add_argument("--split-components", action="store_true")
    p.add_argument("--min-ctas", type=int, choices=(1, 2), default=1)
    p.add_argument("--trail-ring", type=int, choices=(0, 2, 4, 8), default=0)
    p.add_argument("--hoist-ring-bases", action="store_true")
    p.add_argument("--tiled-control", action="store_true")
    p.add_argument("--matched-d2-control", action="store_true")
    p.add_argument("--d2-tile-x", type=int, choices=(16, 32), default=16)
    p.add_argument("--d2-tile-y", type=int, choices=(8, 16), default=8)
    p.add_argument("--d2-threads", type=int, choices=(128, 256, 512), default=256)
    p.add_argument("--three-banks", action="store_true")
    p.add_argument("--face-header", type=Path)
    p.add_argument("--circular-header", type=Path)
    p.add_argument("--face-cache", action="store_true")
    p.add_argument("--circular-cache", action="store_true")
    p.add_argument("--register-cache", action="store_true")
    p.add_argument("--asymmetric-cache", action="store_true")
    p.add_argument("--asymmetric-low-x", action="store_true")
    p.add_argument("--asymmetric-high-x", action="store_true")
    p.add_argument("--streamed-uz", action="store_true")
    p.add_argument("--streamed-interior", action="store_true")
    p.add_argument("--streamed-header", type=Path)
    p.add_argument("--asymmetric-pipeline", action="store_true")
    p.add_argument("--asymmetric-async", action="store_true")
    p.add_argument(
        "--phase-profile",
        action="store_true",
        help="Instrument moving CTAs; timings are diagnostic, not candidate scores",
    )
    p.add_argument("--register-header", type=Path)
    p.add_argument("--register-threads", type=int, choices=(0, 256, 512), default=0)
    p.add_argument("--register-local-centers", action="store_true")
    p.add_argument("--register-phase-metadata", action="store_true")
    p.add_argument("--register-warp-neighbors", action="store_true")
    p.add_argument("--register-bad-import-phase", action="store_true")
    p.add_argument("--circular-rolling-heads", action="store_true")
    p.add_argument("--circular-cache-bad-import-phase", action="store_true")
    p.add_argument("--face-cache-bad-h-phase", action="store_true")
    p.add_argument("--unsafe-two-epoch-negative-control", action="store_true")
    p.add_argument(
        "--predict",
        nargs=2,
        type=int,
        action="append",
        default=[],
        metavar=("SIZE", "STEPS"),
        help="CPU-only memory prediction; may be used without a build",
    )
    p.add_argument(
        "--run",
        nargs=2,
        type=int,
        action="append",
        default=[],
        metavar=("SIZE", "STEPS"),
    )
    p.add_argument(
        "--admission-only",
        nargs=2,
        type=int,
        action="append",
        default=[],
        metavar=("SIZE", "STEPS"),
        help="Create a CUDA context and query memory, but run no kernels",
    )
    p.add_argument(
        "--largest-run",
        type=int,
        choices=(4, 8, 16, 32),
        action="append",
        default=[],
        metavar="STEPS",
        help="Find and run the largest live-memory-admitted size within int32 indexing",
    )
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--fresh-processes", type=int, default=1)
    p.add_argument("--order-rotation", type=int, default=0)
    p.add_argument("--run-timeout", type=int, default=900)
    a = p.parse_args()
    cpu_predictions = [
        predict_memory(
            n, t, a.trail_ring, 3 if a.three_banks else 4, a.tile, a.phase_profile
        )
        for n, t in a.predict
    ]
    if (
        a.predict
        and not a.output_dir
        and not (a.run or a.admission_only or a.largest_run)
    ):
        print(json.dumps({"cpu_memory_predictions": cpu_predictions}, indent=2))
        return
    if not a.output_dir or not a.nvcc:
        p.error("--output-dir and --nvcc are required for a build")
    if (a.trail_ring == 2) != a.unsafe_two_epoch_negative_control:
        p.error("Two epochs require the explicit unsafe negative-control option")
    if a.hoist_ring_bases and (not a.trail_ring or not a.relative_addressing):
        p.error("--hoist-ring-bases requires relative addressing and a ring")
    owned = a.tile[0] * a.tile[1] * a.tile[2]
    if min(a.tile) < 1 or owned > (
        (1 << 20)
        if a.streamed_uz
        else 8192
        if a.register_cache or a.asymmetric_cache
        else 1024
    ):
        p.error("Owned tile must contain1..1024 cells")
    if not (
        a.face_cache
        or a.circular_cache
        or a.register_cache
        or a.asymmetric_cache
        or a.streamed_uz
    ) and (owned != 1024 or any(v & (v - 1) for v in a.tile)):
        p.error(
            "Direct history requires exactly1024 owned cells and power-of-two dimensions"
        )
    if (
        sum(
            [
                a.face_cache,
                a.circular_cache,
                a.register_cache,
                a.asymmetric_cache,
                a.streamed_uz,
            ]
        )
        > 1
    ):
        p.error("Cache candidates are mutually exclusive")
    if a.circular_rolling_heads and not a.circular_cache:
        p.error("Rolling heads require --circular-cache")
    if a.circular_cache_bad_import_phase and not a.circular_cache:
        p.error("The bad import phase negative requires --circular-cache")
    if (
        a.face_cache
        or a.circular_cache
        or a.register_cache
        or a.asymmetric_cache
        or a.streamed_uz
    ) and a.split_components:
        p.error(
            "Cache candidates require their fixed thread mapping without component splitting"
        )
    if a.face_cache_bad_h_phase and not a.face_cache:
        p.error("The wrong-H-phase negative control requires --face-cache")
    if not 1 <= a.samples <= 21 or a.fresh_processes < 1 or a.order_rotation < 0:
        p.error("Expected samples1..21, positive fresh-processes, nonnegative rotation")
    if a.register_cache and a.tile not in ([15, 15, 15], [15, 15, 31]):
        p.error("Register cache requires owned tile15x15x15 or15x15x31")
    if (
        a.register_threads
        or a.register_header
        or a.register_local_centers
        or a.register_phase_metadata
        or a.register_bad_import_phase
    ) and not a.register_cache:
        p.error("Register cache diagnostics require --register-cache")
    if (
        a.register_cache
        and a.tile[2] == 31
        and any((a.min_ctas != 1, a.register_threads == 256))
    ):
        p.error("Owned15x15x31 requires512threads and oneCTA launch bound")
    validate_streamed(p, a)
    validate_asymmetric_cache(p, a)
    validate_warp_neighbors(p, a)
    repo = Path(__file__).resolve().parents[1]
    a.output_dir.mkdir(parents=True, exist_ok=False)
    if a.matched_d2_control and not all((a.three_banks, a.tiled_control)):
        p.error("Matched D2 requires --three-banks --tiled-control")
    phase_helper = a.output_dir / "cuda_moving_phase_profile.cuh"
    shutil.copy2(repo / "scripts/cuda_moving_phase_profile.cuh", phase_helper)
    source, header = a.output_dir / "source.cu", a.output_dir / "yee_cell_generated.cuh"
    shutil.copy2(repo / "scripts/cuda_moving_box_probe.cu", source)
    shutil.copy2(repo / "cuda/src/yee_cell_generated.cuh", header)
    helper = None
    if a.face_cache or a.circular_cache:
        helper = a.output_dir / "cuda_moving_face_cache.cuh"
        shutil.copy2(
            a.face_header or repo / "scripts/cuda_moving_face_cache.cuh", helper
        )
    circular_helper = None
    if a.circular_cache:
        circular_helper = a.output_dir / "cuda_moving_circular_cache.cuh"
        shutil.copy2(
            a.circular_header or repo / "scripts/cuda_moving_circular_cache.cuh",
            circular_helper,
        )
    register_helper = None
    if a.register_cache:
        register_helper = a.output_dir / "cuda_moving_register_cache.cuh"
        shutil.copy2(
            a.register_header or repo / "scripts/cuda_moving_register_cache.cuh",
            register_helper,
        )
    d2_helper = copy_optional_helper(
        a.matched_d2_control,
        repo / "scripts/cuda_moving_d2_control.cuh",
        a.output_dir / "cuda_moving_d2_control.cuh",
    )
    asymmetric_helper = copy_optional_helper(
        a.asymmetric_cache,
        repo / "scripts/cuda_moving_asymmetric_cache.cuh",
        a.output_dir / "cuda_moving_asymmetric_cache.cuh",
    )
    pipeline_helper = copy_optional_helper(
        a.asymmetric_pipeline,
        repo / "scripts/cuda_moving_asymmetric_pipeline.cuh",
        a.output_dir / "cuda_moving_asymmetric_pipeline.cuh",
    )
    streamed_helper = copy_optional_helper(
        a.streamed_uz,
        repo / "scripts/cuda_moving_streamed_uz.cuh",
        a.output_dir / "cuda_moving_streamed_uz.cuh",
    )
    streamed_generated = copy_optional_helper(
        a.streamed_uz,
        a.streamed_header or repo / "scripts/cuda_streamed_uz_generated.cuh",
        a.output_dir / "cuda_streamed_uz_generated.cuh",
    )
    low_x_helper = copy_optional_helper(
        a.asymmetric_low_x,
        repo / "scripts/cuda_moving_asymmetric_low_x.cuh",
        a.output_dir / "cuda_moving_asymmetric_low_x.cuh",
    )
    binary = a.output_dir / "probe"
    command = [
        a.nvcc,
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "--extended-lambda",
        "--relocatable-device-code=true",
        "-lineinfo",
        f"-arch=sm_{a.arch}",
        "-Xptxas=-v",
        "-Xcompiler=-ffp-contract=off",
        "-I",
        str(a.output_dir.resolve()),
    ]
    for name, value in {
        "BZ": a.tile[0],
        "BY": a.tile[1],
        "BX": a.tile[2],
        "RELATIVE_ADDRESSING": a.relative_addressing,
        "SPLIT_COMPONENTS": a.split_components,
        "MIN_CTAS": a.min_ctas,
        "TRAIL_RING": a.trail_ring,
        "RING_HOIST": a.hoist_ring_bases,
        "TILED_CONTROL": a.tiled_control,
        "MATCHED_D2_CONTROL": a.matched_d2_control,
        "D2_TILE_X": a.d2_tile_x,
        "D2_TILE_Y": a.d2_tile_y,
        "D2_THREADS": a.d2_threads,
        "THREE_BANKS": a.three_banks,
        "FACE_CACHE": a.face_cache,
        "CIRCULAR_CACHE": a.circular_cache,
        "REGISTER_CACHE": a.register_cache,
        "ASYMMETRIC_CACHE": a.asymmetric_cache,
        "ASYMMETRIC_LOW_X": a.asymmetric_low_x,
        "ASYMMETRIC_HIGH_X": a.asymmetric_high_x,
        "STREAMED_UZ": a.streamed_uz,
        "STREAMED_INTERIOR": a.streamed_interior,
        "ASYMMETRIC_PIPELINE": a.asymmetric_pipeline,
        "ASYMMETRIC_ASYNC": a.asymmetric_async,
        "PHASE_PROFILE": a.phase_profile,
        "REGISTER_THREADS": a.register_threads,
        "REGISTER_LOCAL_CENTERS": a.register_local_centers,
        "REGISTER_PHASE_METADATA": a.register_phase_metadata,
        "REGISTER_WARP_NEIGHBORS": a.register_warp_neighbors,
        "REGISTER_BAD_IMPORT_PHASE": a.register_bad_import_phase,
        "CIRCULAR_ROLLING_HEADS": a.circular_rolling_heads,
        "CIRCULAR_BAD_IMPORT_PHASE": a.circular_cache_bad_import_phase,
        "FACE_CACHE_BAD_H_PHASE": a.face_cache_bad_h_phase,
        "UNSAFE_TWO_EPOCH_NEGATIVE_CONTROL": a.unsafe_two_epoch_negative_control,
    }.items():
        command.append(f"-DMOVING_{name}={int(value)}")
    command.extend([str(source), "-o", str(binary)])
    manifest = {
        "scope": __doc__,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "source_sha256": digest(source),
        "wrapper_sha256": digest(Path(__file__)),
        "generated_cell_sha256": digest(header),
        "face_cache_header_sha256": digest(helper) if helper else None,
        "relative_addressing": a.relative_addressing,
        "tile_zyx": a.tile,
        "split_components": a.split_components,
        "minimum_ctas_launch_bound": a.min_ctas,
        "trail_ring_epochs": a.trail_ring,
        "hoist_ring_bases": a.hoist_ring_bases,
        "tiled_control": a.tiled_control,
        "matched_d2_control": a.matched_d2_control,
        "matched_d2_tile_xy": [a.d2_tile_x, a.d2_tile_y],
        "matched_d2_threads": a.d2_threads,
        "matched_d2_header_sha256": digest(d2_helper) if d2_helper else None,
        "harness_field_banks": 3 if a.three_banks else 4,
        "face_cache": a.face_cache,
        "circular_cache": a.circular_cache,
        "register_cache": a.register_cache,
        "asymmetric_cache": a.asymmetric_cache,
        "asymmetric_low_x": a.asymmetric_low_x,
        "asymmetric_high_x": a.asymmetric_high_x,
        "low_x_header_sha256": digest(low_x_helper) if low_x_helper else None,
        "streamed_uz": a.streamed_uz,
        "streamed_interior": a.streamed_interior,
        "streamed_wrapper_sha256": digest(streamed_helper) if streamed_helper else None,
        "streamed_generated_sha256": digest(streamed_generated)
        if streamed_generated
        else None,
        "asymmetric_pipeline": a.asymmetric_pipeline,
        "asymmetric_async": a.asymmetric_async,
        "asymmetric_pipeline_header_sha256": digest(pipeline_helper)
        if pipeline_helper
        else None,
        "phase_profile": a.phase_profile,
        "phase_profile_header_sha256": digest(phase_helper),
        "asymmetric_header_sha256": digest(asymmetric_helper)
        if asymmetric_helper
        else None,
        "register_threads": a.register_threads,
        "register_local_centers": a.register_local_centers,
        "register_phase_metadata": a.register_phase_metadata,
        "register_warp_neighbors": a.register_warp_neighbors,
        "register_bad_import_phase": a.register_bad_import_phase,
        "register_cache_header_sha256": digest(register_helper)
        if register_helper
        else None,
        "circular_rolling_heads": a.circular_rolling_heads,
        "circular_cache_bad_import_phase": a.circular_cache_bad_import_phase,
        "circular_cache_header_sha256": digest(circular_helper)
        if circular_helper
        else None,
        "face_cache_bad_h_phase": a.face_cache_bad_h_phase,
        "unsafe_two_epoch_negative_control": a.unsafe_two_epoch_negative_control,
        "fresh_processes": a.fresh_processes,
        "samples": a.samples,
        "cases": [],
        "admission_queries": [],
        "cpu_memory_predictions": cpu_predictions,
    }

    def save():
        (a.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    save()
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    (a.output_dir / "build.log").write_text(result.stdout + result.stderr)
    manifest.update(
        compile_exit_code=result.returncode,
        compiler_version=subprocess.check_output([a.nvcc, "--version"], text=True),
    )
    save()
    result.check_returncode()
    manifest["binary_sha256"] = digest(binary)
    save()

    def admission(size, steps):
        result = subprocess.run(
            [str(binary), "--admission-only", str(size), str(steps)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        stem = (
            a.output_dir
            / f"admission-{len(manifest['admission_queries'])}-{size}-{steps}"
        )
        stem.with_suffix(".stdout").write_text(result.stdout)
        stem.with_suffix(".stderr").write_text(result.stderr)
        result.check_returncode()
        value = json.loads(result.stdout)
        prediction = predict_memory(
            size,
            steps,
            a.trail_ring,
            3 if a.three_banks else 4,
            a.tile,
            a.phase_profile,
        )
        for key in (
            "device_payload_bytes",
            "host_workspace_bound_bytes",
            "max_frontier",
            "field_bytes",
            "trail_bytes",
            "solver_payload_bytes",
            "harness_field_banks",
            "tile_zyx",
            "owned_cells_per_box",
            "surface_cells_per_box",
        ):
            if value[key] != prediction[key]:
                raise ValueError(f"CPU/native memory model differs for {key}")
        manifest["admission_queries"].append(value)
        save()
        return value

    for size, steps in a.admission_only:
        admission(size, steps)
    runs = list(a.run)
    for steps in a.largest_run:
        lo, hi = 3, maximum_index_size()
        if not admission(lo, steps)["admitted"]:
            raise RuntimeError("No size is admitted")
        while lo < hi:
            middle = (lo + hi + 1) // 2
            if admission(middle, steps)["admitted"]:
                lo = middle
            else:
                hi = middle - 1
        runs.append((lo, steps))
    for size, steps in dict.fromkeys(map(tuple, runs)):
        plan = admission(size, steps)
        if not plan["admitted"]:
            raise RuntimeError(f"Size {size} is not currently admitted")
        prediction = predict_memory(
            size,
            steps,
            a.trail_ring,
            3 if a.three_banks else 4,
            a.tile,
            a.phase_profile,
        )
        if not prediction["int32_admitted"]:
            raise ValueError("Requested six-field offsets exceed int32")
        for process in range(a.fresh_processes):
            rotation = a.order_rotation + process * (
                4 if a.matched_d2_control else 3 if a.tiled_control else 1
            )
            profile_path = (
                a.output_dir / f"case-{size}-{steps}-process-{process}.phase.bin"
            )
            run = subprocess.run(
                [str(binary), str(size), str(steps), str(a.samples), str(rotation)],
                env={
                    **os.environ,
                    "MOVING_PROFILE_OUTPUT": str(profile_path.resolve()),
                },
                capture_output=True,
                text=True,
                timeout=a.run_timeout,
            )
            stem = a.output_dir / f"case-{size}-{steps}-process-{process}"
            stem.with_suffix(".stdout").write_text(run.stdout)
            stem.with_suffix(".stderr").write_text(run.stderr)
            record = {
                "size": size,
                "steps": steps,
                "process": process,
                "order_rotation": rotation,
                "exit_code": run.returncode,
                "prediction": prediction,
                "admission": plan,
            }
            if run.returncode == 0:
                record["result"] = json.loads(run.stdout)
            manifest["cases"].append(record)
            save()
            run.check_returncode()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
