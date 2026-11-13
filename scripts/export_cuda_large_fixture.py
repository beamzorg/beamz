#!/usr/bin/env python3
"""Stream complete realistic CUDA fixtures to NPY files, with pre-init admission.

--convert rewrites an existing NPZ fixture on CPU without importing JAX.
The new scaling recipe fixes80nm mesh,CPML12,two mode monitors32/64cells
downstream,three frequencies,binary guide and512 useful updates by default.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cuda_fixture_store import ArrayReader, ArrayWriter, digest_array
from scripts.cuda_native_geometry_ir import verify_component_shapes
from scripts.export_cuda_native_fixture import (
    PERSISTENT_INPUTS,
    WORKSPACE_INPUTS,
    ZERO_WORKSPACE_INPUTS,
    abi_input_names,
    capture_mstep,
    flatten_state,
    state_output_mapping,
)
from scripts.export_fdtdx_comparison import file_digest


def recipe(args):
    """Resolve physical work on CPU before importing a device framework."""
    if args.recipe == "guide":
        steps = args.steps if args.steps is not None else 512
        options = SimpleNamespace(
            shape=tuple(args.shape), steps=steps, pml=12, monitors=2, frequencies=3,
            material="binary", source="mode", monitor_type="mode", boundary="all",
            resolution_nm=80.0, monitor_offset_cells=32,
        )
        return options, None, steps, {
            "kind": "realistic_mode_fixed_aperture_offset32_64", "options": vars(options),
            "physical_box_um": [n * .08 for n in args.shape], "not_Mie": True,
        }
    from scripts.benchmark_cuda_mie import case_spec
    from scripts.benchmark_cuda_mie import parser as mie_parser

    if len(set(args.shape)) != 1:
        raise ValueError("The canonical Mie recipe requires a cube")
    resolution = 680. / args.shape[0]
    argv = ["--output", "/tmp/unused-streaming-mie.json", "--resolutions-nm", str(resolution),
            "--duration-fs", str(args.duration_fs)]
    if args.steps is not None:
        argv += ["--steps", str(args.steps)]
    options = mie_parser().parse_args(argv)
    spec = case_spec(options, resolution)
    steps = args.reference_steps if args.reference_steps is not None else spec["timesteps"]
    if not 2 <= steps <= spec["timesteps"]:
        raise ValueError("Reference steps must fit the immutable waveform horizon")
    return options, spec, steps, {
        "kind": "common_mie_binary_sphere", "not_Mie": False, "spec": spec,
        "waveform_horizon_steps": spec["timesteps"], "reference_steps": steps,
        "source_prefix_policy": "Full prepared waveforms retained; only executed reference step count changes",
    }


def moving_packet_admission(captured):
    """Check moving's important data-class predicates before a large solve."""
    values, attributes = captured["arguments"], captured["attributes"]
    minimum = verify_component_shapes(tuple(tuple(v.shape) for v in values[:6]))
    if min(minimum) < 24 or any(np.dtype(v.dtype) != np.dtype("float32") for v in values[:6]):
        raise ValueError("Moving requires canonical FP32 supports and CPML12 room")
    if int(attributes["metric_kind"]) != 0 or int(attributes["boundary_code"]) != 12 << 8:
        raise ValueError("Mie packet requires Cartesian CPML12 without PEC")
    layouts = {len(values[40 + c].shape) for c in range(3)}
    if len(layouts) != 1 or not layouts.issubset({1, 2}):
        raise ValueError("Moving requires one canonical packed E layout")
    if any(values[k].shape != () or np.dtype(values[k].dtype) != np.dtype("float32") for k in range(6, 12)):
        raise ValueError("Moving requires scalar FP32 H coefficients")
    groups = []
    mask = int(attributes["coincident_source_group_mask"])
    for g in range(9):
        coeff, waveform, starts = values[86 + 3 * g:89 + 3 * g]
        count = coeff.shape[0]
        if (len(coeff.shape) != 4 or len(waveform.shape) != 2 or starts.shape != (count, 3)
                or np.dtype(coeff.dtype) != np.dtype("float32") or np.dtype(waveform.dtype) != np.dtype("float32")
                or np.dtype(starts.dtype) != np.dtype("int32")):
            raise ValueError("Moving requires canonical FP32/S32 slab source descriptors")
        if count and (g < 3 or count <= 1 or not mask & (1 << g) or waveform.shape[0] != count):
            raise ValueError("Moving requires post-H/E coincident multi-slab sources")
        if count and not np.all(np.asarray(starts) == np.asarray(starts)[0]):
            raise ValueError("Coincident source flag disagrees with actual uploaded starts")
        groups.append({"group": g, "count": count, "shape": list(coeff.shape), "waveform_shape": list(waveform.shape)})
    return {"checked": True, "source_groups": groups, "E_layout": next(iter(layouts)),
            "scope": "Canonical data-class check; exact GPU memory/workspace admission remains in native worker"}


def available_host_bytes():
    values = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    return int(values["MemAvailable"].split()[0]) * 1024


def admit(shape):
    z, y, x = shape
    field_shapes = [
        (z, y, x + 1),
        (z, y + 1, x),
        (z + 1, y, x),
        (z + 1, y + 1, x),
        (z + 1, y, x + 1),
        (z, y + 1, x + 1),
    ]
    fields = sum(int(np.prod(s)) * 4 for s in field_shapes)
    logical = (z + 1) * (y + 1) * (x + 1)
    psi_cap = 12 * 24 * max((z + 1) * (y + 1), (y + 1) * (x + 1), (z + 1) * (x + 1)) * 4
    readonly_cap = 12 * logical + 8 * logical + (16 << 20)
    # Includes an extra field bank for allocator fragmentation/captured-buffer reuse.
    gpu_required = 4 * fields + 2 * psi_cap + readonly_cap + (2 << 30)
    host_required = 4 * fields + 2 * psi_cap + (4 << 30)
    gpu_free = (
        int(
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        << 20
    )
    host_free = available_host_bytes()
    result = {
        "field_bytes_exact": fields,
        "cpml_bytes_upper_bound": psi_cap,
        "GPU_required_conservative_bytes": gpu_required,
        "GPU_free_bytes": gpu_free,
        "host_required_with4GiB_reserve_bytes": host_required,
        "host_available_bytes": host_free,
    }
    if (
        logical >= np.iinfo(np.int32).max
        or gpu_required > gpu_free
        or host_required > host_free
    ):
        raise MemoryError(f"Pre-init admission rejected: {result}")
    return result


def finish_fixture(fixture, output, writer):
    manifest = writer.finish()
    fixture.update(
        data_format="npy_directory_v1",
        data_directory=writer.directory.name,
        data_file=f"{writer.directory.name}/manifest.json",
        data_sha256=file_digest(manifest),
        streaming_exporter_sha256=file_digest(__file__),
    )
    output.write_text(json.dumps(fixture, indent=2) + "\n")


def convert(args):
    fixture = json.loads(args.convert.read_text())
    path = args.convert.parent / fixture["data_file"]
    if file_digest(path) != fixture["data_sha256"]:
        raise ValueError("Input NPZ hash differs")
    writer = ArrayWriter(args.output.parent / f"{args.output.stem}.arrays")
    dedup = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            value = data[key]
            checksum = digest_array(value)
            identity = (value.dtype.str, tuple(value.shape), checksum)
            if identity in dedup:
                writer.alias(key, dedup[identity])
            else:
                writer.put(key, value)
                dedup[identity] = key
    fixture["converted_from"] = {
        "path": str(args.convert),
        "sha256": file_digest(args.convert),
        "NPZ_sha256": fixture["data_sha256"],
    }
    finish_fixture(fixture, args.output, writer)


def export(args):
    options, mie_spec, steps, fixture_recipe = recipe(args)
    admission = admit(tuple(args.shape))
    # No JAX/backend import is performed before memory admission above.
    import beamz._cuda as extension
    import jax

    from beamz.simulation.execute import build_scan, initial_program_state

    if file_digest(extension.__file__) != args.expected_component_sha:
        raise ValueError("Loaded FFI component hash differs")
    if args.recipe == "mie":
        from scripts.benchmark_cuda_mie import build_simulation

        sim = build_simulation(options, mie_spec)
        program = sim.compile(num_steps=mie_spec["timesteps"], backend="cuda_mstep")
        program = replace(program, config=replace(program.config, num_steps=steps))
    else:
        from scripts.benchmark_cuda_realistic import build_simulation

        sim = build_simulation(options)
        program = sim.compile(num_steps=steps, backend="cuda_mstep")
    if not program.boundary.cpml.enabled or program.config.cuda_storage_axes != (
        0,
        1,
        2,
    ):
        raise ValueError("Large packets require unrotated CPML12")
    state = jax.block_until_ready(
        initial_program_state(program, t=0, current_step=0, monitor_steps=steps)
    )
    captured = capture_mstep(program, state, steps)
    packet_admission = moving_packet_admission(captured) if args.recipe == "mie" else None
    writer = ArrayWriter(args.output.parent / f"{args.output.stem}.arrays")
    inputs = []
    for index, (name, value) in enumerate(
        zip(abi_input_names(), captured["arguments"], strict=True)
    ):
        if np.dtype(value.dtype) not in (np.dtype(np.float32), np.dtype(np.int32)):
            raise ValueError("Large fixture workflow requires FP32/S32")
        key = f"input_{index:03d}"
        descriptor = (
            writer.put(key, shape=value.shape, dtype=value.dtype, zero=True)
            if index in WORKSPACE_INPUTS
            else writer.put(key, value)
        )
        role = (
            "persistent_state"
            if index in PERSISTENT_INPUTS
            else "zero_each_call"
            if index in ZERO_WORKSPACE_INPUTS
            else "uninitialized_workspace"
            if index in WORKSPACE_INPUTS
            else "clock"
            if index in (126, 127, 128)
            else "readonly"
        )
        inputs.append(
            {
                "index": index,
                "name": name,
                "key": key,
                "shape": list(value.shape),
                "dtype": np.dtype(value.dtype).str,
                "role": role,
                "sha256": descriptor["sha256"],
            }
        )
    leaves = []
    for index, (name, value) in enumerate(flatten_state(state)):
        key = f"initial_{index:02d}"
        mapping = state_output_mapping(name)
        source_index = None
        if mapping["kind"] == "field":
            source_index = mapping["component"]
        elif mapping["kind"] == "output":
            raw = mapping["index"]
            source_index = (
                31 + raw - 6
                if raw < 12
                else 62 + raw - 12
                if raw < 18
                else 120 + raw - 30
            )
        elif name == "current_step":
            source_index = 127
        elif name == "t":
            source_index = 126
        if source_index is None:
            writer.put(key, value)
        else:
            writer.alias(key, f"input_{source_index:03d}")
        leaves.append(
            {
                "name": name,
                "initial_key": key,
                "shape": list(value.shape),
                "dtype": np.dtype(value.dtype).str,
                "mapping": mapping,
                "reference_key": f"reference_{index:02d}",
            }
        )
    if len(leaves) != 30:
        raise ValueError("Expected all30 state leaves")
    attributes = {k: np.asarray(v).item() for k, v in captured["attributes"].items()}
    attribute_dtypes = {
        k: np.asarray(v).dtype.str for k, v in captured["attributes"].items()
    }
    outputs = [
        {"index": i, "shape": list(v.shape), "dtype": np.dtype(v.dtype).str}
        for i, v in enumerate(captured["result_values"])
    ]
    target, aliases, plan = (
        captured["target"],
        captured["aliases"],
        captured["plan_summary"],
    )
    monitor_counts = np.asarray(captured["arguments"][117]).copy()
    # Persist a recoverable initial packet before a possibly donated oracle run.
    prepared_manifest = writer.finish()
    prepared = {
        "schema": "beamz.cuda-native-prepared.v1",
        "status": "awaiting_reference",
        "target": target,
        "attributes": attributes,
        "attribute_dtypes": attribute_dtypes,
        "inputs": inputs,
        "outputs": outputs,
        "output_aliases": aliases,
        "state_leaves": leaves,
        "plan": plan,
        "data_directory": writer.directory.name,
        "data_file": f"{writer.directory.name}/manifest.json",
        "data_sha256": file_digest(prepared_manifest),
        "reference_ownership": args.reference_ownership,
        "fixture": fixture_recipe,
        "moving_data_admission": packet_admission,
    }
    args.output.with_suffix(".prepared.json").write_text(
        json.dumps(prepared, indent=2) + "\n"
    )
    # Release captured final/workspace device arrays before allocating the oracle solve.
    del captured
    reference_program = (replace(program, config=replace(program.config, backend="cuda_streamed", cuda_schedule="streamed"))
                         if args.recipe == "mie" else program)
    reference_scan = build_scan(reference_program, donate_state=args.reference_ownership == "owned")
    reference_schedule = reference_scan.cuda_schedule_decision
    if args.recipe == "mie" and reference_schedule["selected"] != "streamed":
        raise ValueError("Mie oracle must use the independent ordinary Program schedule")
    reference = jax.block_until_ready(reference_scan(state, reference_program.coefficients))
    for leaf, (name, value) in zip(leaves, flatten_state(reference), strict=True):
        if leaf["name"] != name:
            raise ValueError("Reference state structure changed")
        writer.put(leaf["reference_key"], value, require_finite=True)
    if args.reference_ownership == "owned":
        # Donation consumes only the private GPU state, never the saved packet.
        initial = ArrayReader(writer.directory)
        for leaf in leaves:
            initial.verify(leaf["initial_key"])
    else:
        for leaf, (_, value) in zip(leaves, flatten_state(state), strict=True):
            if digest_array(value) != writer.arrays[leaf["initial_key"]]["sha256"]:
                raise ValueError(f"Reference mutated immutable input: {leaf['name']}")
    by_name = {leaf["name"]: leaf for leaf in leaves}
    real = np.load(
        writer.directory
        / writer.arrays[by_name["dft_vec_re"]["reference_key"]]["file"],
        mmap_mode="r",
    )
    imag = np.load(
        writer.directory
        / writer.arrays[by_name["dft_vec_im"]["reference_key"]]["file"],
        mmap_mode="r",
    )
    monitor_norms = []
    for frequencies, points, _, offset, _ in monitor_counts:
        begin, end = int(offset), int(offset + 6 * frequencies * points)
        monitor_norms.append(
            float(
                np.sqrt(
                    np.sum(
                        real[begin:end].astype(np.float64) ** 2
                        + imag[begin:end].astype(np.float64) ** 2
                    )
                )
            )
        )
    stats = jax.devices()[0].memory_stats() or {}
    admission["post_reference_jax_allocator"] = {
        key: int(stats[key])
        for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
        if key in stats
    }
    admission["host_peak_RSS_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    fixture = {
        "schema": "beamz.cuda-native-fixture.v1",
        "kind": "mstep",
        "target": target,
        "attributes": attributes,
        "attribute_dtypes": attribute_dtypes,
        "inputs": inputs,
        "outputs": outputs,
        "output_aliases": aliases,
        "state_leaves": leaves,
        "continuation_steps": [],
        "seed": None,
        "fixture": fixture_recipe,
        "moving_data_admission": packet_admission,
        "plan": plan,
        "admission": admission,
        "native_component_sha256": file_digest(extension.__file__),
        "jax_version": jax.__version__,
        "reference_ownership": args.reference_ownership,
        "reference_schedule": reference_schedule,
        "reference_backend": reference_program.config.backend,
        "reference_policy_note": "Only the one-off oracle may donate GPU state; immutable initial packet remains on disk. Timed backend ownership is selected separately.",
        "monitor_DFT_signal_norms": monitor_norms,
        "reference_max_abs": {
            leaf["name"]: writer.arrays[leaf["reference_key"]]["max_abs"]
            for leaf in leaves
        },
    }
    finish_fixture(fixture, args.output, writer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--convert", type=Path)
    parser.add_argument("--shape", type=int, nargs=3, default=(256, 256, 256))
    parser.add_argument("--recipe", choices=("guide", "mie"), default="guide")
    parser.add_argument("--steps", type=int, help="Waveform horizon override; guide default512, Mie defaultfull30fs")
    parser.add_argument("--reference-steps", type=int, help="Mie-only shorter reference with unchanged full waveform horizon")
    parser.add_argument("--duration-fs", type=float, default=30.)
    parser.add_argument("--expected-component-sha")
    parser.add_argument("--admission-only", action="store_true")
    parser.add_argument(
        "--reference-ownership", choices=("preserve", "owned"), default="preserve"
    )
    args = parser.parse_args()
    if not args.convert and not args.admission_only and not args.expected_component_sha:
        parser.error("Export requires --expected-component-sha")
    if (args.steps is not None and args.steps < 2) or min(args.shape) < 112:
        parser.error("Need>=2steps and>=112cells for both fixed-offset monitors")
    if args.recipe == "guide" and args.reference_steps is not None:
        parser.error("--reference-steps is Mie-only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.admission_only:
        _, _, _, physical = recipe(args)
        print(json.dumps({"status": "admitted", "memory": admit(tuple(args.shape)), "fixture": physical}))
        return
    (convert if args.convert else export)(args)
    print(json.dumps({"status": "ok", "fixture": str(args.output)}))


if __name__ == "__main__":
    main()
