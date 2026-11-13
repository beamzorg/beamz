#!/usr/bin/env python3
"""Fixed BeamZ CUDA adapter for the common-discrete-data FDTDX comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_fdtdx_comparison import (
    COMPONENTS,
    NATIVE_SHA,
    array_digest,
    build_program,
    canonical_work,
    collect_arrays,
    file_digest,
)


def parameter_difference(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (
        actual.dtype != np.float32
        or expected.dtype != np.float32
        or actual.shape != expected.shape
    ):
        raise ValueError("Canonical parameter shape or FP32 dtype differs")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise ValueError("Nonfinite canonical parameter")

    def ordered(value):
        bits = value.view(np.uint32)
        return np.where(
            bits & np.uint32(0x80000000), ~bits, bits ^ np.uint32(0x80000000)
        ).astype(np.int64)

    ulps = np.abs(ordered(actual) - ordered(expected))
    result = {
        "max_ulp": int(ulps.max(initial=0)),
        "changed_values": int(np.count_nonzero(ulps)),
        "max_abs": float(np.max(np.abs(actual - expected), initial=0)),
    }
    if result["max_ulp"] > 8:
        raise ValueError(f"Prepared parameter differs by more than 8 ULP: {result}")
    return result


def configure_common_parameters(program, contract, data):
    """Configure shared data, retaining strict static and numerical audits."""
    import jax.numpy as jnp

    configuration_start = time.perf_counter()
    comparison_s = 0.0
    differences, pinned = {}, []
    families = {}
    for family in ("E", "H"):
        original = getattr(program.boundary.cpml, family.lower() + "_terms")
        specs = [
            term for term in contract["cpml_original_terms"] if term["family"] == family
        ]
        if len(original) != len(specs):
            raise ValueError("Canonical CPML term count differs")
        terms = []
        for term, spec in zip(original, specs, strict=True):
            static = {
                "component": term.component,
                "axis_zyx": term.axis,
                "sign": term.sign,
                "low": term.slab.low,
                "high": term.slab.high,
                "shape": list(term.slab.shape),
                "logical_stop": term.slab.logical_stop,
            }
            if any(value != spec[key] for key, value in static.items()):
                raise ValueError("Canonical CPML term geometry differs")
            changes = {}
            for kind in ("a", "b", "inv_kappa"):
                key = spec[kind]
                value = np.asarray(data[key])
                comparison_start = time.perf_counter()
                differences[key] = parameter_difference(getattr(term, kind), value)
                comparison_s += time.perf_counter() - comparison_start
                changes[kind] = jnp.asarray(value)
                pinned.append(key)
            terms.append(replace(term, **changes))
        families[family.lower() + "_terms"] = tuple(terms)
    cpml = replace(program.boundary.cpml, **families)
    boundary = replace(program.boundary, cpml=cpml)
    if len(program.sources) != len(contract["sources"]):
        raise ValueError("Canonical source term count differs")
    sources = []
    for source, spec in zip(program.sources, contract["sources"], strict=True):
        static = {
            "component": source.component,
            "timing": source.timing,
            "slab_starts": list(source.slab_starts),
            "slab_sizes": list(source.slab_sizes),
            "source_index": source.source_index,
        }
        if not source.is_slab or any(
            value != spec[key] for key, value in static.items()
        ):
            raise ValueError("Canonical source term geometry or timing differs")
        changes = {}
        for attribute, kind in (("coeff", "coeff"), ("waveform", "waveform")):
            key = spec[kind]
            value = np.asarray(data[key])
            comparison_start = time.perf_counter()
            differences[key] = parameter_difference(getattr(source, attribute), value)
            comparison_s += time.perf_counter() - comparison_start
            changes[attribute] = jnp.asarray(value)
            pinned.append(key)
        sources.append(replace(source, **changes))
    configured = replace(program, boundary=boundary, sources=tuple(sources))
    return configured, {
        "policy": "Explicit canonical CPML/source configuration; static metadata exact, raw FP32 preparation within 8 ULP, post-configuration all-array hashes exact",
        "parameters": differences,
        "pinned_array_keys": pinned,
        "configuration_s_in_cold": time.perf_counter()
        - configuration_start
        - comparison_s,
    }


def gather_fields(state, monitor, data):
    values = []
    for component in COMPONENTS:
        item = monitor["components"][component]
        indices = np.asarray(data[item["indices"]], np.int64)
        weights = np.asarray(data[item["weights"]], np.float32)
        field = np.asarray(getattr(state, component.lower())).reshape(-1)
        values.append(np.sum(field[indices] * weights, axis=1, dtype=np.float32))
    return np.stack(values)


def export_state(path, contract, data, program, initial, final):
    import jax

    from beamz.simulation.execute import build_scan, initial_program_state

    steps = int(contract["steps"])
    short = replace(program, config=replace(program.config, num_steps=steps - 1))
    short_initial = initial_program_state(
        short, t=0, current_step=0, monitor_steps=steps - 1
    )
    penultimate = jax.block_until_ready(
        build_scan(short)(short_initial, short.coefficients)
    )
    shape = tuple(contract["shape_zyx"])
    crop = tuple(slice(0, n) for n in shape)
    payload = {
        "E_xyz": np.stack(
            [
                np.asarray(getattr(penultimate, c.lower()))[crop].transpose(2, 1, 0)
                for c in COMPONENTS[:3]
            ]
        ),
        "H_si_xyz": np.stack(
            [
                np.asarray(getattr(final, c.lower()))[crop].transpose(2, 1, 0)
                for c in COMPONENTS[3:]
            ]
        ),
        "E_final_xyz": np.stack(
            [
                np.asarray(getattr(final, c.lower()))[crop].transpose(2, 1, 0)
                for c in COMPONENTS[:3]
            ]
        ),
        "steps": np.asarray(steps, np.int32),
        "dt_s": np.asarray(contract["dt_s"], np.float64),
    }
    phase = np.complex64(
        np.exp(2j * np.pi * contract["frequency_hz"] * contract["dt_s"] * steps)
    )
    for monitor in contract["monitors"]:
        offset, points = monitor["dft_value_offset"], monitor["points"]
        raw = (
            np.asarray(final.dft_vec_re[offset : offset + 6 * points])
            + 1j * np.asarray(final.dft_vec_im[offset : offset + 6 * points])
        ).reshape(6, points)
        samples = gather_fields(final, monitor, data)
        common = raw.copy()
        common[:3] -= samples[:3] * phase
        payload[monitor["name"] + "_phasor_raw"] = raw
        payload[monitor["name"] + "_phasor"] = common
        payload[monitor["name"] + "_final_e_samples"] = samples[:3]
    if not all(np.all(np.isfinite(value)) for value in payload.values()):
        raise ValueError("Nonfinite common state export")
    np.savez_compressed(path, **payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument(
        "--phase", choices=("validate", "benchmark"), default="validate"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-output", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--public-samples", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--expected-native-sha", default=NATIVE_SHA)
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import beamz._cuda as extension
    import jax

    from beamz.simulation.execute import build_scan, initial_program_state
    from scripts.benchmark_cuda_revision import DeviceMemorySampler

    if jax.devices()[0].platform != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("Comparison requires exactly one visible GPU")
    contract = json.loads(args.contract.read_text())
    data_path = args.data or Path(contract["data_file"])
    if not data_path.is_absolute():
        data_path = args.contract.parent / data_path
    if file_digest(data_path) != contract["data_sha256"]:
        raise ValueError("Canonical NPZ does not match its manifest")
    started = time.perf_counter()
    sim, program, spec = build_program(
        contract["mie_arguments"], expected_native_sha=args.expected_native_sha
    )
    audit_start = time.perf_counter()
    with np.load(data_path, allow_pickle=False) as data:
        program, parameter_audit = configure_common_parameters(program, contract, data)
    actual_arrays, _ = collect_arrays(program, spec)
    actual_hashes = {key: array_digest(value) for key, value in actual_arrays.items()}
    if actual_hashes != contract["array_sha256"]:
        bad = [
            key
            for key in actual_hashes
            if actual_hashes[key] != contract["array_sha256"].get(key)
        ]
        raise ValueError(f"Prepared BeamZ arrays differ from shared contract: {bad}")
    if canonical_work(contract, actual_arrays) != contract["canonical"]:
        raise ValueError("Resolved BeamZ canonical work differs")
    del actual_arrays
    audit_s = (
        time.perf_counter() - audit_start - parameter_audit["configuration_s_in_cold"]
    )
    initial = initial_program_state(
        program, t=0, current_step=0, monitor_steps=contract["steps"]
    )
    jax.block_until_ready(initial)
    setup_s = time.perf_counter() - started - audit_s
    sampler = DeviceMemorySampler(1)
    scan = build_scan(program, donate_state=False)
    started = time.perf_counter()
    lowered = scan.lower(initial, program.coefficients)
    lower_s = time.perf_counter() - started
    started = time.perf_counter()
    executable = lowered.compile()
    compile_s = time.perf_counter() - started
    started = time.perf_counter()
    final = jax.block_until_ready(executable(initial, program.coefficients))
    first_s = time.perf_counter() - started
    cold_s = setup_s + lower_s + compile_s + first_s
    solve, public = [], []
    if args.phase == "benchmark":
        for index in range(args.warmups + args.samples):
            final = None
            started = time.perf_counter()
            final = jax.block_until_ready(executable(initial, program.coefficients))
            if index >= args.warmups:
                solve.append(time.perf_counter() - started)
        # Match the release's warmed jitted solver-wrapper dispatch scope.
        final = None
        final = jax.block_until_ready(scan(initial, program.coefficients))
        for _ in range(args.public_samples):
            final = None
            started = time.perf_counter()
            final = jax.block_until_ready(scan(initial, program.coefficients))
            public.append(time.perf_counter() - started)
    memory = {
        "device_global": sampler.finish(),
        "allocator": jax.devices()[0].memory_stats(),
        "device_global_scope": "Compile and executable/public execution after setup/input audits; excludes state exports and penultimate solve",
        "allocator_scope": "JAX process-lifetime high-water mark including setup; not an isolated execution peak",
    }
    report = {
        "status": "ok",
        "phase": args.phase,
        "engine": {
            "name": "beamz",
            "version": extension.__version__,
            "abi": extension.__abi_version__,
            "native_sha256": file_digest(extension.__file__),
        },
        "contract_sha256": file_digest(args.contract),
        "data_sha256": file_digest(data_path),
        "actual_work": {
            "canonical": contract["canonical"],
            "native": {
                "backend": program.config.backend,
                "cuda_flags": program.config.cuda_flags,
                "component_shapes": contract["component_shapes"],
                "storage_axes": list(program.config.cuda_storage_axes),
                "field_units": "SI electric and magnetic fields",
                "steps": program.config.num_steps,
                "dt_s": program.config.dt,
                "array_readback": "exact",
                "canonical_parameters": "Original CPML terms and source arrays explicitly configured from shared NPZ",
                "source_clock": "H[n] after H update; E[n] after E update",
                "detector_points": [m["points"] for m in contract["monitors"]],
            },
        },
        "timing": {
            "setup_s": setup_s,
            "lower_s": lower_s,
            "compile_s": compile_s,
            "first_executable_s": first_s,
            "cold_scene_to_first_s": cold_s,
            "warm_solve_samples_s": solve,
            "warm_public_samples_s": public,
            "synchronization": "all_returned_device_leaves",
            "reset_policy": "same_immutable_initial_state_each_call",
            "public_scope": "Warmed jitted BeamZ scan(initial_state, coefficients) solver-wrapper dispatch; excludes CPU result assembly",
        },
        "memory": memory,
        "input_audit_s_excluded_from_cold": audit_s,
        "prepared_parameter_audit": parameter_audit,
        "jax_version": jax.__version__,
        "adapter_sha256": file_digest(__file__),
        "schedule_decision": scan.cuda_schedule_decision,
    }
    if args.phase == "validate" and args.state_output:
        with np.load(data_path, allow_pickle=False) as data:
            export_state(args.state_output, contract, data, program, initial, final)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "phase": args.phase, "status": "ok"}))


if __name__ == "__main__":
    main()
