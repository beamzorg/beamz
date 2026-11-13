#!/usr/bin/env python3
"""Export immutable BeamZ Mie discretization for a released-FDTDX comparison.

This is setup/export code, not an engine optimization. CUDA is initialized only
inside main/build_program, using the installed, fixed production component.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
NATIVE_SHA = "34de360a045eceb3343a9fc88bb94f4c4b2ee7e65a09b6fc8a8d66ef59e5f836"


def array_digest(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            value.update(chunk)
    return value.hexdigest()


def decode_e_source(coefficients, component, shape):
    axis = component[-1].lower()
    values = np.asarray(getattr(coefficients, "e_source_" + axis))
    if values.dtype != np.int32:
        return np.broadcast_to(values, shape).astype(np.float32).copy()
    pool = np.asarray(getattr(coefficients, "e_decay_" + axis), np.float32)
    if values.ndim == 2:
        from beamz.simulation.cuda.material_blocks import unpack_material_blocks

        return unpack_material_blocks(pool, values, shape)
    result = np.empty(int(np.prod(shape)), np.float32)
    words = values.view(np.uint32)
    for begin in range(0, result.size, 1 << 20):
        linear = np.arange(begin, min(begin + (1 << 20), result.size), dtype=np.int64)
        ids = (words[linear >> 2] >> (8 * (linear & 3)).astype(np.uint32)) & 255
        result[begin : begin + linear.size] = pool[ids]
    return result.reshape(shape)


def build_program(mie_arguments, *, expected_native_sha=NATIVE_SHA):
    import beamz._cuda as extension

    from scripts.benchmark_cuda_mie import build_simulation, case_spec, parser

    if file_digest(extension.__file__) != expected_native_sha:
        raise ValueError("Comparison requires the frozen production-v13 component")
    argv = [
        "--output",
        "/tmp/comparison-unused.json",
        "--resolutions-nm",
        str(mie_arguments["resolution_nm"]),
    ]
    for name in ("steps", "particle_index", "duration_fs"):
        if mie_arguments.get(name) is not None:
            argv += ["--" + name.replace("_", "-"), str(mie_arguments[name])]
    args = parser().parse_args(argv)
    spec = case_spec(args, args.resolutions_nm[0])
    sim = build_simulation(args, spec)
    program = sim.compile(num_steps=spec["timesteps"], backend="cuda_mstep")
    if program.config.cuda_storage_axes != (0, 1, 2):
        raise ValueError("Shared contract requires canonical z/y/x field storage")
    return sim, program, spec


def collect_arrays(program, spec):
    from beamz.const import EPS_0, MU_0
    from beamz.simulation.cuda import pack_dft_monitors
    from beamz.simulation.execute import compiled_source_batches

    arrays, epsilon, inverse, masks = {}, {}, {}, {}
    shapes = {
        name: list(program.boundary.logical_component_shapes[name])
        for name in COMPONENTS
    }
    for component in COMPONENTS[:3]:
        source = decode_e_source(
            program.coefficients, component, tuple(shapes[component])
        )
        inv = (source.astype(np.float64) * EPS_0 / program.config.dt).astype(np.float32)
        eps = np.divide(np.float32(1), inv, out=np.ones_like(inv), where=inv > 0)
        for mapping, stem, value in (
            (epsilon, "epsilon", eps),
            (inverse, "inv_epsilon", inv),
            (masks, "e_update_mask", inv != 0),
        ):
            key = f"{stem}_{component}"
            arrays[key] = value
            mapping[component] = key
        arrays[f"e_update_source_{component}"] = source
    for component in COMPONENTS[3:]:
        axis = component[-1].lower()
        arrays[f"h_update_source_{component}"] = np.asarray(
            getattr(program.coefficients, "h_source_" + axis)
        )
        arrays[f"h_update_decay_{component}"] = np.asarray(
            getattr(program.coefficients, "h_decay_" + axis)
        )
    arrays["cell_epsilon"] = np.asarray(
        program.grid.material_grid.permittivity, np.float32
    )
    cpml = {axis: {side: {} for side in ("low", "high")} for axis in "xyz"}
    terms = []
    for family in ("E", "H"):
        seen = {}
        for index, term in enumerate(
            getattr(program.boundary.cpml, family.lower() + "_terms")
        ):
            if (term.slab.low, term.slab.high) != (12, 12):
                raise ValueError(
                    "The shared workload requires twelve cells on every CPML face"
                )
            descriptor = {
                "family": family,
                "component": term.component,
                "axis_zyx": term.axis,
                "sign": term.sign,
                "low": term.slab.low,
                "high": term.slab.high,
                "shape": list(term.slab.shape),
                "logical_stop": term.slab.logical_stop,
            }
            axis = "zyx"[term.axis]
            values = {}
            for kind, attribute in (("a", "a"), ("b", "b"), ("inv_kappa", "inv_kappa")):
                raw = np.asarray(getattr(term, attribute), np.float32)
                key = f"cpml_{family}_term_{index}_{kind}"
                arrays[key] = raw
                descriptor[kind] = key
                values[kind] = raw.reshape(-1)
            if axis in seen:
                for kind in values:
                    if not np.array_equal(seen[axis][kind], values[kind]):
                        raise ValueError(
                            "CPML profiles differ between components on one derivative axis"
                        )
            else:
                seen[axis] = values
                for side in ("low", "high"):
                    cpml[axis][side][family] = {}
                    for kind, vector in values.items():
                        face = vector[:12] if side == "low" else vector[-12:]
                        if family == "E" and side == "high":
                            identity = 0 if kind == "a" else 1
                            face = np.concatenate(
                                (np.array([identity], np.float32), face[:11])
                            )
                        key = f"cpml_common_{axis}_{side}_{family}_{kind}"
                        arrays[key] = face.copy()
                        cpml[axis][side][family][kind] = key
            terms.append(descriptor)
    sources = []
    for index, source in enumerate(program.sources):
        if source.timing == "pre_e":
            raise ValueError(
                "Unexpected pre_E source term; report and extend the clock map before benchmarking"
            )
        if not source.is_slab:
            raise ValueError(
                "The common Mie source is expected to lower to exact slabs"
            )
        coeff, waveform = f"source_{index}_coeff", f"source_{index}_waveform"
        arrays[coeff] = np.asarray(source.coeff, np.float32)
        arrays[waveform] = np.asarray(source.waveform, np.float32)
        sources.append(
            {
                "component": source.component,
                "timing": source.timing,
                "coeff": coeff,
                "waveform": waveform,
                "slab_starts": list(source.slab_starts),
                "slab_sizes": list(source.slab_sizes),
                "source_index": source.source_index,
            }
        )
    groups = []
    for (timing, component), (group, rest) in compiled_source_batches(
        program.sources
    ).items():
        if group is None:
            continue
        prefix = f"source_group_{timing}_{component}"
        for name in ("waveforms", "coeffs", "starts"):
            arrays[prefix + "_" + name] = np.asarray(getattr(group, name))
        groups.append(
            {
                "timing": timing,
                "component": component,
                "prefix": prefix,
                "n": group.n,
                "max_sizes": list(group.max_sizes),
                "unbatched_terms": len(rest),
            }
        )
    monitors = []
    for monitor in program.monitors:
        if (
            monitor.freq_count != 1
            or monitor.dft_record_interval != 1
            or monitor.dft_window_code != 0
        ):
            raise ValueError(
                "Expected one-frequency, every-step, unapodized field DFT monitors"
            )
        components = {}
        for component, indices, weights in zip(
            COMPONENTS, monitor.dft_flat_idx, monitor.dft_weights, strict=True
        ):
            idx_key, w_key = (
                f"{monitor.name}_{component}_indices",
                f"{monitor.name}_{component}_weights",
            )
            arrays[idx_key] = np.asarray(indices, np.int32)
            arrays[w_key] = np.asarray(weights, np.float32)
            components[component] = {
                "indices": idx_key,
                "weights": w_key,
                "layout": "point_neighbor",
            }
        monitors.append(
            {
                "name": monitor.name,
                "components": components,
                "points": monitor.dft_point_count,
                "dft_value_offset": monitor.dft_value_offset,
                "record_interval": 1,
                "component_mask": np.asarray(monitor.dft_component_mask).tolist(),
            }
        )
    for name, value in zip(
        (
            "indices",
            "weights",
            "frequencies",
            "component_masks",
            "counts",
            "codes",
            "windows",
        ),
        pack_dft_monitors(program.monitors),
        strict=True,
    ):
        arrays["packed_dft_" + name] = np.asarray(value)
    metadata = {
        "component_shapes": shapes,
        "epsilon": epsilon,
        "inv_epsilon": inverse,
        "electric_update_masks": masks,
        "cpml": cpml,
        "cpml_original_terms": terms,
        "sources": sources,
        "source_groups": groups,
        "monitors": monitors,
        "frequency_hz": float(np.asarray(program.monitors[0].freq_hz)[0]),
        "constants": {"epsilon0_si": EPS_0, "mu0_si": MU_0},
        "pre_e_terms": [],
    }
    return arrays, metadata


def canonical_work(contract, arrays):
    keys = {
        "electric_coefficients": [v for v in contract["inv_epsilon"].values()],
        "sources": [s[k] for s in contract["sources"] for k in ("coeff", "waveform")],
        "monitors": [
            v[k]
            for m in contract["monitors"]
            for v in m["components"].values()
            for k in ("indices", "weights")
        ],
        "cpml": [
            v
            for axis in contract["cpml"].values()
            for side in axis.values()
            for family in side.values()
            for v in family.values()
        ],
    }
    hashes = {
        name: hashlib.sha256(
            json.dumps(
                {key: array_digest(arrays[key]) for key in sorted(names)},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        for name, names in keys.items()
    }
    return {
        "comparison_mode": "adapter-configured common discrete data; released kernels unchanged",
        "shape_zyx": contract["shape_zyx"],
        "dx_m": contract["dx_m"],
        "dt_s": contract["dt_s"],
        "steps": contract["steps"],
        "pml_cells_each_face": 12,
        "field_dtype": "float32",
        "frequency_hz": contract["frequency_hz"],
        "source_terms": len(contract["sources"]),
        "detector_points": [m["points"] for m in contract["monitors"]],
        "detector_components": list(COMPONENTS),
        "dft_interval": 1,
        "dft_complex_accumulations": contract["steps"]
        * 6
        * sum(m["points"] for m in contract["monitors"]),
        "detector_phase": "exp(+i*omega*(n+1)*dt), all six; no H time averaging",
        "clock_map": "FD H[n]=BeamZ H[n]; FD E[n]=BeamZ E[n-1]; E[-1]=0",
        "data_hashes": hashes,
        "gpu_count": 1,
        "gradients": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution-nm", type=float, default=5)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--vacuum", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    mie = {
        "resolution_nm": args.resolution_nm,
        "steps": args.steps,
        "particle_index": 1 if args.vacuum else 2,
        "duration_fs": 30,
    }
    _, program, spec = build_program(mie)
    arrays, metadata = collect_arrays(program, spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_path = args.output.with_suffix(".npz")
    np.savez_compressed(data_path, **arrays)
    contract = {
        "schema": "beamz.fdtdx-common-discrete.v1",
        "mie_arguments": mie,
        "shape_zyx": spec["shape_zyx"],
        "dx_m": spec["resolution_nm"] * 1e-9,
        "dt_s": program.config.dt,
        "steps": program.config.num_steps,
        "pml_cells": 12,
        "data_file": data_path.name,
        "data_sha256": file_digest(data_path),
        "beamz_native_sha256": NATIVE_SHA,
        **metadata,
    }
    contract["array_sha256"] = {
        name: array_digest(value) for name, value in arrays.items()
    }
    contract["canonical"] = canonical_work(contract, arrays)
    args.output.write_text(json.dumps(contract, indent=2) + "\n")
    print(
        json.dumps(
            {
                "contract": str(args.output),
                "data": str(data_path),
                "canonical": contract["canonical"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
