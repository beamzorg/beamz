"""Matched origin/main and shared-native CUDA benchmarks from four examples.

The photonic variants keep their geometries but use regular FP32 grids,
CPML12, fixed time horizons and flux/field outputs. Ring Q and final MMI design
quality are not convergence claims. The solver never receives a geometry name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main() -> int:
    """Build and execute one declared workload in a fresh measured process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("mie", "cosine", "ring", "mmi"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("main", "bounded"), required=True)
    parser.add_argument("--module", type=Path)
    parser.add_argument("--warm-runs", type=int, default=1)
    args = parser.parse_args()
    started = time.perf_counter()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    os.environ.pop("BEAMZ_DOCS_TEST", None)
    os.environ.update(MPLBACKEND="Agg", XLA_PYTHON_CLIENT_PREALLOCATE="false",
        BEAMZ_EXECUTION_BACKEND="cuda_streamed", BEAMZ_TRACE_PREPARATION=str(output / "frontend-preparation.jsonl"))
    if args.variant == "bounded":
        os.environ.update(BEAMZ_NATIVE_HOST_RESET="1", BEAMZ_SETUP_DEVICE="cpu")
    else:
        os.environ["BEAMZ_NATIVE_HOST_RESET"] = "0"
        os.environ.pop("BEAMZ_SETUP_DEVICE", None)
    sys.path[:0] = [str(root), str(repo)]
    import numpy as np
    import beamz as bz
    from scripts.cuda_notebook_runtime import Trial
    trial = Trial(args.variant, output, str(args.module.resolve()) if args.module else None)
    ns = dict(np=np, bz=bz, __name__="__benchmark__")
    setup_start = time.perf_counter()
    if args.case == "cosine":
        path = repo / "examples/notebooks/cosine_waveguide_crossing_cuda_comparison.ipynb"
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            code = "".join(cell["source"])
            if cell["cell_type"] == "code":
                if "sim_data = sim.run" in code or "sim.run(" in code:
                    break
                exec(compile(code, str(path), "exec"), ns)
        sim = ns["sim"]
        freqs = ns["freqs"]
        source_file = str(path)
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    elif args.case in ("ring", "mmi"):
        name = "ring_resonator_add_drop" if args.case == "ring" else "mmi1x4_power_splitter"
        path = repo / f"examples/notebooks/{name}.ipynb"
        notebook = json.loads(path.read_text())
        code = "\n".join("".join(notebook["cells"][i]["source"]) for i in (1, 3, 5))
        dx = 25e-9 if args.case == "ring" else 30e-9
        if args.case == "ring":
            a = code.index("    grid_spec = (")
            b = code.index("    source_size =", a)
            code = code[:a] + "    grid_spec = bz.GridSpec.uniform(25e-9)\n" + code[b:]
            code = code.replace("3 if test_mode else 201", "31")
            code = code.replace("0.5 if test_mode else 4.0", "1.0")
            code = code.replace("thickness=0.6 * um", "thickness=12 * 25e-9")
            code = code.replace("(8 / freq0) if test_mode else 30e-12", "1e-12")
        else:
            a = code.index("    grid_spec = bz.GridSpec.auto(")
            b = code.index("    # Keep the physical PML", a)
            code = code[:a] + "    grid_spec = bz.GridSpec.uniform(30e-9)\n" + code[b:]
            code = code.replace("3 if test_mode else 401", "31")
            code = code.replace("pml_thickness = lambda0 / n_si", "pml_thickness = 12 * 30e-9")
            code = code.replace("0.5 if test_mode else 4.0", "1.0")
            code = code.replace("(*flux_monitors, *mode_monitors, *((field_monitor,) if include_field else ()))",
                                "(*flux_monitors, *((field_monitor,) if include_field else ()))")
            code = code.replace('fields=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")', 'fields=("Ez",)')
            code = code.replace("(8 if test_mode else 5_000) / freq0", "0.7e-12")
        exec(compile(code, str(path), "exec"), ns)
        sim, setup = ns["make_sim"]() if args.case == "ring" else ns["make_sim"](ns["L_MMI"], ns["W_MMI"])
        freqs = ns["freqs"]
        source_file, source_hash = str(path), hashlib.sha256(code.encode()).hexdigest()
    else:
        from scripts.benchmark_cuda_mie import build_simulation, parser as mie_parser, case_spec
        spec_args = mie_parser().parse_args(["--output", str(output / "mie-unused.json"),
            "--resolutions-nm", str(680 / 384), "--pml-cells", "12", "--duration-fs", "30"])
        spec = case_spec(spec_args, 680 / 384)
        sim = build_simulation(spec_args, spec)
        freqs = np.array([bz.LIGHT_SPEED / (500e-9)])
        source_file, source_hash = "scripts/benchmark_cuda_mie.py", hashlib.sha256((repo / "scripts/benchmark_cuda_mie.py").read_bytes()).hexdigest()
    construction_s = time.perf_counter() - setup_start
    print(json.dumps(dict(stage="constructed", case=args.case, shape=sim.grid.shape, steps=sim.num_steps,
                          construction_s=construction_s)), flush=True)
    if sim.grid.metric_kind != "isotropic_uniform":
        raise ValueError("Matched temporal comparison requires a regular Cartesian grid")
    cold_start = time.perf_counter()
    sim_data = sim.run(progress=True, backend="cuda_streamed")
    cold_public_s = time.perf_counter() - cold_start
    raw = sim_data.renormalize(None)
    arrays = dict(freqs=freqs, launched_power=np.asarray(sim_data.launched_power()))
    for monitor in sim.monitors:
        value = raw[monitor.name]
        if hasattr(value, "flux"):
            arrays[monitor.name + ".flux"] = np.asarray(value.flux)
        if hasattr(value, "dft_fields"):
            for field, data in value.dft_fields.items():
                arrays[monitor.name + "." + field] = np.asarray(data)
    namespace = dict(sim=sim, sim_data=sim_data, freqs=freqs, benchmark_observables=arrays)
    cold_wall_s = time.perf_counter() - started
    trial.export(namespace)
    for repeat in range(args.warm_runs):
        repeated = sim.run(progress=True, backend="cuda_streamed").renormalize(None)
        for monitor in sim.monitors:
            expected, actual = raw[monitor.name], repeated[monitor.name]
            if hasattr(expected, "flux"):
                np.testing.assert_array_equal(np.asarray(actual.flux), np.asarray(expected.flux))
            if hasattr(expected, "dft_fields"):
                for field, data in expected.dft_fields.items():
                    np.testing.assert_array_equal(np.asarray(actual.dft_fields[field]), np.asarray(data))
    result = trial.export(namespace, finish=True)
    result.update(case=args.case, construction_s=construction_s, cold_public_s=cold_public_s,
        total_cold_wall_s=cold_wall_s, source_file=source_file, source_sha256=source_hash,
        frontend_source_root=str(root),
        material_packer_sha256=hashlib.sha256((root / "beamz/simulation/cuda/material_blocks.py").read_bytes()).hexdigest(),
        benchmark_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime_driver_sha256=hashlib.sha256((repo / "scripts/cuda_notebook_runtime.py").read_bytes()).hexdigest(),
        cpml_cells_per_face=12, fixed_horizon=True,
        benchmark_variations="Regular grid, fixed duration; ring/MMI31 spectral samples and flux+field monitors; MMI omits mode projection monitors")
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(stage="complete", case=args.case, variant=args.variant,
        engine_s=[x["engine_s"] for x in result["runs"]], cold_wall_s=cold_wall_s,
        host_peak=result["memory"]["host_maxrss_bytes"], gpu_peak=result["memory"]["peak_gpu_process_bytes"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
