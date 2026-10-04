"""Search same-physics CUDA configurations with donation and exact state parity."""

import argparse
import gc
import hashlib
import itertools
import json
import os
import random
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", type=int, nargs=3, default=(128, 256, 512))
    p.add_argument("--presteps", type=int, default=0)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--configs", nargs="*")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if min(a.shape) <= 24 or a.presteps < 0 or a.samples < 7:
        p.error("Need a nonempty CPML interior, nonnegative presteps and >=7 samples")
    if a.output.exists():
        p.error("Use a fresh output path")
    os.environ.update(
        BEAMZ_CUDA_AUTOTUNE="off",
        BEAMZ_CUDA_CPML_PSI_PRECISION="fp32",
        BEAMZ_CUDA_TEMPORAL_STEPS="1",
        BEAMZ_CUDA_CPML_TEMPORAL="0",
        BEAMZ_CUDA_CPML_SPATIAL="0",
        BEAMZ_CUDA_FIELD_PADDING="none",
        BEAMZ_CUDA_CPML_SHELL_TILE="64x4",
        BEAMZ_CUDA_STORAGE_AXES="012",
        BEAMZ_CUDA_CPML_TILE="32x8x8",
        BEAMZ_CUDA_CPML_CORE_FUSION="0",
    )
    import beamz._cuda as extension
    import jax
    import numpy as np

    from beamz.simulation import _cuda_abi as abi
    from beamz.simulation.execute import build_scan, initial_program_state
    from scripts.benchmark_cuda_realistic import build_simulation

    assert "H100" in jax.devices()[0].device_kind
    sim = build_simulation(
        SimpleNamespace(
            shape=a.shape,
            steps=256,
            presteps=a.presteps,
            pml=12,
            monitors=1,
            frequencies=3,
            material="binary",
            source="mode",
            monitor_type="mode",
        )
    )
    base = sim.compile(num_steps=256, backend="cuda_streamed")
    pristine = initial_program_state(base, t=0, current_step=0, monitor_steps=256)
    if a.presteps:
        pristine = sim.advance(
            state=pristine, num_steps=a.presteps, backend="cuda_streamed"
        ).state
    jax.block_until_ready(pristine)
    variants = {}
    tiles = {"64x4": 0, "32x8": abi.CUDA_SHELL32X8, "32x4": abi.CUDA_SHELL32X4}
    for axes, tile, fusion in itertools.product(("012", "120", "201"), tiles, (0, 1)):
        key = f"{axes}-{tile}-f{fusion}"
        variants[key] = dict(
            axes=tuple(map(int, axes)), tile=tile, fusion=fusion, donate=True
        )
    variants["012-64x4-f0-preserve"] = dict(variants["012-64x4-f0"], donate=False)
    if a.configs:
        unknown = set(a.configs) - variants.keys()
        if unknown:
            p.error(f"Unknown configs {unknown}")
        variants = {
            k: v for k, v in variants.items() if k in set(a.configs) | {"012-64x4-f0"}
        }
    records, executables = {}, {}
    reference = None

    def run(key):
        os.environ["BEAMZ_CUDA_CPML_CORE_FUSION"] = str(variants[key]["fusion"])
        state = jax.tree.map(lambda x: x.copy(), pristine)
        jax.block_until_ready(state)
        start = time.perf_counter()
        result = executables[key](state, base.coefficients)
        jax.block_until_ready(result)
        return result, time.perf_counter() - start

    data = dict(
        shape=a.shape,
        steps=256,
        presteps=a.presteps,
        pml=12,
        source="mode",
        monitors=1,
        monitor_type="mode",
        frequencies=3,
        precision="fp32",
        variants=variants,
        records=records,
        native_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        jax_version=jax.__version__,
        device=jax.devices()[0].device_kind,
        allocation_outside_timer=True,
        timing_orders=[],
        telemetry=[],
    )

    def save():
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(data, indent=2) + "\n")

    for key, v in variants.items():
        flags = (
            base.config.cuda_flags
            & ~(
                abi.CUDA_SHELL32X8
                | abi.CUDA_SHELL32X4
                | abi.CUDA_TEMPORAL_PAIR
                | abi.CUDA_CPML_PAIR
            )
        ) | tiles[v["tile"]]
        program = replace(
            base,
            config=replace(base.config, cuda_flags=flags, cuda_storage_axes=v["axes"]),
        )
        start = time.perf_counter()
        executables[key] = (
            build_scan(program, donate_state=v["donate"])
            .lower(pristine, base.coefficients)
            .compile()
        )
        records[key] = dict(compile_s=time.perf_counter() - start, samples_s=[])
        result, _ = run(key)
        host = jax.device_get(result)
        if reference is None:
            reference = host
        for i, (ref, got) in enumerate(
            zip(jax.tree.leaves(reference), jax.tree.leaves(host), strict=True)
        ):
            assert np.isfinite(np.asarray(got)).all(), f"{key} nonfinite leaf {i}"
            np.testing.assert_array_equal(got, ref, err_msg=f"{key} leaf {i}")
        records[key]["exact_complete_state_parity"] = True
        del result, host
        for _ in range(4):
            result, _ = run(key)
            del result
        print(key, "compiled, parity passed, warmed", flush=True)
        save()
    del reference
    gc.collect()
    names = list(variants)
    random.Random(20260924).shuffle(names)
    for r in range(a.samples):
        order = names[r % len(names) :] + names[: r % len(names)]
        if r % 2:
            order = order[::-1]
        data["timing_orders"].append(order)
        for key in order:
            result, seconds = run(key)
            records[key]["samples_s"].append(seconds)
            records[key]["gcups"] = (
                int(np.prod(a.shape))
                * 256
                / statistics.median(records[key]["samples_s"])
                / 1e9
            )
            del result
        telemetry = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        data["telemetry"].append(telemetry.stdout.strip())
        save()
        print(
            "round",
            r + 1,
            "best",
            max(records, key=lambda k: records[k]["gcups"]),
            flush=True,
        )
    print(json.dumps({k: r["gcups"] for k, r in records.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
