"""Paired H100 CPML12 fusion/tile study with complete-state parity checks."""

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import beamz._cuda as extension
import jax
import numpy as np

from beamz.simulation import _cuda_abi as abi
from beamz.simulation.execute import build_scan, initial_program_state
from scripts.benchmark_cuda_realistic import build_simulation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", nargs=3, type=int, required=True)
    p.add_argument("--material", choices=("binary", "smooth"), default="binary")
    p.add_argument("--presteps", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--study", choices=("layouts", "fusion", "shells"), default="fusion")
    p.add_argument("--profile", choices=("queue", "fused"))
    a = p.parse_args()
    if min(a.shape) <= 24 or a.presteps < 0:
        p.error("need a nonempty 12-cell CPML interior and nonnegative presteps")
    os.environ.update(
        BEAMZ_CUDA_CPML_PSI_PRECISION="fp32",
        BEAMZ_CUDA_TEMPORAL_STEPS="1",
        BEAMZ_CUDA_CPML_TEMPORAL="0",
        BEAMZ_CUDA_CPML_PAIR_TILE="oriented",
        BEAMZ_CUDA_CPML_SPATIAL="0",
        BEAMZ_CUDA_FIELD_PADDING="none",
        BEAMZ_CUDA_CPML_SHELL_TILE="64x4",
        BEAMZ_CUDA_STORAGE_AXES="012",
    )
    os.environ.pop("BEAMZ_CUDA_CPML_CORE_FUSION", None)
    os.environ.pop("BEAMZ_CUDA_CPML_TILE", None)
    sim = build_simulation(
        SimpleNamespace(
            shape=tuple(a.shape),
            steps=256,
            presteps=a.presteps,
            pml=12,
            monitors=1,
            frequencies=3,
            material=a.material,
            source="mode",
            monitor_type="mode",
        )
    )
    base = sim.compile(num_steps=256, backend="cuda_streamed")
    state = initial_program_state(base, t=0, current_step=0, monitor_steps=256)
    if a.presteps:
        state = sim.advance(
            state=state, num_steps=a.presteps, backend="cuda_streamed"
        ).state
    jax.block_until_ready(state)
    variants = {}
    if a.study == "layouts":
        for name, extra in [
            ("one_step", 0),
            ("cpml_pair", abi.CUDA_TEMPORAL_PAIR | abi.CUDA_CPML_PAIR),
        ]:
            for axes in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                key = name + "_" + "".join(map(str, axes))
                variants[key] = (axes, extra, None)
    elif a.study == "shells":
        variants = {
            "queue_64x4": ((0, 1, 2), 0, "0"),
            "queue_32x8": ((0, 1, 2), abi.CUDA_SHELL32X8, "0"),
            "queue_32x4": ((0, 1, 2), abi.CUDA_SHELL32X4, "0"),
        }
    else:
        variants = {
            "canonical_queue": ((0, 1, 2), 0, "0"),
            "canonical_fused": ((0, 1, 2), 0, "1"),
            "fused_64x4x8": ((0, 1, 2), 0, "1"),
            "fused_32x4x8": ((0, 1, 2), 0, "1"),
        }
    if a.profile and a.study != "fusion":
        p.error("--profile requires --study fusion")
    if a.profile:
        name = "canonical_queue" if a.profile == "queue" else "canonical_fused"
        variants = {name: variants[name]}
    executables = {}
    for key, (axes, extra, _fusion) in variants.items():
        flags = (
            base.config.cuda_flags & ~(abi.CUDA_TEMPORAL_PAIR | abi.CUDA_CPML_PAIR)
        ) | extra
        program = replace(
            base, config=replace(base.config, cuda_flags=flags, cuda_storage_axes=axes)
        )
        executables[key] = (
            build_scan(program).lower(state, program.coefficients).compile()
        )

    def run(key):
        os.environ["BEAMZ_CUDA_CPML_TILE"] = {
            "fused_64x4x8": "64x4x8",
            "fused_32x4x8": "32x4x8",
        }.get(key, "32x8x8")
        fusion = variants[key][2]
        if fusion is None:
            os.environ.pop("BEAMZ_CUDA_CPML_CORE_FUSION", None)
        else:
            os.environ["BEAMZ_CUDA_CPML_CORE_FUSION"] = fusion
        return jax.block_until_ready(executables[key](state, base.coefficients))

    if a.profile:
        import ctypes

        name = next(iter(variants))
        for _ in range(4):
            run(name)
        cuda = ctypes.CDLL("libcudart.so")
        cuda.cudaProfilerStart()
        run(name)
        cuda.cudaProfilerStop()
        return

    reference = jax.device_get(run(next(iter(variants))))
    for leaf in jax.tree.leaves(reference):
        assert np.isfinite(np.asarray(leaf)).all(), "Non-finite reference state"
    hashes = [
        hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
        for x in jax.tree.leaves(reference)
    ]
    for key in executables:
        result = jax.device_get(run(key))
        for i, (ref, got) in enumerate(
            zip(jax.tree.leaves(reference), jax.tree.leaves(result), strict=True)
        ):
            np.testing.assert_array_equal(got, ref, err_msg=f"{key} leaf {i}")
        del result
        for _ in range(3):
            run(key)
        print(key, "exact complete-state parity", flush=True)
    del reference
    samples = {key: [] for key in executables}
    names = list(executables)
    random.Random(20260918).shuffle(names)
    timing_orders = []
    for reverse in (False, True):
        for offset in range(len(names)):
            order = names[offset:] + names[:offset]
            if reverse:
                order = order[::-1]
            timing_orders.append(order)
            for key in order:
                start = time.perf_counter()
                result = run(key)
                elapsed = time.perf_counter() - start
                samples[key].append(elapsed)
                del result
    cells = int(np.prod(a.shape))
    data = dict(
        study=a.study,
        variants=variants,
        shape=a.shape,
        steps=256,
        presteps=a.presteps,
        material=a.material,
        pml=12,
        precision="fp32",
        source="mode",
        monitors=1,
        frequencies=3,
        monitor_type="mode",
        logical_cells=cells,
        conversion_cost_included=True,
        device=str(jax.devices()[0]),
        device_kind=jax.devices()[0].device_kind,
        buffer_donation=False,
        exact_complete_state_parity=True,
        native_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
        storage_sha256=hashlib.sha256(
            (
                Path(__file__).resolve().parents[1] / "beamz/simulation/cuda/storage.py"
            ).read_bytes()
        ).hexdigest(),
        state_sha256=hashes,
        samples_s=samples,
        timing_orders=timing_orders,
        median_gcups={
            k: cells * 256 / statistics.median(v) / 1e9 for k, v in samples.items()
        },
    )
    Path(a.output).write_text(json.dumps(data, indent=2) + "\n")
    print(data["median_gcups"], flush=True)


if __name__ == "__main__":
    main()
