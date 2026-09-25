#!/usr/bin/env python3
"""Sequential, fresh-process H100 backend/scaling comparison with raw evidence.

Run from the repo root with PYTHONPATH=. No kernels or default policies change.
Only streamed CUDA and JAX are supported. Failed trials are retained.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace


@dataclasses.dataclass(frozen=True)
class ModalWorkload:
    name: str = "modal_cpml12"
    shape_zyx: tuple[int, int, int] = (128, 256, 384)
    timesteps: int = 256

    def resized(self, *, shape_zyx=None, timesteps=None):
        return dataclasses.replace(
            self,
            shape_zyx=shape_zyx or self.shape_zyx,
            timesteps=self.timesteps if timesteps is None else timesteps,
        )

    @property
    def feature_labels(self):
        return {
            "boundaries": ("CPML(all,12 cells)",),
            "sources": ("ModeSource(TE)",),
            "monitors": ("ModeMonitor(2 planes,3 frequencies)",),
        }

    def build(self):
        from benchmark_cuda_realistic import build_simulation

        return build_simulation(
            SimpleNamespace(
                shape=self.shape_zyx,
                steps=self.timesteps,
                pml=12,
                monitors=2,
                frequencies=3,
                material="binary",
                source="mode",
                monitor_type="mode",
            )
        )


def worker(args):
    import benchmark_h100 as harness
    import jax
    import numpy as np

    harness.H100_WORKLOADS["modal_cpml12"] = ModalWorkload()
    final_states = []

    def synchronize(state):
        jax.block_until_ready(state)
        final_states[:] = [state]

    harness._block = synchronize
    devices = jax.devices()
    if len(devices) != args.devices or any(
        "H100" not in d.device_kind for d in devices
    ):
        raise RuntimeError(f"Expected {args.devices} full H100 devices, got {devices}")
    args.allow_cpu = False
    started = time.perf_counter()
    record = harness.run_benchmark(args)
    if record.backend != args.backend or record.device_count != args.devices:
        raise RuntimeError("Requested and resolved execution do not match")
    data = record.as_dict()
    final_state = final_states[0]
    for leaf in jax.tree.leaves(final_state):
        if not np.isfinite(np.asarray(leaf)).all():
            raise RuntimeError("Final state contains non-finite values")
    data.update(
        final_state_finite=True,
        worker_measurement_wall_s=time.perf_counter() - started,
        shard_axis=args.shard_axis,
        allocator={k: v for k, v in os.environ.items() if k.startswith("XLA_")},
        cuda_env={k: v for k, v in os.environ.items() if k.startswith("BEAMZ_CUDA_")},
        nccl_env={k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
        device_memory=[{"id": d.id, "stats": d.memory_stats()} for d in devices],
    )
    if args.backend.startswith("cuda"):
        import beamz._cuda as extension

        data["extension_sha256"] = hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest()
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--timesteps", type=int, default=256)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workload", default="realistic_3d")
    parser.add_argument("--shape", type=int, nargs=3)
    parser.add_argument(
        "--backend", choices=("jax", "cuda_streamed"), default="cuda_streamed"
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--shard-axis", choices=["auto", "x", "y", "z"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if (
        args.samples < 5
        or args.timesteps < 32
        or any(n not in (1, 2, 4, 8) for n in args.counts)
    ):
        parser.error("need >=5 samples, >=32 steps, counts from 1/2/4/8")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    cases = [
        ("bare_3d", (128, 256, 384)),
        ("realistic_3d", (128, 256, 384)),
        ("realistic_3d", (256, 256, 256)),
        ("realistic_3d", (96, 256, 768)),
        ("modal_cpml12", (128, 256, 384)),
    ]
    env = {k: v for k, v in os.environ.items() if not k.startswith("BEAMZ_CUDA_")}
    env.update(
        PYTHONPATH=str(root),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION=".80",
        LD_LIBRARY_PATH="",
        BEAMZ_CUDA_CPML_PSI_PRECISION="fp32",
        NUMPY_MADVISE_HUGEPAGE="0",
    )
    summary = []
    manifest = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    source_hash = hashlib.sha256()
    for path in sorted([*root.glob("beamz/**/*.py"), *root.glob("cuda/src/*")]):
        if path.is_file():
            source_hash.update(str(path.relative_to(root)).encode())
            source_hash.update(path.read_bytes())
    manifest["solver_source_sha256"] = source_hash.hexdigest()
    (args.output / "solver.patch").write_text(
        subprocess.check_output(
            ["git", "diff", "--", "beamz", "cuda"], cwd=root, text=True
        )
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for index, (workload, shape) in enumerate(cases):
        for count in args.counts:
            backends = ["jax", "cuda_streamed"]
            if index % 2:
                backends.reverse()
            for backend in backends:
                name = f"{workload}-{'x'.join(map(str, shape))}-{backend}-{count}gpu"
                output = args.output / f"{name}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--output",
                    str(output.resolve()),
                    "--workload",
                    workload,
                    "--shape",
                    *map(str, shape),
                    "--backend",
                    backend,
                    "--devices",
                    str(count),
                    "--shard-axis",
                    args.shard_axis,
                    "--timesteps",
                    str(args.timesteps),
                    "--samples",
                    str(args.samples),
                ]
                print(name, flush=True)
                if args.dry_run:
                    summary.append({"name": name, "command": command})
                else:
                    trial_env = dict(
                        env, CUDA_VISIBLE_DEVICES=",".join(map(str, range(count)))
                    )
                    started = time.perf_counter()
                    with (args.output / f"{name}.log").open("w") as log:
                        try:
                            result = subprocess.run(
                                command,
                                env=trial_env,
                                cwd=root,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                timeout=args.timeout,
                            )
                            status = (
                                "ok"
                                if result.returncode == 0 and output.exists()
                                else "failed"
                            )
                        except subprocess.TimeoutExpired:
                            status = "timeout"
                    summary.append(
                        {
                            "name": name,
                            "status": status,
                            "wall_s": time.perf_counter() - started,
                        }
                    )
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
