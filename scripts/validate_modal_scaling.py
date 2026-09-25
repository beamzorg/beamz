#!/usr/bin/env python3
"""Run a propagated mode-source/12-cell-CPML case and save raw modal DFT data.

Run in fresh processes for each backend/device count, then compare the .npz
artifacts. This is a physics gate, separate from the short throughput sweep.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from benchmark_cuda_realistic import build_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "cuda_streamed"), required=True)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--shape", type=int, nargs=3, default=(64, 96, 256))
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--frequencies", type=int, default=101)
    parser.add_argument("--shard-axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 4 * max(args.shape) or min(args.shape) <= 24:
        parser.error(
            "need a CPML interior and >=4*max(shape) steps to exercise propagation"
        )
    sim = build_simulation(
        SimpleNamespace(
            shape=args.shape,
            steps=args.steps,
            pml=12,
            monitors=2,
            frequencies=args.frequencies,
            material="binary",
            source="mode",
            monitor_type="mode",
        )
    )
    cfg = (
        None
        if args.devices == 1
        else dict(axis=args.shard_axis, num_devices=args.devices, backend="gpu")
    )
    started = time.perf_counter()
    run = sim.advance(
        num_steps=args.steps, backend=args.backend, sharding=cfg, progress=False
    )
    jax.block_until_ready(run.state)
    finite = all(
        bool(jax.device_get(jnp.all(jnp.isfinite(leaf))))
        for leaf in jax.tree.leaves(run.state)
    )
    if not finite:
        raise RuntimeError("Non-finite state")
    arrays = {
        name: np.asarray(getattr(run.state, name))
        for name in ("dft_vec_re", "dft_vec_im", "dft_weight_sum")
    }
    fluxes = {f"flux_{i}": np.asarray(run.results[f"mode_{i}"].flux) for i in range(2)}
    arrays.update(fluxes)
    if not all(np.linalg.norm(flux) > 0 for flux in fluxes.values()):
        raise RuntimeError("Pulse has not produced a nonzero monitor spectrum")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    data = dict(
        backend=args.backend,
        devices=args.devices,
        shape=args.shape,
        steps=args.steps,
        frequencies=args.frequencies,
        cpml_cells=12,
        finite_state=finite,
        total_wall_s=time.perf_counter() - started,
        shard_axis=args.shard_axis,
        fluxes={name: value.tolist() for name, value in fluxes.items()},
    )
    if args.backend == "cuda_streamed":
        import beamz._cuda as native

        data["native_sha256"] = hashlib.sha256(
            Path(native.__file__).read_bytes()
        ).hexdigest()
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
