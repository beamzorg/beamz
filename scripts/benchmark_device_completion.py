#!/usr/bin/env python3
"""Propagate a finite pulse through an S-bend and measure time to field decay.

All backend/device comparisons must use identical shape, steps and checkpoints.
Convergence is reported, never assumed from a nonzero spectrum. The field norm
is an E/H-scaled decay proxy, not an electromagnetic energy integral.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import beamz as bz
from beamz.design import MaterialGrid


def build_device(shape, steps, frequencies=101):
    nz, ny, nx = shape
    dx = 80e-9
    size = np.array((nx, ny, nz)) * dx
    dt = 0.95 * dx / (bz.LIGHT_SPEED * np.sqrt(3))
    # Fixed 480 x 320 nm Si core in silica; smooth 2 um transverse offset.
    x = (np.arange(nx) + 0.5) * dx
    u = np.clip((x / size[0] - 0.35) / 0.3, 0, 1)
    center = size[1] / 2 + 2e-6 * (0.5 - 0.5 * np.cos(np.pi * u))
    eps = np.full(shape, 1.44**2, np.float32)
    y = (np.arange(ny) + 0.5) * dx
    core = np.abs(y[:, None] - center[None, :]) <= 240e-9
    eps[nz // 2 - 2 : nz // 2 + 2] = np.where(core, 3.48**2, 1.44**2)
    freq = bz.LIGHT_SPEED / 1.55e-6
    indices = np.arange(steps)
    signal = np.exp(-(((indices - 480) / 120) ** 2)) * np.sin(
        2 * np.pi * freq * indices * dt
    )
    signal[indices >= 1024] = 0  # finite pulse: all decay checks occur after this
    port = (0, 1.6e-6, 0.8e-6)
    source = bz.ModeSource(
        center=(0.15 * size[0], size[1] / 2, size[2] / 2),
        size=port,
        source_time=bz.SampledSignal(signal.astype(np.float32), dt=dt, freq0=freq),
        direction="+",
        mode_spec=bz.ModeSpec(polarization="te"),
    )
    monitors = [
        bz.ModeMonitor(
            center=(fraction * size[0], size[1] / 2 + offset, size[2] / 2),
            size=port,
            freqs=np.linspace(freq * 0.97, freq * 1.03, frequencies),
            mode_spec=bz.ModeSpec(polarization="te"),
            name=f"mode_{i}",
        )
        for i, (fraction, offset) in enumerate(((0.25, 0), (0.8, 2e-6)))
    ]
    return bz.Simulation(
        material_grid=MaterialGrid(
            permittivity=eps,
            conductivity=np.float32(0),
            permeability=np.float32(1),
            resolution=dx,
            shape=shape,
        ),
        size=tuple(size),
        sources=[source],
        monitors=monitors,
        boundaries=[bz.PML(edges="all", thickness=12 * dx, formulation="cpml")],
        time=indices * dt,
    ), dt


@jax.jit
def field_norm(state):
    return sum(
        jnp.sum(getattr(state, c) ** 2) for c in ("ex", "ey", "ez")
    ) + 376.730313668**2 * sum(
        jnp.sum(getattr(state, c) ** 2) for c in ("hx", "hy", "hz")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "cuda_streamed"), required=True)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--shape", nargs=3, type=int, default=(128, 512, 1024))
    parser.add_argument("--steps", type=int, default=16384)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--frequencies", type=int, default=101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.shape) <= 24
        or args.shape[1] * 80e-9 < 8e-6
        or args.steps % args.chunk
        or args.steps < 2048
    ):
        parser.error(
            "need CPML interior, >=8 um y extent, >=2048 steps divisible by chunk"
        )
    if len(jax.devices()) < args.devices:
        parser.error("insufficient visible devices")
    started = time.perf_counter()
    sim, dt = build_device(tuple(args.shape), args.steps, args.frequencies)
    setup_s = time.perf_counter() - started
    cfg = (
        None
        if args.devices == 1
        else dict(axis="x", num_devices=args.devices, backend="gpu")
    )
    state, rows, previous_flux = None, [], None
    peak = 0.0
    for step in range(args.chunk, args.steps + 1, args.chunk):
        tick = time.perf_counter()
        run = sim.advance(
            state=state,
            num_steps=args.chunk,
            backend=args.backend,
            sharding=cfg,
            donate_state=state is not None,
            progress=False,
            performance=False,
        )
        jax.block_until_ready(run.state)
        elapsed = time.perf_counter() - tick
        state = run.state
        norm = float(field_norm(state))
        peak = max(peak, norm)
        flux = np.stack([np.asarray(run.results[f"mode_{i}"].flux) for i in range(2)])
        change = (
            None
            if previous_flux is None
            else float(
                np.linalg.norm(flux - previous_flux) / max(np.linalg.norm(flux), 1e-300)
            )
        )
        previous_flux = flux
        rows.append(
            dict(
                step=step,
                call_s=elapsed,
                field_norm=norm,
                relative_field_norm=norm / max(peak, 1e-300),
                flux_change=change,
            )
        )
        print(json.dumps(rows[-1]), flush=True)
    finite = all(
        bool(jax.device_get(jnp.all(jnp.isfinite(x)))) for x in jax.tree.leaves(state)
    )
    tail = rows[-3:]
    converged = (
        finite
        and len(tail) == 3
        and all(
            r["step"] > 1024
            and r["relative_field_norm"] < 1e-6
            and r["flux_change"] is not None
            and r["flux_change"] < 1e-4
            for r in tail
        )
    )
    arrays = {
        name: np.asarray(getattr(state, name))
        for name in ("dft_vec_re", "dft_vec_im", "dft_weight_sum")
    }
    arrays.update({f"flux_{i}": flux[i] for i in range(2)})
    result = dict(
        workload="s_bend_completion",
        backend=args.backend,
        devices=args.devices,
        shape=args.shape,
        resolution_m=80e-9,
        steps=args.steps,
        chunk=args.chunk,
        frequencies=args.frequencies,
        cpml_cells=12,
        finite_state=finite,
        converged=converged,
        setup_s=setup_s,
        total_wall_s=time.perf_counter() - started,
        simulated_time_s=args.steps * dt,
        checkpoints=rows,
        convergence_criterion="last 3 checkpoints: scaled field norm / sampled peak < 1e-6 and relative flux L2 change < 1e-4; source off after step 1024",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k != "checkpoints"}), flush=True
    )
    if not finite:
        raise SystemExit("Non-finite state")


if __name__ == "__main__":
    main()
