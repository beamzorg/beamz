#!/usr/bin/env python3
"""Isolate normal-incidence packet return with 12-cell CPML and FP32/BF16 psi.

Uses the existing analytical validation's characteristic-energy measurement,
with an exact 12-cell absorber and explicit storage-only mixed precision.
This small 2D accuracy experiment is not a throughput benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from beamz import EPS_0, LIGHT_SPEED, MU_0, PEC, PML, Design, Material, Simulation, um
from beamz.simulation.execute import build_scan, initial_program_state


def measure(index, ppw):
    wavelength = um / index
    dx = wavelength / ppw
    dt = 0.95 * dx / (LIGHT_SPEED * np.sqrt(2.0))
    times = np.arange(0.0, 12.0 * um / LIGHT_SPEED, dt)
    impedance = np.sqrt(MU_0 / EPS_0) / index
    sim = Simulation(
        design=Design(
            width=12 * wavelength,
            height=26 * wavelength,
            material=Material(permittivity=index**2),
        ),
        sources=[],
        boundaries=[
            PML(edges=["left", "right"], thickness=12 * dx, formulation="cpml"),
            PEC(edges=["top", "bottom"]),
        ],
        time=times,
        resolution=dx,
    )
    program = sim.compile(num_steps=len(times), backend="jax")
    state = initial_program_state(
        program, t=0, current_step=0, monitor_steps=len(times)
    )
    electric_x = np.arange(state.ez.shape[1]) * dx
    magnetic_x = (np.arange(state.hy.shape[1]) + 0.5) * dx

    def packet(x):
        return np.exp(-0.5 * ((x - 3 * wavelength) / (0.8 * wavelength)) ** 2) * np.cos(
            2 * np.pi * (x - 3 * wavelength) / wavelength
        )

    state = state._replace(
        ez=jnp.asarray(
            np.broadcast_to(packet(electric_x), state.ez.shape), dtype=jnp.float32
        ),
        hy=jnp.asarray(
            np.broadcast_to(
                -packet(magnetic_x + 0.5 * LIGHT_SPEED / index * dt) / impedance,
                state.hy.shape,
            ),
            dtype=jnp.float32,
        ),
    )
    initial_e = 0.5 * (np.asarray(state.ez)[:, 1:] + np.asarray(state.ez)[:, :-1])
    incident = 0.5 * (initial_e - impedance * np.asarray(state.hy))
    x = (np.arange(incident.shape[1]) + 0.5) * dx
    y = np.arange(incident.shape[0]) * dx
    # Exclude the absorber and avoid transversely returning PEC disturbances.
    ix = (x > max(1.7 * wavelength, 13 * dx)) & (
        x < min(10.3 * wavelength, 12 * wavelength - 13 * dx)
    )
    iy = (y > 11 * wavelength) & (y < 15 * wavelength)
    incident_energy = np.sum(incident[np.ix_(iy, ix)] ** 2)
    result = dict(
        refractive_index=index,
        points_per_wavelength=ppw,
        steps=len(times),
        cpml_cells=12,
        shape=list(state.ez.shape),
        variants={},
    )
    for precision, dtype in [("fp32", jnp.float32), ("bf16", jnp.bfloat16)]:
        initial = state._replace(
            **{
                key: tuple(a.astype(dtype) for a in getattr(state, key))
                for key in ("cpml_psi_h_terms", "cpml_psi_e_terms")
            }
        )
        final = build_scan(program, donate_state=False)(initial, program.coefficients)
        jax.block_until_ready(final)
        final_e = 0.5 * (np.asarray(final.ez)[:, 1:] + np.asarray(final.ez)[:, :-1])
        reflected = 0.5 * (final_e + impedance * np.asarray(final.hy))
        ratio = float(np.sum(reflected[np.ix_(iy, ix)] ** 2) / incident_energy)
        result["variants"][precision] = dict(
            reflected_energy_ratio=ratio, reflection_db=float(10 * np.log10(ratio))
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, ppw in [(1.0, 10), (1.0, 20), (1.5, 15)]:
        data = measure(index, ppw)
        report.append(data)
        print(json.dumps(data), flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
