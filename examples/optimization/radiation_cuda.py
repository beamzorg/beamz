"""One small radiation value/gradient on native CUDA or portable JAX.

Run from the repository root:
    python -m examples.optimization.radiation_cuda --backend cuda
    python -m examples.optimization.radiation_cuda --backend jax

This is a 14,400-cell, 40 fs API demonstration, not a converged device design.
It performs one value/gradient evaluation and no optimization iterations.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

import beamz as bz
from beamz.analysis import FarFieldProjection, near_field_box
from beamz.optimization import CollectionConePower, InverseDesign, TopologyDesignRegion


def make_design(*, backend="cuda", run_time=40e-15):
    """Build a small dielectric design inside a homogeneous recording box."""
    frequency = bz.LIGHT_SPEED / 1e-6
    monitors, surfaces = near_field_box(
        center=(1.5e-6, 1e-6, 1e-6),
        size=(1.2e-6, 1.2e-6, 1.2e-6),
        freqs=[frequency],
    )
    simulation = bz.Simulation(
        design=bz.Design(
            width=3e-6, height=2e-6, depth=2.4e-6, material=bz.Material(1)
        ),
        resolution=0.1e-6,
        run_time=run_time,
        sources=[
            bz.GaussianBeamSource(
                center=(1.5e-6, 1e-6, 1.9e-6),
                size=(1.2e-6, 1.2e-6),
                direction="-z",
                waist_radius=0.4e-6,
                source_time=bz.GaussianPulse(
                    freq0=frequency, fwidth=0.2 * frequency, offset=1.5
                ),
            )
        ],
        monitors=monitors,
        boundaries=[bz.PML(thickness=0.3e-6, formulation="cpml")],
    )
    design = InverseDesign(
        simulation=simulation,
        design_region=TopologyDesignRegion(
            center=(1.5e-6, 1e-6, 1e-6),
            size=(0.6e-6, 0.8e-6, 0.4e-6),
            pixel_size=0.1e-6,
            eps_bounds=(1, 4),
        ),
        backend=backend,
        gradient_backend="autodiff",
        checkpoint_interval=16,
    )
    cosine, polar_weights = np.polynomial.legendre.leggauss(6)
    phi = np.arange(12) * 2 * np.pi / 12
    cosine_grid, phi_grid = np.meshgrid(cosine, phi, indexing="ij")
    sine_grid = np.sqrt(1 - cosine_grid**2)
    directions = np.stack(
        (sine_grid * np.cos(phi_grid), sine_grid * np.sin(phi_grid), cosine_grid), -1
    ).reshape(-1, 3)
    weights = np.broadcast_to(
        polar_weights[:, None] * 2 * np.pi / len(phi), cosine_grid.shape
    ).ravel()
    return design, FarFieldProjection(surfaces, directions, weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "cuda"), default="cuda")
    args = parser.parse_args()
    design, projection = make_design(backend=args.backend)
    objective = CollectionConePower(projection, numerical_aperture=0.8)
    params = jnp.asarray(design.design_region.initial_parameters)
    value, gradient = jax.jit(jax.value_and_grad(design.make_objective_fn(objective)))(
        params
    )
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite radiation objective or gradient.")
    print(
        f"backend={args.backend} collection={float(value):.6g} "
        f"gradient_norm={float(jnp.linalg.norm(gradient)):.6g}"
    )


if __name__ == "__main__":
    main()
