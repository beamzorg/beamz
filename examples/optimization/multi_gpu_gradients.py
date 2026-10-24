"""One resident distributed inverse-design step; run on a host with 2+ GPUs."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

import beamz as bz
from beamz.design import MaterialGrid


def main():
    dx, shape = 80e-9, (24, 24, 32)
    epsilon = np.ones(shape, dtype=np.float32)
    epsilon[8:16, 8:16, 12:20] = 2
    materials = MaterialGrid(epsilon, np.zeros_like(epsilon), np.float32(1), dx, shape)
    sim = bz.Simulation(
        material_grid=materials,
        time=np.arange(256) * 1e-16,
        boundaries=[bz.PML(thickness=4 * dx, formulation="cpml")],
        monitors=[
            bz.FluxMonitor(
                center=(20 * dx, 12 * dx, 12 * dx),
                size=(0, 8 * dx, 8 * dx),
                freqs=(2e14,),
                name="output",
            )
        ],
    )
    sim = sim.updated_copy(
        sources=[
            bz.CustomSource(
                component="Ez",
                timing="e",
                index=(slice(8, 16), slice(8, 16), slice(15, 17)),
                coeff=np.full((8, 8, 2), 0.03, dtype=np.float32),
                waveform=np.sin(np.arange(256) * 0.1).astype(np.float32),
                target_shape=sim.initial_state().ez.shape,
            )
        ]
    )
    solver = sim.differentiable(sharding=dict(axis="x", num_devices=2, backend="gpu"))
    parameters = solver.prepare_permittivity(epsilon)
    mask = solver.prepare_permittivity(np.where(epsilon == 2, 1, 0).astype(np.float32))
    objective = jax.jit(lambda eps: -solver.monitor_flux(solver(eps), "output")[0])
    value, gradient = jax.jit(jax.value_and_grad(objective))(parameters)
    optimizer = optax.adam(0.01)
    updates, state = jax.jit(optimizer.update)(
        gradient * mask, optimizer.init(parameters), parameters
    )
    parameters = jnp.clip(parameters + updates, 1, 4)
    print(f"Objective: {float(value):.6g}; parameter layout: {parameters.sharding}")
    return parameters, state


if __name__ == "__main__":
    main()
