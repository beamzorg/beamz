"""Compact PMC checks on two logical CPU devices, including storage padding."""

import os
import subprocess
import sys


def _check_sharding():
    import jax
    import jax.numpy as jnp
    import numpy as np

    import beamz as bz
    from beamz.simulation import jax_sharding

    rng = np.random.default_rng(77)
    for cpml in (False, True):
        boundaries = [bz.PMC()]
        if cpml:
            boundaries.append(
                bz.PML(
                    edges=("front", "bottom", "right"),
                    thickness=0.1 * bz.um,
                    formulation="cpml",
                )
            )
        simulation = bz.Simulation(
            domain=(0.8 * bz.um, 0.7 * bz.um, 0.6 * bz.um),
            resolution=0.1 * bz.um,
            time=np.arange(7) * 2e-17,
            boundaries=boundaries,
        )
        state = simulation.initial_state()
        names = ("ex", "ey", "ez", "hx", "hy", "hz")
        state = state._replace(
            **{
                name: jnp.asarray(
                    rng.normal(size=getattr(state, name).shape), dtype=jnp.float32
                )
                * (1 if name.startswith("e") else 0.002)
                for name in names
            }
        )
        reference = simulation.advance(
            state=state, num_steps=6, backend="jax", performance=False
        ).state
        for axis in "xyz":
            cfg = dict(axis=axis, num_devices=2, backend="cpu")
            program = simulation.compile(num_steps=6, backend="jax", sharding=cfg)
            assert jax_sharding.supported(program) == cpml
            actual = simulation.advance(
                state=state,
                num_steps=6,
                backend="jax",
                sharding=cfg,
                performance=False,
            ).state
            arrays = (*names, "cpml_psi_h_terms", "cpml_psi_e_terms")
            for name in arrays:
                for result, expected in zip(
                    jax.tree.leaves(getattr(actual, name)),
                    jax.tree.leaves(getattr(reference, name)),
                    strict=True,
                ):
                    np.testing.assert_allclose(
                        result, expected, rtol=3e-5, atol=3e-7, err_msg=f"{axis} {name}"
                    )


def test_pmc_sharding_matches_single_device():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.unit.test_pmc_sharding import _check_sharding; _check_sharding()",
        ],
        env=dict(
            os.environ,
            XLA_FLAGS="--xla_force_host_platform_device_count=2",
            JAX_PLATFORMS="cpu",
        ),
        check=True,
        timeout=120,
    )
