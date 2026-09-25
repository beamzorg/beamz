"""Pure-JAX local stencils retain fields, CPML, continuation and derivatives."""

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.simulation import sharding
from beamz.simulation.execute import build_scan, runtime_inputs
from tests.unit.test_cuda_sharded_features import assert_state_close, mode_simulation
from tests.unit.test_cuda_sharding import seed_state


def check(case):
    from tests.hardware.test_cuda_backends import _feature_simulation
    from tests.performance.h100_workloads import H100Workload

    sim = (
        mode_simulation()
        if case == "mode"
        else _feature_simulation(case)
        if case != "gradient"
        else H100Workload(
            name="gradient",
            shape_zyx=(9, 10, 11),
            timesteps=4,
            resolution=80e-9,
            pml_cells=2,
            cpml=True,
            heterogeneous=True,
        ).build()
    )
    state = seed_state(sim)
    steps = 4 if case == "gradient" else 12
    with patch(
        "beamz.simulation.cuda.runtime._ffi_phase",
        side_effect=AssertionError("JAX called native FFI"),
    ):
        if case != "gradient":
            reference = sim.advance(state=state, num_steps=steps, backend="jax").state
            for count in (2, 4):
                for axis in "xyz":
                    cfg = dict(axis=axis, num_devices=count, backend="cpu")
                    actual = sim.advance(
                        state=state, num_steps=steps, backend="jax", sharding=cfg
                    ).state
                    assert_state_close(reference, actual)
                    first = sim.advance(
                        state=state, num_steps=steps // 2, backend="jax", sharding=cfg
                    ).state
                    cfg["axis"] = {"x": "y", "y": "z", "z": "x"}[axis]
                    continued = sim.advance(
                        state=first, num_steps=steps // 2, backend="jax", sharding=cfg
                    ).state
                    assert_state_close(actual, continued)
            return
        for conductivity in (0.0, 1e3, 1e6):
            results = []
            for cfg in (None, dict(axis="x", num_devices=4, backend="cpu")):
                program = sim.compile(num_steps=steps, backend="jax", sharding=cfg)
                prepared = sharding.prepare_state(
                    program,
                    runtime_inputs(program, state, monitor_steps=steps),
                    replicated_fields=(),
                )
                coeffs = program.coefficients._replace(
                    **{
                        **{
                            f"e_conductivity_{c}": jnp.asarray(
                                conductivity, dtype=jnp.float32
                            )
                            for c in "xyz"
                        },
                        **{
                            f"h_sigma_m_{c}": jnp.asarray(
                                conductivity * 1e3, dtype=jnp.float32
                            )
                            for c in "xyz"
                        },
                    }
                )
                coeffs = sharding.place_tree(program, coeffs)
                scan = build_scan(program)

                def loss(factor, scan=scan, coeffs=coeffs, prepared=prepared):
                    varied = coeffs._replace(
                        **{
                            f"e_permittivity_{c}": getattr(
                                coeffs, f"e_permittivity_{c}"
                            )
                            * factor
                            for c in "xyz"
                        }
                    )
                    result = scan(prepared, varied)
                    return sum(
                        jnp.sum(getattr(result, c) ** 2) for c in ("ex", "ey", "ez")
                    )

                results.append(jax.value_and_grad(loss)(jnp.float32(1.0)))
            np.testing.assert_allclose(results[1], results[0], rtol=1e-4, atol=1e-10)


@pytest.mark.parametrize("case", ["asymmetric_cpml", "mixed_faces", "mode", "gradient"])
def test_local_jax_contract(case):
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tests.unit.test_jax_local_sharding import check; check(sys.argv[1])",
            case,
        ],
        env=dict(
            os.environ,
            XLA_FLAGS="--xla_force_host_platform_device_count=4",
            JAX_PLATFORMS="cpu",
        ),
        check=True,
        timeout=300,
    )
