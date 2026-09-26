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
        if case in {"mode", "local_dft", "dft_gradient"}
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
    if case in {"local_dft", "dft_gradient"}:
        sim = sim.updated_copy(monitors=sim.monitors[:2])
    state = seed_state(sim)
    gradient = case in {"gradient", "dft_gradient"}
    steps = 4 if gradient else 12
    if case in {"local_dft", "dft_gradient"}:
        from beamz.simulation.distributed_monitors import supported

        assert supported(
            sim.compile(
                backend="jax", sharding=dict(axis="x", num_devices=2, backend="cpu")
            )
        )
    with patch(
        "beamz.simulation.cuda.runtime._ffi_phase",
        side_effect=AssertionError("JAX called native FFI"),
    ):
        if not gradient:
            reference = sim.advance(state=state, num_steps=steps, backend="jax").state
            for count in (2, 4, 8) if case == "local_dft" else (2, 4):
                for axis in "xyz":
                    cfg = dict(axis=axis, num_devices=count, backend="cpu")
                    actual = sim.advance(
                        state=state, num_steps=steps, backend="jax", sharding=cfg
                    ).state
                    assert_state_close(reference, actual)
                    for name in ("ex", "ey", "ez", "hx", "hy", "hz"):
                        field = getattr(actual, name)
                        if any(size % count == 0 for size in field.shape):
                            assert not field.is_fully_replicated
                            assert (
                                sum(
                                    shard.data.size
                                    for shard in field.addressable_shards
                                )
                                == field.size
                            )
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

                def loss(
                    factor, scan=scan, coeffs=coeffs, prepared=prepared, program=program
                ):
                    varied = coeffs._replace(
                        **{
                            f"e_permittivity_{c}": getattr(
                                coeffs, f"e_permittivity_{c}"
                            )
                            * factor
                            for c in "xyz"
                        }
                    )
                    result = sharding.crop_state(program, scan(prepared, varied))
                    if case == "dft_gradient":
                        return jnp.sum(result.dft_vec_re**2 + result.dft_vec_im**2)
                    return sum(
                        jnp.sum(getattr(result, c) ** 2) for c in ("ex", "ey", "ez")
                    )

                results.append(jax.value_and_grad(loss)(jnp.float32(1.0)))
            np.testing.assert_allclose(results[1], results[0], rtol=1e-4, atol=1e-10)


@pytest.mark.parametrize(
    "case",
    ["asymmetric_cpml", "mixed_faces", "mode", "gradient", "local_dft", "dft_gradient"],
)
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
            XLA_FLAGS="--xla_force_host_platform_device_count="
            + ("8" if case == "local_dft" else "4"),
            JAX_PLATFORMS="cpu",
        ),
        check=True,
        timeout=300,
    )
