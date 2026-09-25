"""CPU evidence for CUDA orchestration; this does not execute native CUDA."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import jax
import numpy as np
import pytest

import beamz as bz
from beamz.simulation.cuda import runtime
from beamz.simulation.cuda import sharding as cuda_sharding
from beamz.simulation.model import ShardingConfig
from beamz.simulation.sharding import build_sharding_plan


def make_simulation():
    return bz.Simulation(
        design=bz.Design(
            width=7.0, height=8.0, depth=9.0, material=bz.Material(permittivity=2.0)
        ),
        sources=[],
        monitors=[],
        boundaries=[bz.PEC(edges="all")],
        time=np.arange(9) * 2e-10,
        resolution=1.0,
    )


def seed_state(sim):
    state = sim.initial_state()
    rng = np.random.default_rng(904)
    return state._replace(
        **{
            name: rng.normal(size=getattr(state, name).shape).astype(np.float32)
            * (1e-3 if name.startswith("e") else 1e-6)
            for name in ("ex", "ey", "ez", "hx", "hy", "hz")
        }
    )


def test_aligned_layout_preserves_component_supports():
    fields = SimpleNamespace(
        permittivity=np.zeros((8, 8, 8)),
        **{
            name: np.zeros(shape)
            for name, shape in {
                "Ex": (9, 9, 8),
                "Ey": (9, 8, 9),
                "Ez": (8, 9, 9),
                "Hx": (8, 8, 9),
                "Hy": (8, 9, 8),
                "Hz": (9, 8, 8),
            }.items()
        },
    )
    # Mesh construction needs distinct real devices, so the arithmetic assertions
    # run in the subprocess below. A disabled plan must remain shape preserving.
    plan = build_sharding_plan(
        fields, ShardingConfig(enabled=False), is_3d=True, aligned_components=True
    )
    assert plan.layout.logical_shapes == plan.layout.padded_shapes


def run_cpu_contract():
    from beamz.simulation import backend
    from beamz.simulation.execute import build_scan, runtime_inputs
    from beamz.simulation.sharding import place_tree, prepare_state
    from tests.performance.h100_workloads import H100Workload

    sim = make_simulation()
    state = seed_state(sim)
    expected = sim.advance(num_steps=6, state=state, backend="jax", progress=False)
    from tests.unit.cuda_host_contract import register_host_sharded_ffi

    register_host_sharded_ffi()
    with (
        patch.object(backend, "resolve_backend", lambda _: "cuda_streamed"),
        patch.object(cuda_sharding, "_validate_devices", lambda _: None),
        patch.object(
            runtime, "run_steps", side_effect=AssertionError("native graph used")
        ),
    ):
        for count in (2, 4):
            for axis in ("z", "y", "x"):
                cfg = ShardingConfig(
                    enabled=True, axis=axis, num_devices=count, backend="cpu"
                )
                program = sim.compile(
                    num_steps=6, backend="cuda_streamed", sharding=cfg
                )
                i = ("z", "y", "x").index(axis)
                extents = {
                    shape[i] for shape in program.sharding.layout.padded_shapes.values()
                }
                assert len(extents) == 1
                assert next(iter(extents)) % count == 0
                assert all(
                    getattr(program.coefficients, f"e_source_{c}").ndim != 1
                    for c in "xyz"
                )
                prepared = prepare_state(
                    program,
                    runtime_inputs(program, state, monitor_steps=6),
                    replicated_fields=(),
                )
                assert len(prepared.ex.addressable_shards) == count
                assert (
                    prepared.ex.addressable_shards[0].data.shape[i]
                    == prepared.ex.shape[i] // count
                )
                got = sim.advance(
                    num_steps=6,
                    state=state,
                    backend="cuda_streamed",
                    sharding=cfg,
                    progress=False,
                )
                first = sim.advance(
                    num_steps=3,
                    state=state,
                    backend="cuda_streamed",
                    sharding=cfg,
                    progress=False,
                )
                continued = sim.advance(
                    num_steps=3,
                    state=first.state,
                    backend="cuda_streamed",
                    sharding=cfg,
                    progress=False,
                )
                for name in ("ex", "ey", "ez", "hx", "hy", "hz"):
                    np.testing.assert_allclose(
                        getattr(got.state, name),
                        getattr(expected.state, name),
                        rtol=1e-5,
                        atol=1e-9,
                    )
                    np.testing.assert_allclose(
                        getattr(continued.state, name),
                        getattr(got.state, name),
                        rtol=1e-5,
                        atol=1e-9,
                    )
                assert int(continued.state.current_step) == 6
                ir = (
                    build_scan(program)
                    .lower(prepared, place_tree(program, program.coefficients))
                    .as_text()
                )
                assert "stablehlo.custom_call @beamz_cuda_sharded" in ir
                assert "collective_permute" in ir
                assert "all_gather" not in ir

        workload = H100Workload(
            name="sharded_cpu_contract",
            shape_zyx=(9, 10, 11),
            timesteps=32,
            resolution=80e-9,
            pml_cells=1,
            heterogeneous=True,
            cpml=False,
            source=True,
            monitor=True,
        )
        # Build the reference outside the backend-selection mock.
        with patch.object(backend, "resolve_backend", lambda _: "jax"):
            sim = workload.build()
            state = seed_state(sim)
            reference = sim.advance(
                num_steps=32, state=state, backend="jax", progress=False
            )
        for axis in ("z", "y", "x"):
            cfg = ShardingConfig(enabled=True, axis=axis, num_devices=4, backend="cpu")
            got = sim.advance(
                num_steps=32,
                state=state,
                backend="cuda_streamed",
                sharding=cfg,
                progress=False,
            )
            first = sim.advance(
                num_steps=16,
                state=state,
                backend="cuda_streamed",
                sharding=cfg,
                progress=False,
            )
            continued = sim.advance(
                num_steps=16,
                state=first.state,
                backend="cuda_streamed",
                sharding=cfg,
                progress=False,
            )
            for name in (
                "ex",
                "ey",
                "ez",
                "hx",
                "hy",
                "hz",
                "dft_vec",
                "dft_weight_sum",
            ):
                if name == "dft_vec":
                    # Real/imaginary sums can cancel independently. Compare the
                    # complex signal so its error budget does not depend on phase.
                    # JAX 0.6.2 on AVX2 differs from the portable host kernel by
                    # ~4e-9 here, below 1 ppm of the complex signal's peak.
                    expected, actual, resumed = (
                        np.asarray(result.state.dft_vec_re)
                        + 1j * np.asarray(result.state.dft_vec_im)
                        for result in (reference, got, continued)
                    )
                else:
                    expected, actual, resumed = (
                        np.asarray(getattr(result.state, name))
                        for result in (reference, got, continued)
                    )
                # Use a one-ppm absolute floor relative to the signal's range.
                atol = max(1e-12, 1e-6 * float(np.max(np.abs(expected), initial=0)))
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=3e-5,
                    atol=atol,
                    err_msg=name,
                )
                np.testing.assert_allclose(
                    resumed,
                    actual,
                    rtol=3e-5,
                    atol=atol,
                    err_msg=name,
                )


def test_cpu_devices_execute_sharded_cuda_orchestration():
    env = dict(
        os.environ,
        XLA_FLAGS="--xla_force_host_platform_device_count=4",
        JAX_PLATFORMS="cpu",
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.unit.test_cuda_sharding import run_cpu_contract; run_cpu_contract()",
        ],
        env=env,
        check=True,
        timeout=180,
    )


@pytest.mark.parametrize("unsupported", ["jax", "2d"])
def test_unsupported_sharded_domains_fail_before_execution(unsupported):
    program = make_simulation().compile(backend="jax")
    cfg = replace(
        program.config,
        backend="jax" if unsupported == "jax" else "cuda_streamed",
        is_3d=unsupported != "2d",
    )
    with pytest.raises(RuntimeError, match="cuda_streamed and a 3D grid"):
        cuda_sharding.validate_sharded_config(cfg, program.boundary, None)


def test_cpu_mesh_rejected_for_native_cuda():
    mesh = jax.sharding.Mesh(np.asarray(jax.devices("cpu")[:1]), ("fdtd",))
    with pytest.raises(RuntimeError, match="GPU device mesh"):
        cuda_sharding._validate_devices(mesh)
