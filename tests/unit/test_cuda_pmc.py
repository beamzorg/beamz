"""Exercise CUDA's production C++ cells and FFI on compact logical CPU shards."""

import os
import subprocess
import sys


def _check_native_pmc():
    import numpy as np

    import beamz as bz
    from beamz.design import MaterialGrid, RectilinearGrid
    from tests.unit.test_cuda_sharded_features import (
        assert_state_close,
        native_cpu_backend,
    )
    from tests.unit.test_cuda_sharding import seed_state

    dx = 0.1 * bz.um
    shape = (6, 7, 8)
    cases = ("pmc", "mixed", "cpml", "uniform_cpml", "axis_uniform", "stretched")
    with native_cpu_backend():
        for case in cases:
            x = np.arange(shape[2] + 1) * dx
            y = np.arange(shape[1] + 1) * dx
            z = np.arange(shape[0] + 1) * dx
            if case == "stretched":
                x *= 1 + 0.03 * np.arange(x.size)
                y *= 1 + 0.04 * np.arange(y.size)
                z *= 1 + 0.02 * np.arange(z.size)
            if case == "axis_uniform":
                y *= 1.2
                z *= 1.4
            boundaries = [bz.PMC()]
            if case == "mixed":
                boundaries.append(bz.PEC(edges=("front", "top", "right")))
            if case == "cpml":
                boundaries.append(
                    bz.PML(
                        edges=("front", "top", "right"),
                        thickness=dx,
                        formulation="cpml",
                    )
                )
            if case == "uniform_cpml":
                boundaries = [bz.PML(thickness=dx, formulation="cpml"), bz.PMC()]
            materials = MaterialGrid(
                np.ones(shape, np.float32),
                np.float32(0),
                np.float32(1),
                dx,
                shape,
                grid=RectilinearGrid(x, y, z),
            )
            sim = bz.Simulation(
                material_grid=materials,
                time=np.arange(6) * 2e-17,
                boundaries=boundaries,
            )
            state = seed_state(sim)
            reference = sim.advance(
                state=state, num_steps=5, backend="jax", performance=False
            ).state
            for axis in "xyz":
                cfg = dict(axis=axis, num_devices=2, backend="cpu")
                actual = sim.advance(
                    state=state,
                    num_steps=5,
                    backend="cuda_streamed",
                    sharding=cfg,
                    performance=False,
                ).state
                assert_state_close(reference, actual)
            # Three steps followed by two must preserve both fields and CPML memory.
            first = sim.advance(
                state=state,
                num_steps=3,
                backend="cuda_streamed",
                sharding=cfg,
                performance=False,
            ).state
            continued = sim.advance(
                state=first,
                num_steps=2,
                backend="cuda_streamed",
                sharding=cfg,
                performance=False,
            ).state
            assert_state_close(reference, continued)
            program = sim.compile(num_steps=5, backend="cuda_streamed", sharding=cfg)
            assert program.boundary.cpml.magnetic_edges
            if case == "uniform_cpml":
                from beamz.simulation import _cuda_abi as abi
                from beamz.simulation.cuda.runtime import _boundary_code

                code = _boundary_code(frozenset(), program.boundary.cpml.h_terms)
                assert ((code >> 8) & abi.BOUNDARY_CPML_THICKNESS_MASK) > 0


def test_native_pmc_matches_jax_on_all_shard_axes():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.unit.test_cuda_pmc import _check_native_pmc; _check_native_pmc()",
        ],
        env=dict(
            os.environ,
            XLA_FLAGS="--xla_force_host_platform_device_count=2",
            JAX_PLATFORMS="cpu",
        ),
        check=True,
        timeout=180,
    )


def test_production_streamed_boundary_stencil_on_cpu(tmp_path):
    import shutil
    from pathlib import Path

    compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    assert compiler, "The native boundary test requires a C++17 compiler"
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "pmc_boundary"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-fsanitize=undefined",
            f"-I{root / 'cuda/src'}",
            str(root / "tests/unit/cuda_pmc_boundary.cc"),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=90,
    )
    subprocess.run([str(executable)], check=True, timeout=30)
