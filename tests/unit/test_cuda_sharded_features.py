"""Numerical coverage of the production sharded C++ cells through CPU FFI."""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from unittest.mock import patch

import jax
import numpy as np
import pytest

import beamz as bz
from beamz.design import MaterialGrid
from beamz.simulation import backend
from beamz.simulation.cuda import sharding
from tests.unit.cuda_host_contract import register_host_sharded_ffi
from tests.unit.test_cuda_sharding import seed_state


@contextmanager
def native_cpu_backend():
    register_host_sharded_ffi()
    with (
        patch.object(
            backend,
            "resolve_backend",
            lambda name: (
                "cuda_streamed" if name in {"cuda", "cuda_streamed"} else "jax"
            ),
        ),
        patch.object(sharding, "_validate_devices", lambda _: None),
    ):
        yield


def assert_state_close(expected, actual, *, tolerance=2e-6):
    for name in expected._fields:
        refs = jax.tree_util.tree_leaves(getattr(expected, name))
        values = jax.tree_util.tree_leaves(getattr(actual, name))
        assert len(refs) == len(values), name
        for ref, value in zip(refs, values, strict=True):
            ref, value = np.asarray(ref), np.asarray(value)
            assert ref.shape == value.shape, (name, ref.shape, value.shape)
            if ref.dtype.kind in "iu":
                np.testing.assert_array_equal(value, ref, err_msg=name)
            else:
                scale = float(np.max(np.abs(ref.astype(np.float32)), initial=0))
                np.testing.assert_allclose(
                    value.astype(np.float32),
                    ref.astype(np.float32),
                    rtol=3e-5,
                    atol=max(1e-12, tolerance * scale),
                    err_msg=name,
                )


def compare(sim, *, state=None, steps=None, counts=(4,), continuation=False):
    state = seed_state(sim) if state is None else state
    steps = sim.num_steps if steps is None else steps
    reference = sim.advance(
        num_steps=steps, state=state, backend="jax", progress=False
    ).state
    for count in counts:
        for axis in ("z", "y", "x"):
            cfg = dict(axis=axis, num_devices=count, backend="cpu")
            actual = sim.advance(
                num_steps=steps,
                state=state,
                backend="cuda_streamed",
                sharding=cfg,
                progress=False,
            ).state
            assert_state_close(reference, actual)
            if continuation:
                first = sim.advance(
                    num_steps=steps // 2,
                    state=state,
                    backend="cuda_streamed",
                    sharding=cfg,
                    progress=False,
                ).state
                # Change the partition axis for the second segment. Packed CPML
                # continuation must not silently reset when storage changes.
                cfg["axis"] = {"x": "y", "y": "z", "z": "x"}[axis]
                continued = sim.advance(
                    num_steps=steps - steps // 2,
                    state=first,
                    backend="cuda_streamed",
                    sharding=cfg,
                    progress=False,
                ).state
                assert_state_close(actual, continued)
                resumed_jax = sim.advance(
                    num_steps=steps - steps // 2,
                    state=first,
                    backend="jax",
                    progress=False,
                ).state
                assert_state_close(reference, resumed_jax)


def mode_simulation():
    resolution = 100e-9
    shape = (12, 12, 20)
    eps = np.ones(shape, dtype=np.float32)
    eps[4:8, 4:8, :] = 4.0
    grid = MaterialGrid(eps, np.float32(0), np.float32(1), resolution, shape)
    dt = 0.8 * resolution / (bz.LIGHT_SPEED * np.sqrt(3.0))
    freq = bz.LIGHT_SPEED / 1.55e-6
    time = np.arange(24) * dt
    source = bz.ModeSource(
        center=(0.6e-6, 0.6e-6, 0.6e-6),
        size=(0.0, 0.8e-6, 0.8e-6),
        source_time=bz.SampledSignal(
            np.sin(2 * np.pi * freq * time).astype(np.float32), dt=dt, freq0=freq
        ),
        direction="+",
        mode_spec=bz.ModeSpec(polarization="te"),
    )
    plane = dict(center=(1.3e-6, 0.6e-6, 0.6e-6), size=(0.0, 0.8e-6, 0.8e-6))
    monitors = [
        bz.ModeMonitor(
            **plane,
            freqs=np.asarray([freq]),
            name="mode",
            mode_spec=bz.ModeSpec(polarization="te"),
        ),
        bz.FluxMonitor(**plane, freqs=np.asarray([freq]), name="flux"),
        bz.FieldRecorder(
            **plane, components=("Ey", "Hz"), interval=3, name="recording"
        ),
    ]
    return bz.Simulation(
        material_grid=grid,
        sources=[source],
        monitors=monitors,
        boundaries=[bz.PML(thickness=2 * resolution, formulation="cpml")],
        time=time,
    )


def run_case(case):
    from tests.hardware.test_cuda_backends import (
        _feature_simulation,
        _nonuniform_simulation,
    )
    from tests.performance.h100_workloads import H100Workload

    with native_cpu_backend():
        if case == "cpml":
            sim = H100Workload(
                name="cpml_sharding",
                shape_zyx=(9, 10, 11),
                timesteps=32,
                resolution=80e-9,
                pml_cells=2,
                cpml=True,
                heterogeneous=True,
                source=True,
                monitor=True,
            ).build()
            compare(sim, counts=(2, 4), continuation=True)
        elif case == "boundaries":
            for profile in (
                "conductive",
                "sponge",
                "mixed_faces",
                "asymmetric_cpml",
                "multiple_sources",
                "overlapping_sources",
                "h_source",
                "gaussian_beam",
                "multiple_monitors",
                "cpml_multiple_monitors",
            ):
                print(profile, flush=True)
                compare(_feature_simulation(profile))
        elif case == "nonuniform":
            for metric in ("axis_uniform", "rectilinear"):
                for cpml in (False, True):
                    sim, state = _nonuniform_simulation(metric_kind=metric, cpml=cpml)
                    compare(sim, state=state, continuation=cpml)
        elif case == "tensor":
            from tests.unit.test_execution_backend import _full_tensor_3d_simulation

            base = _full_tensor_3d_simulation()
            for cpml in (False, True):
                sim = bz.Simulation(
                    material_grid=base._material_grid(),
                    time=np.arange(12) * 1e-10,
                    boundaries=[bz.PML(thickness=0.05, formulation="cpml")]
                    if cpml
                    else [bz.PEC(edges="all")],
                )
                compare(sim, continuation=cpml)
        elif case == "mode":
            sim = mode_simulation()
            compare(sim)
        elif case == "recording":
            sim = H100Workload(
                name="recorded_sharding",
                shape_zyx=(9, 10, 11),
                timesteps=12,
                resolution=80e-9,
                pml_cells=2,
                cpml=True,
                source=True,
            ).build()
            sim = sim.updated_copy(
                monitors=[
                    bz.FieldRecorder(
                        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"), interval=3
                    ),
                ]
            )
            compare(sim)
        elif case == "precision":
            sim = H100Workload(
                name="bf16_sharding",
                shape_zyx=(9, 10, 11),
                timesteps=16,
                resolution=80e-9,
                pml_cells=2,
                cpml=True,
            ).build()
            state = seed_state(sim)
            with patch.dict(os.environ, {"BEAMZ_CUDA_CPML_PSI_PRECISION": "bf16"}):
                results = []
                for axis in ("z", "y", "x"):
                    results.append(
                        sim.advance(
                            num_steps=16,
                            state=state,
                            backend="cuda_streamed",
                            sharding=dict(axis=axis, num_devices=4, backend="cpu"),
                            progress=False,
                        ).state
                    )
                for result in results:
                    assert all(
                        value.dtype == jax.numpy.bfloat16
                        for value in result.cpml_psi_h_terms
                    )
                    assert_state_close(results[0], result)
        else:
            raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    ["cpml", "boundaries", "nonuniform", "tensor", "mode", "recording", "precision"],
)
def test_native_sharded_feature_parity_on_cpu(case):
    env = dict(
        os.environ,
        XLA_FLAGS="--xla_force_host_platform_device_count=4",
        JAX_PLATFORMS="cpu",
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tests.unit.test_cuda_sharded_features import run_case; run_case(sys.argv[1])",
            case,
        ],
        check=True,
        env=env,
        timeout=300,
    )
