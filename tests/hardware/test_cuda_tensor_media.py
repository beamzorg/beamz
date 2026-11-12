"""Hardware parity of tensor permeability and tensor-pole media on streamed CUDA.

The streamed backend runs the tensor-media corrections through the native
``beamz_cuda_tensor_media`` kernels; JAX is the reference. Non-symmetric
tensors and conductivity tensors exercise the inverse launch's antisymmetric
cross terms.

Run explicitly on a CUDA host with the matching native component:
JAX_PLATFORMS=cuda PYTHONPATH=. pytest -q tests/hardware/test_cuda_tensor_media.py
"""

from __future__ import annotations

import collections
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
import beamz.simulation.cuda.tensor_media as native
from beamz.design import MaterialGrid
from beamz.design.raster import (
    Box,
    Grid,
    Material,
    Object,
    RasterOptions,
    Scene,
    rasterize,
)
from beamz.design.raster.dispersion import rasterize_tensor_dispersion
from beamz.simulation.backend import cuda_backend_status
from beamz.simulation.execute import build_step_context
from beamz.simulation.kernels import select_update_kernel

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(
    not STATUS.available,
    reason=STATUS.reason or "CUDA backend unavailable",
)

FIELDS = ("ex", "ey", "ez", "hx", "hy", "hz")
WAVELENGTH = 1.5e-6
OMEGA = 2 * np.pi * bz.LIGHT_SPEED / WAVELENGTH


def _media(kind):
    """Two non-overlapping media exercising one family of native kernels each."""
    if kind == "electric_coupled":
        pole = bz.AnisotropicMedium(
            2.0,
            electric_poles=[
                bz.TensorPole.lorentz(
                    (1.5, 0.5, 0.2), resonance=2 * OMEGA, damping=0.02 * OMEGA
                )
            ],
        ).rotated((0.2, 0.3, 0.4))
        other = bz.AnisotropicMedium(1.5, (2.0, 1.4, 1.2)).rotated((0.1, 0.5, 0.2))
    elif kind == "magnetic_pole":
        pole = bz.AnisotropicMedium(
            2.0,
            (1.3, 1.1, 1.2),
            magnetic_poles=[
                bz.TensorPole.lorentz(
                    (0.6, 0.2, 0.3), resonance=2.5 * OMEGA, damping=0.05 * OMEGA
                )
            ],
        ).rotated((0.3, 0.1, 0.5))
        other = bz.AnisotropicMedium(2.5, (1.5, 1.2, 1.8))
    elif kind == "nonsymmetric":
        # Hermitian (gyrotropic) eps with loss, rotated: antisymmetric and
        # off-diagonal conductivity on a full-tensor grid, plus a tensor pole
        # whose inverse inherits the antisymmetric cross terms.
        frequency = OMEGA / (2 * np.pi)
        pole = bz.AnisotropicMedium.from_complex_permittivity(
            ((2.0 + 0.05j, 0.3j, 0.1), (-0.3j, 2.5 + 0.02j, 0.0), (0.1, 0.0, 3.0)),
            frequency=frequency,
            electric_poles=[
                bz.TensorPole.lorentz(
                    (0.5, 0.2, 0.1), resonance=2 * OMEGA, damping=0.02 * OMEGA
                )
            ],
        ).rotated((0.2, 0.3, 0.4))
        other = bz.AnisotropicMedium.from_complex_permittivity(
            2.0,
            permeability=(
                (1.3 + 0.04j, 0.2j, 0.0),
                (-0.2j, 1.2, 0.05j),
                (0.0, -0.05j, 1.1 + 0.01j),
            ),
            frequency=frequency,
        ).rotated((0.1, 0.5, 0.2))
    else:  # two uncoupled diagonal poles and diagonal permeability
        pole = bz.AnisotropicMedium(
            (2.0, 2.5, 3.0),
            (1.2, 1.4, 1.1),
            electric_poles=[
                bz.TensorPole.lorentz(
                    (1.0, 0.5, 0.2), resonance=2 * OMEGA, damping=0.02 * OMEGA
                ),
                bz.TensorPole.lorentz(
                    (0.3, 0.6, 0.4), resonance=3 * OMEGA, damping=0.1 * OMEGA
                ),
            ],
        )
        other = bz.AnisotropicMedium(1.5, (2.0, 1.4, 1.2))
    return pole, other


def _simulation(kind, grid_kind, cells=20):
    length = cells * WAVELENGTH / 12
    u = np.linspace(0, 1, cells + 1)
    edges = [length * u] * 3
    if grid_kind == "graded":
        edges = [length * (u + a * np.sin(2 * np.pi * u)) for a in (0.04, 0.06, 0.08)]
    pole, other = _media(kind)
    scene = Scene(
        (Material(), pole, other),
        (
            Object(Box((0.25 * length,) * 3, (0.6 * length,) * 3), 1),
            Object(
                Box(
                    (0.65 * length, 0.3 * length, 0.3 * length),
                    (0.85 * length, 0.6 * length, 0.7 * length),
                ),
                2,
            ),
        ),
    )
    grid = Grid(*edges)
    options = RasterOptions(smoothing="farjadpour_full")
    material = MaterialGrid.from_raster_result(rasterize(scene, grid, options=options))
    # The low-level raster API leaves pole media to an explicit second pass.
    material = replace(
        material,
        tensor_dispersion=rasterize_tensor_dispersion(
            scene,
            grid,
            kind="3d",
            polarization="tm",
            quality=options.quality,
            resolution=material.resolution,
            cache_directory=None,
        ),
    )
    assert material.tensor_dispersion
    return bz.Simulation(
        material_grid=material,
        time=np.arange(41) * material.grid.cfl_time_step(0.5),
        boundaries=[bz.PML(edges="all", thickness=0.15 * length, formulation="cpml")],
    )


def _seeded(sim):
    state = sim.initial_state()
    rng = np.random.default_rng(7)
    return state._replace(
        **{
            name: jnp.asarray(
                rng.normal(size=getattr(state, name).shape), dtype=jnp.float32
            )
            * (1.0 if name[0] == "e" else 1 / 377.0)
            for name in FIELDS
        }
    )


def _flat(tree):
    return np.concatenate([np.asarray(v).ravel() for v in jax.tree.leaves(tree)])


@pytest.mark.parametrize("grid_kind", ["uniform", "graded"])
@pytest.mark.parametrize(
    "kind", ["electric_coupled", "magnetic_pole", "diagonal", "nonsymmetric"]
)
def test_native_tensor_media_match_jax(kind, grid_kind, monkeypatch):
    sim = _simulation(kind, grid_kind)
    program = sim.compile(backend="cuda_streamed")
    assert program.config.backend == "cuda_streamed"
    assert select_update_kernel(build_step_context(program)).kind == "cuda_streamed"
    calls = collections.Counter()
    for name in ("advance_poles", "apply_inverse"):
        original = getattr(native, name)

        def counted(*args, _original=original, _name=name, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(native, name, counted)

    state = _seeded(sim)
    reference = sim.advance(
        state=state, num_steps=40, backend="jax", performance=False
    ).state
    assert not calls
    actual = sim.advance(
        state=state, num_steps=40, backend="cuda_streamed", performance=False
    ).state
    # One pole region; the other medium is a static tensor (inverse only).
    assert calls["advance_poles"] >= 1 and calls["apply_inverse"] >= 2, calls
    for name in (*FIELDS, "polarization"):
        expected = _flat(getattr(reference, name))
        result = _flat(getattr(actual, name))
        scale = np.abs(expected).max()
        assert scale > 0 and np.isfinite(result).all(), name
        assert np.abs(result - expected).max() < 2e-5 * scale, name


def test_native_inverse_needs_the_antisymmetric_cross_terms(monkeypatch):
    """Negative control: dropping the native skew terms breaks JAX parity."""
    from beamz.simulation.compile import clear_program_cache
    from beamz.simulation.execute import clear_execution_cache

    sim = _simulation("nonsymmetric", "uniform")
    state = _seeded(sim)
    reference = sim.advance(
        state=state, num_steps=40, backend="jax", performance=False
    ).state
    original = native.apply_inverse

    def dropped(state, box, inverse, *args, **kwargs):
        return original(state, box, replace(inverse, skew=None), *args, **kwargs)

    monkeypatch.setattr(native, "apply_inverse", dropped)
    clear_program_cache()
    clear_execution_cache()
    actual = sim.advance(
        state=state, num_steps=40, backend="cuda_streamed", performance=False
    ).state
    error = max(
        np.abs(_flat(getattr(actual, n)) - _flat(getattr(reference, n))).max()
        / np.abs(_flat(getattr(reference, n))).max()
        for n in FIELDS
    )
    assert error > 1e-3, error


def test_tensor_media_streamed_cuda_reexecutes():
    sim = _simulation("electric_coupled", "uniform", cells=24)
    first = sim.advance(num_steps=40, backend="cuda_streamed", performance=False)
    # Re-executing the cached executable used to crash inside XLA's CUDA-graph
    # update once per-step FFI phases were captured into command buffers.
    second = sim.advance(num_steps=40, backend="cuda_streamed", performance=False)
    for name in FIELDS:
        np.testing.assert_array_equal(
            np.asarray(getattr(first.state, name)),
            np.asarray(getattr(second.state, name)),
        )
