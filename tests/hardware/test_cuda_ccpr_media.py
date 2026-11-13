"""Hardware parity of CCPR (E'-coupled) tensor poles on streamed CUDA.

The centred ``B`` term ``(dt / (2 d)) B (E^{n+1} - E^{n-1})`` runs natively:
the pole launch (op 0) applies its explicit half from the stored previous
field change, and a third launch (op 2) the implicit half
``(dt / (2 d)) B (E^{n+1} - E^n)``. JAX is the reference; the compared
polarization includes the stored change.

Run explicitly on a CUDA host with the matching native component:
JAX_PLATFORMS=cuda PYTHONPATH=. pytest -q tests/hardware/test_cuda_ccpr_media.py
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
from beamz.design.anisotropic import pole_residue_tensor_poles
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
from beamz.simulation.ccpr import CCPRPolarizationRegion

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(
    not STATUS.available,
    reason=STATUS.reason or "CUDA backend unavailable",
)

FIELDS = ("ex", "ey", "ez", "hx", "hy", "hz")
WAVELENGTH = 1.5e-6
OMEGA = 2 * np.pi * bz.LIGHT_SPEED / WAVELENGTH
BAND = (0.3 * OMEGA / (2 * np.pi), 3 * OMEGA / (2 * np.pi))


def _model(epsilon_inf, scale=1.0):
    """Passive CCPR pair with Re(residue) != 0 plus a Debye term."""
    return bz.PoleResidue(
        epsilon_inf,
        [
            (-0.1 * OMEGA - 2j * scale * OMEGA, 0.05 * OMEGA + 0.6j * OMEGA),
            (-0.3 * scale * OMEGA, 0.2 * OMEGA),
        ],
        frequency_range=BAND,
    )


def _media(kind):
    """A CCPR medium and a static tensor medium that never overlap."""
    if kind == "magnetic":
        pole = bz.AnisotropicMedium(
            2.0,
            (1.3, 1.1, 1.2),
            magnetic_poles=pole_residue_tensor_poles(
                _model(1.0), tensor=(0.6, 0.2, 0.3)
            ),
        )
        return pole, bz.AnisotropicMedium(2.5, (1.5, 1.2, 1.8))
    other = 2.0 if kind == "coupled" else 3.0
    crystal = bz.AnisotropicMedium.from_pole_residue(
        _model(2.0), _model(other, 1.4), _model(2.0, 0.8)
    ).rotated((0.2, 0.3, 0.4))
    static = bz.AnisotropicMedium(1.5, (2.0, 1.4, 1.2))
    if kind == "full":  # an off-diagonal static tensor elsewhere, too
        static = static.rotated((0.1, 0.5, 0.2))
    return crystal, static


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
    return bz.Simulation(
        material_grid=material,
        time=np.arange(41) * material.grid.cfl_time_step(0.5),
        boundaries=[bz.PML(edges="all", thickness=0.15 * length, formulation="cpml")],
    )


def _seeded(sim):
    state = sim.initial_state()
    rng = np.random.default_rng(11)
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
@pytest.mark.parametrize("kind", ["coupled", "full", "magnetic"])
def test_native_ccpr_poles_match_jax(kind, grid_kind, monkeypatch):
    sim = _simulation(kind, grid_kind)
    program = sim.compile(backend="cuda_streamed")
    plan = program.dispersion.anisotropic
    regions = plan.magnetic_regions if kind == "magnetic" else plan.electric_regions
    (region,) = regions
    assert isinstance(region, CCPRPolarizationRegion)
    assert (region.static is None) == (kind == "full")
    assert region.edot_offdiagonal is not None or kind == "magnetic"
    calls = collections.Counter()
    for name in ("advance_poles", "apply_inverse", "add_edot"):
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
    assert calls["add_edot"] >= 1 and calls["advance_poles"] >= 1, calls
    for name in (*FIELDS, "polarization"):
        expected = _flat(getattr(reference, name))
        result = _flat(getattr(actual, name))
        scale = np.abs(expected).max()
        assert scale > 0 and np.isfinite(result).all(), name
        assert np.abs(result - expected).max() < 2e-5 * scale, name
