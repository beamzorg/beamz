"""Bloch public configuration and compilation contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.devices.boundaries import (
    bloch_storage_phases,
    validate_boundary_compatibility,
)


@pytest.mark.contract
def test_bloch_validation_and_physical_axes():
    with pytest.raises(ValueError, match="three finite"):
        bz.Bloch(wavevector=(np.nan, 0, 0))
    with pytest.raises(ValueError, match="outside"):
        bz.Bloch(axes="x", wavevector=(0, 1, 0))
    with pytest.raises(ValueError, match="Conflicting"):
        validate_boundary_compatibility(
            [bz.Periodic(axes="x"), bz.Bloch(axes="x", wavevector=(1, 0, 0))],
            is_3d=True,
        )
    with pytest.raises(ValueError, match="conflicting edges"):
        validate_boundary_compatibility(
            [bz.Bloch(axes="x"), bz.PML(edges="left")], is_3d=True
        )
    grid = bz.RectilinearGrid.from_spacing((2, 3, 4), 0.1)
    phases = bloch_storage_phases(
        [bz.Bloch(axes=("x", "z"), wavevector=(2, 0, -3))],
        grid,
        is_3d=False,
        plane_2d="xz",
    )
    np.testing.assert_allclose(phases, np.exp(1j * np.array([-3 * 0.3, 2 * 0.2])))


@pytest.mark.contract
def test_zero_bloch_matches_periodic_and_cache_distinguishes_wavevectors():
    kwargs = dict(size=(160e-9, 160e-9, 0), resolution=40e-9, run_time=1e-15)
    ordinary = bz.Simulation(**kwargs, boundaries=[bz.Periodic()])
    zero = bz.Simulation(**kwargs, boundaries=[bz.Bloch()])
    a, b = ordinary.initial_state(), zero.initial_state()
    assert a.ez.dtype == b.ez.dtype == jnp.float32
    for x, y in zip(
        jax.tree.leaves(ordinary.step(a)), jax.tree.leaves(zero.step(b)), strict=True
    ):
        np.testing.assert_array_equal(x, y)
    first = bz.Simulation(
        **kwargs, boundaries=[bz.Bloch(wavevector=(1e6, 0, 0))]
    ).compile()
    second = bz.Simulation(
        **kwargs, boundaries=[bz.Bloch(wavevector=(-1e6, 0, 0))]
    ).compile()
    assert first is not second
    assert first.boundary.periodic_phases != second.boundary.periodic_phases


@pytest.mark.parametrize(
    "plane,vector,axes",
    [
        ("xy", (2e6, -1e6, 0), ("x", "y")),
        ("xz", (2e6, 0, -1e6), ("x", "z")),
        ("yz", (0, 2e6, -1e6), ("y", "z")),
    ],
)
def test_bloch_compiled_phases_map_all_2d_planes(plane, vector, axes):
    sim = bz.Simulation(
        design=bz.Design(width=200e-9, height=160e-9),
        resolution=40e-9,
        run_time=1e-15,
        plane_2d=plane,
        boundaries=[bz.Bloch(axes=axes, wavevector=vector)],
    )
    program = sim.compile()
    np.testing.assert_allclose(
        program.boundary.periodic_phases,
        np.exp(1j * np.array([-1e6 * 160e-9, 2e6 * 200e-9])),
    )
    state = sim.initial_state()._replace(ez=sim.initial_state().ez.at[0, 0].set(1 + 2j))
    evolved = sim.step(state)
    np.testing.assert_allclose(
        evolved.ez[-1], program.boundary.periodic_phases[0] * evolved.ez[0], atol=1e-6
    )
    np.testing.assert_allclose(
        evolved.ez[:, -1],
        program.boundary.periodic_phases[1] * evolved.ez[:, 0],
        atol=1e-6,
    )


@pytest.mark.parametrize("case", ["mismatch", "broad", "evanescent", "partial"])
def test_oblique_source_rejects_incompatible_configurations(case):
    frequency = 5e14
    k = 2 * np.pi * frequency / bz.LIGHT_SPEED * 0.4
    source_k = (k * 4, 0, 0) if case == "evanescent" else (k, 0, 0)
    boundary_k = (k / 2, 0, 0) if case == "mismatch" else source_k
    aperture = (80e-9 if case == "partial" else 160e-9, 160e-9, 0)
    sim = bz.Simulation(
        size=(160e-9, 160e-9, 1.6e-6),
        resolution=40e-9,
        run_time=1e-15,
        sources=[
            bz.PlaneWaveSource(
                center=(0, 0, -0.3e-6),
                size=aperture,
                source_time=bz.GaussianPulse(
                    frequency, 0.2 * frequency if case == "broad" else 0.05 * frequency
                ),
                direction="+z",
                transverse_wavevector=source_k,
            )
        ],
        boundaries=[
            bz.Bloch(axes=("x", "y"), wavevector=boundary_k),
            bz.PML(edges=("front", "back"), thickness=160e-9),
        ],
    )
    match = {
        "mismatch": "must match",
        "broad": "narrowband",
        "evanescent": "propagating",
        "partial": "full transverse",
    }[case]
    with pytest.raises(ValueError, match=match):
        sim.compile()


def test_bloch_rejects_cuda_and_sharding_explicitly():
    sim = bz.Simulation(
        size=(160e-9, 160e-9, 160e-9),
        resolution=40e-9,
        run_time=1e-15,
        boundaries=[bz.Bloch(wavevector=(1e6, 0, 0))],
    )
    from beamz.simulation.backend import CudaBackendUnavailable

    with pytest.raises(CudaBackendUnavailable, match="periodic"):
        sim.compile(backend="cuda_streamed")
    from beamz.simulation.compile import compile_simulation
    from beamz.simulation.model import ShardingConfig

    request = sim.to_request(sharding=ShardingConfig(enabled=True, num_devices=2))
    with pytest.raises(ValueError, match="single-device JAX"):
        compile_simulation(request)
