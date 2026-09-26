"""Causal media, ownership, and ADE constitutive physics checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.simulation.dispersion import update_dispersion

BAND = (4e14, 8e14)


def test_drude_and_lorentz_match_analytic_susceptibilities():
    f = np.linspace(*BAND, 101)
    w = 2 * np.pi * f
    drude = bz.PoleResidue.drude(
        1.0, plasma_frequency=1.2e16, damping=2e14, frequency_range=BAND
    )
    np.testing.assert_allclose(
        drude.eps_model(f), 1 - 1.2e16**2 / (w * w + 2e14 * 1j * w), rtol=1e-12
    )
    lorentz = bz.PoleResidue.lorentz(
        2.0, strength=3.0, resonance=6e15, damping=3e14, frequency_range=BAND
    )
    np.testing.assert_allclose(
        lorentz.eps_model(f),
        2 + 3 * 6e15**2 / (6e15**2 - w * w - 3e14 * 1j * w),
        rtol=1e-12,
    )
    restored = bz.PoleResidue.from_spec(drude.to_spec())
    assert restored.cache_spec() == drude.cache_spec()
    assert drude.eps_model(5e14).real < 0


def test_rejects_gain_and_unstable_poles():
    with pytest.raises(ValueError, match="real part"):
        bz.PoleResidue(1.0, [(1e14, 1e14)], frequency_range=BAND)
    with pytest.raises(ValueError, match="passivity"):
        bz.PoleResidue(1.0, [(-1e14, -1e14)], frequency_range=BAND)


def test_raster_keeps_same_epsilon_materials_distinct_and_painter_order():
    a = bz.PoleResidue.drude(
        1.0, plasma_frequency=1e16, damping=1e14, frequency_range=BAND
    )
    b = bz.PoleResidue.drude(
        1.0, plasma_frequency=2e16, damping=1e14, frequency_range=BAND
    )
    design = bz.Design(width=1e-6, height=1e-6, background=a)
    design += bz.Rectangle(
        position=(0.25e-6, 0.25e-6), width=0.5e-6, height=0.5e-6, material=b
    )
    design += bz.Rectangle(
        position=(0.4e-6, 0.4e-6),
        width=0.2e-6,
        height=0.2e-6,
        material=bz.Material(1.0),
    )
    grid = design.rasterize(50e-9)
    assert len(grid.dispersion) == 2
    first, second = [supports["Ez"] for _, supports in grid.dispersion]
    assert first[2, 2] == 1 and second[2, 2] == 0
    assert first[7, 7] == 0 and second[7, 7] == 1
    assert first[10, 10] == second[10, 10] == 0
    assert grid.smoothing == "volume"


@pytest.mark.parametrize("polarization", ["tm", "te"])
def test_ade_harmonic_response_and_continuation(polarization):
    m = bz.PoleResidue.lorentz(
        1.0, strength=2.0, resonance=6e15, damping=8e14, frequency_range=BAND
    )
    dt = 1e-17
    sim = bz.Simulation(
        design=bz.Design(width=0.2e-6, height=0.2e-6, background=m),
        resolution=50e-9,
        time=np.arange(6000) * dt,
        polarization=polarization,
        boundaries=[bz.Periodic(axes=("x", "y"))],
    )
    program = sim.compile(backend="jax")
    state = sim.initial_state()
    # Prescribe harmonic D/epsilon0. Ampere's update is its exact difference.
    # This checks the constitutive solve independently of curl dispersion.
    name = "ez" if polarization == "tm" else "ex"
    omega = 2 * np.pi * 5e14

    def step(state, i):
        old = state
        delta = jnp.cos(omega * dt * (i + 1)) - jnp.cos(omega * dt * i)
        free = state._replace(**{name: getattr(state, name) + delta})
        state = update_dispersion(old, free, program.dispersion)
        return state, jnp.mean(getattr(state, name))

    state, trace = jax.jit(lambda s: jax.lax.scan(step, s, jnp.arange(6000)))(state)
    t = np.arange(1, 6001) * dt
    matrix = np.c_[np.cos(omega * t[-2000:]), np.sin(omega * t[-2000:]), np.ones(2000)]
    amplitude = np.linalg.lstsq(matrix, np.asarray(trace)[-2000:], rcond=None)[0]
    measured = amplitude[0] + 1j * amplitude[1]
    expected = 1 / m.eps_model(omega / (2 * np.pi))
    np.testing.assert_allclose(measured, expected, rtol=0.003, atol=0.001)
    initial = sim.initial_state()._replace(
        **{name: jnp.ones_like(getattr(sim.initial_state(), name))}
    )
    full = sim.advance(
        state=initial, num_steps=20, backend="jax", performance=False
    ).state
    half = sim.advance(
        state=initial, num_steps=10, backend="jax", performance=False
    ).state
    resumed = sim.advance(
        state=half, num_steps=10, backend="jax", performance=False
    ).state
    for actual, reference in zip(
        jax.tree.leaves(resumed), jax.tree.leaves(full), strict=True
    ):
        np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)


def test_passive_fit_recovers_synthetic_optics():
    m = bz.PoleResidue.lorentz(
        2.0, strength=1.2, resonance=6e15, damping=5e14, frequency_range=BAND
    )
    wl = np.linspace(400e-9, 700e-9, 80)
    nk = np.sqrt(m.eps_model(bz.LIGHT_SPEED / wl))
    fit, report = bz.fit_nk(wl, nk.real, nk.imag, num_poles=1, max_nfev=500)
    np.testing.assert_allclose(fit.eps_model(bz.LIGHT_SPEED / wl), nk**2, rtol=0.001)
    assert report["rms_k"] < 0.001


def test_broadband_plane_wave_normalization_is_independent_of_run_duration():
    size = (0.16e-6, 0.16e-6, 2.4e-6)
    frequencies = np.array([4.5e14, 5.5e14, 7e14])
    sim = bz.Simulation(
        size=size,
        sources=[
            bz.PlaneWaveSource(
                center=(0, 0, 0.7e-6),
                size=(size[0], size[1], 0),
                source_time=bz.GaussianPulse(5.75e14, 2e14),
                direction="-z",
            )
        ],
        monitors=[
            bz.FluxMonitor(
                center=(0, 0, 0),
                size=(size[0], size[1], 0),
                freqs=frequencies,
                name="flux",
            )
        ],
        boundaries=[
            bz.Periodic(axes=("x", "y")),
            bz.PML(edges=("front", "back"), thickness=0.3e-6),
        ],
        grid_spec=bz.GridSpec.uniform(20e-9, courant=0.7),
        run_time=120e-15,
    )
    half = sim.advance(num_steps=sim.num_steps // 2, backend="jax", performance=False)
    final = sim.advance(
        state=half.state,
        num_steps=sim.num_steps - sim.num_steps // 2,
        backend="jax",
        performance=False,
    )
    np.testing.assert_allclose(
        half.results["flux"].flux, final.results["flux"].flux, rtol=0.01
    )
    assert np.all(-final.results["flux"].flux > 0.9)


def test_periodic_material_supports_join_across_seam():
    m = bz.PoleResidue.lorentz(
        2.0, strength=1.0, resonance=6e15, damping=8e14, frequency_range=BAND
    )
    design = bz.Design(width=0.4e-6, height=0.4e-6)
    design += bz.Rectangle(position=(0, 0), width=0.2e-6, height=0.4e-6, material=m)
    sim = bz.Simulation(
        design=design,
        resolution=50e-9,
        time=np.arange(10) * 1e-17,
        boundaries=[bz.Periodic(axes=("x", "y"))],
    )
    plan = sim.compile(backend="jax")
    np.testing.assert_allclose(plan.grid.eps_z[:, 0], plan.grid.eps_z[:, -1])
    assert np.allclose(plan.grid.eps_z[:, 0], 1.5)
    region = plan.dispersion.regions[0]
    np.testing.assert_allclose(region.weights[:, 0], region.weights[:, -1])


def test_sensor_shield_opening_overwrites_metal_at_correct_height():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "cmos_model", root / "examples/cmos_rgb_sensor.py"
    )
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    design, _, _ = model.make_sensor_design()
    hole = next(s for s in design.structures if isinstance(s, bz.Circle))
    shield = next(
        s
        for s in design.structures
        if isinstance(s, bz.Box) and s.size == (model.Lx, model.Ly, model.t_shield)
    )
    assert hole.z == pytest.approx(shield.center[2] - shield.size[2] / 2, abs=1e-20)


@pytest.mark.parametrize("kind", ["aluminum", "silicon"])
@pytest.mark.simulation
def test_broadband_dispersive_slab_matches_fresnel(kind):
    import json
    from pathlib import Path

    from scripts.validation.dispersive_slabs import analytic_slab, run_slab

    root = Path(__file__).resolve().parents[3]
    specs = json.loads((root / "examples/data/cmos_rgb/materials.json").read_text())
    name, thickness = (
        ("Al_Rakic1995", 40e-9) if kind == "aluminum" else ("aSi_Horiba", 100e-9)
    )
    m = bz.PoleResidue.from_spec(specs[name])
    wavelength = np.linspace(400e-9, 700e-9, 9)
    reference, _ = run_slab(None, 5e-9, wavelength, thickness)
    device, _ = run_slab(m, 5e-9, wavelength, thickness)
    expected_R, expected_T = analytic_slab(
        m.eps_model(bz.LIGHT_SPEED / wavelength), wavelength, thickness
    )
    R = 1 - device["input"] / reference["input"]
    T = device["output"] / reference["output"]
    np.testing.assert_allclose(R, expected_R, atol=0.02, rtol=0)
    np.testing.assert_allclose(T, expected_T, atol=0.003, rtol=0)
    assert np.min(1 - R - T) > 0


def test_dispersion_rejects_unsupported_execution_and_energy_stopping():
    m = bz.PoleResidue.drude(
        1.0, plasma_frequency=1e16, damping=1e14, frequency_range=BAND
    )
    sim = bz.Simulation(
        design=bz.Design(width=0.4e-6, height=0.4e-6, background=m),
        resolution=50e-9,
        time=np.arange(20) * 1e-17,
    )
    with pytest.raises(RuntimeError, match="Dispersive media require"):
        sim.compile(backend="cuda_streamed")
    with pytest.raises(ValueError, match="material energy"):
        sim.run(backend="jax", termination=bz.AutoTermination(chunk_steps=10))


def test_imported_scene_keeps_dispersive_response():
    from beamz.design.raster import Grid, Scene

    m = bz.PoleResidue.drude(
        1.0, plasma_frequency=1e16, damping=1e14, frequency_range=BAND
    )
    scene = Scene(materials=(m,), objects=())
    grid = Grid.uniform((0.0, 0.0, 0.0), (0.4e-6, 0.4e-6, 50e-9), (8, 8, 1))
    sim = bz.Simulation(scene=scene, raster_grid=grid, time=np.arange(20) * 1e-17)
    program = sim.compile(backend="jax")
    assert len(program.dispersion.regions) == 1
    assert program.dispersion.regions[0].shape[0] == len(m.poles)


def test_memory_estimate_includes_auxiliary_polarization():
    m = bz.PoleResidue.drude(
        1.0, plasma_frequency=1e16, damping=1e14, frequency_range=BAND
    )
    sim = bz.Simulation(
        design=bz.Design(width=0.4e-6, height=0.4e-6, background=m),
        resolution=50e-9,
        time=np.arange(20) * 1e-17,
    )
    state = sim.initial_state()
    expected = sum(q.size * q.dtype.itemsize for q in state.polarization)
    report = sim.memory_estimate()["compiled"]
    assert report["totals_by_category"]["polarization_state"] == expected
    assert report["totals_by_category"]["dispersion_coefficients"] > 0
