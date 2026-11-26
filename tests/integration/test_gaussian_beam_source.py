from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PML,
    Box,
    Design,
    FluxMonitor,
    GaussianBeamSource,
    GaussianPulse,
    GridSpec,
    Material,
    RectilinearGrid,
    Simulation,
    um,
)
from beamz.design import MaterialGrid
from beamz.devices.sources.compiler import (
    compile_source_specs,
    field_profile_phasor_residuals,
    gaussian_beam_field_profile,
)
from beamz.devices.sources.planar_tfsf import (
    advance_incident_e_3d,
    advance_incident_h_3d,
    build_incident_3d_phasor_state,
    launched_side_component_mask_3d,
    mask_incident_3d_state_to_launched_side,
)
from tests.utils import compiled_grid


def _empty_3d_fields(shape=(10, 10, 10)):
    nz, ny, nx = (int(v) for v in shape)
    ex = np.zeros((nz, ny, nx - 1), dtype=np.float64)
    ey = np.zeros((nz, ny - 1, nx), dtype=np.float64)
    ez = np.zeros((nz - 1, ny, nx), dtype=np.float64)
    hx = np.zeros((nz - 1, ny - 1, nx), dtype=np.float64)
    hy = np.zeros((nz - 1, ny, nx - 1), dtype=np.float64)
    hz = np.zeros((nz, ny - 1, nx - 1), dtype=np.float64)
    return SimpleNamespace(
        boundaries=[],
        permittivity=np.ones((nz, ny, nx), dtype=np.float64),
        permeability=np.ones((nz, ny, nx), dtype=np.float64),
        conductivity=np.zeros((nz, ny, nx), dtype=np.float64),
        total_conductivity=np.zeros((nz, ny, nx), dtype=np.float64),
        Ex=ex,
        Ey=ey,
        Ez=ez,
        Hx=hx,
        Hy=hy,
        Hz=hz,
        eps_x=np.ones_like(ex),
        eps_y=np.ones_like(ey),
        eps_z=np.ones_like(ez),
        sig_x=np.zeros_like(ex),
        sig_y=np.zeros_like(ey),
        sig_z=np.zeros_like(ez),
        region_x=(slice(None), slice(None), slice(None)),
        region_y=(slice(None), slice(None), slice(None)),
        region_z=(slice(None), slice(None), slice(None)),
        sigma_m_hx=np.zeros_like(hx),
        sigma_m_hy=np.zeros_like(hy),
        sigma_m_hz=np.zeros_like(hz),
    )


def _pulse(wavelength):
    freq = LIGHT_SPEED / float(wavelength)
    return GaussianPulse(
        freq0=freq,
        fwidth=freq,
        offset=0.0,
        remove_dc_component=False,
    )


def _expand_residuals(residuals, fields, components):
    expanded = {
        component: np.zeros_like(getattr(fields, component), dtype=np.complex128)
        for component in components
    }
    for residual in residuals:
        if residual.component in expanded:
            expanded[residual.component][residual.index] += residual.residual
    return expanded


def test_gaussian_beam_source_emits_compiled_planar_tfsf_specs():
    fields = _empty_3d_fields()
    source = GaussianBeamSource(
        center=(1.0 * um, 1.0 * um, 1.0 * um),
        size=(1.2 * um, 1.2 * um),
        source_time=_pulse(1.55 * um),
        direction="+x",
        angle_theta=0.0,
        angle_phi=0.0,
        pol_angle=0.0,
        waist_radius=0.35 * um,
    )
    dt = 0.15 * um / (LIGHT_SPEED * np.sqrt(3.0))

    specs = compile_source_specs(
        (source,),
        fields,
        dt=dt,
        resolution=0.25 * um,
        num_steps=3,
        t0=0.0,
        total_steps=3,
    )

    assert specs
    assert {"h", "e"} <= {spec.timing for spec in specs}
    assert any(spec.component.startswith("H") for spec in specs)
    assert any(spec.component.startswith("E") for spec in specs)
    assert any(spec.is_slab for spec in specs)


def test_gaussian_beam_source_empty_space_propagation_smoke():
    wavelength = 1.55 * um
    dx = 0.35 * um
    dt = 0.45 * dx / (LIGHT_SPEED * np.sqrt(3.0))
    source = GaussianBeamSource(
        center=(-0.45 * um, 0.0, 0.0),
        size=(1.4 * um, 1.4 * um),
        source_time=_pulse(wavelength),
        direction="+x",
        angle_theta=0.0,
        angle_phi=0.0,
        pol_angle=0.0,
        waist_radius=0.45 * um,
        power=1.0,
    )
    sim = Simulation(
        domain=(2.8 * um, 2.8 * um, 2.8 * um),
        background=Material(permittivity=1.0),
        sources=[source],
        monitors=[],
        boundaries=[],
        time=np.arange(0.0, 5 * dt, dt),
        resolution=dx,
    )

    state = sim.initial_state()
    for _ in range(3):
        state = sim.step(state)

    energy = sum(
        float(np.sum(np.asarray(getattr(state, component.lower())) ** 2))
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    )
    assert np.isfinite(energy)
    assert energy > 0.0


def test_gaussian_beam_source_has_low_source_plane_residual_error():
    fields = _empty_3d_fields(shape=(11, 11, 11))
    dt = 0.12 * um / (LIGHT_SPEED * np.sqrt(3.0))
    resolution = 0.25 * um
    source = GaussianBeamSource(
        center=(1.25 * um, 1.25 * um, 1.25 * um),
        size=(1.5 * um, 1.5 * um),
        source_time=_pulse(1.55 * um),
        direction="+x",
        angle_theta=0.08,
        angle_phi=0.2,
        pol_angle=0.1,
        waist_radius=0.45 * um,
    )
    profile = gaussian_beam_field_profile(source, fields, resolution=resolution)

    full_prev = build_incident_3d_phasor_state(
        profile,
        fields,
        resolution=resolution,
        t_e=0.0,
        t_h=-0.5 * dt,
        masked=False,
        max_shift=source.max_shift,
    )
    masked_prev = build_incident_3d_phasor_state(
        profile,
        fields,
        resolution=resolution,
        t_e=0.0,
        t_h=-0.5 * dt,
        masked=True,
        max_shift=source.max_shift,
    )
    h_full_next = advance_incident_h_3d(
        fields,
        full_prev,
        dt,
        resolution=resolution,
    )
    h_target_next = mask_incident_3d_state_to_launched_side(
        profile,
        h_full_next,
        resolution=resolution,
    )
    h_mask_next = advance_incident_h_3d(
        fields,
        masked_prev,
        dt,
        resolution=resolution,
    )
    e_full_next = advance_incident_e_3d(
        fields,
        full_prev,
        h_full_next,
        dt,
        resolution=resolution,
    )
    e_target_next = mask_incident_3d_state_to_launched_side(
        profile,
        e_full_next,
        resolution=resolution,
    )
    e_mask_next = advance_incident_e_3d(
        fields,
        masked_prev,
        h_target_next,
        dt,
        resolution=resolution,
    )

    residuals = field_profile_phasor_residuals(
        profile,
        fields,
        dt=dt,
        resolution=resolution,
        max_shift=source.max_shift,
    )
    h_delta = _expand_residuals(residuals, fields, ("Hx", "Hy", "Hz"))
    e_delta = _expand_residuals(residuals, fields, ("Ex", "Ey", "Ez"))

    for component in ("Hx", "Hy", "Hz"):
        np.testing.assert_allclose(
            h_mask_next[component] + h_delta[component],
            h_target_next[component],
            rtol=1e-6,
            atol=1e-8,
        )
    for component in ("Ex", "Ey", "Ez"):
        np.testing.assert_allclose(
            e_mask_next[component] + e_delta[component],
            e_target_next[component],
            rtol=1e-6,
            atol=1e-8,
        )


@pytest.mark.parametrize("direction", ["+x", "-x", "+y", "-y", "+z", "-z"])
def test_planar_phasor_residuals_support_rectilinear_metrics_and_padding(direction):
    resolution = 0.25 * um
    material = np.ones((9, 10, 11), dtype=np.float64)
    fields = compiled_grid(
        material,
        np.zeros_like(material),
        material,
        resolution,
    )

    def graded_edges(count, phase):
        angles = np.linspace(phase, phase + np.pi, count)
        widths = resolution * (0.85 + 0.3 * np.sin(angles) ** 2)
        return np.concatenate(([0.0], np.cumsum(widths)))

    grid = RectilinearGrid(
        graded_edges(11, 0.0),
        graded_edges(10, 0.2),
        graded_edges(9, 0.4),
    )
    fields.geometry = grid
    source = GaussianBeamSource(
        center=(1.25 * um, 1.2 * um, 1.0 * um),
        size=(1.2 * um, 1.1 * um),
        source_time=_pulse(1.55 * um),
        direction=direction,
        angle_theta=0.05,
        angle_phi=0.1,
        pol_angle=0.2,
        waist_radius=0.4 * um,
    )
    profile = gaussian_beam_field_profile(
        source, fields, resolution=resolution, grid=grid
    )
    axis = direction[1]
    axis_slot = "zyx".index(axis)
    component = "Ey" if axis == "x" else "Ex"
    assert profile.indices[component][axis_slot] == np.argmin(
        abs(grid.axis_edges(axis) - source.center["xyz".index(axis)])
    )
    dt = 0.1 * um / (LIGHT_SPEED * np.sqrt(3.0))

    residuals = field_profile_phasor_residuals(
        profile,
        fields,
        dt=dt,
        resolution=resolution,
        max_shift=source.max_shift,
    )

    assert residuals
    assert {residual.timing for residual in residuals} == {"e", "h"}
    assert all(np.all(np.isfinite(residual.residual)) for residual in residuals)

    padded_shape = list(fields.Ex.shape)
    padded_shape[axis_slot] += 2
    padded_mask = launched_side_component_mask_3d(
        profile,
        "Ex",
        tuple(padded_shape),
        resolution=resolution,
    )
    expected_shape = [1, 1, 1]
    expected_shape[axis_slot] = padded_shape[axis_slot]
    assert padded_mask.shape == tuple(expected_shape)


def test_gaussian_beam_source_compiled_engine_support():
    wavelength = 1.55 * um
    dx = 0.4 * um
    dt = 0.45 * dx / (LIGHT_SPEED * np.sqrt(3.0))
    source = GaussianBeamSource(
        center=(0.0, 0.0, 0.35 * um),
        size=(1.2 * um, 1.2 * um),
        source_time=_pulse(wavelength),
        direction="-z",
        angle_theta=0.1,
        angle_phi=0.2,
        pol_angle=0.0,
        waist_radius=0.45 * um,
    )
    sim = Simulation(
        domain=(2.4 * um, 2.4 * um, 2.4 * um),
        background=Material(permittivity=1.0),
        sources=[source],
        monitors=[],
        boundaries=[],
        time=np.arange(0.0, 4 * dt, dt),
        resolution=dx,
    )

    program = cast(Any, sim.compile(num_steps=2))

    assert program.sources
    assert {"h", "e"} <= {spec.timing for spec in program.sources}
    result = sim.advance(num_steps=1, progress=False)
    total = sum(
        float(np.sum(np.asarray(getattr(result.state, component.lower())) ** 2))
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    )
    assert np.isfinite(total)


def _empty_space_gaussian_flux(
    *,
    steps_per_wavelength=8,
    angle_theta=0.0,
    angle_phi=0.0,
    mesh="uniform",
    backend="jax",
):
    wavelength = 1.55 * um
    freq0 = LIGHT_SPEED / wavelength
    source = GaussianBeamSource(
        center=(3.5 * um, 3.5 * um, 2.5 * um),
        size=(5.5 * um, 5.5 * um),
        source_time=GaussianPulse(freq0=freq0, fwidth=freq0 / 5.0, offset=4.0),
        direction="-z",
        angle_theta=angle_theta,
        angle_phi=angle_phi,
        pol_angle=np.pi / 2.0,
        waist_radius=1.2 * um,
        wavelength=wavelength,
        power=1.0,
    )
    monitors = cast(
        Any,
        [
            FluxMonitor(
                center=(3.5 * um, 3.5 * um, 1.25 * um),
                size=(5.8 * um, 5.8 * um, 0.0),
                freqs=[freq0],
                name="forward",
            ),
            FluxMonitor(
                center=(3.5 * um, 3.5 * um, 3.75 * um),
                size=(5.8 * um, 5.8 * um, 0.0),
                freqs=[freq0],
                name="backward",
            ),
        ],
    )
    spacing = wavelength / steps_per_wavelength
    if mesh in ("graded", "axis_uniform"):

        def edges(length, displacement):
            u = np.linspace(0, 1, int(np.ceil(length / spacing)) + 1)
            return length * u + displacement * np.sin(2 * np.pi * u)

        strength = float(mesh == "graded")
        grid = RectilinearGrid(
            edges(7 * um, strength * 0.21 * um),
            edges(7 * um, strength * 0.15 * um),
            edges(6 * um, strength * 0.11 * um),
        )
        model = dict(
            material_grid=MaterialGrid(
                permittivity=np.ones(grid.shape_zyx, dtype=np.float32),
                conductivity=0.0,
                permeability=1.0,
                resolution=grid.minimum_spacing,
                shape=grid.shape_zyx,
                grid=grid,
            ),
        )
    else:
        model = dict(
            design=Design(
                width=7.0 * um,
                height=7.0 * um,
                depth=6.0 * um,
                material=Material(permittivity=1.0),
            ),
            grid_spec=(
                GridSpec.auto(
                    wavelength=wavelength,
                    courant=0.48,
                    snapping_points=((3.1 * um, 3.7 * um, 2.61 * um),),
                )
                if mesh == "auto"
                else GridSpec.uniform(spacing, courant=0.48)
            ),
        )
    sim = Simulation(
        **model,
        sources=[source],
        monitors=monitors,
        boundaries=[PML(thickness=0.6 * um)],
        run_time=260e-15,
    )

    if mesh != "uniform":
        assert sim.grid.metric_kind == (
            "axis_uniform" if mesh == "axis_uniform" else "rectilinear"
        )
    results = sim.run(backend=backend, progress=False, performance=False)
    assert results is not None
    if backend == "cuda":
        assert results.cuda_plan["binding"] == "native", results.cuda_plan
    forward_result = results.monitors["forward"]
    backward_result = results.monitors["backward"]
    assert forward_result is not None
    assert backward_result is not None
    forward = -float(np.asarray(forward_result.flux)[0])
    backward = float(np.asarray(backward_result.flux)[0])
    assert results.launched_power() == pytest.approx(source.power)
    sim.clear_compiled_cache()
    return forward, backward


@pytest.fixture(scope="module")
def empty_space_gaussian_flux():
    return _empty_space_gaussian_flux()


def test_gaussian_beam_source_is_forward_directional(empty_space_gaussian_flux):
    forward, backward = empty_space_gaussian_flux

    assert forward > 0.0
    assert backward >= 0.0
    assert backward / forward < 0.05


@pytest.mark.parametrize("angle_deg", [0.0, 14.5, 25.0])
def test_gaussian_beam_is_forward_directional_on_graded_grid(angle_deg):
    forward, backward = _empty_space_gaussian_flux(
        mesh="graded",
        angle_theta=np.deg2rad(angle_deg),
        angle_phi=0.7,
    )

    assert forward == pytest.approx(1.0, rel=0.15)
    assert backward >= 0.0
    assert backward / forward < 0.05


def test_gaussian_beam_uses_the_realized_automatic_grid():
    source = GaussianBeamSource(
        center=(1.2 * um, 1.2 * um, 1.95 * um),
        size=(1.4 * um, 1.4 * um),
        source_time=_pulse(1.55 * um),
        direction="-z",
    )
    design = Design(width=2.4 * um, height=2.4 * um, depth=2.8 * um)
    grid_spec = GridSpec.auto(
        wavelength=1.55 * um,
        snapping_points=((1.07 * um, 0.92 * um, 1.43 * um),),
    )
    sim = Simulation(
        design=design, grid_spec=grid_spec, sources=[source], run_time=4e-15
    )
    assert sim.grid == grid_spec.realize(design)
    assert sim.grid.metric_kind == "rectilinear"
    assert sim.compile(backend="jax").sources


def test_grating_angle_gaussian_beam_source_is_forward_directional():
    forward, backward = _empty_space_gaussian_flux(
        angle_theta=np.deg2rad(14.5), angle_phi=np.pi
    )

    assert forward == pytest.approx(1.0, rel=0.15)
    assert backward >= 0.0
    assert backward / forward < 0.06


def test_gaussian_beam_flux_is_grid_convergent(empty_space_gaussian_flux):
    coarse_forward, coarse_backward = empty_space_gaussian_flux
    fine_forward, fine_backward = _empty_space_gaussian_flux(steps_per_wavelength=12)

    assert fine_forward == pytest.approx(coarse_forward, rel=0.25)
    assert fine_backward / fine_forward < coarse_backward / coarse_forward


def test_gaussian_beam_requested_power_matches_normalized_forward_flux(
    empty_space_gaussian_flux,
):
    forward, _backward = empty_space_gaussian_flux
    assert forward == pytest.approx(1.0, rel=0.15)


def test_gaussian_beam_source_grating_coupler_like_compile_smoke():
    wavelength = 1.55 * um
    dx = 0.35 * um
    dt = 0.45 * dx / (LIGHT_SPEED * np.sqrt(3.0))
    silicon = Material(permittivity=12.0)
    design = Design(
        width=3.0 * um,
        height=2.0 * um,
        depth=2.4 * um,
        material=Material(permittivity=1.0),
    )
    design += Box(
        center=(1.5 * um, 1.0 * um, 0.55 * um),
        size=(1.8 * um, 0.5 * um, 0.22 * um),
        material=silicon,
    )
    for idx in range(3):
        design += Box(
            center=((1.0 + 0.28 * idx) * um, 1.0 * um, 0.75 * um),
            size=(0.12 * um, 0.5 * um, 0.12 * um),
            material=silicon,
        )
    source = GaussianBeamSource(
        center=(1.5 * um, 1.0 * um, 1.65 * um),
        size=(1.5 * um, 1.1 * um),
        source_time=_pulse(wavelength),
        direction="-z",
        angle_theta=0.25,
        angle_phi=0.0,
        pol_angle=0.0,
        waist_radius=0.55 * um,
    )
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[],
        boundaries=[],
        time=np.arange(0.0, 3 * dt, dt),
        resolution=dx,
    )

    program = cast(Any, sim.compile(num_steps=1))

    assert program.sources
    assert any(spec.component.startswith("E") for spec in program.sources)
    assert any(spec.component.startswith("H") for spec in program.sources)
