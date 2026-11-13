"""Mode sources and monitors in magnetic (tensor-permeability) media.

Oracles: electromagnetic duality (eps <-> mu maps E -> H, H -> -E), analytic
slab dispersion, the actual leapfrog update, and static-equivalent guides for
magnetic tensor poles. Each check fails by a wide margin if a port ignores mu
(measured during development): 2D TM modal/flux falls to 0.30-0.82, 3D ports
without the tensor fall to 0.78-0.82, and a 3D launch without the mu^-1 map of
its incident H increment carries 68% excess modal power.
"""

from __future__ import annotations

from functools import lru_cache
from typing import cast

import numpy as np
import pytest
from scipy.optimize import brentq

import beamz as bz
from beamz.const import LIGHT_SPEED
from beamz.design.discretization import MaterialGrid
from beamz.design.raster import (
    Box,
    Grid,
    Material,
    Object,
    RasterOptions,
    Scene,
    rasterize,
)
from beamz.devices.modes.plane import solve_mode_plane_3d
from beamz.devices.sources.planar_tfsf import (
    advance_incident_e_3d,
    advance_incident_h_3d,
    build_incident_3d_phasor_state,
)
from beamz.devices.sources.specs import FieldProfile3D

FREQUENCY = LIGHT_SPEED / 1e-6
U = 1e-6
# A full symmetric positive-definite permeability tensor and its dual partner.
MU = ((1.6, 0.25, 0.1), (0.25, 1.4, 0.2), (0.1, 0.2, 1.3))
ISO = ((3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 3.0))


def _guide_2d(core, polarization, monitors=(), freqs=(FREQUENCY,)):
    spacing = 50e-9
    design = bz.Design(width=6e-6, height=3e-6, material=bz.Material(1.0))
    design += bz.Rectangle(
        position=(0.0, 1.3e-6), width=6e-6, height=0.4e-6, material=core
    )
    dt = 0.8 * spacing / (LIGHT_SPEED * np.sqrt(2))
    center, size = (4.5e-6, 1.5e-6, 0.0), (0.0, 2.0e-6, 1.0e-6)
    spec = bz.ModeSpec(polarization=polarization)
    return bz.Simulation(
        design=design,
        resolution=spacing,
        time=np.arange(1400) * dt,
        polarization=polarization,
        sources=[
            bz.ModeSource(
                center=(1.2e-6, 1.5e-6, 0.0),
                size=(0.0, 2.0e-6, 1.0e-6),
                direction="+",
                mode_spec=spec,
                source_time=bz.GaussianPulse(freq0=FREQUENCY, fwidth=FREQUENCY / 4),
            )
        ],
        monitors=list(monitors)
        or [
            bz.ModeMonitor(
                center=center,
                size=size,
                mode_spec=spec,
                freqs=np.asarray(freqs),
                name="port",
            ),
            bz.FluxMonitor(
                center=center, size=size, freqs=np.asarray(freqs), name="flux"
            ),
        ],
        boundaries=[bz.PML(thickness=0.5e-6, formulation="cpml")],
        normalize_source=None,
    )


def _modal_and_flux(result):
    modal = result.mode("port").amps.sel(direction="+", mode_index=0).values
    flux = np.asarray(result.monitor("flux").get_dft_flux()).ravel()
    return np.abs(modal) ** 2, np.abs(flux)


@pytest.mark.parametrize(
    ("magnetic", "dielectric"),
    [("tm", "te"), ("te", "tm")],
    ids=["tm-mu-vs-te-eps", "te-mu-vs-tm-eps"],
)
def test_2d_magnetic_core_ports_are_dual_to_dielectric_core_ports(magnetic, dielectric):
    """A mu=4 core in one polarization is the exact dual of an eps=4 core in
    the other: launched modal power and modal/flux ratio must agree."""
    freqs = FREQUENCY * np.array([0.9, 1.0, 1.1])
    results = [
        _modal_and_flux(
            _guide_2d(core, polarization, freqs=freqs).run(
                backend="jax", performance=False
            )
        )
        for core, polarization in (
            (bz.AnisotropicMedium(permittivity=1.0, permeability=4.0), magnetic),
            (bz.Material(4.0), dielectric),
        )
    ]
    (modal, flux), (dual_modal, dual_flux) = results
    # Measured: <=0.5% (power) and <=0.15% (ratio); ignoring mu gives 0.30-0.82.
    np.testing.assert_allclose(modal, dual_modal, rtol=1e-2)
    np.testing.assert_allclose(modal / flux, dual_modal / dual_flux, rtol=4e-3)


def _slab_neff(eps_zz, mu_xx, mu_yy, half_width, k0):
    """Fundamental even TM (Ez) mode of a symmetric slab in vacuum."""

    def mismatch(n):
        beta = n * k0
        ky = np.sqrt(mu_xx * (k0**2 * eps_zz - beta**2 / mu_yy))
        return ky / mu_xx * np.tan(ky * half_width) - np.sqrt(beta**2 - k0**2)

    cutoff = mu_yy * (eps_zz - (0.5 * np.pi / (half_width * k0)) ** 2 / mu_xx)
    return float(
        cast(
            float,
            brentq(
                mismatch,
                np.sqrt(max(1.0, cutoff)) + 1e-9,
                np.sqrt(eps_zz * mu_yy) - 1e-9,
            ),
        )
    )


@pytest.mark.parametrize("mu", [(1.5, 3.0, 1.0), (3.0, 1.5, 1.0)])
def test_2d_anisotropic_permeability_slab_matches_analytic_dispersion(mu):
    """mu_xx and mu_yy play distinct roles: swapping them moves neff by 0.46."""
    simulation = _guide_2d(
        bz.AnisotropicMedium(permittivity=2.0, permeability=mu), "tm"
    )
    source = cast(bz.ModeSource, simulation.sources[0])
    data = source.solve_modes(simulation, freqs=[FREQUENCY])
    neff = float(np.real(np.asarray(data.neffs)).ravel()[0])
    exact = _slab_neff(2.0, mu[0], mu[1], 0.2e-6, 2 * np.pi * FREQUENCY / LIGHT_SPEED)
    # Discretization error at 8 cells across the core is ~0.5%.
    assert abs(neff - exact) < 0.015, (neff, exact)


def test_2d_tm_port_rejects_in_plane_permeability_coupling():
    from beamz.design.raster import RasterOptions

    simulation = _guide_2d(
        bz.AnisotropicMedium(permeability=(2.0, 3.0, 1.0, 0.3, 0.0, 0.0)), "tm"
    ).updated_copy(
        monitors=[], raster_options=RasterOptions(smoothing="farjadpour_full")
    )
    with pytest.raises(ValueError, match=r"off-diagonal \(xy\) permeability"):
        simulation.compile(backend="jax")


@lru_cache(maxsize=8)
def _guide_3d(case, axis="x", cells=16):
    eps, mu = {"mu": (ISO, MU), "dual": (MU, ISO)}[case]
    axes = {"x": "xyz", "y": "yxz", "z": "zxy"}[axis]
    order = [axes.index(a) for a in "xyz"]

    def oriented(tensor):
        return tuple(map(tuple, np.asarray(tensor)[np.ix_(order, order)]))

    length = np.array([4.0, 2.0, 2.0])[order] * U
    scene = Scene(
        (Material(), Material(epsilon_r=oriented(eps), mu_r=oriented(mu))),
        (
            Object(
                Box(
                    tuple(np.array([0.0, 0.6, 0.6])[order] * U),
                    tuple(np.array([4.0, 1.4, 1.4])[order] * U),
                ),
                1,
            ),
        ),
    )
    materials = MaterialGrid.from_raster_result(
        rasterize(
            scene,
            Grid.uniform(
                (0, 0, 0), tuple(length), tuple(np.array([2, 1, 1])[order] * cells)
            ),
            options=RasterOptions(smoothing="farjadpour_full"),
        )
    )
    spacing = 2 * U / cells
    return materials, spacing, 0.4 * spacing / LIGHT_SPEED


def _mode_3d(case, axis="x", cells=16):
    materials, spacing, dt = _guide_3d(case, axis, cells)
    plane = round(1e-6 / spacing)
    return solve_mode_plane_3d(
        np.asarray(materials.permittivity),
        np.asarray(materials.permeability),
        material_tensors=materials.tensors,
        yee_materials=materials.yee_materials,
        yee_tensors=materials.yee_tensors,
        magnetic_tensors=materials.magnetic_tensors,
        frequency=LIGHT_SPEED / (1.55 * U),
        resolution=spacing,
        dt=dt,
        axis=axis,
        direction="+" + axis,
        grid_shape=materials.shape,
        center=(1e-6,) * 3,
        width=1.6e-6,
        height=1.6e-6,
        plane_index=plane,
        offset_index=plane - 1,
        mode_index=0,
        polarization=None,
        target_neff=2.0,
        num_modes=3,
        snapped_region=None,
        solver_padding_cells=cells,
    )


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_3d_full_permeability_tensor_port_satisfies_actual_leapfrog_update(axis):
    """The coupled mode must be an eigenvector of the propagation's update,
    including its mu^-1 map (H-site diagonals and cell-centred cross terms).
    Solving without the magnetic tensors leaves residuals up to 0.46."""
    materials, spacing, dt = _guide_3d("mu", axis)
    frequency = LIGHT_SPEED / (1.55 * U)
    mode = _mode_3d("mu", axis)
    assert mode.diagnostics["full_tensor_operator"]
    assert cast(float, mode.diagnostics["tensor_faraday_residual"]) < 1e-7
    assert cast(float, mode.diagnostics["tensor_ampere_residual"]) < 1e-7
    # Rotated ports share the canonical x-normal problem.
    assert abs(mode.neff - _mode_3d("mu", "x").neff) < 1e-9
    fields = (
        bz.Simulation(
            material_grid=materials, time=np.arange(3) * dt, setup_device="cpu"
        )
        .compile()
        .grid
    )
    state = build_incident_3d_phasor_state(
        FieldProfile3D(
            components=mode.profiles,
            indices=mode.component_indices,
            axis=mode.axis,
            direction_sign=1.0,
            omega=2 * np.pi * frequency,
            k_axis=mode.k_num_axis,
            phase_ref_coord=mode.phase_reference_coord,
            phase_plane_coord=mode.phase_plane_coord,
            power_weights=mode.integration_weights,
        ),
        fields,
        resolution=spacing,
        t_e=0,
        t_h=-0.5 * dt,
        masked=False,
        max_shift=8,
    )
    h = advance_incident_h_3d(fields, state, dt, resolution=spacing)
    e = advance_incident_e_3d(fields, state, h, dt, resolution=spacing)
    selection = [slice(2, -2)] * 3
    selection[{"z": 0, "y": 1, "x": 2}[axis]] = slice(8, 12)
    for name, values in {**e, **h}.items():
        target = state[name] * np.exp(2j * np.pi * frequency * dt)
        relative = np.linalg.norm((values - target)[tuple(selection)]) / np.linalg.norm(
            target[tuple(selection)]
        )
        assert relative < 3e-6, (axis, name, relative)


def test_3d_magnetic_tensor_guide_is_dual_to_dielectric_tensor_guide():
    """(eps=3, mu=M) and (eps=M, mu=3) share every continuum mode. On the Yee
    lattice they converge to it from either side (gap 0.0084 at 16 cells,
    0.0036 at 24); dropping the mu cross terms moves neff by 0.16."""
    magnetic, dielectric = _mode_3d("mu"), _mode_3d("dual")
    assert magnetic.diagnostics["full_tensor_operator"]
    assert abs(magnetic.neff - dielectric.neff) < 0.015


def _ports_3d(case):
    spacing = 100e-9
    length = (4.0 * U, 3.0 * U, 3.0 * U)
    eps, mu = {"mu": (ISO, MU), "dual": (MU, ISO)}[case]
    scene = Scene(
        (Material(), Material(epsilon_r=eps, mu_r=mu)),
        (Object(Box((0.0, 1.1 * U, 1.1 * U), (4.0 * U, 1.9 * U, 1.9 * U)), 1),),
    )
    materials = MaterialGrid.from_raster_result(
        rasterize(
            scene,
            Grid.uniform(
                (0, 0, 0),
                length,
                (
                    round(length[0] / spacing),
                    round(length[1] / spacing),
                    round(length[2] / spacing),
                ),
            ),
            options=RasterOptions(smoothing="farjadpour_full"),
        )
    )
    f0 = LIGHT_SPEED / (1.55 * U)
    freqs = f0 * np.array([0.95, 1.0, 1.05])
    center, size = (3.0 * U, 1.5 * U, 1.5 * U), (0.0, 2.0 * U, 2.0 * U)
    simulation = bz.Simulation(
        material_grid=materials,
        time=np.arange(1100) * 0.5 * spacing / LIGHT_SPEED,
        sources=[
            bz.ModeSource(
                center=(1.0 * U, 1.5 * U, 1.5 * U),
                size=(0.0, 2.0 * U, 2.0 * U),
                direction="+",
                mode_spec=bz.ModeSpec(),
                source_time=bz.GaussianPulse(freq0=f0, fwidth=f0 / 5),
            )
        ],
        monitors=[
            bz.ModeMonitor(
                center=center,
                size=size,
                mode_spec=bz.ModeSpec(),
                freqs=freqs,
                name="port",
            ),
            bz.FluxMonitor(center=center, size=size, freqs=freqs, name="flux"),
        ],
        boundaries=[bz.PML(thickness=0.5 * U, formulation="cpml")],
        normalize_source=None,
        setup_device="cpu",
    )
    return _modal_and_flux(simulation.run(backend="jax", performance=False))


def test_3d_mode_ports_in_full_permeability_tensor_core():
    """A ModeSource and ModeMonitor inside a full-mu-tensor core: the launch is
    pure (modal power equals flux) and matches the dual dielectric-tensor
    guide's launched power (measured agreement ~3e-4 on both)."""
    modal, flux = _ports_3d("mu")
    dual_modal, dual_flux = _ports_3d("dual")
    np.testing.assert_allclose(modal / flux, dual_modal / dual_flux, rtol=2e-3)
    np.testing.assert_allclose(modal / flux, 1.0, atol=5e-3)
    np.testing.assert_allclose(modal, dual_modal, rtol=5e-3)


def _magnetic_pole_core(omega):
    # mu(f0) = 2 + (16/9) * 9 / (9 - 1) = 4, mirroring the electric-pole tests.
    return bz.AnisotropicMedium(
        1.0,
        permeability=2.0,
        magnetic_poles=[
            bz.TensorPole.lorentz(16.0 / 9.0, resonance=3 * omega, damping=0.0)
        ],
    )


def test_mode_ports_in_magnetic_pole_core_fold_mu_per_frequency(tmp_path):
    """Ports inside a magnetic Lorentz core use mu_inf + Re chi_m(f) at each
    frequency, also after a result-archive round trip."""
    from beamz.design.material_views import static_material_grid
    from beamz.simulation.result_io import load_results, save_results

    core = _magnetic_pole_core(2 * np.pi * FREQUENCY)
    np.testing.assert_allclose(core.permeability_tensor(FREQUENCY)[2, 2], 4.0)
    freqs = FREQUENCY * np.array([0.9, 1.1])
    dispersive = _guide_2d(core, "tm", freqs=freqs)
    for f in (*freqs, FREQUENCY):
        mu_f = float(core.permeability_tensor(f)[0, 0].real)
        static = _guide_2d(bz.AnisotropicMedium(1.0, permeability=mu_f), "tm")
        view = static_material_grid(dispersive._material_grid(), f)
        np.testing.assert_allclose(
            np.asarray(view.permeability),
            np.asarray(static._material_grid().permeability),
        )
        np.testing.assert_allclose(
            np.asarray(view.magnetic_tensors["mu_cell"])[:3],
            np.broadcast_to(
                np.asarray(static._material_grid().magnetic_tensors["mu_cell"]),
                np.asarray(view.magnetic_tensors["mu_cell"])[:3].shape,
            ),
        )
    result = dispersive.run(backend="jax", performance=False)
    path = tmp_path / "result.npz"
    save_results(result, path)
    loaded = load_results(path)
    region = loaded.monitor("port").material_region
    original = result.monitor("port").material_region
    assert region is not None and original is not None
    assert region.dispersion_folding is not None
    assert region.dispersion_folding.magnetic_poles
    np.testing.assert_allclose(
        region.at_frequency(freqs[0]).material_tensors["mu"],
        original.at_frequency(freqs[0]).material_tensors["mu"],
    )
    np.testing.assert_allclose(
        loaded.mode("port").amps.values, result.mode("port").amps.values
    )
    modal, flux = _modal_and_flux(result)
    for index, f in enumerate(freqs):
        mu_f = float(core.permeability_tensor(f)[0, 0].real)
        static_modal, static_flux = _modal_and_flux(
            _guide_2d(
                bz.AnisotropicMedium(1.0, permeability=mu_f), "tm", freqs=freqs
            ).run(backend="jax", performance=False)
        )
        # Measured 2.6e-3 and 1.3e-3: folded H-site interface coefficients
        # differ slightly from a freshly smoothed static raster.
        np.testing.assert_allclose(
            modal[index] / flux[index],
            static_modal[index] / static_flux[index],
            rtol=5e-3,
        )


def test_mode_source_in_magnetic_pole_core_matches_static_equivalent():
    """TE launches through mu_zz(f0): the transmitted Hz equals that of the
    static mu(f0) guide (measured 0.36%)."""

    def transmitted(core):
        simulation = _guide_2d(
            core,
            "te",
            monitors=[
                bz.FieldMonitor(
                    center=(4.5e-6, 1.5e-6, 0.0),
                    size=(0.0, 3e-6, 0.0),
                    freqs=np.array([FREQUENCY]),
                    fields=("Hz",),
                    name="line",
                )
            ],
        )
        result = simulation.run(backend="jax", performance=False)
        return np.asarray(result.monitor("line").dft_fields["Hz"]).ravel()

    dispersive = transmitted(_magnetic_pole_core(2 * np.pi * FREQUENCY))
    static = transmitted(bz.AnisotropicMedium(1.0, permeability=4.0))
    error = np.linalg.norm(dispersive - static) / np.linalg.norm(static)
    assert error < 0.01, error
