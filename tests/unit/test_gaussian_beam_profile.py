import numpy as np
import pytest

from beamz.const import EPS_0, MU_0
from beamz.design.grid import RectilinearGrid
from beamz.devices.sources.compiler import GaussianBeamProfile
from beamz.devices.sources.specs import FieldProfile3D
from beamz.lattice import component_coordinates_rectilinear


def test_gaussian_beam_profile_has_transverse_gaussian_envelope():
    beam = GaussianBeamProfile(
        center=(10.0, 10.0, 10.0),
        size=(6.0, 6.0),
        direction="+x",
        angle_theta=0.0,
        angle_phi=0.0,
        pol_angle=0.0,
        waist_radius=1.0,
        waist_distance=0.0,
        wavelength=5.0,
    )

    profile = beam.field_profile(resolution=0.25, grid_shape=(80, 80, 80))

    assert isinstance(profile, FieldProfile3D)
    assert profile.axis == "x"
    assert profile.direction_sign == 1.0
    ey = np.abs(profile.components["Ey"])
    assert ey.shape[0] > 8 and ey.shape[1] > 8
    center_value = float(np.max(ey))
    edge_value = float(max(np.max(ey[0, :]), np.max(ey[-1, :])))
    assert center_value > 0.0
    assert edge_value < 0.02 * center_value


def test_gaussian_beam_profile_vectors_are_orthogonal_to_each_other_and_k():
    beam = GaussianBeamProfile(
        center=(8.0, 8.0, 8.0),
        size=(5.0, 5.0),
        direction="+z",
        angle_theta=0.31,
        angle_phi=0.47,
        pol_angle=0.62,
        waist_radius=1.4,
        waist_distance=0.3,
        wavelength=3.0,
        background_index=1.5,
    )

    k_hat = beam.propagation_unit_vector()
    e_hat = beam.electric_unit_vector()
    h_hat = beam.magnetic_unit_vector()

    np.testing.assert_allclose(np.linalg.norm(k_hat), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(e_hat), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(h_hat), 1.0, atol=1e-12)
    assert abs(float(np.dot(k_hat, e_hat))) < 1e-12
    assert abs(float(np.dot(k_hat, h_hat))) < 1e-12
    assert abs(float(np.dot(e_hat, h_hat))) < 1e-12
    assert float(np.dot(np.cross(e_hat, h_hat), k_hat)) == pytest.approx(1.0)

    profile = beam.field_profile(resolution=0.25, grid_shape=(64, 64, 64))
    assert profile.k_axis == pytest.approx(beam.propagation_vector()[2])


def test_gaussian_beam_profile_transverse_phase_matches_tilt_direction():
    beam = GaussianBeamProfile(
        center=(10.0, 10.0, 10.0),
        size=(7.0, 7.0),
        direction="+z",
        angle_theta=0.24,
        angle_phi=0.0,
        pol_angle=0.0,
        waist_radius=3.0,
        waist_distance=0.0,
        wavelength=10.0,
    )
    profile = beam.field_profile(resolution=0.25, grid_shape=(80, 80, 80))

    ex = profile.components["Ex"]
    y_mid = ex.shape[0] // 2
    row = ex[y_mid, :]
    x_slice = profile.indices["Ex"][2]
    assert isinstance(x_slice, slice)
    x_coords = (np.arange(x_slice.start, x_slice.stop) + 0.5) * 0.25
    phase = np.unwrap(np.angle(row))
    slope, _intercept = np.polyfit(x_coords, phase, deg=1)

    assert slope == pytest.approx(-beam.propagation_vector()[0], rel=2e-2, abs=2e-3)


def test_gaussian_beam_profile_power_normalization_is_sane():
    beam = GaussianBeamProfile(
        center=(10.0, 10.0, 10.0),
        size=(8.0, 8.0),
        direction="+x",
        angle_theta=0.0,
        angle_phi=0.0,
        pol_angle=0.0,
        waist_radius=1.5,
        waist_distance=0.0,
        wavelength=5.0,
        power=2.0,
    )
    resolution = 0.2
    profile = beam.field_profile(resolution=resolution, grid_shape=(100, 100, 100))

    ey = profile.components["Ey"]
    ez = profile.components["Ez"]
    hy = profile.components["Hy"]
    hz = profile.components["Hz"]
    flux = 0.5 * np.real(np.sum(ey * np.conjugate(hz) - ez * np.conjugate(hy)))
    power = float(flux * resolution**2)

    assert np.isfinite(power)
    assert power == pytest.approx(2.0, rel=0.15)


def test_grating_coupler_beam_direction_phase_and_polarization():
    theta = np.deg2rad(14.5)
    beam = GaussianBeamProfile(
        center=(5.0, 5.0, 5.0),
        size=(8.0, 8.0, 0.0),
        direction="-z",
        angle_theta=theta,
        angle_phi=np.pi,
        pol_angle=np.pi / 2.0,
        waist_radius=3.0,
        waist_distance=0.0,
        wavelength=10.0,
    )

    expected_k_hat = np.array((-np.sin(theta), 0.0, -np.cos(theta)))
    expected_e_hat = np.array((0.0, 1.0, 0.0))
    expected_h_hat = np.array((np.cos(theta), 0.0, -np.sin(theta)))
    np.testing.assert_allclose(
        beam.propagation_unit_vector(), expected_k_hat, atol=1e-12
    )
    np.testing.assert_allclose(beam.electric_unit_vector(), expected_e_hat, atol=1e-12)
    np.testing.assert_allclose(beam.magnetic_unit_vector(), expected_h_hat, atol=1e-12)

    resolution = 0.1
    profile = beam.field_profile(resolution=resolution, grid_shape=(100, 100, 100))
    ey = profile.components["Ey"]
    row = ey[ey.shape[0] // 2]
    x_slice = profile.indices["Ey"][2]
    assert isinstance(x_slice, slice)
    x = (np.arange(x_slice.start, x_slice.stop) + 0.5) * resolution
    phase_slope, _ = np.polyfit(x, np.unwrap(np.angle(row)), deg=1)
    assert phase_slope == pytest.approx(
        -beam.propagation_vector()[0], rel=5e-3, abs=2e-3
    )


@pytest.mark.parametrize("theta_degrees", (0.0, 14.5, 35.0, 45.0))
def test_unclipped_gaussian_power_normalization_is_angle_invariant(theta_degrees):
    waist = 1.0
    power = 2.5
    background_index = 1.4
    beam = GaussianBeamProfile(
        center=(6.0, 6.0, 6.0),
        size=(10.0, 10.0, 0.0),
        direction="-z",
        angle_theta=np.deg2rad(theta_degrees),
        angle_phi=0.37,
        pol_angle=0.81,
        waist_radius=waist,
        waist_distance=0.0,
        wavelength=3.0,
        background_index=background_index,
        power=power,
    )
    resolution = 0.05
    slices = beam._transverse_slices(resolution, (240, 240, 240))
    sampled_scale = beam._power_amplitude_scale(
        resolution,
        slices,
        k_normal_abs=abs(beam.propagation_unit_vector()[2]),
    )

    impedance = np.sqrt(MU_0 / EPS_0) / background_index
    infinite_plane_flux_per_amplitude_sq = 0.5 / impedance * np.pi * waist**2 / 2.0
    analytical_scale = np.sqrt(power / infinite_plane_flux_per_amplitude_sq)
    assert sampled_scale == pytest.approx(analytical_scale, rel=2e-3)


def test_gaussian_requested_power_scales_all_fields_by_square_root():
    common = dict(
        center=(4.0, 4.0, 4.0),
        size=(6.0, 6.0, 0.0),
        direction="-z",
        angle_theta=0.27,
        angle_phi=1.2,
        pol_angle=0.4,
        waist_radius=1.3,
        waist_distance=0.2,
        wavelength=4.0,
    )
    unit = GaussianBeamProfile(**common, power=1.0).field_profile(
        resolution=0.2, grid_shape=(40, 40, 40)
    )
    four_watts = GaussianBeamProfile(**common, power=4.0).field_profile(
        resolution=0.2, grid_shape=(40, 40, 40)
    )

    for component in unit.components:
        np.testing.assert_allclose(
            four_watts.components[component],
            2.0 * unit.components[component],
            rtol=1e-13,
            atol=1e-13,
        )


@pytest.mark.parametrize(
    ("aperture", "theta_degrees"),
    ((10.0, 14.5), (2.5, 35.0)),
    ids=("unclipped_grating_angle", "clipped_oblique_beam"),
)
def test_discrete_profile_flux_matches_requested_power(aperture, theta_degrees):
    requested_power = 2.5
    resolution = 0.05
    beam = GaussianBeamProfile(
        center=(6.0, 6.0, 6.0),
        size=(aperture, aperture, 0.0),
        direction="-z",
        angle_theta=np.deg2rad(theta_degrees),
        angle_phi=np.pi,
        pol_angle=np.pi / 2.0,
        waist_radius=1.0,
        waist_distance=0.0,
        wavelength=3.0,
        power=requested_power,
    )
    profile = beam.field_profile(resolution=resolution, grid_shape=(240, 240, 240))
    ex = profile.components["Ex"]
    ey = profile.components["Ey"]
    hx = profile.components["Hx"]
    hy = profile.components["Hy"]
    power_toward_negative_z = (
        -0.5
        * np.real(np.sum(ex * np.conjugate(hy) - ey * np.conjugate(hx)))
        * resolution**2
    )

    assert power_toward_negative_z == pytest.approx(requested_power, rel=1e-2)


def test_waist_distance_matches_paraxial_radius_and_curvature():
    waist = 1.1
    wavelength = 3.2
    waist_distance = 1.7
    beam = GaussianBeamProfile(
        center=(5.0, 5.0, 5.0),
        size=(8.0, 8.0, 0.0),
        direction="+z",
        angle_theta=0.0,
        angle_phi=0.0,
        pol_angle=0.0,
        waist_radius=waist,
        waist_distance=waist_distance,
        wavelength=wavelength,
    )
    radius, curvature, gouy = beam._beam_radius_curvature_gouy()
    rayleigh = np.pi * waist**2 / wavelength

    assert radius == pytest.approx(
        waist * np.sqrt(1.0 + (waist_distance / rayleigh) ** 2)
    )
    assert curvature == pytest.approx(
        waist_distance * (1.0 + (rayleigh / waist_distance) ** 2)
    )
    assert gouy == pytest.approx(np.arctan2(waist_distance, rayleigh))


@pytest.mark.parametrize("direction", ("+x", "-x", "+y", "-y", "+z", "-z"))
@pytest.mark.parametrize("theta", (0.0, 0.37))
@pytest.mark.parametrize("waist_distance", (-0.8, 0.0, 0.8))
def test_gaussian_phasor_matches_dense_formula_on_graded_grid(
    direction, theta, waist_distance
):
    grid = RectilinearGrid(
        np.linspace(0, 1, 19) ** 1.2 * 6,
        np.linspace(0, 1, 23) ** 1.4 * 7,
        np.linspace(0, 1, 17) ** 1.6 * 8,
    )
    beam = GaussianBeamProfile(
        center=(3.1, 3.3, 4.2),
        size=(5.0, 4.0),
        direction=direction,
        angle_theta=theta,
        angle_phi=0.59,
        pol_angle=0.81,
        waist_radius=1.1,
        waist_distance=waist_distance,
        wavelength=3.2,
        background_index=1.4,
    )
    profile = beam.field_profile(resolution=0.3, grid_shape=grid.shape_zyx, grid=grid)
    normal = "xyz".index(beam.axis)
    k_hat = beam.propagation_unit_vector()
    k_tangent = beam.propagation_vector()
    k_tangent[normal] = 0.0
    radius, curvature, gouy = beam._beam_radius_curvature_gouy()
    amplitude = beam._power_amplitude_scale(
        0.3,
        beam._transverse_slices(0.3, grid.shape_zyx, grid=grid),
        k_normal_abs=abs(k_hat[normal]),
        grid=grid,
    )
    electric = beam.electric_unit_vector()
    magnetic = beam.background_index / np.sqrt(MU_0 / EPS_0) * np.cross(k_hat, electric)
    for component, values in profile.components.items():
        index = profile.indices[component]
        coords = component_coordinates_rectilinear(component, grid)
        z, y, x = np.meshgrid(
            *(coords[axis][item] for axis, item in zip("zyx", index, strict=True)),
            indexing="ij",
        )
        displacement = np.stack((x, y, z), axis=-1) - beam.center
        rho2 = np.sum(np.cross(displacement, k_hat) ** 2, axis=-1)
        phase = (
            -(displacement @ k_tangent)
            - beam.medium_wavenumber * rho2 / (2 * curvature)
            + gouy
        )
        vector = electric if component.startswith("E") else magnetic
        factor = vector["xyz".index(component[1].lower())]
        expected = amplitude * factor * np.exp(-rho2 / radius**2 + 1j * phase)
        np.testing.assert_allclose(
            values, expected.squeeze(axis=2 - normal), rtol=2e-13, atol=2e-14
        )
