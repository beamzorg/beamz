"""Physical power, directionality and aperture checks on graded Yee grids."""

import numpy as np
import pytest

import beamz as bz


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_graded_plane_wave_power_and_backward_leakage(axis, sign):
    normal = "xyz".index(axis)
    size = np.full(3, 160e-9)
    size[normal] = 2.4e-6
    source_center = np.zeros(3)
    source_center[normal] = -sign * 0.7e-6
    aperture = size.copy()
    aperture[normal] = 0
    override_size = size.copy()
    override_size[normal] = 0.5e-6
    spacing = [None] * 3
    spacing[normal] = 8e-9
    back = np.zeros(3)
    back[normal] = -sign * 0.85e-6
    edges = {"x": ("left", "right"), "y": ("bottom", "top"), "z": ("front", "back")}[
        axis
    ]
    sim = bz.Simulation(
        size=tuple(size),
        sources=[
            bz.PlaneWaveSource(
                center=tuple(source_center),
                size=tuple(aperture),
                direction=("+" if sign > 0 else "-") + axis,
                source_time=bz.GaussianPulse(5.75e14, 2e14),
            )
        ],
        monitors=[
            bz.FluxMonitor(
                center=tuple(center),
                size=tuple(aperture),
                freqs=[4.5e14, 5.5e14, 7e14],
                name=name,
            )
            for center, name in [(np.zeros(3), "forward"), (back, "backward")]
        ],
        boundaries=[
            bz.Periodic(axes=tuple(a for a in "xyz" if a != axis)),
            bz.PML(edges=edges, thickness=0.3e-6),
        ],
        grid_spec=bz.GridSpec.auto(
            wavelength=500e-9,
            courant=0.7,
            dl_max=20e-9,
            overrides=(
                bz.MeshOverride(
                    center=(0, 0, 0), size=tuple(override_size), dl=tuple(spacing)
                ),
            ),
        ),
        run_time=100e-15,
    )
    assert sim.grid.metric_kind == "rectilinear"
    result = sim.run(backend="jax", progress=False)
    np.testing.assert_allclose(sign * result["forward"].flux, 1, atol=0.02)
    assert np.max(abs(result["backward"].flux)) < 1e-4


def test_partial_cell_monitor_aperture_area():
    # Uniform irradiance must integrate over the requested aperture, including
    # fractional edge cells, rather than the enclosing rectangle of grid cells.
    from types import SimpleNamespace

    from beamz.lattice import compile_yee_plane_quadrature_3d, component_shape_3d

    grid = bz.RectilinearGrid.from_spacing((8, 8, 8), 20e-9)
    region = SimpleNamespace(axis_interval=lambda axis: slice(1, 7))
    q = compile_yee_plane_quadrature_3d(
        center=(80e-9, 80e-9, 80e-9),
        size=(103e-9, 97e-9, 0),
        normal_axis="z",
        region=region,
        resolution=20e-9,
        grid_shape=(8, 8, 8),
        component_shapes={
            c: component_shape_3d(c, (8, 8, 8))
            for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        },
        grid=grid,
    )
    np.testing.assert_allclose(q.integration_weights.sum(), 103e-9 * 97e-9, rtol=1e-14)


def test_centered_mesh_override_tracks_structure_coordinates():
    sim = bz.Simulation(
        size=(0.4e-6, 0.4e-6, 2e-6),
        grid_spec=bz.GridSpec.auto(
            wavelength=500e-9,
            dl_max=50e-9,
            overrides=(
                bz.MeshOverride(
                    center=(0, 0, 0.35e-6),
                    size=(0.4e-6, 0.4e-6, 0.2e-6),
                    dl=(None, None, 8e-9),
                    enforced=True,
                ),
            ),
        ),
        run_time=1e-15,
    )
    centers = sim.grid.centers("z") - 1e-6
    widths = sim.grid.cell_widths("z")
    assert widths[(centers > 0.25e-6) & (centers < 0.45e-6)].max() <= 8e-9 * 1.001
    assert widths[centers < -0.4e-6].max() > 20e-9


def test_periodic_dispersive_seam_uses_physical_support_volumes():
    from beamz.simulation.dispersion import periodic_support_average

    grid = bz.RectilinearGrid(
        np.array([0, 1, 3]) * 1e-9,
        np.array([0, 1, 2]) * 1e-9,
        np.array([0, 1, 2]) * 1e-9,
    )
    values = np.zeros((2, 2, 3))
    values[:, :, -1] = 1
    joined = periodic_support_average(values, (2, 2, 2), frozenset({2}), grid)
    np.testing.assert_allclose(joined[:, :, 0], 2 / 3)
    np.testing.assert_allclose(joined[:, :, -1], 2 / 3)


def test_reference_models_preserve_embedded_coefficients_and_passbands():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    data = root / "examples/data/cmos_rgb"
    snapshot = json.loads((data / "tidy3d_reference_simulation.json").read_text())
    cases = {
        "red": "red filter",
        "green": "green1 filter",
        "blue": "blue filter",
        "Al_Rakic1995": "metal shield",
        "aSi_Horiba": "silicon layer",
        "SiN_Horiba": "anti-reflection",
        "SiO2_Palik_LowLoss": "len blue",
    }
    models = json.loads((data / "filters.json").read_text()) | json.loads(
        (data / "materials.json").read_text()
    )
    frequencies = np.linspace(bz.LIGHT_SPEED / 700e-9, bz.LIGHT_SPEED / 400e-9, 200)
    for name, structure in cases.items():
        medium = next(
            s["medium"] for s in snapshot["structures"] if s["name"] == structure
        )
        s = -2j * np.pi * frequencies
        exact = np.full(200, medium["eps_inf"], dtype=complex)
        for a, c in medium["poles"]:
            a, c = complex(**a), complex(**c)
            exact += c / (s - a) + c.conjugate() / (s - a.conjugate())
        model = bz.PoleResidue.from_spec(models[name])
        np.testing.assert_allclose(model.eps_model(frequencies), exact, rtol=1e-12)
        if name in ("red", "green", "blue"):
            wl, n, k = np.loadtxt(data / f"{name}_eps.csv", delimiter=",").T
            fit = np.sqrt(model.eps_model(bz.LIGHT_SPEED / (wl * 1e-6)))
            assert fit.imag[k < 0.001].max() < 0.02


def test_empty_reference_reuses_sensor_grid_and_time():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "examples/cmos_rgb_sensor.py"
    spec = importlib.util.spec_from_file_location("sensor_reference_grid", path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    policy = model.sensor_grid(lateral_nm=150, metal_nm=50, bulk_nm=200)
    sensor = model.make_sim(grid_spec=policy, run_time=1e-15, fields=False)
    reference = model.make_sim(
        grid_spec=policy, run_time=1e-15, reference=True, fields=False
    )
    for a, b in zip(sensor.grid.edges, reference.grid.edges, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(sensor.time, reference.time)
    assert reference.sources[0].center == sensor.sources[0].center


def test_rectilinear_source_rejects_dispersive_injection_sheet():
    medium = bz.PoleResidue.lorentz(
        1.0, strength=0.5, resonance=6e15, damping=1e14, frequency_range=(4e14, 8e14)
    )
    sim = bz.Simulation(
        size=(0.2e-6, 0.2e-6, 1e-6),
        background=medium,
        sources=[
            bz.PlaneWaveSource(
                center=(0, 0, 0.2e-6),
                size=(0.2e-6, 0.2e-6, 0),
                source_time=bz.GaussianPulse(6e14, 2e14),
                direction="-z",
            )
        ],
        grid_spec=bz.GridSpec.auto(
            wavelength=500e-9,
            dl_max=30e-9,
            overrides=(
                bz.MeshOverride(
                    center=(0, 0, 0),
                    size=(0.2e-6, 0.2e-6, 0.2e-6),
                    dl=(None, None, 10e-9),
                ),
            ),
        ),
        run_time=1e-15,
    )
    with pytest.raises(ValueError, match="lossless nondispersive"):
        sim.compile(backend="jax")


def test_uniform_cell_center_quadrature_has_exact_aperture_area():
    from types import SimpleNamespace

    from beamz.lattice import compile_yee_plane_quadrature_3d, component_shape_3d

    region = SimpleNamespace(axis_interval=lambda axis: slice(1, 7))
    q = compile_yee_plane_quadrature_3d(
        center=(80e-9, 80e-9, 80e-9),
        size=(103e-9, 97e-9, 0),
        normal_axis="z",
        region=region,
        resolution=20e-9,
        grid_shape=(8, 8, 8),
        component_shapes={
            c: component_shape_3d(c, (8, 8, 8))
            for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        },
    )
    np.testing.assert_allclose(
        q.sample_area * q.point_count, 103e-9 * 97e-9, rtol=1e-14
    )
